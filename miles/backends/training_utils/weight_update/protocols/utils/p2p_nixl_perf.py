"""NIXL P2P trainer-side perf log. Rank 0 appends one section per full weight transfer.

The metric definitions, loops and log format are specified in
`docs/advanced/nixl_weight_transfer_feature/p2p-nixl-perf-metrics.md`.
"""

import os
import time
from collections.abc import Iterator, Sequence
from concurrent.futures import Future
from contextlib import AbstractContextManager, contextmanager, nullcontext
from pathlib import Path
from typing import Any, NamedTuple

import torch.distributed as dist

from miles.backends.training_utils.weight_update.gpu_prep_timing import (
    GPU_PREP_CONVERT,
    GPU_PREP_CPU_LOAD,
    GPU_PREP_GATHER,
    GPU_PREP_GATHER_EP,
    GPU_PREP_GATHER_LOAD,
    GPU_PREP_GATHER_PP,
    GPU_PREP_GATHER_TP,
    GPU_PREP_GATHER_TP_CONCAT,
    GPU_PREP_GATHER_TP_START,
    GPU_PREP_GATHER_TP_WAIT,
    GPU_PREP_STAGE,
)
from miles.backends.training_utils.weight_update.protocols.transports.remote_shard import WriteStats
from miles.utils.distributed_utils import get_gloo_group

LOG_FILENAME = "p2p_nixl_perf.log"
# Decimal GB (10^9), matching how NIC line rates are quoted, so wire bytes can be
# divided by wire_time and compared against link bandwidth without a base change.
_GB = 1000**3
# The first load of every param runs the byte-coverage check (two extra loads), so version 1 is not steady state.
WARM_UP_WEIGHT_VERSION = 1

# Timed on the main thread, but not part of the gather/convert/stage/cpu_load split.
KIND_NEXT = "next"
KIND_CPU_PREP = "cpu_prep"
KIND_BUFFER_WAIT = "buffer_wait"
KIND_DRAIN = "drain"

# Printed order: the parent gather, its chunks, then convert / stage / cpu load.
GPU_PREP_PARTS = (
    GPU_PREP_GATHER,
    GPU_PREP_GATHER_LOAD,
    GPU_PREP_GATHER_PP,
    GPU_PREP_GATHER_TP,
    GPU_PREP_GATHER_TP_START,
    GPU_PREP_GATHER_TP_WAIT,
    GPU_PREP_GATHER_TP_CONCAT,
    GPU_PREP_GATHER_EP,
    GPU_PREP_CONVERT,
    GPU_PREP_STAGE,
    GPU_PREP_CPU_LOAD,
)
# `gather` is the sum of these. Their time is not wrapped by one clock: on the ETP == 1 expert path convert sits
# between them. The TP chunks are inside `gather_tp`, so they are not added again.
_GATHER_CHILDREN = (GPU_PREP_GATHER_LOAD, GPU_PREP_GATHER_PP, GPU_PREP_GATHER_TP, GPU_PREP_GATHER_EP)
_KINDS = frozenset((*GPU_PREP_PARTS, KIND_NEXT, KIND_CPU_PREP, KIND_BUFFER_WAIT, KIND_DRAIN))


class WireFold(NamedTuple):
    nbytes: int
    busy: float
    sum_groups: float
    queue: float


def fold_write_groups(groups: Sequence[tuple[float, Sequence[WriteStats]]]) -> WireFold:
    """Fold write groups, each `(t_submit_return, stats of its shards)`.

    `busy` is the wall clock with at least one write in flight (union of the intervals). `sum_groups` takes the
    longest shard of each group and sums the groups, the old serial definition. `queue` takes the shard that waited
    longest between `write()` returning and starting, and sums the groups.
    """
    all_stats = [stats for _, shard_stats in groups for stats in shard_stats]
    return WireFold(
        nbytes=sum(stats.nbytes for stats in all_stats),
        busy=_union_seconds([(stats.t_start, stats.t_end) for stats in all_stats]),
        sum_groups=sum(
            max((stats.t_end - stats.t_start for stats in shard_stats), default=0.0) for _, shard_stats in groups
        ),
        queue=sum(
            max((max(0.0, stats.t_start - t_submit_return) for stats in shard_stats), default=0.0)
            for t_submit_return, shard_stats in groups
        ),
    )


def _union_seconds(intervals: Sequence[tuple[float, float]]) -> float:
    total = 0.0
    merged_end = None
    merged_start = None
    for start, end in sorted(intervals):
        if merged_end is None or start > merged_end:
            if merged_end is not None:
                total += merged_end - merged_start
            merged_start, merged_end = start, end
        else:
            merged_end = max(merged_end, end)
    if merged_end is not None:
        total += merged_end - merged_start
    return total


class P2PNixlPerfCollector:
    """Per-rank counters for one `update_weights` NIXL P2P transfer.

    Everything is recorded on the main thread. The write threads only return `WriteStats` on their futures, which
    `finish` folds after every write has returned.
    """

    def __init__(self, *, gpu: int) -> None:
        self.gpu = gpu
        self.reset(0)

    def reset(self, weight_version: int) -> None:
        """Clear the counters; each full transfer is its own log section."""
        self.weight_version = weight_version
        self._seconds_by_kind: dict[str, float] = {}
        self._active_start: float | None = None
        self._active = 0.0
        self._write_groups: list[tuple[float, list[Future]]] = []
        self._fold = WireFold(0, 0.0, 0.0, 0.0)

    def add(self, kind: str, seconds: float) -> None:
        if kind not in _KINDS:
            raise ValueError(f"unknown perf kind: {kind}")
        self._seconds_by_kind[kind] = self._seconds_by_kind.get(kind, 0.0) + seconds

    @contextmanager
    def measure(self, kind: str):
        start = time.monotonic()
        try:
            yield
        finally:
            self.add(kind, time.monotonic() - start)

    def wrap_buckets(self, buckets: Iterator[Any]) -> Iterator[Any]:
        """Time each `next()` of the HF iterator as gpu_prep and start the active clock at the first one."""
        iterator = iter(buckets)
        while True:
            start = time.monotonic()
            if self._active_start is None:
                self._active_start = start
            try:
                bucket = next(iterator)
            except StopIteration:
                self.add(KIND_NEXT, time.monotonic() - start)
                return
            self.add(KIND_NEXT, time.monotonic() - start)
            yield bucket

    def record_write_group(self, writes: Sequence[Future]) -> None:
        """Call right after `transport.write` returns; `writes` are that group's per-shard futures."""
        self._write_groups.append((time.monotonic(), list(writes)))

    def finish(self) -> None:
        """Call right after every write has returned: stop the active clock and fold the write stats."""
        if self._active_start is not None:
            self._active = time.monotonic() - self._active_start
            self._active_start = None
        self._fold = fold_write_groups(
            [(t_submit_return, [write.result() for write in writes]) for t_submit_return, writes in self._write_groups]
        )
        self._write_groups = []

    def payload(self) -> dict[str, int | float]:
        seconds = self._seconds_by_kind
        parts = {part: seconds.get(part, 0.0) for part in GPU_PREP_PARTS}
        parts[GPU_PREP_GATHER] += sum(seconds.get(child, 0.0) for child in _GATHER_CHILDREN)
        payload: dict[str, int | float] = {
            "gpu": self.gpu,
            "weight_version": self.weight_version,
            "wire_bytes": self._fold.nbytes,
            "wire_time": self._fold.busy,
            "wire_time_sum": self._fold.sum_groups,
            "write_queue": self._fold.queue,
            "gpu_prep": seconds.get(KIND_NEXT, 0.0) + parts[GPU_PREP_STAGE] + parts[GPU_PREP_CPU_LOAD],
            "cpu_prep": seconds.get(KIND_CPU_PREP, 0.0),
            "buffer_wait": seconds.get(KIND_BUFFER_WAIT, 0.0),
            "drain": seconds.get(KIND_DRAIN, 0.0),
            "active": self._active,
        }
        for part, value in parts.items():
            payload[f"gpu_{part}"] = value
        return payload


def measure(collector: P2PNixlPerfCollector | None, kind: str) -> AbstractContextManager:
    """`collector.measure(kind)`, or a no-op without a collector (Mooncake, or a rank that sends nothing)."""
    return nullcontext() if collector is None else collector.measure(kind)


def perf_log_path() -> Path:
    """The shared append-only log rank 0 writes: `MILES_LOG_DIR` if set, else the working directory."""
    return Path(os.environ.get("MILES_LOG_DIR") or os.getcwd()) / LOG_FILENAME


def format_gb(num_bytes: int) -> str:
    return f"{num_bytes / _GB:.2f}GB"


def format_seconds(num_seconds: float) -> str:
    return f"{num_seconds:.3f}s"


def format_prep_seconds(num_seconds: float) -> str:
    return f"{num_seconds:.2f}s"


def format_perf_section(payloads: list[dict[str, int | float]]) -> str:
    """One log section: the bottleneck GPU of each metric. Three `trainer_prep_time` picks come from three argmaxes
    (the third, gpu_prep + cpu_prep, on one GPU), and the part split is that third GPU's."""
    weight_version = int(payloads[0]["weight_version"])
    best_bytes = max(payloads, key=lambda payload: payload["wire_bytes"])
    best_wire = max(payloads, key=lambda payload: payload["wire_time"])
    best_gpu_prep = max(payloads, key=lambda payload: payload["gpu_prep"])
    best_cpu_prep = max(payloads, key=lambda payload: payload["cpu_prep"])
    best_total = max(payloads, key=lambda payload: payload["gpu_prep"] + payload["cpu_prep"])
    best_exposed = max(payloads, key=lambda payload: payload["buffer_wait"] + payload["drain"])
    best_active = max(payloads, key=lambda payload: payload["active"])
    total_gpu_prep = float(best_total["gpu_prep"])
    total_cpu_prep = float(best_total["cpu_prep"])
    exposed = float(best_exposed["buffer_wait"]) + float(best_exposed["drain"])
    part_lines = "".join(
        f"trainer_prep_time_gpu_{part}: gpu={best_total['gpu']} "
        f"time={format_prep_seconds(float(best_total[f'gpu_{part}']))}\n"
        for part in GPU_PREP_PARTS
    )
    warm_up = " (warm-up)" if weight_version == WARM_UP_WEIGHT_VERSION else ""
    return (
        f"=== p2p nixl perf  weight_version={weight_version}{warm_up} ===\n"
        f"max_num_wire_bytes_per_trainer: gpu={best_bytes['gpu']} bytes={format_gb(int(best_bytes['wire_bytes']))}\n"
        f"wire_time: gpu={best_wire['gpu']} busy={format_seconds(float(best_wire['wire_time']))} "
        f"sum_groups={format_seconds(float(best_wire['wire_time_sum']))}\n"
        f"write_queue: gpu={best_wire['gpu']} time={format_prep_seconds(float(best_wire['write_queue']))}\n"
        f"trainer_prep_time_gpu: gpu={best_gpu_prep['gpu']} time={format_prep_seconds(float(best_gpu_prep['gpu_prep']))}\n"
        f"trainer_prep_time_cpu: gpu={best_cpu_prep['gpu']} time={format_prep_seconds(float(best_cpu_prep['cpu_prep']))}\n"
        f"trainer_prep_time_total: gpu={best_total['gpu']} gpu_prep={format_prep_seconds(total_gpu_prep)} "
        f"cpu_prep={format_prep_seconds(total_cpu_prep)} total={format_prep_seconds(total_gpu_prep + total_cpu_prep)}\n"
        f"{part_lines}"
        f"trainer_exposed_wire_time: gpu={best_exposed['gpu']} "
        f"buffer_wait={format_prep_seconds(float(best_exposed['buffer_wait']))} "
        f"drain={format_prep_seconds(float(best_exposed['drain']))} total={format_prep_seconds(exposed)}\n"
        f"trainer_active_time: gpu={best_active['gpu']} time={format_seconds(float(best_active['active']))}\n"
    )


def write_perf_log(collector: P2PNixlPerfCollector | None, *, log_path: Path | None = None) -> None:
    """Gloo-gather every rank's payload; rank 0 appends one section.

    Every rank must call this once per transfer, a rank that sends nothing with `collector=None`: the gather is a
    collective.
    """
    gathered = _gather_payloads(None if collector is None else collector.payload())
    if gathered is None:
        return
    senders = [payload for payload in gathered if payload is not None]
    if not senders:
        return
    path = log_path if log_path is not None else perf_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(format_perf_section(senders))


def _gather_payloads(payload: dict[str, int | float] | None) -> list[dict[str, int | float] | None] | None:
    """`dist.gather_object` on gloo. Rank 0 gets the list; everyone else gets None."""
    group = get_gloo_group()
    dst = dist.get_global_rank(group, 0)
    gathered = [None] * dist.get_world_size(group) if dist.get_rank(group) == 0 else None
    dist.gather_object(payload, object_gather_list=gathered, dst=dst, group=group)
    return gathered
