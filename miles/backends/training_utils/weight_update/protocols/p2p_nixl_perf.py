"""NIXL P2P trainer-side perf log. Rank 0 appends one section per full weight transfer."""

import os
import time
from collections.abc import Callable, Iterable, Iterator
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

LOG_FILENAME = "p2p_nixl_perf.log"
# Decimal GB (10^9), matching how NIC line rates are quoted, so wire bytes can be
# divided by wire_time and compared against link bandwidth without a base change.
_GB = 1000**3


class TimeMonitor:
    """`time.monotonic()` start/stop around a region.

    `wire_time` uses this around `_do_nixl_write`. `trainer_prep_time` gpu uses
    one instance on the collector from `next(iterator)` until after staging;
    each `load_weights` is added into that total separately. `trainer_prep_time`
    cpu starts on the main thread immediately after `load_weights` and stops
    in the session before `_do_nixl_write`. `trainer_active_time` uses
    one instance from the first `next(iterator)` until after `wait_transfers()`.
    `timed_call` uses a fresh instance around one function.
    `timed_gpu_prep_part` uses `time.monotonic()` around gather / convert / stage.
    """

    def __init__(self) -> None:
        self._start: float | None = None

    def start(self) -> None:
        self._start = time.monotonic()

    def stop(self) -> float:
        elapsed = time.monotonic() - self._start
        self._start = None
        return elapsed

    def cancel(self) -> None:
        """Drop a start without counting elapsed (failed or extra `next()`)."""
        self._start = None

    def running(self) -> bool:
        return self._start is not None


GPU_PREP_GATHER = "gather"
GPU_PREP_GATHER_LOAD = "gather_load"
GPU_PREP_GATHER_PP = "gather_pp"
GPU_PREP_GATHER_TP = "gather_tp"
GPU_PREP_GATHER_TP_START = "gather_tp_start"
GPU_PREP_GATHER_TP_WAIT = "gather_tp_wait"
GPU_PREP_GATHER_TP_CONCAT = "gather_tp_concat"
GPU_PREP_GATHER_EP = "gather_ep"
GPU_PREP_CONVERT = "convert"
GPU_PREP_STAGE = "stage"
GPU_PREP_CPU_LOAD = "cpu_load"
# Parent gather first, then its raw-path chunks, then convert / stage / cpu load.
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

_T = TypeVar("_T")
_add_part: ContextVar[Callable[[str, float], None] | None] = ContextVar("gpu_prep_add_part", default=None)


def bind_gpu_prep_parts(add_part: Callable[[str, float], None] | None) -> None:
    """Route `add_gpu_prep_part` to this transfer's collector, or None on mooncake."""
    _add_part.set(add_part)


def add_gpu_prep_part(part: str, duration: float) -> None:
    """No-op unless a NIXL collector is bound. Adds `duration` to that GPU's part total."""
    add_part = _add_part.get()
    if add_part is not None:
        add_part(part, duration)


def timed_gpu_prep_part(part: str, fn: Callable[..., _T], *args: Any, **kwargs: Any) -> _T:
    """Run `fn` and add elapsed seconds to `part`. Re-raises without counting if `fn` raises."""
    start = time.monotonic()
    result = fn(*args, **kwargs)
    add_gpu_prep_part(part, time.monotonic() - start)
    return result


def iter_timed_gpu_prep_part(part: str, inner: Iterable[_T]) -> Iterator[_T]:
    """Time each `next()` of a lazy export (bridge gather) and add it to `part`."""
    iterator = iter(inner)
    while True:
        start = time.monotonic()
        try:
            item = next(iterator)
        except StopIteration:
            return
        add_gpu_prep_part(part, time.monotonic() - start)
        yield item


@dataclass
class P2PNixlPerfCollector:
    """Per-rank counters for one `update_weights` NIXL P2P transfer.

    Identity fields (`gpu`, `pp_rank`, `gathered_dp_rank`) ride on every payload so
    rank 0 can label the bottleneck GPU in the log. Wire-byte parts feed
    `max_num_wire_bytes_per_trainer`. Per-replica session times feed `wire_time`
    (max within a replica, then sum across replicas and `send_bucket` steps).
    `gpu_prep` sums outer `next(iterator)` intervals plus every `load_weights`.
    `gpu_gather` / `gpu_convert` / `gpu_stage` sum the functions inside that
    outer clock. Gather also has load / PP / TP / EP chunks on the raw path.
    `gpu_cpu_load` is just those `load_weights` calls. `cpu_prep` is max host
    time per replica group from right after `load_weights` until before the
    RDMA write.
    `active_time` is one wall clock from the first `next(iterator)` until
    `wait_transfers()`.
    """

    gpu: int
    pp_rank: int
    gathered_dp_rank: int
    weight_version: int = 0
    _wire_bytes_parts: list[int] = field(default_factory=list)
    _next_wire_group: int = 0
    _current_wire_group: int | None = None
    _wire_groups: dict[int, list[float]] = field(default_factory=dict)
    _cpu_setup_groups: dict[int, list[float]] = field(default_factory=dict)
    _gpu_prep: float = 0.0
    _gpu_parts: dict[str, float] = field(default_factory=dict)
    _active_time: float = 0.0
    _active_started: bool = False
    _gpu_prep_monitor: TimeMonitor = field(default_factory=TimeMonitor)
    _active_monitor: TimeMonitor = field(default_factory=TimeMonitor)

    def reset(self, weight_version: int) -> None:
        """Clear counters at the start of this `update_weights`.

        Each full transfer is its own log section; we do not add versions together.
        Used by every metric that is summed or maxed over one transfer.
        """
        self.weight_version = weight_version
        self._wire_bytes_parts = []
        self._next_wire_group = 0
        self._current_wire_group = None
        self._wire_groups = {}
        self._cpu_setup_groups = {}
        self._gpu_prep = 0.0
        self._gpu_parts = {}
        self._active_time = 0.0
        self._active_started = False
        self._gpu_prep_monitor.cancel()
        self._active_monitor.cancel()

    def add_wire_bytes(self, num_bytes: int) -> None:
        """Record RDMA payload size for one inner session write.

        `max_num_wire_bytes_per_trainer`: one `sum(source_lens)` per
        `_do_p2p_write_one_session`. Sessions of one replica, sequential replicas,
        and outer `send_bucket` steps all append; the total is the GPU's wire bytes.
        """
        self._wire_bytes_parts.append(num_bytes)

    def begin_wire_group(self) -> int:
        """Open one replica's parallel-session group for `wire_time` and cpu setup.

        Sessions in this group are one replica's RDMA targets (they share a buffer
        and run in the thread pool together). `wire_time` and cpu pointer-setup
        each count the group once (max session), then sum groups across replicas
        and outer `send_bucket`s.
        """
        group_id = self._next_wire_group
        self._next_wire_group += 1
        self._wire_groups[group_id] = []
        self._cpu_setup_groups[group_id] = []
        self._current_wire_group = group_id
        return group_id

    def add_session_wire_time(self, group_id: int | None, duration: float) -> None:
        """Record one `_do_nixl_write` duration into its replica group.

        `wire_time`: per-event timer around the NIXL write. Parallel sessions of
        one replica land in the same group so we can take max, not sum.
        """
        if group_id is None:
            group_id = -1
            self._wire_groups.setdefault(group_id, [])
        self._wire_groups[group_id].append(duration)

    def add_cpu_load(self, duration: float) -> None:
        """Record one `load_weights` duration inside `gpu_prep`.

        Replicas on one GPU run one after the other, so these add. Outer
        `send_bucket` steps add as well. Also stored as the `cpu_load` part so
        the log can show this copy alone. Call after `stop_gpu_prep`: the outer
        clock stops before this copy.
        """
        self.add_gpu_prep_part(GPU_PREP_CPU_LOAD, duration)

    def add_gpu_prep_part(self, part: str, duration: float) -> None:
        """Add one timed gather / convert / stage / cpu-load call to this GPU's part total.

        Rank 0 prints these from the `trainer_prep_time_total` GPU, not as
        independent argmax lines. Gather load / PP / TP / EP are extra counters
        inside `gather` on the raw path. TP start / wait / concat are extra
        counters inside `gather_tp`. `cpu_load` is outside the outer clock, so
        that part is also added into `gpu_prep`.
        """
        if part not in GPU_PREP_PARTS:
            raise ValueError(f"unknown gpu_prep part: {part}")
        self._gpu_parts[part] = self._gpu_parts.get(part, 0.0) + duration
        if part == GPU_PREP_CPU_LOAD:
            self._gpu_prep += duration

    def add_session_cpu_setup(self, group_id: int | None, duration: float) -> None:
        """Record one session's host time from after `load_weights` until `_do_nixl_write`.

        `trainer_prep_time` cpu: the main thread stamps the start. Each session
        reports its own stop. Parallel sessions of one replica land in the same
        group so we take max, not sum, like `wire_time`.
        """
        if group_id is None:
            group_id = -1
            self._cpu_setup_groups.setdefault(group_id, [])
        self._cpu_setup_groups[group_id].append(duration)

    def start_gpu_prep(self) -> None:
        """Start `trainer_prep_time` gpu at `next(iterator)` for this outer step."""
        self._gpu_prep_monitor.start()

    def stop_gpu_prep(self) -> None:
        """Stop the outer `trainer_prep_time` gpu clock after staging, before `load_weights`."""
        if not self._gpu_prep_monitor.running():
            return
        self._gpu_prep += self._gpu_prep_monitor.stop()

    def cancel_gpu_prep(self) -> None:
        """Drop a gpu-prep start that did not yield a bucket (`StopIteration`)."""
        self._gpu_prep_monitor.cancel()

    def start_active_time(self) -> None:
        """Start `trainer_active_time` once, at the first `next(iterator)` of this transfer."""
        if self._active_started:
            return
        self._active_started = True
        self._active_monitor.start()

    def stop_active_time(self) -> None:
        """Stop `trainer_active_time` after `wait_transfers()` and keep that one elapsed."""
        if not self._active_monitor.running():
            return
        self._active_time = self._active_monitor.stop()

    def wire_bytes(self) -> int:
        """Total bytes this GPU put on the wire in this transfer.

        Fold for `max_num_wire_bytes_per_trainer` (sum of every inner session).
        """
        return sum(self._wire_bytes_parts)

    def wire_time(self) -> float:
        """RDMA work on this GPU in this transfer.

        `wire_time`: max session time per replica group, then sum groups (replicas
        and outer `send_bucket`s). Rank 0 logs the GPU with the largest work.
        """
        return sum(max(times) for times in self._wire_groups.values() if times)

    def gpu_prep(self) -> float:
        """All-gather + HF convert + quant + staging, plus every `load_weights`.

        `trainer_prep_time` gpu: sum every outer `next(iterator)` until after
        staging, then add each `load_weights`. Rank 0 logs the GPU with the
        largest gpu_prep. The `cpu_load` part is just those loads.
        """
        return self._gpu_prep

    def gpu_gather(self) -> float:
        """Sum of `_materialize_*` / bridge `export_hf_weights` `next()` on this GPU."""
        return self._gpu_parts.get(GPU_PREP_GATHER, 0.0)

    def gpu_convert(self) -> float:
        """Sum of `convert_to_hf` / `_postprocess_and_quantize` on this GPU."""
        return self._gpu_parts.get(GPU_PREP_CONVERT, 0.0)

    def gpu_stage(self) -> float:
        """Sum of `_get_transfer_ready_params` on this GPU."""
        return self._gpu_parts.get(GPU_PREP_STAGE, 0.0)

    def cpu_prep(self) -> float:
        """Host time on this GPU from after `load_weights` until before RDMA.

        `trainer_prep_time` cpu: max per replica group (wire-group open, pool
        submit, and pointer setup), then sum groups across replicas and outer
        `send_bucket`s. The CPU copy itself is `gpu_prep` / `cpu_load`.
        """
        return sum(max(times) for times in self._cpu_setup_groups.values() if times)

    def active_time(self) -> float:
        """Wall clock of this full transfer on this GPU.

        `trainer_active_time`: first `next(iterator)` until `wait_transfers()`
        returns. Not a sum of outer or inner times (those overlap). Rank 0 logs
        the GPU with the largest active time.
        """
        return self._active_time

    def payload(self) -> dict[str, int | float]:
        """Snapshot this rank sends on the gloo gather.

        Rank 0 uses `wire_bytes` for `max_num_wire_bytes_per_trainer`, `wire_time`
        for `wire_time`, `gpu_prep` / `cpu_prep` for the three `trainer_prep_time`
        picks, `gpu_{part}` for that total GPU's gather / convert / stage /
        cpu_load split,
        and `active_time` for `trainer_active_time`, and keeps `gpu` /
        `pp_rank` / `gathered_dp_rank` to label those lines in the log.
        """
        payload: dict[str, int | float] = {
            "gpu": self.gpu,
            "pp_rank": self.pp_rank,
            "gathered_dp_rank": self.gathered_dp_rank,
            "wire_bytes": self.wire_bytes(),
            "wire_time": self.wire_time(),
            "gpu_prep": self.gpu_prep(),
            "cpu_prep": self.cpu_prep(),
            "active_time": self.active_time(),
            "weight_version": self.weight_version,
        }
        for part in GPU_PREP_PARTS:
            payload[f"gpu_{part}"] = self._gpu_parts.get(part, 0.0)
        return payload


class _GpuPrepIter:
    """Starts gpu-prep on each `next()` and active time on the first `next()`.

    Cancels gpu-prep if the inner iterator is done. Leaves active time running
    until `stop_active_time` after `wait_transfers()`.
    """

    def __init__(self, collector: P2PNixlPerfCollector, inner: Iterator[Any]) -> None:
        self._collector = collector
        self._inner = iter(inner)

    def __iter__(self) -> "_GpuPrepIter":
        return self

    def __next__(self) -> Any:
        self._collector.start_active_time()
        self._collector.start_gpu_prep()
        try:
            return next(self._inner)
        except BaseException:
            self._collector.cancel_gpu_prep()
            raise


def new_collector(*, gpu: int, pp_rank: int, gathered_dp_rank: int) -> P2PNixlPerfCollector:
    """Build the per-rank collector when `--update-weight-transfer-backend nixl`.

    `gpu` is gloo global_rank (the `gpu=` in the log). Shared by all metrics.
    """
    return P2PNixlPerfCollector(gpu=gpu, pp_rank=pp_rank, gathered_dp_rank=gathered_dp_rank)


def reset_collector(collector: P2PNixlPerfCollector | None, weight_version: int) -> None:
    """No-op on mooncake. On NIXL, start a clean transfer for this `weight_version`."""
    if collector is not None:
        collector.reset(weight_version)
        bind_gpu_prep_parts(collector.add_gpu_prep_part)
        return
    bind_gpu_prep_parts(None)


def add_wire_bytes(collector: P2PNixlPerfCollector | None, num_bytes: int) -> None:
    """No-op on mooncake. On NIXL, count this session toward `max_num_wire_bytes_per_trainer`."""
    if collector is not None:
        collector.add_wire_bytes(num_bytes)


def begin_wire_group(collector: P2PNixlPerfCollector | None) -> int:
    """No-op on mooncake. On NIXL, start a replica group for `wire_time` / cpu-setup max-then-sum."""
    if collector is None:
        return 0
    return collector.begin_wire_group()


def current_wire_group(collector: P2PNixlPerfCollector | None) -> int | None:
    """Replica group opened by the latest `begin_wire_group` on this rank.

    `_do_p2p_write_one_session` snapshots this at entry so `wire_time` and cpu
    setup can max parallel sessions of one replica without wrapping the submitted write.
    """
    if collector is None:
        return None
    return collector._current_wire_group


def wrap_gpu_prep_iter(collector: P2PNixlPerfCollector | None, buckets: Iterator[Any]) -> Iterator[Any]:
    """Start gpu-prep at each `next(iterator)` and active time on the first. No-op on mooncake.

    The updater wraps the HF iterator through the protocol so it does not import NIXL.
    """
    if collector is None:
        return buckets
    return _GpuPrepIter(collector, buckets)


def stop_gpu_prep(collector: P2PNixlPerfCollector | None) -> None:
    """No-op on mooncake. On NIXL, stop the outer gpu-prep clock after staging, before `load_weights`."""
    if collector is not None:
        collector.stop_gpu_prep()


def stop_active_time(collector: P2PNixlPerfCollector | None) -> None:
    """No-op on mooncake. On NIXL, stop `trainer_active_time` after `wait_transfers()`."""
    if collector is not None:
        collector.stop_active_time()


def add_cpu_load(collector: P2PNixlPerfCollector | None, duration: float) -> None:
    """No-op on mooncake. On NIXL, add this `load_weights` time to `gpu_prep` and the `cpu_load` part."""
    if collector is not None:
        collector.add_cpu_load(duration)


def add_session_cpu_setup(
    collector: P2PNixlPerfCollector | None, duration: float, *, group_id: int | None = None
) -> None:
    """No-op on mooncake. On NIXL, record host time after `load_weights` into a replica group for cpu-prep."""
    if collector is None:
        return
    collector.add_session_cpu_setup(group_id, duration)


def timed_call(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> float:
    """Run `fn` and return elapsed seconds. Re-raises if `fn` raises.

    `wire_time`: clock around `_do_nixl_write` only (not pointer setup).
    `trainer_prep_time` gpu `cpu_load`: clock around `load_weights`.
    """
    monitor = TimeMonitor()
    monitor.start()
    try:
        fn(*args, **kwargs)
        return monitor.stop()
    except BaseException:
        monitor.cancel()
        raise


def add_session_wire_time(
    collector: P2PNixlPerfCollector | None, duration: float, *, group_id: int | None = None
) -> None:
    """No-op on mooncake. On NIXL, record this write into a replica group for `wire_time`."""
    if collector is None:
        return
    collector.add_session_wire_time(group_id, duration)


def perf_log_path() -> Path:
    """Path of the shared append-only log rank 0 writes (`p2p_nixl_perf.log`).

    All metrics in that file share this path. `MILES_LOG_DIR` if set, else cwd.
    """
    return Path(os.environ.get("MILES_LOG_DIR") or os.getcwd()) / LOG_FILENAME


def format_gb(num_bytes: int) -> str:
    """Render wire bytes as `13.31GB` for the `max_num_wire_bytes_per_trainer` log line."""
    return f"{num_bytes / _GB:.2f}GB"


def format_seconds(num_seconds: float) -> str:
    """Render seconds as `2.110s` / `3.410s` for `wire_time` and `trainer_active_time`."""
    return f"{num_seconds:.3f}s"


def format_prep_seconds(num_seconds: float) -> str:
    """Render seconds as `0.91s` for `trainer_prep_time` log lines."""
    return f"{num_seconds:.2f}s"


def format_perf_section(weight_version: int, payloads: list[dict[str, int | float]]) -> str:
    """One log section for this `weight_version`.

    Prints `max_num_wire_bytes_per_trainer` (GPU with the most wire bytes),
    `wire_time` (GPU with the most RDMA work), three `trainer_prep_time` picks,
    that total GPU's gather / convert / stage / cpu_load split (gather also
    prints load / PP / TP / EP), and `trainer_active_time` (GPU with the
    longest first-`next` to `wait_transfers()` wall clock).
    """
    best_bytes = max(payloads, key=lambda payload: payload["wire_bytes"])
    best_wire = max(payloads, key=lambda payload: payload["wire_time"])
    best_gpu_prep = max(payloads, key=lambda payload: payload["gpu_prep"])
    best_cpu_prep = max(payloads, key=lambda payload: payload["cpu_prep"])
    best_total = max(payloads, key=lambda payload: payload["gpu_prep"] + payload["cpu_prep"])
    best_active = max(payloads, key=lambda payload: payload["active_time"])
    total_gpu_prep = float(best_total["gpu_prep"])
    total_cpu_prep = float(best_total["cpu_prep"])
    total_gpu = best_total["gpu"]
    part_lines = "".join(
        f"trainer_prep_time_gpu_{part}: gpu={total_gpu} "
        f"time={format_prep_seconds(float(best_total[f'gpu_{part}']))}\n"
        for part in GPU_PREP_PARTS
    )
    return (
        f"=== p2p nixl perf  weight_version={weight_version} ===\n"
        f"max_num_wire_bytes_per_trainer: gpu={best_bytes['gpu']} bytes={format_gb(int(best_bytes['wire_bytes']))}\n"
        f"wire_time: gpu={best_wire['gpu']} work={format_seconds(float(best_wire['wire_time']))}\n"
        f"trainer_prep_time_gpu: gpu={best_gpu_prep['gpu']} time={format_prep_seconds(float(best_gpu_prep['gpu_prep']))}\n"
        f"trainer_prep_time_cpu: gpu={best_cpu_prep['gpu']} time={format_prep_seconds(float(best_cpu_prep['cpu_prep']))}\n"
        f"trainer_prep_time_total: gpu={total_gpu} gpu_prep={format_prep_seconds(total_gpu_prep)} "
        f"cpu_prep={format_prep_seconds(total_cpu_prep)} total={format_prep_seconds(total_gpu_prep + total_cpu_prep)}\n"
        f"{part_lines}"
        f"trainer_active_time: gpu={best_active['gpu']} time={format_seconds(float(best_active['active_time']))}\n"
    )


def gather_and_write_perf_log(
    collector: P2PNixlPerfCollector | None,
    *,
    is_sender: bool,
    log_path: Path | None = None,
) -> None:
    """Gloo-gather every rank's payload; rank 0 appends the bottleneck GPU to the log.

    Non-senders send `None`. Rank 0 writes `max_num_wire_bytes_per_trainer`,
    `wire_time`, the three `trainer_prep_time` lines, and `trainer_active_time`
    so many processes do not append the same file.
    """
    if collector is None:
        return
    payload = collector.payload() if is_sender else None
    gathered = _gather_payloads(payload)
    if gathered is None:
        return
    senders = [item for item in gathered if item is not None]
    if not senders:
        return
    path = log_path if log_path is not None else perf_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(format_perf_section(collector.weight_version, senders))


def _gather_payloads(payload: dict[str, int | float] | None) -> list[dict[str, int | float] | None] | None:
    """`dist.gather_object` on gloo. Rank 0 gets the list; everyone else returns None.

    Local import so format/collector helpers stay importable without torch.
    """
    import torch.distributed as dist

    from miles.utils.distributed_utils import get_gloo_group

    group = get_gloo_group()
    dst = dist.get_global_rank(group, 0)
    gathered = [None] * dist.get_world_size(group) if dist.get_rank(group) == 0 else None
    dist.gather_object(payload, object_gather_list=gathered, dst=dst, group=group)
    return gathered
