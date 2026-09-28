"""NIXL P2P trainer-side perf log. Rank 0 appends one section per full weight transfer."""

import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LOG_FILENAME = "p2p_nixl_perf.log"
_GIB = 1024**3


@dataclass
class P2PNixlPerfCollector:
    """Per-rank counters for one `update_weights` NIXL P2P transfer.

    Identity fields (`gpu`, `pp_rank`, `gathered_dp_rank`) ride on every payload so
    rank 0 can label the bottleneck GPU in the log. Wire-byte parts feed
    `max_num_wire_bytes_per_trainer`. Per-replica session times feed `wire_time`
    (max within a replica, then sum across replicas and `send_bucket` steps).
    """

    gpu: int
    pp_rank: int
    gathered_dp_rank: int
    weight_version: int = 0
    _wire_bytes_parts: list[int] = field(default_factory=list)
    _next_wire_group: int = 0
    _current_wire_group: int | None = None
    _wire_groups: dict[int, list[float]] = field(default_factory=dict)

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

    def add_wire_bytes(self, num_bytes: int) -> None:
        """Record RDMA payload size for one inner session write.

        `max_num_wire_bytes_per_trainer`: one `sum(source_lens)` per
        `_do_p2p_write_one_session`. Sessions of one replica, sequential replicas,
        and outer `send_bucket` steps all append; the total is the GPU's wire bytes.
        """
        self._wire_bytes_parts.append(num_bytes)

    def begin_wire_group(self) -> int:
        """Open one replica's parallel-session group for `wire_time`.

        Sessions in this group are one replica's RDMA targets (they share a buffer
        and run in the thread pool together). `wire_time` counts the group once
        (max session), then sums groups across replicas and outer `send_bucket`s.
        """
        group_id = self._next_wire_group
        self._next_wire_group += 1
        self._wire_groups[group_id] = []
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

    def payload(self) -> dict[str, int | float]:
        """Snapshot this rank sends on the gloo gather.

        Rank 0 uses `wire_bytes` for `max_num_wire_bytes_per_trainer` and `wire_time`
        for `wire_time`, and keeps `gpu` / `pp_rank` / `gathered_dp_rank` to label
        those lines in the log.
        """
        return {
            "gpu": self.gpu,
            "pp_rank": self.pp_rank,
            "gathered_dp_rank": self.gathered_dp_rank,
            "wire_bytes": self.wire_bytes(),
            "wire_time": self.wire_time(),
            "weight_version": self.weight_version,
        }


class TimeMonitor:
    """`time.monotonic()` start/stop around a region.

    `wire_time` uses this around `_do_nixl_write` (start right before, stop when
    it returns). Later metrics (`trainer_prep_time`, `trainer_active_time`) reuse it.
    """

    def __init__(self) -> None:
        self._start: float | None = None

    def start(self) -> None:
        self._start = time.monotonic()

    def stop(self) -> float:
        elapsed = time.monotonic() - self._start
        self._start = None
        return elapsed


def new_collector(*, gpu: int, pp_rank: int, gathered_dp_rank: int) -> P2PNixlPerfCollector:
    """Build the per-rank collector when `--update-weight-transfer-backend nixl`.

    `gpu` is gloo global_rank (the `gpu=` in the log). Shared by all metrics.
    """
    return P2PNixlPerfCollector(gpu=gpu, pp_rank=pp_rank, gathered_dp_rank=gathered_dp_rank)


def reset_collector(collector: P2PNixlPerfCollector | None, weight_version: int) -> None:
    """No-op on mooncake. On NIXL, start a clean transfer for this `weight_version`."""
    if collector is not None:
        collector.reset(weight_version)


def add_wire_bytes(collector: P2PNixlPerfCollector | None, num_bytes: int) -> None:
    """No-op on mooncake. On NIXL, count this session toward `max_num_wire_bytes_per_trainer`."""
    if collector is not None:
        collector.add_wire_bytes(num_bytes)


def begin_wire_group(collector: P2PNixlPerfCollector | None) -> int:
    """No-op on mooncake. On NIXL, start a replica group for `wire_time` max-then-sum."""
    if collector is None:
        return 0
    return collector.begin_wire_group()


def current_wire_group(collector: P2PNixlPerfCollector | None) -> int | None:
    """Replica group opened by the latest `begin_wire_group` on this rank.

    `_do_p2p_write_one_session` snapshots this at entry so `wire_time` can max
    parallel sessions of one replica without wrapping the submitted write.
    """
    if collector is None:
        return None
    return collector._current_wire_group


def timed_call(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> float:
    """Run `fn` and return elapsed seconds.

    `wire_time`: clock around `_do_nixl_write` only (not pointer setup / load_weights).
    """
    monitor = TimeMonitor()
    monitor.start()
    try:
        fn(*args, **kwargs)
    finally:
        elapsed = monitor.stop()
    return elapsed


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


def format_gib(num_bytes: int) -> str:
    """Render wire bytes as `12.40GiB` for the `max_num_wire_bytes_per_trainer` log line."""
    return f"{num_bytes / _GIB:.2f}GiB"


def format_seconds(num_seconds: float) -> str:
    """Render seconds as `2.110s` for the `wire_time` log line."""
    return f"{num_seconds:.3f}s"


def format_perf_section(weight_version: int, payloads: list[dict[str, int | float]]) -> str:
    """One log section for this `weight_version`.

    Prints `max_num_wire_bytes_per_trainer` (GPU with the most wire bytes) and
    `wire_time` (GPU with the most RDMA work). Later metrics add lines here.
    """
    best_bytes = max(payloads, key=lambda payload: payload["wire_bytes"])
    best_wire = max(payloads, key=lambda payload: payload["wire_time"])
    return (
        f"=== p2p nixl perf  weight_version={weight_version} ===\n"
        f"max_num_wire_bytes_per_trainer: gpu={best_bytes['gpu']} bytes={format_gib(int(best_bytes['wire_bytes']))}\n"
        f"wire_time: gpu={best_wire['gpu']} work={format_seconds(float(best_wire['wire_time']))}\n"
    )


def gather_and_write_perf_log(
    collector: P2PNixlPerfCollector | None,
    *,
    is_sender: bool,
    log_path: Path | None = None,
) -> None:
    """Gloo-gather every rank's payload; rank 0 appends the bottleneck GPU to the log.

    Non-senders send `None`. Rank 0 writes `max_num_wire_bytes_per_trainer` and
    `wire_time` so many processes do not append the same file.
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
