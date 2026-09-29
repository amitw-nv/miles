"""NIXL P2P trainer-side perf log. Rank 0 appends one section per full weight transfer."""

import os
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LOG_FILENAME = "p2p_nixl_perf.log"
# Decimal GB (10^9), matching how NIC line rates are quoted, so wire bytes can be
# divided by wire_time and compared against link bandwidth without a base change.
_GB = 1000**3


class TimeMonitor:
    """`time.monotonic()` start/stop around a region.

    `wire_time` uses this around `_do_nixl_write`. `trainer_prep_time` gpu uses
    one instance on the collector from `next(iterator)` until before
    `load_weights`. `trainer_prep_time` cpu uses a per-session instance for
    pointer setup until before `_do_nixl_write`. `trainer_active_time` uses one
    instance from the first `next(iterator)` until after `wait_transfers()`.
    `timed_call` uses a fresh instance around one function.
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


@dataclass
class P2PNixlPerfCollector:
    """Per-rank counters for one `update_weights` NIXL P2P transfer.

    Identity fields (`gpu`, `pp_rank`, `gathered_dp_rank`) ride on every payload so
    rank 0 can label the bottleneck GPU in the log. Wire-byte parts feed
    `max_num_wire_bytes_per_trainer`. Per-replica session times feed `wire_time`
    (max within a replica, then sum across replicas and `send_bucket` steps).
    `gpu_prep` sums outer `next(iterator)` intervals. `cpu_prep` sums
    `load_weights` plus max pointer-setup per replica group. `active_time` is
    one wall clock from the first `next(iterator)` until `wait_transfers()`.
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
    _cpu_load: float = 0.0
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
        self._cpu_load = 0.0
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
        """Record one `load_weights` duration.

        `trainer_prep_time` cpu: replicas on one GPU run one after the other, so
        these add. Outer `send_bucket` steps add as well.
        """
        self._cpu_load += duration

    def add_session_cpu_setup(self, group_id: int | None, duration: float) -> None:
        """Record pointer setup in `_do_p2p_write_one_session` until `_do_nixl_write`.

        `trainer_prep_time` cpu: parallel sessions of one replica land in the same
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
        """Stop `trainer_prep_time` gpu right before `load_weights` and add elapsed."""
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
        """All-gather + HF convert + quant on this GPU in this transfer.

        `trainer_prep_time` gpu: sum every outer `next(iterator)` until before
        `load_weights`. Rank 0 logs the GPU with the largest gpu_prep.
        """
        return self._gpu_prep

    def cpu_prep(self) -> float:
        """Load to CPU plus pointer setup on this GPU in this transfer.

        `trainer_prep_time` cpu: sum `load_weights`, plus max setup per replica
        group, then sum groups across replicas and outer `send_bucket`s.
        """
        setup = sum(max(times) for times in self._cpu_setup_groups.values() if times)
        return self._cpu_load + setup

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
        picks, and `active_time` for `trainer_active_time`, and keeps `gpu` /
        `pp_rank` / `gathered_dp_rank` to label those lines in the log.
        """
        return {
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
    """No-op on mooncake. On NIXL, stop gpu-prep right before `load_weights`."""
    if collector is not None:
        collector.stop_gpu_prep()


def stop_active_time(collector: P2PNixlPerfCollector | None) -> None:
    """No-op on mooncake. On NIXL, stop `trainer_active_time` after `wait_transfers()`."""
    if collector is not None:
        collector.stop_active_time()


def add_cpu_load(collector: P2PNixlPerfCollector | None, duration: float) -> None:
    """No-op on mooncake. On NIXL, add this `load_weights` time to `trainer_prep_time` cpu."""
    if collector is not None:
        collector.add_cpu_load(duration)


def add_session_cpu_setup(
    collector: P2PNixlPerfCollector | None, duration: float, *, group_id: int | None = None
) -> None:
    """No-op on mooncake. On NIXL, record pointer setup into a replica group for cpu-prep."""
    if collector is None:
        return
    collector.add_session_cpu_setup(group_id, duration)


def timed_call(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> float:
    """Run `fn` and return elapsed seconds. Re-raises if `fn` raises.

    `wire_time`: clock around `_do_nixl_write` only (not pointer setup).
    `trainer_prep_time` cpu: clock around `load_weights`.
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
    and `trainer_active_time` (GPU with the longest first-`next` to
    `wait_transfers()` wall clock).
    """
    best_bytes = max(payloads, key=lambda payload: payload["wire_bytes"])
    best_wire = max(payloads, key=lambda payload: payload["wire_time"])
    best_gpu_prep = max(payloads, key=lambda payload: payload["gpu_prep"])
    best_cpu_prep = max(payloads, key=lambda payload: payload["cpu_prep"])
    best_total = max(payloads, key=lambda payload: payload["gpu_prep"] + payload["cpu_prep"])
    best_active = max(payloads, key=lambda payload: payload["active_time"])
    total_gpu_prep = float(best_total["gpu_prep"])
    total_cpu_prep = float(best_total["cpu_prep"])
    return (
        f"=== p2p nixl perf  weight_version={weight_version} ===\n"
        f"max_num_wire_bytes_per_trainer: gpu={best_bytes['gpu']} bytes={format_gb(int(best_bytes['wire_bytes']))}\n"
        f"wire_time: gpu={best_wire['gpu']} work={format_seconds(float(best_wire['wire_time']))}\n"
        f"trainer_prep_time_gpu: gpu={best_gpu_prep['gpu']} time={format_prep_seconds(float(best_gpu_prep['gpu_prep']))}\n"
        f"trainer_prep_time_cpu: gpu={best_cpu_prep['gpu']} time={format_prep_seconds(float(best_cpu_prep['cpu_prep']))}\n"
        f"trainer_prep_time_total: gpu={best_total['gpu']} gpu_prep={format_prep_seconds(total_gpu_prep)} "
        f"cpu_prep={format_prep_seconds(total_cpu_prep)} total={format_prep_seconds(total_gpu_prep + total_cpu_prep)}\n"
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
