"""NIXL P2P trainer-side perf log. Rank 0 appends one section per full weight transfer."""

import os
from dataclasses import dataclass, field
from pathlib import Path

LOG_FILENAME = "p2p_nixl_perf.log"
_GIB = 1024**3


@dataclass
class P2PNixlPerfCollector:
    """Per-rank counters for one `update_weights` NIXL P2P transfer.

    Identity fields (`gpu`, `pp_rank`, `gathered_dp_rank`) ride on every payload so
    rank 0 can label the bottleneck GPU in the log. Wire-byte parts feed
    `max_num_wire_bytes_per_trainer`; later timers (wire_time, trainer_prep_time,
    trainer_active_time) will live on this same object.
    """

    gpu: int
    pp_rank: int
    gathered_dp_rank: int
    weight_version: int = 0
    _wire_bytes_parts: list[int] = field(default_factory=list)

    def reset(self, weight_version: int) -> None:
        """Clear counters at the start of this `update_weights`.

        Each full transfer is its own log section; we do not add versions together.
        Used by every metric that is summed or maxed over one transfer.
        """
        self.weight_version = weight_version
        self._wire_bytes_parts = []

    def add_wire_bytes(self, num_bytes: int) -> None:
        """Record RDMA payload size for one inner session write.

        `max_num_wire_bytes_per_trainer`: one `sum(source_lens)` per
        `_do_p2p_write_one_session`. Sessions of one replica, sequential replicas,
        and outer `send_bucket` steps all append; the total is the GPU's wire bytes.
        """
        self._wire_bytes_parts.append(num_bytes)

    def wire_bytes(self) -> int:
        """Total bytes this GPU put on the wire in this transfer.

        Fold for `max_num_wire_bytes_per_trainer` (sum of every inner session).
        """
        return sum(self._wire_bytes_parts)

    def payload(self) -> dict[str, int]:
        """Snapshot this rank sends on the gloo gather.

        Rank 0 uses `wire_bytes` to pick `max_num_wire_bytes_per_trainer`, and keeps
        `gpu` / `pp_rank` / `gathered_dp_rank` to label that line in the log.
        """
        return {
            "gpu": self.gpu,
            "pp_rank": self.pp_rank,
            "gathered_dp_rank": self.gathered_dp_rank,
            "wire_bytes": self.wire_bytes(),
            "weight_version": self.weight_version,
        }


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


def perf_log_path() -> Path:
    """Path of the shared append-only log rank 0 writes (`p2p_nixl_perf.log`).

    All metrics in that file share this path. `MILES_LOG_DIR` if set, else cwd.
    """
    return Path(os.environ.get("MILES_LOG_DIR") or os.getcwd()) / LOG_FILENAME


def format_gib(num_bytes: int) -> str:
    """Render wire bytes as `12.40GiB` for the `max_num_wire_bytes_per_trainer` log line."""
    return f"{num_bytes / _GIB:.2f}GiB"


def format_perf_section(weight_version: int, payloads: list[dict[str, int]]) -> str:
    """One log section for this `weight_version`.

    Currently prints `max_num_wire_bytes_per_trainer` (GPU with the most wire bytes).
    Later metrics (`wire_time`, `trainer_prep_time_*`, `trainer_active_time`) add lines here.
    """
    best = max(payloads, key=lambda payload: payload["wire_bytes"])
    return (
        f"=== p2p nixl perf  weight_version={weight_version} ===\n"
        f"max_num_wire_bytes_per_trainer: gpu={best['gpu']} bytes={format_gib(best['wire_bytes'])}\n"
    )


def gather_and_write_perf_log(
    collector: P2PNixlPerfCollector | None,
    *,
    is_sender: bool,
    log_path: Path | None = None,
) -> None:
    """Gloo-gather every rank's payload; rank 0 appends the bottleneck GPU to the log.

    Non-senders send `None`. Rank 0 writes `max_num_wire_bytes_per_trainer` (and later
    the other metric lines) so many processes do not append the same file.
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


def _gather_payloads(payload: dict[str, int] | None) -> list[dict[str, int] | None] | None:
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
