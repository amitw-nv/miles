import time
from pathlib import Path
from unittest.mock import patch

from miles.backends.training_utils.weight_update.protocols.p2p_nixl_perf import (
    add_session_wire_time,
    add_wire_bytes,
    begin_wire_group,
    format_gib,
    format_perf_section,
    format_seconds,
    gather_and_write_perf_log,
    new_collector,
    perf_log_path,
    reset_collector,
    timed_call,
)

_MODULE = "miles.backends.training_utils.weight_update.protocols.p2p_nixl_perf"
_GIB = 1024**3


def _payload(
    *,
    gpu: int,
    wire_bytes: int,
    weight_version: int = 1,
    pp_rank: int = 0,
    gathered_dp_rank: int | None = None,
    wire_time: float = 0.0,
) -> dict[str, int | float]:
    return {
        "gpu": gpu,
        "pp_rank": pp_rank,
        "gathered_dp_rank": gathered_dp_rank if gathered_dp_rank is not None else gpu,
        "wire_bytes": wire_bytes,
        "wire_time": wire_time,
        "weight_version": weight_version,
    }


def _section(*, weight_version: int, bytes_gpu: int, bytes_gib: str, wire_gpu: int, wire_s: str) -> str:
    return (
        f"=== p2p nixl perf  weight_version={weight_version} ===\n"
        f"max_num_wire_bytes_per_trainer: gpu={bytes_gpu} bytes={bytes_gib}\n"
        f"wire_time: gpu={wire_gpu} work={wire_s}\n"
    )


class TestP2PNixlPerfHelpers:
    """Byte accounting, wire_time folds, and the rank-0 log section."""

    def test_format_gib_matches_the_log_example(self):
        """The log prints GiB with two decimals so a 12.40GiB line is readable without raw byte counts."""
        assert format_gib(int(12.40 * _GIB)) == "12.40GiB"
        assert format_gib(0) == "0.00GiB"

    def test_format_seconds_matches_the_log_example(self):
        """wire_time is printed as work=2.110s so the bottleneck GPU is comparable across ranks."""
        assert format_seconds(2.110) == "2.110s"
        assert format_seconds(0) == "0.000s"

    def test_log_section_picks_the_gpu_that_sent_the_most_and_the_gpu_with_most_rdma_work(self):
        """Rank 0 prints one bottleneck GPU per metric; they can be different GPUs."""
        text = format_perf_section(
            12,
            [
                _payload(gpu=2, wire_bytes=10, wire_time=2.110),
                _payload(gpu=5, wire_bytes=int(12.40 * _GIB), wire_time=0.4),
                _payload(gpu=1, wire_bytes=3, wire_time=1.0),
            ],
        )
        assert text == _section(weight_version=12, bytes_gpu=5, bytes_gib="12.40GiB", wire_gpu=2, wire_s="2.110s")

    def test_session_bytes_sum_and_reset_drops_the_previous_transfer(self):
        """Inner sessions and outer send_bucket steps add; the next update_weights must not inherit them."""
        collector = new_collector(gpu=3, pp_rank=1, gathered_dp_rank=4)
        reset_collector(collector, weight_version=7)
        add_wire_bytes(collector, 40)
        add_wire_bytes(collector, 60)
        assert collector.payload() == _payload(gpu=3, wire_bytes=100, weight_version=7, pp_rank=1, gathered_dp_rank=4)

        reset_collector(collector, weight_version=8)
        add_wire_bytes(collector, 5)
        assert collector.wire_bytes() == 5
        assert collector.weight_version == 8

    def test_parallel_sessions_count_once_as_max_then_replicas_sum(self):
        """wire_time: max within a replica (parallel sessions), then sum sequential replicas / buckets."""
        collector = new_collector(gpu=5, pp_rank=0, gathered_dp_rank=5)
        reset_collector(collector, weight_version=1)
        replica_a = begin_wire_group(collector)
        collector.add_session_wire_time(replica_a, 1.0)
        collector.add_session_wire_time(replica_a, 3.0)
        replica_b = begin_wire_group(collector)
        collector.add_session_wire_time(replica_b, 2.0)
        assert collector.wire_time() == 5.0

        reset_collector(collector, weight_version=2)
        assert collector.wire_time() == 0.0

    def test_timed_call_measures_until_the_function_returns(self):
        """wire_time starts right before `_do_nixl_write` and stops when it returns."""

        def _sleep() -> None:
            time.sleep(0.02)

        assert timed_call(_sleep) >= 0.02

    def test_helpers_are_noops_without_a_collector(self):
        """Mooncake P2P never constructs a collector, so the hooks must not throw."""
        reset_collector(None, weight_version=1)
        add_wire_bytes(None, 100)
        add_session_wire_time(None, 0.1)
        assert begin_wire_group(None) == 0

    def test_log_path_prefers_miles_log_dir(self, tmp_path: Path, monkeypatch):
        """These launchers bind MILES_LOG_DIR to the shared signals dir so the file is tail-able on Lustre."""
        monkeypatch.setenv("MILES_LOG_DIR", str(tmp_path))
        assert perf_log_path() == tmp_path / "p2p_nixl_perf.log"
        monkeypatch.delenv("MILES_LOG_DIR")
        monkeypatch.chdir(tmp_path)
        assert perf_log_path() == tmp_path / "p2p_nixl_perf.log"


class TestGatherAndWritePerfLog:
    """Rank 0 is the only writer; every gloo rank must still enter the gather."""

    def test_rank0_writes_the_max_and_skips_non_senders(self, tmp_path: Path):
        """Non-senders contribute None so one shared log is not a mix of empty ranks and real GPU bytes."""
        log_path = tmp_path / "p2p_nixl_perf.log"
        collector = new_collector(gpu=0, pp_rank=0, gathered_dp_rank=0)
        reset_collector(collector, weight_version=12)
        add_wire_bytes(collector, 10)
        gathered = [
            None,
            _payload(gpu=2, wire_bytes=55, weight_version=12, wire_time=0.5),
            _payload(gpu=5, wire_bytes=int(12.40 * _GIB), weight_version=12, wire_time=2.110),
        ]
        with patch(f"{_MODULE}._gather_payloads", return_value=gathered):
            gather_and_write_perf_log(collector, is_sender=True, log_path=log_path)
        assert log_path.read_text(encoding="utf-8") == _section(
            weight_version=12, bytes_gpu=5, bytes_gib="12.40GiB", wire_gpu=5, wire_s="2.110s"
        )

    def test_a_later_transfer_appends_a_new_section(self, tmp_path: Path):
        """Each update_weights is its own section; we do not add those transfers together."""
        log_path = tmp_path / "p2p_nixl_perf.log"
        collector = new_collector(gpu=0, pp_rank=0, gathered_dp_rank=0)
        reset_collector(collector, weight_version=1)
        with patch(
            f"{_MODULE}._gather_payloads",
            return_value=[_payload(gpu=1, wire_bytes=_GIB, weight_version=1, wire_time=1.0)],
        ):
            gather_and_write_perf_log(collector, is_sender=True, log_path=log_path)
        reset_collector(collector, weight_version=2)
        with patch(
            f"{_MODULE}._gather_payloads",
            return_value=[_payload(gpu=1, wire_bytes=2 * _GIB, weight_version=2, wire_time=2.0)],
        ):
            gather_and_write_perf_log(collector, is_sender=True, log_path=log_path)
        assert log_path.read_text(encoding="utf-8") == (
            _section(weight_version=1, bytes_gpu=1, bytes_gib="1.00GiB", wire_gpu=1, wire_s="1.000s")
            + _section(weight_version=2, bytes_gpu=1, bytes_gib="2.00GiB", wire_gpu=1, wire_s="2.000s")
        )

    def test_non_rank0_gathers_but_does_not_write(self, tmp_path: Path):
        """Many trainer processes must not append the same shared file."""
        log_path = tmp_path / "p2p_nixl_perf.log"
        collector = new_collector(gpu=1, pp_rank=0, gathered_dp_rank=1)
        with patch(f"{_MODULE}._gather_payloads", return_value=None) as gather:
            gather_and_write_perf_log(collector, is_sender=True, log_path=log_path)
        gather.assert_called_once()
        assert not log_path.exists()

    def test_mooncake_skips_the_collective(self, tmp_path: Path):
        """Mooncake has no collector; joining a gather would hang if only some ranks called it."""
        log_path = tmp_path / "p2p_nixl_perf.log"
        with patch(f"{_MODULE}._gather_payloads") as gather:
            gather_and_write_perf_log(None, is_sender=True, log_path=log_path)
        gather.assert_not_called()
        assert not log_path.exists()

    def test_non_sender_still_enters_the_gather(self, tmp_path: Path):
        """gather_object is a collective; a non-sender that skipped it would hang rank 0."""
        log_path = tmp_path / "p2p_nixl_perf.log"
        collector = new_collector(gpu=7, pp_rank=0, gathered_dp_rank=7)
        reset_collector(collector, weight_version=3)
        with patch(f"{_MODULE}._gather_payloads", return_value=None) as gather:
            gather_and_write_perf_log(collector, is_sender=False, log_path=log_path)
        gather.assert_called_once_with(None)
        assert not log_path.exists()
