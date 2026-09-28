from pathlib import Path
from unittest.mock import patch

from miles.backends.training_utils.weight_update.protocols.p2p_nixl_perf import (
    add_wire_bytes,
    format_gib,
    format_perf_section,
    gather_and_write_perf_log,
    new_collector,
    perf_log_path,
    reset_collector,
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
) -> dict[str, int]:
    return {
        "gpu": gpu,
        "pp_rank": pp_rank,
        "gathered_dp_rank": gathered_dp_rank if gathered_dp_rank is not None else gpu,
        "wire_bytes": wire_bytes,
        "weight_version": weight_version,
    }


class TestP2PNixlPerfHelpers:
    """Byte accounting and the rank-0 log section for max_num_wire_bytes_per_trainer."""

    def test_format_gib_matches_the_log_example(self):
        """The log prints GiB with two decimals so a 12.40GiB line is readable without raw byte counts."""
        assert format_gib(int(12.40 * _GIB)) == "12.40GiB"
        assert format_gib(0) == "0.00GiB"

    def test_log_section_picks_the_gpu_that_sent_the_most(self):
        """Rank 0 prints one bottleneck GPU, not a sum across trainers."""
        text = format_perf_section(
            12,
            [
                _payload(gpu=2, wire_bytes=10),
                _payload(gpu=5, wire_bytes=int(12.40 * _GIB)),
                _payload(gpu=1, wire_bytes=3),
            ],
        )
        assert text == (
            "=== p2p nixl perf  weight_version=12 ===\n" "max_num_wire_bytes_per_trainer: gpu=5 bytes=12.40GiB\n"
        )

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

    def test_helpers_are_noops_without_a_collector(self):
        """Mooncake P2P never constructs a collector, so the hooks must not throw."""
        reset_collector(None, weight_version=1)
        add_wire_bytes(None, 100)

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
            _payload(gpu=2, wire_bytes=55, weight_version=12),
            _payload(gpu=5, wire_bytes=int(12.40 * _GIB), weight_version=12),
        ]
        with patch(f"{_MODULE}._gather_payloads", return_value=gathered):
            gather_and_write_perf_log(collector, is_sender=True, log_path=log_path)
        assert log_path.read_text(encoding="utf-8") == (
            "=== p2p nixl perf  weight_version=12 ===\n" "max_num_wire_bytes_per_trainer: gpu=5 bytes=12.40GiB\n"
        )

    def test_a_later_transfer_appends_a_new_section(self, tmp_path: Path):
        """Each update_weights is its own section; we do not add those transfers together."""
        log_path = tmp_path / "p2p_nixl_perf.log"
        collector = new_collector(gpu=0, pp_rank=0, gathered_dp_rank=0)
        reset_collector(collector, weight_version=1)
        with patch(f"{_MODULE}._gather_payloads", return_value=[_payload(gpu=1, wire_bytes=_GIB, weight_version=1)]):
            gather_and_write_perf_log(collector, is_sender=True, log_path=log_path)
        reset_collector(collector, weight_version=2)
        with patch(
            f"{_MODULE}._gather_payloads", return_value=[_payload(gpu=1, wire_bytes=2 * _GIB, weight_version=2)]
        ):
            gather_and_write_perf_log(collector, is_sender=True, log_path=log_path)
        assert log_path.read_text(encoding="utf-8") == (
            "=== p2p nixl perf  weight_version=1 ===\n"
            "max_num_wire_bytes_per_trainer: gpu=1 bytes=1.00GiB\n"
            "=== p2p nixl perf  weight_version=2 ===\n"
            "max_num_wire_bytes_per_trainer: gpu=1 bytes=2.00GiB\n"
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
