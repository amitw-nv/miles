import time
from pathlib import Path
from unittest.mock import patch

import pytest

from miles.backends.training_utils.weight_update.protocols.p2p_nixl_perf import (
    GPU_PREP_CONVERT,
    GPU_PREP_GATHER,
    GPU_PREP_STAGE,
    add_cpu_load,
    add_gpu_prep_part,
    add_session_cpu_setup,
    add_session_wire_time,
    add_wire_bytes,
    begin_wire_group,
    format_gb,
    format_perf_section,
    format_prep_seconds,
    format_seconds,
    gather_and_write_perf_log,
    new_collector,
    perf_log_path,
    reset_collector,
    stop_active_time,
    stop_gpu_prep,
    timed_call,
    timed_gpu_prep_part,
    wrap_gpu_prep_iter,
)

_MODULE = "miles.backends.training_utils.weight_update.protocols.p2p_nixl_perf"
_GB = 1000**3


def _payload(
    *,
    gpu: int,
    wire_bytes: int,
    weight_version: int = 1,
    pp_rank: int = 0,
    gathered_dp_rank: int | None = None,
    wire_time: float = 0.0,
    gpu_prep: float = 0.0,
    gpu_gather: float = 0.0,
    gpu_convert: float = 0.0,
    gpu_stage: float = 0.0,
    cpu_prep: float = 0.0,
    active_time: float = 0.0,
) -> dict[str, int | float]:
    return {
        "gpu": gpu,
        "pp_rank": pp_rank,
        "gathered_dp_rank": gathered_dp_rank if gathered_dp_rank is not None else gpu,
        "wire_bytes": wire_bytes,
        "wire_time": wire_time,
        "gpu_prep": gpu_prep,
        "gpu_gather": gpu_gather,
        "gpu_convert": gpu_convert,
        "gpu_stage": gpu_stage,
        "cpu_prep": cpu_prep,
        "active_time": active_time,
        "weight_version": weight_version,
    }


def _section(
    *,
    weight_version: int,
    bytes_gpu: int,
    bytes_gb: str,
    wire_gpu: int,
    wire_s: str,
    gpu_prep_gpu: int = 0,
    gpu_prep_s: str = "0.00s",
    cpu_prep_gpu: int = 0,
    cpu_prep_s: str = "0.00s",
    total_gpu: int = 0,
    total_gpu_prep_s: str = "0.00s",
    total_cpu_prep_s: str = "0.00s",
    total_s: str = "0.00s",
    gather_s: str = "0.00s",
    convert_s: str = "0.00s",
    stage_s: str = "0.00s",
    active_gpu: int = 0,
    active_s: str = "0.000s",
) -> str:
    return (
        f"=== p2p nixl perf  weight_version={weight_version} ===\n"
        f"max_num_wire_bytes_per_trainer: gpu={bytes_gpu} bytes={bytes_gb}\n"
        f"wire_time: gpu={wire_gpu} work={wire_s}\n"
        f"trainer_prep_time_gpu: gpu={gpu_prep_gpu} time={gpu_prep_s}\n"
        f"trainer_prep_time_cpu: gpu={cpu_prep_gpu} time={cpu_prep_s}\n"
        f"trainer_prep_time_total: gpu={total_gpu} gpu_prep={total_gpu_prep_s} "
        f"cpu_prep={total_cpu_prep_s} total={total_s}\n"
        f"trainer_prep_time_gpu_gather: gpu={total_gpu} time={gather_s}\n"
        f"trainer_prep_time_gpu_convert: gpu={total_gpu} time={convert_s}\n"
        f"trainer_prep_time_gpu_stage: gpu={total_gpu} time={stage_s}\n"
        f"trainer_active_time: gpu={active_gpu} time={active_s}\n"
    )


class TestP2PNixlPerfHelpers:
    """Byte accounting, wire_time folds, trainer_prep_time folds, and the rank-0 log section."""

    def test_format_gb_matches_the_log_example(self):
        """The log prints decimal GB with two decimals so a 13.31GB line is readable without raw byte counts."""
        assert format_gb(int(13.31 * _GB)) == "13.31GB"
        assert format_gb(0) == "0.00GB"

    def test_format_gb_is_decimal_not_binary(self):
        """Decimal base keeps wire bytes / wire_time directly comparable to NIC line rates."""
        assert format_gb(1000**3) == "1.00GB"
        assert format_gb(1024**3) == "1.07GB"

    def test_format_seconds_matches_the_log_example(self):
        """wire_time is printed as work=2.110s so the bottleneck GPU is comparable across ranks."""
        assert format_seconds(2.110) == "2.110s"
        assert format_seconds(0) == "0.000s"

    def test_format_prep_seconds_matches_the_log_example(self):
        """trainer_prep_time lines use two decimals, matching time=0.91s in the spec example."""
        assert format_prep_seconds(0.91) == "0.91s"
        assert format_prep_seconds(0) == "0.00s"

    def test_log_section_picks_the_gpu_that_sent_the_most_and_the_gpu_with_most_rdma_work(self):
        """Rank 0 prints one bottleneck GPU per metric; they can be different GPUs."""
        text = format_perf_section(
            12,
            [
                _payload(gpu=2, wire_bytes=10, wire_time=2.110, gpu_prep=0.20, cpu_prep=0.55, active_time=1.0),
                _payload(
                    gpu=5,
                    wire_bytes=int(13.31 * _GB),
                    wire_time=0.4,
                    gpu_prep=0.91,
                    gpu_gather=0.50,
                    gpu_convert=0.30,
                    gpu_stage=0.01,
                    cpu_prep=0.40,
                    active_time=3.410,
                ),
                _payload(gpu=1, wire_bytes=3, wire_time=1.0, gpu_prep=0.10, cpu_prep=0.10, active_time=2.0),
            ],
        )
        assert text == _section(
            weight_version=12,
            bytes_gpu=5,
            bytes_gb="13.31GB",
            wire_gpu=2,
            wire_s="2.110s",
            gpu_prep_gpu=5,
            gpu_prep_s="0.91s",
            cpu_prep_gpu=2,
            cpu_prep_s="0.55s",
            total_gpu=5,
            total_gpu_prep_s="0.91s",
            total_cpu_prep_s="0.40s",
            total_s="1.31s",
            gather_s="0.50s",
            convert_s="0.30s",
            stage_s="0.01s",
            active_gpu=5,
            active_s="3.410s",
        )

    def test_prep_total_is_argmax_on_the_same_gpu_not_sum_of_category_maxes(self):
        """Do not define trainer_prep_time_total as max(gpu_prep)+max(cpu_prep) across different GPUs."""
        text = format_perf_section(
            1,
            [
                _payload(gpu=2, wire_bytes=1, gpu_prep=0.20, cpu_prep=0.55),
                _payload(gpu=5, wire_bytes=1, gpu_prep=0.91, cpu_prep=0.40),
            ],
        )
        assert "trainer_prep_time_total: gpu=5 gpu_prep=0.91s cpu_prep=0.40s total=1.31s" in text
        assert "total=1.46s" not in text

    def test_gpu_prep_parts_come_from_the_total_gpu_not_independent_maxes(self):
        """Print gather/convert/stage from argmax(gpu_prep+cpu_prep), even if another GPU has a larger part."""
        text = format_perf_section(
            1,
            [
                _payload(
                    gpu=2,
                    wire_bytes=1,
                    gpu_prep=0.20,
                    cpu_prep=0.55,
                    gpu_gather=0.90,
                    gpu_convert=0.05,
                    gpu_stage=0.20,
                ),
                _payload(
                    gpu=5,
                    wire_bytes=1,
                    gpu_prep=0.91,
                    cpu_prep=0.40,
                    gpu_gather=0.50,
                    gpu_convert=0.30,
                    gpu_stage=0.01,
                ),
            ],
        )
        assert "trainer_prep_time_gpu_gather: gpu=5 time=0.50s" in text
        assert "trainer_prep_time_gpu_convert: gpu=5 time=0.30s" in text
        assert "trainer_prep_time_gpu_stage: gpu=5 time=0.01s" in text
        assert "gpu_gather: gpu=2" not in text
        assert "time=0.90s" not in text

    def test_gpu_prep_parts_sum_and_reset_with_the_collector(self):
        """Each GPU sums every gather/convert/stage call; the next update_weights drops them."""
        collector = new_collector(gpu=5, pp_rank=0, gathered_dp_rank=5)
        reset_collector(collector, weight_version=1)
        collector.add_gpu_prep_part(GPU_PREP_GATHER, 0.10)
        collector.add_gpu_prep_part(GPU_PREP_GATHER, 0.20)
        collector.add_gpu_prep_part(GPU_PREP_CONVERT, 0.05)
        collector.add_gpu_prep_part(GPU_PREP_STAGE, 0.01)
        assert collector.gpu_gather() == pytest.approx(0.30)
        assert collector.gpu_convert() == pytest.approx(0.05)
        assert collector.gpu_stage() == pytest.approx(0.01)

        reset_collector(collector, weight_version=2)
        assert collector.gpu_gather() == 0.0
        assert collector.gpu_convert() == 0.0
        assert collector.gpu_stage() == 0.0
        reset_collector(None, weight_version=0)

    def test_timed_gpu_prep_part_binds_through_reset_collector(self):
        """Iterator hooks call timed_gpu_prep_part; reset_collector must attach the collector."""
        collector = new_collector(gpu=1, pp_rank=0, gathered_dp_rank=1)
        reset_collector(collector, weight_version=1)

        def _sleep() -> str:
            time.sleep(0.02)
            return "ok"

        assert timed_gpu_prep_part(GPU_PREP_CONVERT, _sleep) == "ok"
        assert collector.gpu_convert() >= 0.02
        reset_collector(None, weight_version=0)

    def test_timed_gpu_prep_part_is_noop_without_a_collector(self):
        """Mooncake / unbound iterator must still run the function."""
        reset_collector(None, weight_version=1)
        assert timed_gpu_prep_part(GPU_PREP_GATHER, lambda: 7) == 7
        add_gpu_prep_part(GPU_PREP_STAGE, 1.0)

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

    def test_cpu_prep_sums_load_weights_and_maxes_parallel_setup(self):
        """cpu_prep: sum load_weights, max pointer-setup per replica, then sum replicas / buckets."""
        collector = new_collector(gpu=2, pp_rank=0, gathered_dp_rank=2)
        reset_collector(collector, weight_version=1)
        add_cpu_load(collector, 0.10)
        replica_a = begin_wire_group(collector)
        add_session_cpu_setup(collector, 0.50, group_id=replica_a)
        add_session_cpu_setup(collector, 0.30, group_id=replica_a)
        add_cpu_load(collector, 0.20)
        replica_b = begin_wire_group(collector)
        add_session_cpu_setup(collector, 0.40, group_id=replica_b)
        assert collector.cpu_prep() == pytest.approx(1.20)

        reset_collector(collector, weight_version=2)
        assert collector.cpu_prep() == 0.0
        assert collector.gpu_prep() == 0.0

    def test_gpu_prep_sums_outer_steps_and_wrap_starts_on_next(self):
        """gpu_prep starts at next(iterator) and accumulates until stop before load_weights."""
        collector = new_collector(gpu=5, pp_rank=0, gathered_dp_rank=5)
        reset_collector(collector, weight_version=1)
        wrapped = wrap_gpu_prep_iter(collector, iter(["a", "b"]))
        assert next(wrapped) == "a"
        time.sleep(0.02)
        stop_gpu_prep(collector)
        assert next(wrapped) == "b"
        time.sleep(0.02)
        stop_gpu_prep(collector)
        with pytest.raises(StopIteration):
            next(wrapped)
        assert collector.gpu_prep() >= 0.04

    def test_active_time_is_one_wall_clock_from_first_next_until_stop(self):
        """trainer_active_time starts once on the first next() and is not a sum of outer steps."""
        collector = new_collector(gpu=5, pp_rank=0, gathered_dp_rank=5)
        reset_collector(collector, weight_version=1)
        wrapped = wrap_gpu_prep_iter(collector, iter(["a", "b"]))
        assert next(wrapped) == "a"
        time.sleep(0.02)
        stop_gpu_prep(collector)
        assert next(wrapped) == "b"
        time.sleep(0.02)
        stop_gpu_prep(collector)
        stop_active_time(collector)
        assert collector.active_time() >= 0.04

        reset_collector(collector, weight_version=2)
        assert collector.active_time() == 0.0

    def test_wrap_gpu_prep_iter_is_identity_without_a_collector(self):
        """Mooncake has no collector; the updater wrap must not replace the iterator."""
        inner = iter([1])
        assert wrap_gpu_prep_iter(None, inner) is inner

    def test_timed_call_measures_until_the_function_returns(self):
        """wire_time starts right before `_do_nixl_write` and stops when it returns."""

        def _sleep() -> None:
            time.sleep(0.02)

        assert timed_call(_sleep) >= 0.02

    def test_timed_call_reraises(self):
        """load_weights / _do_nixl_write errors must not be swallowed by the timer."""

        def _boom() -> None:
            raise RuntimeError("load failed")

        with pytest.raises(RuntimeError, match="load failed"):
            timed_call(_boom)

    def test_helpers_are_noops_without_a_collector(self):
        """Mooncake P2P never constructs a collector, so the hooks must not throw."""
        reset_collector(None, weight_version=1)
        add_wire_bytes(None, 100)
        add_session_wire_time(None, 0.1)
        add_cpu_load(None, 0.1)
        add_session_cpu_setup(None, 0.1)
        stop_gpu_prep(None)
        stop_active_time(None)
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
            _payload(gpu=2, wire_bytes=55, weight_version=12, wire_time=0.5, gpu_prep=0.20, cpu_prep=0.55, active_time=1.0),
            _payload(
                gpu=5,
                wire_bytes=int(13.31 * _GB),
                weight_version=12,
                wire_time=2.110,
                gpu_prep=0.91,
                gpu_gather=0.50,
                gpu_convert=0.30,
                gpu_stage=0.01,
                cpu_prep=0.40,
                active_time=3.410,
            ),
        ]
        with patch(f"{_MODULE}._gather_payloads", return_value=gathered):
            gather_and_write_perf_log(collector, is_sender=True, log_path=log_path)
        assert log_path.read_text(encoding="utf-8") == _section(
            weight_version=12,
            bytes_gpu=5,
            bytes_gb="13.31GB",
            wire_gpu=5,
            wire_s="2.110s",
            gpu_prep_gpu=5,
            gpu_prep_s="0.91s",
            cpu_prep_gpu=2,
            cpu_prep_s="0.55s",
            total_gpu=5,
            total_gpu_prep_s="0.91s",
            total_cpu_prep_s="0.40s",
            total_s="1.31s",
            gather_s="0.50s",
            convert_s="0.30s",
            stage_s="0.01s",
            active_gpu=5,
            active_s="3.410s",
        )

    def test_a_later_transfer_appends_a_new_section(self, tmp_path: Path):
        """Each update_weights is its own section; we do not add those transfers together."""
        log_path = tmp_path / "p2p_nixl_perf.log"
        collector = new_collector(gpu=0, pp_rank=0, gathered_dp_rank=0)
        reset_collector(collector, weight_version=1)
        with patch(
            f"{_MODULE}._gather_payloads",
            return_value=[_payload(gpu=1, wire_bytes=_GB, weight_version=1, wire_time=1.0)],
        ):
            gather_and_write_perf_log(collector, is_sender=True, log_path=log_path)
        reset_collector(collector, weight_version=2)
        with patch(
            f"{_MODULE}._gather_payloads",
            return_value=[_payload(gpu=1, wire_bytes=2 * _GB, weight_version=2, wire_time=2.0)],
        ):
            gather_and_write_perf_log(collector, is_sender=True, log_path=log_path)
        assert log_path.read_text(encoding="utf-8") == (
            _section(
                weight_version=1,
                bytes_gpu=1,
                bytes_gb="1.00GB",
                wire_gpu=1,
                wire_s="1.000s",
                gpu_prep_gpu=1,
                cpu_prep_gpu=1,
                total_gpu=1,
                active_gpu=1,
                active_s="0.000s",
            )
            + _section(
                weight_version=2,
                bytes_gpu=1,
                bytes_gb="2.00GB",
                wire_gpu=1,
                wire_s="2.000s",
                gpu_prep_gpu=1,
                cpu_prep_gpu=1,
                total_gpu=1,
                active_gpu=1,
                active_s="0.000s",
            )
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
