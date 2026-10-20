import time
from concurrent.futures import Future
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import patch

import pytest

from miles.backends.training_utils.weight_update.gpu_prep_timing import (
    GPU_PREP_CONVERT,
    GPU_PREP_GATHER,
    GPU_PREP_GATHER_TP,
    GPU_PREP_GATHER_TP_WAIT,
    bind_gpu_prep_parts,
    iter_timed_gpu_prep_part,
    time_gpu_prep_part,
)

_GB = 1000**3


def _payload(**overrides: Any) -> dict[str, int | float]:
    payload: dict[str, int | float] = {
        "gpu": 0,
        "weight_version": 12,
        "wire_bytes": 0,
        "wire_time": 0.0,
        "wire_time_sum": 0.0,
        "write_queue": 0.0,
        "gpu_prep": 0.0,
        "cpu_prep": 0.0,
        "buffer_wait": 0.0,
        "drain": 0.0,
        "active": 0.0,
    }
    for part in (
        "gather",
        "gather_load",
        "gather_pp",
        "gather_tp",
        "gather_tp_start",
        "gather_tp_wait",
        "gather_tp_concat",
        "gather_ep",
        "convert",
        "stage",
        "cpu_load",
    ):
        payload[f"gpu_{part}"] = 0.0
    payload.update(overrides)
    return payload


@pytest.fixture(autouse=True)
def _unbind_parts():
    yield
    bind_gpu_prep_parts(None)


class TestPartTimers:
    def test_the_timers_do_nothing_until_a_sink_is_bound(self) -> None:
        """The iterators run on every rank and for every protocol; unbound, the timers must cost nothing visible."""
        with time_gpu_prep_part(GPU_PREP_CONVERT):
            pass
        assert list(iter_timed_gpu_prep_part(GPU_PREP_GATHER, iter([1, 2]))) == [1, 2]

    def test_a_bound_sink_gets_the_time_of_the_block(self) -> None:
        """Each part adds into its own bucket."""
        seen: list[tuple[str, float]] = []
        bind_gpu_prep_parts(lambda part, seconds: seen.append((part, seconds)))

        with time_gpu_prep_part(GPU_PREP_CONVERT):
            time.sleep(0.02)

        assert [part for part, _ in seen] == [GPU_PREP_CONVERT]
        assert seen[0][1] >= 0.02

    def test_an_iterator_is_timed_only_while_it_computes_the_next_item(self) -> None:
        """The consumer's work between items (here a sleep) is not the producer's, so it must not be counted."""
        seen: list[float] = []
        bind_gpu_prep_parts(lambda part, seconds: seen.append(seconds))

        def producer():
            yield 1
            yield 2

        for _ in iter_timed_gpu_prep_part(GPU_PREP_GATHER, producer()):
            time.sleep(0.05)

        assert len(seen) == 3 and sum(seen) < 0.05


class TestFoldWriteGroups:
    def test_overlapping_groups_count_once_in_busy_and_twice_in_sum_groups(self, perf_module: ModuleType) -> None:
        """Two buffers overlap their writes: the wire was busy 3s, while the old serial sum would say 4s."""
        stats = perf_module.WriteStats
        fold = perf_module.fold_write_groups([(0.0, [stats(10, 0.0, 2.0)]), (0.5, [stats(20, 1.0, 3.0)])])

        assert fold.nbytes == 30
        assert fold.busy == pytest.approx(3.0)
        assert fold.sum_groups == pytest.approx(4.0)

    def test_a_gap_between_writes_is_not_busy_time(self, perf_module: ModuleType) -> None:
        """Wire time is the wall clock with a write in flight, not the span from the first start to the last end."""
        stats = perf_module.WriteStats
        fold = perf_module.fold_write_groups([(0.0, [stats(1, 0.0, 1.0)]), (2.0, [stats(1, 5.0, 6.0)])])

        assert fold.busy == pytest.approx(2.0)

    def test_parallel_shards_of_a_group_count_once_in_sum_groups(self, perf_module: ModuleType) -> None:
        """Shards of one group run on different engine threads at the same time, like the old parallel sessions."""
        stats = perf_module.WriteStats
        fold = perf_module.fold_write_groups([(0.0, [stats(1, 0.0, 1.0), stats(1, 0.0, 3.0)])])

        assert fold.sum_groups == pytest.approx(3.0)
        assert fold.busy == pytest.approx(3.0)

    def test_queue_is_the_longest_wait_of_a_group_summed_over_groups_and_never_negative(
        self, perf_module: ModuleType
    ) -> None:
        """A write thread can start before `write()` returns; that is no queue, and must not subtract."""
        stats = perf_module.WriteStats
        fold = perf_module.fold_write_groups(
            [
                (1.0, [stats(1, 1.3, 2.0), stats(1, 1.2, 2.0)]),
                (5.0, [stats(1, 5.5, 6.0)]),
                (9.0, [stats(1, 8.9, 9.5)]),
            ]
        )

        assert fold.queue == pytest.approx(0.3 + 0.5 + 0.0)

    def test_no_writes_fold_to_zero(self, perf_module: ModuleType) -> None:
        """A sender whose runner loads nothing still reports."""
        assert perf_module.fold_write_groups([]) == (0, 0.0, 0.0, 0.0)


class TestCollector:
    def test_gather_is_the_sum_of_its_parts_and_the_tp_chunks_are_not_added_again(
        self, perf_module: ModuleType
    ) -> None:
        """A clock around the gather would also hold convert on the ETP == 1 expert path, so gather is a sum."""
        collector = perf_module.P2PNixlPerfCollector(gpu=5)
        collector.add("gather_load", 0.10)
        collector.add("gather_pp", 0.01)
        collector.add("gather_tp", 0.40)
        collector.add("gather_tp_wait", 0.30)
        collector.add("gather_ep", 0.04)
        collector.add("gather", 0.20)

        payload = collector.payload()

        assert payload["gpu_gather"] == pytest.approx(0.75)
        assert payload["gpu_gather_tp_wait"] == pytest.approx(0.30)

    def test_gpu_prep_is_the_iterator_time_plus_stage_plus_load_not_convert_or_gather(
        self, perf_module: ModuleType
    ) -> None:
        """Gather and convert run inside next(iterator); adding them again would count them twice."""
        collector = perf_module.P2PNixlPerfCollector(gpu=5)
        collector.add("next", 1.0)
        collector.add("gather_tp", 0.4)
        collector.add("convert", 0.3)
        collector.add("stage", 0.02)
        collector.add("cpu_load", 0.15)
        collector.add("cpu_prep", 0.05)

        payload = collector.payload()

        assert payload["gpu_prep"] == pytest.approx(1.17)
        assert payload["cpu_prep"] == pytest.approx(0.05)

    def test_an_unknown_kind_is_rejected(self, perf_module: ModuleType) -> None:
        """A typo in a hook name must not silently drop its time."""
        with pytest.raises(ValueError, match="unknown perf kind"):
            perf_module.P2PNixlPerfCollector(gpu=0).add("nope", 1.0)

    def test_reset_drops_the_previous_transfer(self, perf_module: ModuleType) -> None:
        """Each update_weights is its own log section."""
        collector = perf_module.P2PNixlPerfCollector(gpu=0)
        collector.add("drain", 1.0)
        collector.reset(8)

        assert collector.payload() == _payload(weight_version=8)

    def test_active_starts_at_the_first_next_and_stops_at_finish_and_next_time_includes_the_last_next(
        self, perf_module: ModuleType
    ) -> None:
        """Active is one wall clock, not a sum of nexts; the final next that raises StopIteration still did work."""
        collector = perf_module.P2PNixlPerfCollector(gpu=0)

        wrapped = collector.wrap_buckets(iter(["a", "b"]))
        assert next(wrapped) == "a"
        time.sleep(0.02)
        assert next(wrapped) == "b"
        with pytest.raises(StopIteration):
            next(wrapped)
        time.sleep(0.02)
        collector.finish()

        payload = collector.payload()
        assert payload["active"] >= 0.04
        assert payload["gpu_prep"] < payload["active"]

    def test_finish_folds_the_stats_on_the_futures_of_every_group(self, perf_module: ModuleType) -> None:
        """The write threads only return stats; the main thread folds them after every write has returned."""
        stats = perf_module.WriteStats
        collector = perf_module.P2PNixlPerfCollector(gpu=0)
        futures = [Future(), Future()]
        collector.record_write_group(futures)
        time.sleep(0.01)
        futures[0].set_result(stats(7, time.monotonic(), time.monotonic() + 1.0))
        futures[1].set_result(stats(9, time.monotonic(), time.monotonic() + 2.0))

        collector.finish()

        payload = collector.payload()
        assert payload["wire_bytes"] == 16
        assert payload["wire_time"] == pytest.approx(2.0, abs=0.05)
        assert payload["write_queue"] >= 0.0

    def test_no_collector_means_no_measurement(self, perf_module: ModuleType) -> None:
        """Mooncake and ranks that send nothing have no collector; the hooks must still run their block."""
        ran = []
        with perf_module.measure(None, "drain"):
            ran.append(1)
        assert ran == [1]


class TestFormatPerfSection:
    def test_each_metric_names_its_own_bottleneck_gpu(self, perf_module: ModuleType) -> None:
        """The GPU with the most bytes, the busiest wire, the most exposed wire and the longest transfer can differ."""
        text = perf_module.format_perf_section(
            [
                _payload(gpu=2, wire_bytes=10, wire_time=2.11, wire_time_sum=3.48, write_queue=0.62, active=1.0),
                _payload(
                    gpu=5,
                    wire_bytes=int(13.31 * _GB),
                    wire_time=0.4,
                    buffer_wait=0.4,
                    drain=0.35,
                    active=2.6,
                ),
            ]
        )

        assert "max_num_wire_bytes_per_trainer: gpu=5 bytes=13.31GB\n" in text
        assert "wire_time: gpu=2 busy=2.110s sum_groups=3.480s\n" in text
        assert "write_queue: gpu=2 time=0.62s\n" in text
        assert "trainer_exposed_wire_time: gpu=5 buffer_wait=0.40s drain=0.35s total=0.75s\n" in text
        assert "trainer_active_time: gpu=5 time=2.600s\n" in text

    def test_the_prep_total_is_one_gpu_not_the_sum_of_two_gpus_maxes(self, perf_module: ModuleType) -> None:
        """max(gpu_prep) + max(cpu_prep) over different GPUs describes no real GPU."""
        text = perf_module.format_perf_section(
            [
                _payload(gpu=2, gpu_prep=0.20, cpu_prep=0.55),
                _payload(gpu=5, gpu_prep=0.91, cpu_prep=0.40, gpu_convert=0.30, gpu_gather=0.50),
            ]
        )

        assert "trainer_prep_time_gpu: gpu=5 time=0.91s\n" in text
        assert "trainer_prep_time_cpu: gpu=2 time=0.55s\n" in text
        assert "trainer_prep_time_total: gpu=5 gpu_prep=0.91s cpu_prep=0.40s total=1.31s\n" in text
        assert "total=1.46s" not in text
        assert "trainer_prep_time_gpu_convert: gpu=5 time=0.30s\n" in text
        assert "trainer_prep_time_gpu_gather: gpu=5 time=0.50s\n" in text

    def test_only_the_first_transfer_is_marked_warm_up(self, perf_module: ModuleType) -> None:
        """Later transfers are steady state."""
        assert "weight_version=1 (warm-up) ===" in perf_module.format_perf_section([_payload(weight_version=1)])
        assert perf_module.format_perf_section([_payload(weight_version=2)]).startswith(
            "=== p2p nixl perf  weight_version=2 ===\n"
        )


class TestWritePerfLog:
    def test_rank0_appends_one_section_per_transfer_and_skips_non_senders(
        self, perf_module: ModuleType, tmp_path: Path
    ) -> None:
        """Non-senders contribute None; the shared log holds only real GPUs."""
        log_path = tmp_path / "p2p_nixl_perf.log"
        collector = perf_module.P2PNixlPerfCollector(gpu=0)
        for weight_version in (3, 4):
            with patch.object(
                perf_module, "_gather_payloads", return_value=[None, _payload(gpu=2, weight_version=weight_version)]
            ):
                perf_module.write_perf_log(collector, log_path=log_path)

        assert log_path.read_text(encoding="utf-8").count("=== p2p nixl perf") == 2

    def test_a_rank_that_is_not_rank0_gathers_but_writes_nothing(
        self, perf_module: ModuleType, tmp_path: Path
    ) -> None:
        """Many trainer processes must not append to the same file."""
        log_path = tmp_path / "p2p_nixl_perf.log"
        with patch.object(perf_module, "_gather_payloads", return_value=None):
            perf_module.write_perf_log(perf_module.P2PNixlPerfCollector(gpu=1), log_path=log_path)

        assert not log_path.exists()

    def test_a_rank_that_sends_nothing_still_joins_the_gather(self, perf_module: ModuleType, tmp_path: Path) -> None:
        """gather_object is a collective: a rank that skipped it would hang rank 0."""
        with patch.object(perf_module, "_gather_payloads", return_value=None) as gather:
            perf_module.write_perf_log(None, log_path=tmp_path / "p2p_nixl_perf.log")

        gather.assert_called_once_with(None)

    def test_the_log_path_prefers_miles_log_dir(self, perf_module: ModuleType, tmp_path: Path, monkeypatch) -> None:
        """The launchers bind MILES_LOG_DIR to the shared signals dir so the file can be tailed on Lustre."""
        monkeypatch.setenv("MILES_LOG_DIR", str(tmp_path))
        assert perf_module.perf_log_path() == tmp_path / "p2p_nixl_perf.log"
        monkeypatch.delenv("MILES_LOG_DIR")
        monkeypatch.chdir(tmp_path)
        assert perf_module.perf_log_path() == tmp_path / "p2p_nixl_perf.log"
