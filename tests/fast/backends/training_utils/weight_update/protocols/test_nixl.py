import time
from types import ModuleType
from typing import Any

import pytest
import torch

from miles.backends.training_utils.weight_update.gpu_prep_timing import GPU_PREP_CONVERT, time_gpu_prep_part
from miles.backends.training_utils.weight_update.protocols.utils.rollout_engine_rank_assignment import (
    assign_rollout_engine_ranks_for_data_replica,
)


@pytest.fixture
def p2p_transport() -> str:
    return "nixl"


def test_a_write_addresses_the_gpu_its_rollout_engine_rank_published(
    p2p_sender: Any, make_rollout_api: Any, make_bucket: Any
) -> None:
    """A NIXL write names the remote GPU; another rank's would write into the wrong device's memory."""
    protocol = p2p_sender.make_protocol()
    api = make_rollout_api("cell-a", gpu_count=2)
    p2p_sender.connect(protocol, [api])
    p2p_sender.begin_sync(protocol, weight_version=1)

    protocol.send_bucket(make_bucket("hf.w"))
    protocol.after_base_weights()

    assert p2p_sender.nixl_agent.remote_gpu_ids_by_remote_agent == {api.session_id(0): {0}, api.session_id(1): {1}}


def test_a_transfer_still_running_after_the_timeout_fails_its_write(
    p2p_sender: Any, p2p_protocol: ModuleType, make_rollout_api: Any
) -> None:
    """NIXL never abandons a transfer itself: a lost rollout engine would hold its write thread, and every later
    write to that engine, forever."""
    transport = p2p_protocol.NixlTransport(write_timeout=0.05)
    api = make_rollout_api("cell-a", gpu_count=1)
    assignments = assign_rollout_engine_ranks_for_data_replica(
        data_replica_rank=0, data_replica_size=1, engine_gpu_counts=[1]
    )
    (remote_shard,) = transport.connect([api], assignments, ["target"])["target"][0]
    buffer = torch.zeros(4)
    transport.register_memory(buffer)
    p2p_sender.nixl_agent.stuck_remote_agents.add(api.session_id(0))

    (write,) = transport.write([remote_shard], {"w": buffer})

    with pytest.raises(RuntimeError, match=r"NIXL WRITE of 1 weights is PROC after 0.05s"):
        write.result(timeout=5)


def test_a_reconnect_removes_the_agents_of_the_rollout_engines_it_replaces(
    p2p_sender: Any, make_rollout_api: Any
) -> None:
    """Agents left loaded for replaced engines would pile up, one set per rollout engine restart."""
    protocol = p2p_sender.make_protocol()
    p2p_sender.connect(protocol, [make_rollout_api("cell-a", gpu_count=1)])

    replaced_api = make_rollout_api("cell-a", gpu_count=1, generation=2)
    p2p_sender.connect(protocol, [replaced_api])

    assert p2p_sender.nixl_agent.remote_agents == [replaced_api.session_id(0)]


def _run_transfer(p2p_sender: Any, protocol: Any, make_bucket: Any, weight_version: int) -> None:
    p2p_sender.begin_sync(protocol, weight_version=weight_version)
    for bucket in protocol.wrap_bucket_iter(iter([make_bucket("hf.w")])):
        protocol.send_bucket(bucket)
    protocol.after_base_weights()


class TestPerfLog:
    def test_a_transfer_logs_the_bytes_it_wrote_and_the_clocks_it_ran(
        self, p2p_sender: Any, make_rollout_api: Any, make_bucket: Any
    ) -> None:
        """Each of the two shards is written the 16 bytes of `w`; every clock of the transfer has a value."""
        protocol = p2p_sender.make_protocol()
        p2p_sender.connect(protocol, [make_rollout_api("cell-a", gpu_count=2)])

        _run_transfer(p2p_sender, protocol, make_bucket, weight_version=2)

        (payload,) = p2p_sender.perf_payloads
        assert payload["wire_bytes"] == 32
        assert payload["wire_time"] > 0
        assert payload["active"] > 0 and payload["active"] >= payload["drain"]
        assert payload["gpu_prep"] >= payload["gpu_cpu_load"] + payload["gpu_stage"]
        assert payload["cpu_prep"] > 0
        log = p2p_sender.perf_log_path.read_text(encoding="utf-8")
        assert log.startswith("=== p2p nixl perf  weight_version=2 ===\n")
        assert "max_num_wire_bytes_per_trainer: gpu=0 bytes=0.00GB\n" in log

    def test_the_first_transfer_is_marked_warm_up(
        self, p2p_sender: Any, make_rollout_api: Any, make_bucket: Any
    ) -> None:
        """The first load of every param runs the byte-coverage check, so its timings are not steady state."""
        protocol = p2p_sender.make_protocol()
        p2p_sender.connect(protocol, [make_rollout_api("cell-a", gpu_count=1)])

        _run_transfer(p2p_sender, protocol, make_bucket, weight_version=1)

        assert "weight_version=1 (warm-up) ===" in p2p_sender.perf_log_path.read_text(encoding="utf-8")

    def test_a_later_transfer_is_its_own_section_not_added_to_the_first(
        self, p2p_sender: Any, make_rollout_api: Any, make_bucket: Any
    ) -> None:
        """Counters are reset at the end of begin_sync, so a transfer does not inherit the previous one's bytes."""
        protocol = p2p_sender.make_protocol()
        p2p_sender.connect(protocol, [make_rollout_api("cell-a", gpu_count=2)])

        _run_transfer(p2p_sender, protocol, make_bucket, weight_version=1)
        _run_transfer(p2p_sender, protocol, make_bucket, weight_version=2)

        assert [payload["wire_bytes"] for payload in p2p_sender.perf_payloads] == [32, 32]
        log = p2p_sender.perf_log_path.read_text(encoding="utf-8")
        assert log.count("=== p2p nixl perf") == 2

    def test_gather_and_convert_time_recorded_before_begin_sync_returns_is_dropped(
        self, p2p_sender: Any, make_rollout_api: Any, make_bucket: Any
    ) -> None:
        """begin_sync resets at its end, so what an earlier pass of the iterator recorded (the first begin_sync runs a
        probe pass of it) is not billed to the transfer."""
        protocol = p2p_sender.make_protocol()
        p2p_sender.connect(protocol, [make_rollout_api("cell-a", gpu_count=1)])
        p2p_sender.begin_sync(protocol, weight_version=1)
        with time_gpu_prep_part(GPU_PREP_CONVERT):
            time.sleep(0.01)

        p2p_sender.begin_sync(protocol, weight_version=2)
        protocol.send_bucket(make_bucket("hf.w"))
        protocol.after_base_weights()

        assert p2p_sender.perf_payloads[0]["gpu_convert"] == 0.0
