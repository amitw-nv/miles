from types import ModuleType
from typing import Any

import pytest
import torch

from miles.backends.training_utils.weight_update.protocols.transports.remote_shard import (
    RemoteShard,
    RemoteWeightLocation,
)


@pytest.fixture
def p2p_transport() -> str:
    return "mooncake"


def test_a_weight_of_another_size_on_the_shard_is_never_written(p2p_sender: Any, mooncake_module: ModuleType) -> None:
    """Writing a weight whose published size differs would run past the target's memory."""
    transport = mooncake_module.MooncakeTransport()
    remote_shard = RemoteShard(
        rollout_engine_ind=0,
        rollout_engine_rank=0,
        runner_role="target",
        transport_peer="cell-a-r0",
        weight_locations_by_name={"w": RemoteWeightLocation(address=0x1000, numel=2, element_size=4)},
    )

    with pytest.raises(
        AssertionError, match="w is 16 bytes here but 8 bytes on the target of rollout engine 0 rank 0"
    ):
        transport.write([remote_shard], {"w": torch.zeros(4)})
    assert p2p_sender.transfer_engine.writes == []


def test_a_rollout_engine_publishing_for_another_transport_is_rejected_at_connect(
    p2p_sender: Any, make_rollout_api: Any
) -> None:
    """Mooncake cannot reach memory the engine registered with another transport; its writes would all fail."""
    protocol = p2p_sender.make_protocol()

    with pytest.raises(AssertionError, match="publishes its weights for nixl, but --p2p-transport is mooncake"):
        p2p_sender.connect(protocol, [make_rollout_api("cell-a", gpu_count=1, transport="nixl")])


def test_a_mooncake_transfer_writes_no_perf_log(p2p_sender: Any, make_rollout_api: Any, make_bucket: Any) -> None:
    """The perf log is NIXL's: Mooncake's writes return no stats, and its ranks must not join the log's gather."""
    protocol = p2p_sender.make_protocol()
    p2p_sender.connect(protocol, [make_rollout_api("cell-a", gpu_count=1)])
    p2p_sender.begin_sync(protocol, weight_version=1)

    protocol.send_bucket(make_bucket("hf.w"))
    protocol.after_base_weights()

    assert p2p_sender.perf_payloads == []
    assert not p2p_sender.perf_log_path.exists()
