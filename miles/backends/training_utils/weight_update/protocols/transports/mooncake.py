import os
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from typing import Any

import ray
import torch

from miles.backends.training_utils.weight_update.protocols.transports.remote_shard import (
    RemoteShard,
    parse_weight_locations,
    query_remote_shards,
)
from miles.backends.training_utils.weight_update.protocols.transports.rollout_engine_write_threads import (
    RolloutEngineWriteThreads,
)
from miles.backends.training_utils.weight_update.protocols.utils.rollout_engine_rank_assignment import (
    RolloutEngineRankAssignment,
)
from miles.backends.training_utils.weight_update.rollout_cell_updater import _RolloutCellUpdater


class MooncakeTransport:
    """Writes tensors from this trainer process's registered memory into rollout engines over Mooncake.

    The p2p protocol keeps one for the whole trainer process: it registers its transfer buffers once, calls
    `connect` with each new set of rollout engines, then `write` for each rollout engine rank. Each rollout engine
    has its own write thread, so a stuck engine holds up only its own writes.
    """

    # host memory: no trainer GPU memory, and no GPUDirect RDMA needed
    transfer_buffer_device = torch.device("cpu")

    def __init__(self) -> None:
        self._transfer_engine = _create_transfer_engine()
        self._registered_tensors: dict[int, torch.Tensor] = {}
        self._write_threads = RolloutEngineWriteThreads(thread_name_prefix="mooncake-write")

    def connect(
        self,
        cell_updaters_of_cell_id: dict[str, _RolloutCellUpdater],
        engine_cell_ids: Sequence[str],
        assignments: Sequence[RolloutEngineRankAssignment],
        runner_roles: Sequence[str],
    ) -> dict[str, dict[int, list[RemoteShard]]]:
        """Return assigned shards by runner role and rollout rank."""
        self._write_threads.restart()
        return query_remote_shards(
            cell_updaters_of_cell_id, engine_cell_ids, assignments, runner_roles, "mooncake", _build_remote_shard
        )

    def register_memory(self, tensor: torch.Tensor) -> None:
        ret = self._transfer_engine.register_memory(tensor.data_ptr(), tensor.nbytes)
        if ret != 0:
            raise RuntimeError(
                f"Mooncake could not register {tensor.nbytes} bytes at {tensor.data_ptr():#x}: error {ret}"
            )

        self._registered_tensors[tensor.data_ptr()] = tensor

    def unregister_memory(self, tensor: torch.Tensor) -> None:
        if self._write_threads.has_uncertain_writes:
            raise RuntimeError(
                "Cannot release registered memory after a Mooncake write with unknown native completion"
            )
        ret = self._transfer_engine.unregister_memory(tensor.data_ptr())
        if ret != 0:
            raise RuntimeError(f"Mooncake could not unregister the memory at {tensor.data_ptr():#x}: error {ret}")
        del self._registered_tensors[tensor.data_ptr()]

    def write(
        self,
        remote_shards: Sequence[RemoteShard],
        tensors_by_name: dict[str, torch.Tensor],
        *,
        run_write: Callable[[RemoteShard, Callable[[], None]], None],
    ) -> list[Future[None]]:
        """Writes each tensor into the weight of the same name on every shard; returns one future per shard.

        Allocations must stay registered while native writes may still read them. Healthy receivers require
        unchanged bytes until their futures finish.
        """
        return self._write_threads.write_each_shard(
            remote_shards, tensors_by_name, run_write=run_write, write_shard=self._write_shard
        )

    def _write_shard(self, remote_shard: RemoteShard, tensors_by_name: dict[str, torch.Tensor]) -> None:
        names = list(tensors_by_name)
        ret = self._transfer_engine.batch_transfer_sync_write(
            remote_shard.transport_peer,
            [tensors_by_name[name].data_ptr() for name in names],
            [remote_shard.weight_locations_by_name[name].address for name in names],
            [tensors_by_name[name].nbytes for name in names],
        )
        if ret < 0:
            raise RuntimeError(f"Mooncake batch_transfer_sync_write returned {ret}")


def _build_remote_shard(
    transfer_engine_info: dict, cell_id: str, rollout_engine_ind: int, rollout_engine_rank: int, runner_role: str
) -> RemoteShard:
    return RemoteShard(
        cell_id=cell_id,
        rollout_engine_ind=rollout_engine_ind,
        rollout_engine_rank=rollout_engine_rank,
        runner_role=runner_role,
        transport_peer=transfer_engine_info["session_id"],
        weight_locations_by_name=parse_weight_locations(transfer_engine_info),
    )


def _create_transfer_engine() -> Any:
    # not in miles' requirements
    from mooncake.engine import TransferEngine

    transfer_engine = TransferEngine()
    transfer_engine.initialize(
        ray._private.services.get_node_ip_address(),
        "P2PHANDSHAKE",
        os.environ.get("MOONCAKE_PROTOCOL", "rdma"),
        os.environ.get("MOONCAKE_DEVICE", ""),
    )
    return transfer_engine
