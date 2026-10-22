import logging
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from functools import partial

import torch

from miles.backends.training_utils.weight_update.protocols.transports.remote_shard import (
    RemoteShard,
    assert_tensors_fit,
)
from miles.utils.audit_utils.event_logger.logger import get_event_logger, is_event_logger_initialized
from miles.utils.audit_utils.event_logger.models import WeightTransferFailedEvent
from miles.utils.test_utils.fault_injector.actions.base import FaultHookContext
from miles.utils.test_utils.fault_injector.controller import fault_hook_controller, reach_fault_hook
from miles.utils.test_utils.fault_injector.models import FaultHookName

logger = logging.getLogger(__name__)


class RolloutEngineWriteThreads:
    """One write thread per rollout engine, so a stuck engine holds up only its own writes.

    `has_uncertain_writes` turns true once a native write fails: it may still read its source, so the transport
    keeps its registered memory from then on.
    """

    def __init__(self, thread_name_prefix: str) -> None:
        self._thread_name_prefix = thread_name_prefix
        self._executors_by_rollout_engine_ind: dict[int, ThreadPoolExecutor] = {}
        self.has_uncertain_writes = False

    def restart(self) -> None:
        """Called at connect: a rollout engine index now names a new engine, which must not queue behind the writes
        of the one it replaced."""
        for executor in self._executors_by_rollout_engine_ind.values():
            executor.shutdown(wait=False, cancel_futures=True)
        self._executors_by_rollout_engine_ind = {}

    def write_each_shard(
        self,
        remote_shards: Sequence[RemoteShard],
        tensors_by_name: dict[str, torch.Tensor],
        *,
        run_write: Callable[[RemoteShard, Callable[[], None]], None],
        write_shard: Callable[[RemoteShard, dict[str, torch.Tensor]], None],
    ) -> list[Future[None]]:
        """Runs the transport's native `write_shard`, which raises on failure, through `run_write` on each shard's
        rollout engine thread; returns one future per shard. Raises before writing anything if a shard lacks one of
        the weights or holds it in a different number of bytes."""
        for remote_shard in remote_shards:
            assert_tensors_fit(remote_shard, tensors_by_name)
        context = fault_hook_controller.current_context()
        return [
            self._get_or_create_executor(remote_shard.rollout_engine_ind).submit(
                run_write,
                remote_shard,
                partial(self._write_shard, remote_shard, tensors_by_name, write_shard=write_shard, context=context),
            )
            for remote_shard in remote_shards
        ]

    def _write_shard(
        self,
        remote_shard: RemoteShard,
        tensors_by_name: dict[str, torch.Tensor],
        *,
        write_shard: Callable[[RemoteShard, dict[str, torch.Tensor]], None],
        context: FaultHookContext,
    ) -> None:
        if not tensors_by_name:
            return
        reach_fault_hook(FaultHookName.TRAINER_WEIGHT_UPDATE_BEFORE_SEND)
        started_at = datetime.now(timezone.utc)
        try:
            write_shard(remote_shard, tensors_by_name)
        except Exception as error:
            self.has_uncertain_writes = True
            if is_event_logger_initialized() and context.debug_weight_update_id is not None:
                try:
                    get_event_logger().log(
                        WeightTransferFailedEvent,
                        dict(
                            debug_weight_update_id=context.debug_weight_update_id,
                            cell_id=remote_shard.cell_id,
                            workers_hash=context.snapshot_cell_id_to_hashes[remote_shard.cell_id],
                            started_at=started_at,
                            error=repr(error),
                        ),
                        include_context=False,
                    )
                except Exception:
                    logger.exception("Could not record P2P transport failure for %s", remote_shard.transport_peer)
            logger.exception("P2P transport write failed for %s", remote_shard.transport_peer)
            raise

    def _get_or_create_executor(self, rollout_engine_ind: int) -> ThreadPoolExecutor:
        if rollout_engine_ind not in self._executors_by_rollout_engine_ind:
            self._executors_by_rollout_engine_ind[rollout_engine_ind] = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix=f"{self._thread_name_prefix}-rollout-engine-{rollout_engine_ind}"
            )
        return self._executors_by_rollout_engine_ind[rollout_engine_ind]
