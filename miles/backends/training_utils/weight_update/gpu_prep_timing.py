"""Part timers for the trainer-side gather and convert work of a weight update.

The HF weight iterators call these without knowing who listens: they are no-ops until a protocol binds a sink with
`bind_gpu_prep_parts`, so the iterators never import a protocol.
"""

import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TypeVar

GPU_PREP_GATHER = "gather"
GPU_PREP_GATHER_LOAD = "gather_load"
GPU_PREP_GATHER_PP = "gather_pp"
GPU_PREP_GATHER_TP = "gather_tp"
GPU_PREP_GATHER_TP_START = "gather_tp_start"
GPU_PREP_GATHER_TP_WAIT = "gather_tp_wait"
GPU_PREP_GATHER_TP_CONCAT = "gather_tp_concat"
GPU_PREP_GATHER_EP = "gather_ep"
GPU_PREP_CONVERT = "convert"
GPU_PREP_STAGE = "stage"
GPU_PREP_CPU_LOAD = "cpu_load"

_T = TypeVar("_T")
_add_part: ContextVar[Callable[[str, float], None] | None] = ContextVar("gpu_prep_add_part", default=None)


def bind_gpu_prep_parts(add_part: Callable[[str, float], None] | None) -> None:
    """Route the part timers to `add_part(part, seconds)`, or to nothing when None."""
    _add_part.set(add_part)


@contextmanager
def time_gpu_prep_part(part: str):
    """Add the time spent in the block to `part`. Do not `yield` inside the block: a suspended generator would
    count the consumer's time."""
    add_part = _add_part.get()
    if add_part is None:
        yield
        return
    start = time.monotonic()
    try:
        yield
    finally:
        add_part(part, time.monotonic() - start)


def iter_timed_gpu_prep_part(part: str, inner: Iterable[_T]) -> Iterator[_T]:
    """Yield from `inner`, adding only the time each `next()` takes to `part`."""
    iterator = iter(inner)
    while True:
        with time_gpu_prep_part(part):
            try:
                item = next(iterator)
            except StopIteration:
                return
        yield item
