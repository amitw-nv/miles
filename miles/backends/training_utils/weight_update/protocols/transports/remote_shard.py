from dataclasses import dataclass
from typing import NamedTuple


class RemoteWeightLocation(NamedTuple):
    address: int
    numel: int
    element_size: int


@dataclass(frozen=True)
class RemoteShard:
    """Published weight addresses of one engine rank's target or draft runner."""

    rollout_engine_ind: int
    rollout_engine_rank: int
    runner_role: str
    # what the transport names the runner it writes into: a Mooncake session or a NIXL agent
    transport_peer: str
    weight_locations_by_name: dict[str, RemoteWeightLocation]

    @property
    def published_nbytes_by_name(self) -> dict[str, int]:
        return {
            name: location.numel * location.element_size for name, location in self.weight_locations_by_name.items()
        }
