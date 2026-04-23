
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

# Primitives
Position = tuple[int, int]   # (row, col)
Grid = list[list[int]]        # grid[row][col]: 0 = free, 1 = obstacle
Path = list[Position]         # path[t] = position of one agent at timestep t                           # path[0] = start, path[-1] = goal
Plan = dict[int, Path]    # agent_id -> path


# Agent
@dataclass
class Agent:
    id: int
    start: Position
    goal: Position

# Class for local group of agents with a local leader
@dataclass
class LocalGroup:
    leader_id: int
    member_ids: list[int]   # includes leader_id
    local_view: set[Position] = field(default_factory=set) # non-obstacle cells the leader sees


# Config
@dataclass
class LocalLeadersConfig:
    group_radius: int = 5           # Chebyshev radius used to cluster agents
    max_group_size: int = 10        # hard cap on agents per group
    leader_view_radius: int = 3     # cells the leader can see beyond its agents
    time_limit_sec: float = 60.0
    seed: int = 0


# Resfult of a single run
@dataclass
class SolveResult:
    solved: bool
    soc: Optional[int] = None               # sum of individual path costs
    makespan: Optional[int] = None          # longest individual path cost
    comp_time_ms: float = 0.0
    num_groups: Optional[int] = None
    avg_group_size: Optional[float] = None
    num_conflicts_resolved: int = 0         # conflicts fixed during inter-group resolution
    solution: Optional[Plan] = None

    def to_dict(self) -> dict:
        return {
            "solved": self.solved,
            "soc": self.soc,
            "makespan": self.makespan,
            "comp_time_ms": round(self.comp_time_ms, 2),
            "num_groups": self.num_groups,
            "avg_group_size": (
                round(self.avg_group_size, 2) if self.avg_group_size is not None else None
            ),
            "num_conflicts_resolved": self.num_conflicts_resolved,
        }
