from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code"))

from main import MAPFInstance, empty_map
from main.core.algorithm import LocalLeadersMAPF
from main.core.types import LocalLeadersConfig


def test_local_leaders_smoke_empty_map_solves_without_conflicts():
    grid = empty_map(8, 8)

    starts = [(1, 1), (6, 1)]
    goals = [(1, 6), (6, 6)]
    instance = MAPFInstance.from_arrays(grid, starts, goals)

    cfg = LocalLeadersConfig(
        group_radius=10,
        max_group_size=10,
        leader_view_radius=10,
        leader_election="static",
        time_limit_sec=2.0,
        seed=0,
    )

    res = LocalLeadersMAPF(instance, cfg).solve()

    assert res.solution is not None
    assert res.solved is True
    assert res.num_groups == 1
    assert res.makespan is not None and res.makespan >= 0
