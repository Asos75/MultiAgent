from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple


def _ensure_module_loaded():
    """Build+import the C++ solver on-demand via cppimport.

    This keeps the repo lightweight (no committed binaries) and works well with
    existing dependencies (`cppimport`, `pybind11`) already listed in
    `code/pogema_bench/requirements.txt`.
    """

    import cppimport  # type: ignore

    repo_root = Path(__file__).resolve().parents[2]

    # cppimport works with a concrete `.cpp` file path.
    mod_path = repo_root / "code" / "local_leaders_cpp" / "src" / "local_leaders_cpp.cpp"

    # We point to a tiny shim that compiles the pybind11 module.
    return cppimport.imp_from_filepath(str(mod_path))


def solve_local_leaders_cpp(
    grid: List[List[int]],
    starts: List[Tuple[int, int]],
    goals: List[Tuple[int, int]],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    mod = _ensure_module_loaded()

    # Convert grid into uint8 expected by C++.
    grid_u8 = [[int(v) & 1 for v in row] for row in grid]

    c = mod.Config()
    c.group_radius = int(cfg.get("group_radius", 5))
    c.max_group_size = int(cfg.get("max_group_size", 10))
    c.leader_view_radius = int(cfg.get("leader_view_radius", 6))
    c.leader_election = str(cfg.get("leader_election", "static"))
    c.time_limit_sec = float(cfg.get("time_limit_sec", 0.0))
    c.seed = int(cfg.get("seed", 0))

    res = mod.solve(grid_u8, starts, goals, c)

    out: Dict[str, Any] = {
        "solved": bool(res.solved),
        "soc": int(res.soc) if res.soc is not None else None,
        "makespan": int(res.makespan) if res.makespan is not None else None,
        "comp_time_ms": float(res.comp_time_ms),
        "num_groups": int(res.num_groups) if res.num_groups is not None else None,
        "avg_group_size": float(res.avg_group_size) if res.avg_group_size is not None else None,
        "num_conflicts_resolved": int(res.num_conflicts_resolved),
        "solution": res.paths,
    }
    return out
