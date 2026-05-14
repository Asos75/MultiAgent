from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from pogema import GridConfig, pogema_v0

from .lacam_adapter import plan_with_lacam, solution_to_actions
from .matslp_adapter import MatsLPConfig, MatsLPPolicy
from .movingai_io import read_movingai_map, pick_n_agents

import sys


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["lacam", "matslp", "local-leaders"], required=True)
    p.add_argument("--num_agents", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_episode_steps", type=int, default=128)

    # Environment source
    p.add_argument(
        "--env",
        choices=["pogema-random", "movingai"],
        default="movingai",
        help="Environment source: POGEMA random generator or MovingAI map+scen.",
    )

    # POGEMA random maps
    p.add_argument("--map_name", type=str, default="random", help="POGEMA map_name (used for pogema-random)")
    p.add_argument("--size", type=int, default=32, help="Map size (pogema-random)")
    p.add_argument("--density", type=float, default=0.3, help="Obstacle density (pogema-random)")

    # MovingAI
    repo_root = Path(__file__).resolve().parents[2]
    p.add_argument(
        "--movingai_map",
        type=str,
        default=str(repo_root / "code" / "assets" / "moving_ai_maps" / "random-32-32-10.map"),
    )
    p.add_argument(
        "--movingai_scen",
        type=str,
        default=str(repo_root / "code" / "assets" / "moving_ai_scen" / "random-32-32-10-random-1.scen"),
    )
    p.add_argument("--obs_radius", type=int, default=5)

    # LaCAM
    p.add_argument(
        "--lacam_binary",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "lacam" / "build" / "main"),
    )
    p.add_argument("--lacam_time_limit_sec", type=int, default=10)

    # MATS-LP
    p.add_argument("--matslp_num_expansions", type=int, default=250)
    p.add_argument("--matslp_num_threads", type=int, default=4)

    # Local-Leaders
    p.add_argument("--ll_group_radius", type=int, default=5)
    p.add_argument("--ll_max_group_size", type=int, default=10)
    p.add_argument("--ll_leader_view_radius", type=int, default=6)
    p.add_argument("--ll_leader_election", choices=["static", "dynamic"], default="static")
    p.add_argument(
        "--ll_time_limit_sec",
        type=float,
        default=0.0,
        help="Local-Leaders solver time limit in seconds. Set <=0 to disable.",
    )

    return p.parse_args()


def build_env(args: argparse.Namespace):
    if args.env == "pogema-random":
        cfg = GridConfig(
            num_agents=args.num_agents,
            obs_radius=args.obs_radius,
            seed=args.seed,
            max_episode_steps=args.max_episode_steps,
            map_name=args.map_name,
            size=args.size,
            density=args.density,
            observation_type="POMAPF",
            collision_system="soft",
            on_target="finish",
            auto_reset=False,
        )
        return pogema_v0(grid_config=cfg)

    # MovingAI: load a concrete map (reproducible) and use the first N rows from the scen file.
    map_path = Path(args.movingai_map)
    scen_path = Path(args.movingai_scen)
    grid = read_movingai_map(map_path)
    size = len(grid)
    _, starts_xy, goals_xy = pick_n_agents(scen_path, args.num_agents)

    # Coordinate convention mismatch:
    # - MovingAI uses (x, y) = (col, row)
    # - POGEMA stores obstacles as obstacles[x, y] (numpy indexing => row, col)
    #   and expects agents_xy/targets_xy in the same (row, col) convention.
    # Therefore we must swap to (row, col) here.
    starts_xy = [(y, x) for (x, y) in starts_xy]
    goals_xy = [(y, x) for (x, y) in goals_xy]
    cfg = GridConfig(
        num_agents=args.num_agents,
        obs_radius=args.obs_radius,
        seed=args.seed,
        max_episode_steps=args.max_episode_steps,
        size=size,
        map=grid,
        agents_xy=starts_xy,
        targets_xy=goals_xy,
        observation_type="POMAPF",
        collision_system="soft",
        on_target="finish",
        auto_reset=False,
    )
    return pogema_v0(grid_config=cfg)


def inject_global_fields(env, observations: List[Dict[str, Any]]):
    # These are the fields MATS-LP expects (see `ProvideMapWrapper` in MATS-LP).
    obstacles = env.grid.get_obstacles().astype(int).tolist()
    agents_xy = env.grid.get_agents_xy()
    targets_xy = env.grid.get_targets_xy()

    # For fixed-target setting, lifelong target list is just current target repeated.
    lifelong_targets = [[t] for t in targets_xy]

    for i, obs in enumerate(observations):
        obs["global_obstacles"] = obstacles
        obs["global_agent_xy"] = agents_xy[i]
        obs["global_target_xy"] = targets_xy[i]
        obs["global_lifelong_targets_xy"] = lifelong_targets[i]


def run_lacam(env, args: argparse.Namespace) -> Dict[str, Any]:
    obs, info = env.reset(seed=args.seed)

    # Export the obstacle grid we can access from POGEMA.
    # NOTE: In pogema==1.2.2 this appears to be the same grid used by `has_obstacle()`.
    obstacles = env.grid.get_obstacles().astype(int).tolist()
    starts_xy_raw = env.grid.get_agents_xy()
    goals_xy_raw = env.grid.get_targets_xy()

    pad = int(getattr(env.grid_config, "obs_radius", args.obs_radius))
    height = len(obstacles)
    width = len(obstacles[0]) if height else 0

    def in_bounds(x: int, y: int) -> bool:
        return 0 <= x < width and 0 <= y < height

    def is_free(x: int, y: int) -> bool:
        # obstacles is 1 for obstacle, 0 for free
        return in_bounds(x, y) and obstacles[x][y] == 0

    def apply_offset(points, dx: int, dy: int):
        return [(x + dx, y + dy) for (x, y) in points]

    # Try a few common coordinate conventions.
    # For MovingAI-based GridConfig(map=..., agents_xy=..., targets_xy=...), (0,0) should work.
    candidates = [(0, 0), (pad, pad), (-pad, -pad)]

    chosen = None
    for dx, dy in candidates:
        s_xy = apply_offset(starts_xy_raw, dx, dy)
        g_xy = apply_offset(goals_xy_raw, dx, dy)
        if all(is_free(x, y) for x, y in s_xy) and all(is_free(x, y) for x, y in g_xy):
            chosen = (dx, dy)
            starts_xy = s_xy
            goals_xy = g_xy
            break

    if chosen is None:
        # Strict mode (A): fail fast with debugging info.
        def cell_val(xy):
            x, y = xy
            if not in_bounds(x, y):
                return "OOB"
            return obstacles[x][y]

        sample = list(zip(starts_xy_raw, goals_xy_raw))[: min(3, len(starts_xy_raw))]
        return {
            "algo": "lacam",
            "solved": False,
            "error": "Could not align POGEMA agent/target coordinates with obstacle grid (strict mode).",
            "pad": pad,
            "obstacles_shape": (height, width),
            "sample_raw": sample,
            "sample_raw_cell_values": [
                (cell_val(s), cell_val(g)) for s, g in sample
            ],
        }

    # Convert POGEMA (row,col) back to LaCAM/MovingAI (col,row)
    starts_xy_lacam = [(y, x) for (x, y) in starts_xy]
    goals_xy_lacam = [(y, x) for (x, y) in goals_xy]

    rr = plan_with_lacam(
        lacam_binary=Path(args.lacam_binary),
        obstacles=obstacles,
        starts_xy=starts_xy_lacam,
        goals_xy=goals_xy_lacam,
        seed=args.seed,
        time_limit_sec=args.lacam_time_limit_sec,
    )

    if not rr.solved:
        return {
            "algo": "lacam",
            "solved": False,
            "soc": rr.soc,
            "makespan": rr.makespan,
            "comp_time_ms": rr.comp_time_ms,
        }

    # Swap LaCAM solution (col,row) into POGEMA (row,col) so action deltas match POGEMA movement.
    sol_xy_pogema = [[(y, x) for (x, y) in step] for step in rr.solution_xy]
    actions_seq = solution_to_actions(sol_xy_pogema)

    terminated = False
    step = 0
    while not terminated and step < len(actions_seq):
        acts = actions_seq[step]
        obs, rew, done, trunc, infos = env.step(acts)
        terminated = all(done) or all(trunc)
        step += 1

    # POGEMA metrics are exposed in infos[0]['metrics'] if metrics wrapper is used.
    # Here we at least report whether all agents finished.
    return {
        "algo": "lacam",
        "solved": bool(rr.solved) and terminated,
        "soc": rr.soc,
        "makespan": rr.makespan,
        "comp_time_ms": rr.comp_time_ms,
        "steps_replayed": step,
    }


def run_matslp(env, args: argparse.Namespace) -> Dict[str, Any]:
    obs, info = env.reset(seed=args.seed)

    repo_root = Path(__file__).resolve().parents[2]
    policy = MatsLPPolicy(
        MatsLPConfig(
            repo_root=repo_root,
            num_expansions=args.matslp_num_expansions,
            num_threads=args.matslp_num_threads,
            obs_radius=args.obs_radius,
        )
    )

    terminated = False
    step = 0
    while not terminated and step < args.max_episode_steps:
        inject_global_fields(env, obs)
        acts = policy.act(obs)
        obs, rew, done, trunc, infos = env.step(acts)
        terminated = all(done) or all(trunc)
        step += 1

    return {
        "algo": "matslp",
        "steps": step,
        "all_done": terminated,
        "example_metrics": infos[0].get("metrics") if infos else None,
    }


def run_local_leaders(env, args: argparse.Namespace) -> Dict[str, Any]:
    """Plan once with LocalLeadersMAPF (global plan) and replay in POGEMA."""

    # Ensure repository `code/` is importable so `import main` resolves.
    repo_root = Path(__file__).resolve().parents[2]
    code_root = repo_root / "code"
    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))

    from main import MAPFInstance
    from main.core.algorithm import LocalLeadersMAPF
    from main.core.types import LocalLeadersConfig
    from pogema_bench.local_leaders_cpp_adapter import solve_local_leaders_cpp

    obs, info = env.reset(seed=args.seed)

    obstacles_raw = env.grid.get_obstacles().astype(int).tolist()
    # `main` uses grid[row][col]. Depending on pogema internals, obstacles may come as [x][y].
    # Transpose to be safe for map generation here.
    obstacles = [list(row) for row in zip(*obstacles_raw)] if obstacles_raw else []
    starts = env.grid.get_agents_xy()
    goals = env.grid.get_targets_xy()

    instance = MAPFInstance.from_arrays(obstacles, starts, goals)
    cfg_dict = {
        "group_radius": args.ll_group_radius,
        "max_group_size": args.ll_max_group_size,
        "leader_view_radius": args.ll_leader_view_radius,
        "leader_election": args.ll_leader_election,
        "time_limit_sec": args.ll_time_limit_sec,
        "seed": args.seed,
    }

    # Prefer C++ solver (much faster); fall back to Python implementation if build/import fails.
    try:
        out_cpp = solve_local_leaders_cpp(instance.grid, starts, goals, cfg_dict)
        if not out_cpp.get("solved") or not out_cpp.get("solution"):
            return {
                "algo": "local-leaders",
                "solved": False,
                "soc": out_cpp.get("soc"),
                "makespan": out_cpp.get("makespan"),
                "comp_time_ms": out_cpp.get("comp_time_ms"),
                "num_groups": out_cpp.get("num_groups"),
                "avg_group_size": out_cpp.get("avg_group_size"),
                "num_conflicts_resolved": out_cpp.get("num_conflicts_resolved"),
                "backend": "cpp",
            }
        plan = out_cpp["solution"]
        num_groups = out_cpp.get("num_groups")
        avg_group_size = out_cpp.get("avg_group_size")
        num_conflicts_resolved = out_cpp.get("num_conflicts_resolved")
        comp_time_ms = out_cpp.get("comp_time_ms")
        soc = out_cpp.get("soc")
        makespan = out_cpp.get("makespan")
        backend = "cpp"
    except Exception:
        cfg = LocalLeadersConfig(
            group_radius=args.ll_group_radius,
            max_group_size=args.ll_max_group_size,
            leader_view_radius=args.ll_leader_view_radius,
            leader_election=args.ll_leader_election,
            time_limit_sec=args.ll_time_limit_sec,
            seed=args.seed,
        )

        res = LocalLeadersMAPF(instance, cfg).solve()
        if not res.solved or res.solution is None:
            return {
                "algo": "local-leaders",
                "solved": False,
                "soc": res.soc,
                "makespan": res.makespan,
                "comp_time_ms": res.comp_time_ms,
                "num_groups": res.num_groups,
                "avg_group_size": res.avg_group_size,
                "num_conflicts_resolved": res.num_conflicts_resolved,
                "backend": "python",
            }
        plan = [res.solution[i] for i in range(args.num_agents)]
        num_groups = res.num_groups
        avg_group_size = res.avg_group_size
        num_conflicts_resolved = res.num_conflicts_resolved
        comp_time_ms = res.comp_time_ms
        soc = res.soc
        makespan = res.makespan
        backend = "python"

    # Convert plan (paths) into action sequence for replay.
    # POGEMA action mapping (common): 0=stay, 1=up, 2=down, 3=left, 4=right
    def delta_to_action(dr: int, dc: int) -> int:
        if dr == 0 and dc == 0:
            return 0
        if dr == -1 and dc == 0:
            return 1
        if dr == 1 and dc == 0:
            return 2
        if dr == 0 and dc == -1:
            return 3
        if dr == 0 and dc == 1:
            return 4
        return 0

    # plan is: list[agent] -> list[(r,c)]
    max_len = max(len(p) for p in plan) if plan else 0
    actions_seq: list[list[int]] = []
    for t in range(max_len - 1):
        step_actions: list[int] = []
        for aid in range(args.num_agents):
            path = plan[aid]
            p0 = path[min(t, len(path) - 1)]
            p1 = path[min(t + 1, len(path) - 1)]
            dr, dc = p1[0] - p0[0], p1[1] - p0[1]
            step_actions.append(delta_to_action(dr, dc))
        actions_seq.append(step_actions)

    terminated = False
    step = 0
    while not terminated and step < len(actions_seq):
        obs, rew, done, trunc, infos = env.step(actions_seq[step])
        terminated = all(done) or all(trunc)
        step += 1

    return {
        "algo": "local-leaders",
    "solved": bool(terminated),
    "soc": soc,
    "makespan": makespan,
    "comp_time_ms": comp_time_ms,
    "num_groups": num_groups,
    "avg_group_size": avg_group_size,
    "num_conflicts_resolved": num_conflicts_resolved,
        "steps_replayed": step,
    "backend": backend,
        "config": {
            "group_radius": args.ll_group_radius,
            "max_group_size": args.ll_max_group_size,
            "leader_view_radius": args.ll_leader_view_radius,
            "leader_election": args.ll_leader_election,
            "time_limit_sec": args.ll_time_limit_sec,
        },
    }


def main() -> None:
    args = _parse_args()
    env = build_env(args)

    if args.algo == "lacam":
        out = run_lacam(env, args)
    elif args.algo == "matslp":
        out = run_matslp(env, args)
    else:
        out = run_local_leaders(env, args)

    print(out)


if __name__ == "__main__":
    main()
