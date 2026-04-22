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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["lacam", "matslp"], required=True)
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


def main() -> None:
    args = _parse_args()
    env = build_env(args)

    if args.algo == "lacam":
        out = run_lacam(env, args)
    else:
        out = run_matslp(env, args)

    print(out)


if __name__ == "__main__":
    main()
