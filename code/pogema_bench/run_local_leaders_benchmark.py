from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any, Dict
import re
import importlib.util

import numpy as np
import yaml
from pogema import GridConfig, pogema_v0

from pogema_bench.run_benchmark import run_local_leaders

# Use the same environment and map patterns as MATS-LP benchmarks, but avoid
# importing mats-lp MCTS modules.
REPO_ROOT = Path(__file__).resolve().parents[2]
MATS_LP_ROOT = REPO_ROOT / "code" / "mats-lp"


def _load_warehouse_class():
    path = MATS_LP_ROOT / "env" / "warehouse_wfi.py"
    spec = importlib.util.spec_from_file_location("mats_lp_warehouse_wfi", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load WarehouseWFI from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.WarehouseWFI


def _load_maps_registry() -> dict[str, Any]:
    map_files = ["mazes-maps.yaml", "random-pico.yaml"]
    maps: dict[str, Any] = {}
    for fname in map_files:
        path = MATS_LP_ROOT / "env" / fname
        with path.open("r", encoding="utf-8") as f:
            maps.update(**yaml.safe_load(f))
    return maps


MAPS_REGISTRY = _load_maps_registry()


def _pick_map(pattern: str, seed: int) -> tuple[list[list[int]], str]:
    # Mirror mats-lp MultiMapWrapper: pick a random matching map per seed.
    candidates = [name for name in sorted(MAPS_REGISTRY) if re.match(pattern, name)]
    if not candidates:
        raise KeyError(f"No map matching: {pattern}")
    rng = np.random.default_rng(seed)
    name = candidates[int(rng.integers(0, len(candidates)))]
    return MAPS_REGISTRY[name], name


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--out_csv", type=str, required=True)
    p.add_argument("--seeds", type=str, default="0,1,2")

    # Mirrors the CSV labels already present in mats-lp results (CONFIGS_REDUCED).
    p.add_argument(
        "--labels",
        type=str,
        default=
        "random_d0_a8,random_d0_a16,random_d0_a32,"
        "random_d30_a8,random_d30_a16,random_d30_a32,"
        "maze_20x20_a4,maze_20x20_a8,maze_20x20_a16,maze_20x20_a32,"
        "warehouse_a32,warehouse_a96",
    )

    # Local leaders config
    p.add_argument("--ll_group_radius", type=int, default=50)
    p.add_argument("--ll_max_group_size", type=int, default=50)
    p.add_argument("--ll_leader_view_radius", type=int, default=50)
    p.add_argument("--ll_leader_election", choices=["static", "dynamic"], default="static")
    # Set <=0 to disable the solver time limit (recommended for debugging convergence).
    p.add_argument("--ll_time_limit_sec", type=float, default=0.0)

    return p.parse_args()


def _label_to_params(label: str) -> dict[str, Any]:
    # Expected labels like: random_d0_a8, maze_20x20_a4, warehouse_a32
    parts = label.split("_")
    if len(parts) < 2:
        raise ValueError(f"Unsupported label: {label}")

    if label.startswith("random_d"):
        if len(parts) < 3:
            raise ValueError(f"Unsupported label: {label}")
        density_part = parts[1]  # d0 or d30
        agents_part = parts[2]   # a8
        if not density_part.startswith("d") or not agents_part.startswith("a"):
            raise ValueError(f"Unsupported label: {label}")
        density = int(density_part[1:])
        num_agents = int(agents_part[1:])
        map_name = f"pico_s.*_od{density}_na32"
        return {"num_agents": num_agents, "map_name": map_name}

    if label.startswith("maze_"):
        if len(parts) < 3:
            raise ValueError(f"Unsupported label: {label}")
        size_part = parts[1]   # 20x20
        agents_part = parts[2] # a4
        if not agents_part.startswith("a"):
            raise ValueError(f"Unsupported label: {label}")
        num_agents = int(agents_part[1:])
        map_name = f"mazes-s.*_{size_part}"
        return {"num_agents": num_agents, "map_name": map_name}

    if label.startswith("warehouse_"):
        agents_part = parts[1]  # a32
        if not agents_part.startswith("a"):
            raise ValueError(f"Unsupported label: {label}")
        num_agents = int(agents_part[1:])
        return {"num_agents": num_agents, "map_name": "wfi_warehouse"}

    raise ValueError(f"Unsupported label: {label}")


def main() -> None:
    args = _parse_args()

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    labels = [s.strip() for s in args.labels.split(",") if s.strip()]

    rows: list[dict[str, Any]] = []
    total = len(labels) * len(seeds)
    done = 0

    for label in labels:
        lp = _label_to_params(label)
        for seed in seeds:
            done += 1
            print(
                f"[{done}/{total}] label={label} seed={seed} "
                f"(agents={lp['num_agents']}) ...",
                flush=True,
            )
            # Build a tiny args object compatible with build_env/run_local_leaders.
            ns = argparse.Namespace(
                algo="local-leaders",
                env="pogema-random",
                map_name=lp["map_name"],
                size=32,
                density=0.0,
                num_agents=lp["num_agents"],
                seed=seed,
                obs_radius=5,
                max_episode_steps=512,
                movingai_map="",
                movingai_scen="",
                lacam_binary="",
                lacam_time_limit_sec=10,
                matslp_num_expansions=250,
                matslp_num_threads=4,
                ll_group_radius=args.ll_group_radius,
                ll_max_group_size=args.ll_max_group_size,
                ll_leader_view_radius=args.ll_leader_view_radius,
                ll_leader_election=args.ll_leader_election,
                ll_time_limit_sec=args.ll_time_limit_sec,
            )

            if lp["map_name"] == "wfi_warehouse":
                WarehouseWFI = _load_warehouse_class()
                cfg = GridConfig(
                    num_agents=lp["num_agents"],
                    seed=seed,
                    max_episode_steps=ns.max_episode_steps,
                    obs_radius=ns.obs_radius,
                    map_name=lp["map_name"],
                    observation_type="POMAPF",
                    collision_system="soft",
                    on_target="finish",
                    auto_reset=False,
                )
                env = WarehouseWFI(grid_config=cfg)
            else:
                grid, map_name = _pick_map(lp["map_name"], seed)
                cfg = GridConfig(
                    num_agents=lp["num_agents"],
                    seed=seed,
                    max_episode_steps=ns.max_episode_steps,
                    obs_radius=ns.obs_radius,
                    map=grid,
                    map_name=map_name,
                    observation_type="POMAPF",
                    collision_system="soft",
                    on_target="finish",
                    auto_reset=False,
                )
                env = pogema_v0(grid_config=cfg)

            t0 = time.perf_counter()
            out: Dict[str, Any] = run_local_leaders(env, ns)
            elapsed_s = time.perf_counter() - t0

            print(
                f"    solved={bool(out.get('solved', False))} "
                f"elapsed_s={elapsed_s:.2f} "
                f"soc={out.get('soc')} makespan={out.get('makespan')}",
                flush=True,
            )

            # Proxy throughput: agents/sec (still meaningful for "speed" comparison).
            avg_throughput = (lp["num_agents"] / elapsed_s) if elapsed_s > 0 else 0.0

            rows.append(
                {
                    "label": label,
                    "map_name": lp["map_name"],
                    "num_agents": lp["num_agents"],
                    "seed": seed,
                    "avg_throughput": avg_throughput,
                    "elapsed_s": elapsed_s,
                    "solved": bool(out.get("solved", False)),
                }
            )

    # Write CSV (same columns as mats-lp file; we keep solved as an extra column at the end).
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["label", "map_name", "num_agents", "seed", "avg_throughput", "elapsed_s", "solved"],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    **r,
                    "avg_throughput": round(float(r["avg_throughput"]), 9),
                    "elapsed_s": round(float(r["elapsed_s"]), 2),
                }
            )

    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
