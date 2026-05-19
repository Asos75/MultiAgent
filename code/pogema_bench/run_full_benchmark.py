"""
Unified MAPF benchmark: LaCAM vs Local Leaders on MovingAI maps.

Runs both algorithms on random, maze, and warehouse maps with multiple
agent counts and seeds, collecting SoC, makespan, planning time, and
success rate. Results are saved incrementally to CSV.

Usage (from repo root):
    python -m code.pogema_bench.run_full_benchmark
    python -m code.pogema_bench.run_full_benchmark --algos lacam
    python -m code.pogema_bench.run_full_benchmark --dry_run
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
import traceback
from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from pogema_bench.lacam_adapter import plan_with_lacam
from pogema_bench.movingai_io import read_movingai_map, pick_n_agents

MAPS_DIR = CODE_ROOT / "assets" / "moving_ai_maps"
SCEN_DIR = CODE_ROOT / "assets" / "moving_ai_scen"
LACAM_BIN = CODE_ROOT / "lacam" / "build" / "main"
RESULTS_DIR = Path(__file__).parent / "results"

LACAM_TIME_LIMIT = 30  # seconds per instance


@dataclass
class RunResult:
    algo: str
    map_type: str
    map_name: str
    num_agents: int
    seed: int
    solved: bool
    makespan: Optional[float]
    soc: Optional[float]
    comp_time_ms: Optional[float]
    error: Optional[str] = None


# (map_type, map_stem, n_agents)
# Seeds 1-3 are used as scenario file indices (e.g. random-32-32-10-random-1.scen)
SCENARIOS: List[Tuple[str, str, int]] = [
    # Random maps
    ("random", "random-32-32-10", 8),
    ("random", "random-32-32-10", 16),
    ("random", "random-32-32-10", 32),
    ("random", "random-64-64-10", 16),
    ("random", "random-64-64-10", 32),
    ("random", "random-64-64-10", 64),
    # Maze maps
    ("maze", "maze-32-32-2", 4),
    ("maze", "maze-32-32-2", 8),
    ("maze", "maze-32-32-2", 16),
    ("maze", "maze-32-32-4", 8),
    ("maze", "maze-32-32-4", 16),
    # Warehouse maps
    ("warehouse", "warehouse-10-20-10-2-1", 8),
    ("warehouse", "warehouse-10-20-10-2-1", 16),
    ("warehouse", "warehouse-20-40-10-2-1", 16),
    ("warehouse", "warehouse-20-40-10-2-1", 32),
]

SEEDS = [1, 2, 3]


def run_lacam(map_stem: str, n_agents: int, scen_idx: int) -> RunResult:
    map_path = MAPS_DIR / f"{map_stem}.map"
    scen_path = SCEN_DIR / f"{map_stem}-random-{scen_idx}.scen"
    map_type = _map_type(map_stem)

    try:
        grid = read_movingai_map(map_path)
        _, starts_xy, goals_xy = pick_n_agents(scen_path, n_agents)
        rr = plan_with_lacam(
            lacam_binary=LACAM_BIN,
            obstacles=grid,
            starts_xy=starts_xy,
            goals_xy=goals_xy,
            seed=scen_idx,
            time_limit_sec=LACAM_TIME_LIMIT,
        )
        return RunResult(
            algo="lacam",
            map_type=map_type,
            map_name=map_stem,
            num_agents=n_agents,
            seed=scen_idx,
            solved=rr.solved,
            makespan=rr.makespan if rr.solved else None,
            soc=rr.soc if rr.solved else None,
            comp_time_ms=rr.comp_time_ms,
        )
    except Exception as e:
        return RunResult(
            algo="lacam",
            map_type=map_type,
            map_name=map_stem,
            num_agents=n_agents,
            seed=scen_idx,
            solved=False,
            makespan=None,
            soc=None,
            comp_time_ms=None,
            error=str(e),
        )


def run_local_leaders(map_stem: str, n_agents: int, scen_idx: int) -> RunResult:
    map_path = MAPS_DIR / f"{map_stem}.map"
    scen_path = SCEN_DIR / f"{map_stem}-random-{scen_idx}.scen"
    map_type = _map_type(map_stem)

    try:
        from main import MAPFInstance
        from main.core.algorithm import LocalLeadersMAPF
        from main.core.types import LocalLeadersConfig

        grid = read_movingai_map(map_path)
        _, starts_xy, goals_xy = pick_n_agents(scen_path, n_agents)

        # Convert from MovingAI (col,row) to MAPF (row,col)
        starts = [(y, x) for (x, y) in starts_xy]
        goals = [(y, x) for (x, y) in goals_xy]

        instance = MAPFInstance.from_arrays(grid, starts, goals)
        cfg = LocalLeadersConfig(
            group_radius=6,
            max_group_size=12,
            leader_view_radius=7,
            leader_election="static",
            seed=scen_idx,
        )

        t0 = time.perf_counter()
        res = LocalLeadersMAPF(instance, cfg).solve()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return RunResult(
            algo="local-leaders",
            map_type=map_type,
            map_name=map_stem,
            num_agents=n_agents,
            seed=scen_idx,
            solved=res.solved,
            makespan=res.makespan if res.solved else None,
            soc=res.soc if res.solved else None,
            comp_time_ms=res.comp_time_ms if res.comp_time_ms is not None else elapsed_ms,
        )
    except Exception as e:
        return RunResult(
            algo="local-leaders",
            map_type=map_type,
            map_name=map_stem,
            num_agents=n_agents,
            seed=scen_idx,
            solved=False,
            makespan=None,
            soc=None,
            comp_time_ms=None,
            error=str(e),
        )


def _map_type(map_stem: str) -> str:
    if "maze" in map_stem:
        return "maze"
    if "warehouse" in map_stem:
        return "warehouse"
    return "random"


ALGO_RUNNERS = {
    "lacam": run_lacam,
    "local-leaders": run_local_leaders,
}


def run_all(algos: List[str], dry_run: bool, out_csv: Path) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    col_names = [f.name for f in fields(RunResult)]

    existing: set[tuple] = set()
    if out_csv.exists():
        with open(out_csv, newline="") as f:
            for row in csv.DictReader(f):
                key = (row["algo"], row["map_name"], row["num_agents"], row["seed"])
                existing.add(key)

    total = len(SCENARIOS) * len(SEEDS) * len(algos)
    done = 0

    with open(out_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=col_names)
        if out_csv.stat().st_size == 0:
            writer.writeheader()

        for algo in algos:
            runner = ALGO_RUNNERS[algo]
            for map_type, map_stem, n_agents in SCENARIOS:
                for seed in SEEDS:
                    done += 1
                    key = (algo, map_stem, str(n_agents), str(seed))
                    if key in existing:
                        print(f"[{done}/{total}] SKIP  {algo} {map_stem} n={n_agents} seed={seed}")
                        continue

                    label = f"[{done}/{total}] {algo:14s} {map_stem} n={n_agents:3d} seed={seed}"
                    if dry_run:
                        print(f"DRY  {label}")
                        continue

                    print(f"RUN  {label} ...", end=" ", flush=True)
                    t_wall = time.perf_counter()
                    result = runner(map_stem, n_agents, seed)
                    wall_s = time.perf_counter() - t_wall
                    status = "OK " if result.solved else "FAIL"
                    print(f"{status}  wall={wall_s:.1f}s"
                          + (f"  soc={result.soc}" if result.soc else "")
                          + (f"  mk={result.makespan}" if result.makespan else "")
                          + (f"  [{result.error[:60]}]" if result.error else ""))
                    writer.writerow(asdict(result))
                    f.flush()

    if not dry_run:
        print(f"\nResults written to {out_csv}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full MAPF benchmark: LaCAM + Local Leaders")
    p.add_argument(
        "--algos",
        nargs="+",
        choices=list(ALGO_RUNNERS.keys()),
        default=list(ALGO_RUNNERS.keys()),
        help="Algorithms to benchmark (default: all)",
    )
    p.add_argument("--dry_run", action="store_true", help="Print what would run without executing")
    p.add_argument(
        "--out",
        type=Path,
        default=RESULTS_DIR / "movingai_benchmark.csv",
        help="Output CSV path",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_all(algos=args.algos, dry_run=args.dry_run, out_csv=args.out)
