"""
Unified MAPF benchmark: LaCAM vs Local Leaders on MovingAI maps,
plus a POGEMA-based comparison of Local Leaders vs MATS-LP.

Usage (from repo root):
    python -m code.pogema_bench.run_full_benchmark
    python -m code.pogema_bench.run_full_benchmark --algos lacam
    python -m code.pogema_bench.run_full_benchmark --skip_movingai
    python -m code.pogema_bench.run_full_benchmark --skip_pogema
    python -m code.pogema_bench.run_full_benchmark --dry_run
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import time
import traceback
from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import yaml

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
MATS_LP_ROOT = CODE_ROOT / "mats-lp"

LACAM_TIME_LIMIT = 30  # seconds per instance

# ── POGEMA benchmark configs (mirrors MATS-LP CONFIGS_REDUCED) ────────────────
# (map_pattern, num_agents, label)
POGEMA_CONFIGS: List[Tuple[str, int, str]] = [
    ("pico_s.*_od0_na32",  8,  "random_d0_a8"),
    ("pico_s.*_od0_na32",  16, "random_d0_a16"),
    ("pico_s.*_od0_na32",  32, "random_d0_a32"),
    ("pico_s.*_od30_na32", 8,  "random_d30_a8"),
    ("pico_s.*_od30_na32", 16, "random_d30_a16"),
    ("pico_s.*_od30_na32", 32, "random_d30_a32"),
    ("mazes-s.*_20x20",    4,  "maze_20x20_a4"),
    ("mazes-s.*_20x20",    8,  "maze_20x20_a8"),
    ("mazes-s.*_20x20",    16, "maze_20x20_a16"),
    ("mazes-s.*_20x20",    32, "maze_20x20_a32"),
]
POGEMA_SEEDS = [0, 1, 2]
POGEMA_CSV = RESULTS_DIR / "ll_pogema_results.csv"
POGEMA_MAX_STEPS = 512


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
            time_limit_sec=30.0,
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


# ── POGEMA benchmark helpers ──────────────────────────────────────────────────

def _load_mats_maps() -> dict:
    """Load MATS-LP map registries (random + maze maps)."""
    maps: dict = {}
    for fname in ["random-pico.yaml", "mazes-maps.yaml"]:
        path = MATS_LP_ROOT / "env" / fname
        if not path.exists():
            raise FileNotFoundError(f"MATS-LP map file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            maps.update(yaml.safe_load(f))
    return maps


def _pick_map(maps: dict, pattern: str, seed: int) -> tuple:
    """Pick a random map matching pattern, deterministically by seed."""
    candidates = sorted(name for name in maps if re.match(pattern, name))
    if not candidates:
        raise KeyError(f"No map matching pattern: {pattern!r}")
    rng = np.random.default_rng(seed)
    name = candidates[int(rng.integers(0, len(candidates)))]
    return maps[name], name


def _all_free(grid: list, points: list) -> bool:
    if not grid:
        return False
    h, w = len(grid), len(grid[0])
    return all(0 <= r < h and 0 <= c < w and grid[r][c] == 0 for r, c in points)


def _delta_to_action(dr: int, dc: int) -> int:
    if dr == -1 and dc == 0:
        return 1
    if dr == 1 and dc == 0:
        return 2
    if dr == 0 and dc == -1:
        return 3
    if dr == 0 and dc == 1:
        return 4
    return 0  # stay


def run_local_leaders_pogema(
    maps: dict, map_pattern: str, n_agents: int, label: str, seed: int
) -> dict:
    """Run Local Leaders on a POGEMA map and return ISR/throughput metrics.

    Throughput = agents_done / ep_length  (goals per timestep),
    consistent with MATS-LP's avg_throughput metric.
    """
    from pogema import GridConfig, pogema_v0
    from main import MAPFInstance
    from main.core.algorithm import LocalLeadersMAPF
    from main.core.types import LocalLeadersConfig

    grid, map_name = _pick_map(maps, map_pattern, seed)

    cfg = GridConfig(
        num_agents=n_agents,
        seed=seed,
        max_episode_steps=POGEMA_MAX_STEPS,
        obs_radius=5,
        map=grid,
        map_name=map_name,
        observation_type="POMAPF",
        collision_system="soft",
        on_target="finish",
        auto_reset=False,
    )
    env = pogema_v0(grid_config=cfg)
    env.reset(seed=seed)

    obstacles = env.grid.get_obstacles().astype(int).tolist()
    starts = list(env.grid.get_agents_xy())
    goals = list(env.grid.get_targets_xy())

    # Correct coordinate convention if needed
    if not _all_free(obstacles, starts + goals):
        transposed = [list(row) for row in zip(*obstacles)]
        if _all_free(transposed, starts + goals):
            obstacles = transposed

    instance = MAPFInstance.from_arrays(obstacles, starts, goals)
    cfg_ll = LocalLeadersConfig(
        group_radius=6,
        max_group_size=12,
        leader_view_radius=7,
        leader_election="static",
        time_limit_sec=60.0,
        seed=seed,
    )

    t_plan = time.perf_counter()
    res = LocalLeadersMAPF(instance, cfg_ll).solve()
    planning_ms = (time.perf_counter() - t_plan) * 1000.0

    base = {
        "label": label,
        "map_name": map_pattern,
        "num_agents": n_agents,
        "seed": seed,
        "planning_time_ms": round(planning_ms, 2),
    }

    if not res.solved or res.solution is None:
        return {
            **base,
            "avg_throughput": 0.0,
            "elapsed_s": round(planning_ms / 1000, 2),
            "solved": False,
            "ISR": 0.0,
            "CSR": 0.0,
            "ep_length": None,
        }

    paths = [res.solution[i] for i in range(n_agents)]
    max_len = max(len(p) for p in paths)

    actions_seq = []
    for t in range(max_len - 1):
        step_acts = []
        for aid in range(n_agents):
            p = paths[aid]
            p0 = p[min(t, len(p) - 1)]
            p1 = p[min(t + 1, len(p) - 1)]
            step_acts.append(_delta_to_action(p1[0] - p0[0], p1[1] - p0[1]))
        actions_seq.append(step_acts)

    agents_done = [False] * n_agents
    ep_length = max(len(actions_seq), 1)

    t_replay = time.perf_counter()
    for t, acts in enumerate(actions_seq):
        _, _, done, trunc, _ = env.step(acts)
        for i, d in enumerate(done):
            if d:
                agents_done[i] = True
        if all(done) or all(trunc):
            ep_length = t + 1
            break

    total_elapsed = round(planning_ms / 1000 + (time.perf_counter() - t_replay), 2)
    n_done = sum(agents_done)
    ISR = n_done / n_agents
    CSR = n_done == n_agents
    throughput = n_done / ep_length

    return {
        **base,
        "avg_throughput": round(throughput, 9),
        "elapsed_s": total_elapsed,
        "solved": CSR,
        "ISR": round(ISR, 6),
        "CSR": float(CSR),
        "ep_length": ep_length,
    }


def run_pogema_benchmark(log_fh=None) -> None:
    """Run Local Leaders on POGEMA maps matching MATS-LP configs."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    existing: set = set()
    if POGEMA_CSV.exists():
        with open(POGEMA_CSV, newline="") as f:
            for row in csv.DictReader(f):
                existing.add((row["label"], row["seed"]))

    col_names = [
        "label", "map_name", "num_agents", "seed",
        "avg_throughput", "elapsed_s", "solved", "ISR", "CSR",
        "ep_length", "planning_time_ms",
    ]

    try:
        maps = _load_mats_maps()
    except FileNotFoundError as e:
        _log(log_fh, f"[POGEMA] Skipping — {e}")
        return

    total = len(POGEMA_CONFIGS) * len(POGEMA_SEEDS)
    done = 0

    with open(POGEMA_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=col_names)
        if POGEMA_CSV.stat().st_size == 0:
            writer.writeheader()

        _log(log_fh, f"\n=== POGEMA benchmark: {len(POGEMA_CONFIGS)} configs × {len(POGEMA_SEEDS)} seeds = {total} runs ===")
        _log(log_fh, f"Output CSV: {POGEMA_CSV}")

        for map_pattern, n_agents, label in POGEMA_CONFIGS:
            for seed in POGEMA_SEEDS:
                done += 1
                key = (label, str(seed))
                if key in existing:
                    _log(log_fh, f"[{done}/{total}] SKIP  {label} seed={seed}")
                    continue

                _log(log_fh, f"[{done}/{total}] RUN  {label}  n={n_agents:>2d}  seed={seed} ...")
                t_wall = time.perf_counter()
                try:
                    result = run_local_leaders_pogema(maps, map_pattern, n_agents, label, seed)
                except Exception as e:
                    result = {
                        "label": label, "map_name": map_pattern,
                        "num_agents": n_agents, "seed": seed,
                        "avg_throughput": 0.0, "elapsed_s": 0.0,
                        "solved": False, "ISR": 0.0, "CSR": 0.0,
                        "ep_length": None, "planning_time_ms": None,
                    }
                    _log(log_fh, f"     -> ERROR: {str(e)[:120]}")

                wall_s = time.perf_counter() - t_wall
                _log(
                    log_fh,
                    f"     -> ISR={result.get('ISR', 0):.2f}  "
                    f"tp={result.get('avg_throughput', 0):.4f}  "
                    f"ep={result.get('ep_length', '—')}  "
                    f"wall={wall_s:.1f}s",
                )
                writer.writerow(result)
                f.flush()

        _log(log_fh, f"=== POGEMA benchmark done. Results: {POGEMA_CSV} ===")


ALGO_RUNNERS = {
    "lacam": run_lacam,
    "local-leaders": run_local_leaders,
}


def _log(log_fh, msg: str) -> None:
    """Write a line to both stdout and the log file immediately."""
    import datetime
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if log_fh is not None:
        log_fh.write(line + "\n")
        log_fh.flush()


def run_all(algos: List[str], dry_run: bool, out_csv: Path, log_path: Optional[Path] = None,
            skip_pogema: bool = False) -> None:
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

    log_fh = open(log_path, "a") if log_path else None
    try:
        with open(out_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=col_names)
            if out_csv.stat().st_size == 0:
                writer.writeheader()

            _log(log_fh, f"=== Benchmark start: {len(algos)} algos × {len(SCENARIOS)} scenarios × {len(SEEDS)} seeds = {total} runs ===")
            _log(log_fh, f"Output CSV: {out_csv}")

            for algo in algos:
                runner = ALGO_RUNNERS[algo]
                for map_type, map_stem, n_agents in SCENARIOS:
                    for seed in SEEDS:
                        done += 1
                        key = (algo, map_stem, str(n_agents), str(seed))
                        if key in existing:
                            _log(log_fh, f"[{done}/{total}] SKIP  {algo} {map_stem} n={n_agents} seed={seed}")
                            continue

                        label = f"[{done}/{total}] {algo:14s} {map_stem} n={n_agents:3d} seed={seed}"
                        if dry_run:
                            _log(log_fh, f"DRY  {label}")
                            continue

                        _log(log_fh, f"RUN  {label} ...")
                        t_wall = time.perf_counter()
                        result = runner(map_stem, n_agents, seed)
                        wall_s = time.perf_counter() - t_wall
                        status = "OK " if result.solved else "FAIL"
                        detail = (
                            f"{status}  wall={wall_s:.1f}s"
                            + (f"  soc={result.soc}" if result.soc else "")
                            + (f"  mk={result.makespan}" if result.makespan else "")
                            + (f"  [{result.error[:80]}]" if result.error else "")
                        )
                        _log(log_fh, f"     -> {detail}")

                        writer.writerow(asdict(result))
                        f.flush()

            _log(log_fh, f"=== Done. Results written to {out_csv} ===")

        # POGEMA comparison: Local Leaders vs MATS-LP
        if not skip_pogema and not dry_run and "local-leaders" in algos:
            run_pogema_benchmark(log_fh)
    finally:
        if log_fh:
            log_fh.close()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full MAPF benchmark: LaCAM + Local Leaders + POGEMA vs MATS-LP")
    p.add_argument(
        "--algos",
        nargs="+",
        choices=list(ALGO_RUNNERS.keys()),
        default=list(ALGO_RUNNERS.keys()),
        help="Algorithms to benchmark (default: all)",
    )
    p.add_argument("--dry_run", action="store_true", help="Print what would run without executing")
    p.add_argument(
        "--skip_movingai",
        action="store_true",
        help="Skip the MovingAI benchmark (run only POGEMA comparison)",
    )
    p.add_argument(
        "--skip_pogema",
        action="store_true",
        help="Skip the POGEMA comparison benchmark",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=RESULTS_DIR / "movingai_benchmark.csv",
        help="Output CSV path",
    )
    p.add_argument(
        "--log",
        type=Path,
        default=RESULTS_DIR / "benchmark_run.log",
        help="Log file path (appended to on each run)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.skip_movingai:
        log_fh = open(args.log, "a") if args.log else None
        try:
            run_pogema_benchmark(log_fh)
        finally:
            if log_fh:
                log_fh.close()
    else:
        run_all(
            algos=args.algos,
            dry_run=args.dry_run,
            out_csv=args.out,
            log_path=args.log,
            skip_pogema=args.skip_pogema,
        )
