"""
Local Leaders MAPF – benchmark runner.

Mirrors code/mats-lp/benchmark.py so results can be directly compared.
Same CONFIGS_REDUCED, same SEEDS, same MAX_EPISODE_STEPS, same maps.
"""

import csv
import sys
import time
import argparse
from itertools import product
from pathlib import Path

# ── make sure sibling packages are importable ──────────────────────────────
_ROOT = Path(__file__).resolve().parents[2]
_MATS = _ROOT / "code" / "mats-lp"
if str(_MATS) not in sys.path:
    sys.path.insert(0, str(_MATS))

from env.create_env import create_env_base, DecMAPFConfig
from pogema.wrappers.metrics import LifeLongAverageThroughputMetric

from policy import LocalLeadersPolicy

# ── benchmark configurations (identical to mats-lp/benchmark.py) ──────────
CONFIGS_REDUCED = [
    *[("pico_s.*_od0_na32",  n, f"random_d0_a{n}")  for n in [8, 16, 32]],
    *[("pico_s.*_od30_na32", n, f"random_d30_a{n}") for n in [8, 16, 32]],
    *[("mazes-s.*_20x20",    n, f"maze_20x20_a{n}") for n in [4, 8, 16, 32]],
    *[("wfi_warehouse",      n, f"warehouse_a{n}")  for n in [32, 96]],
]

CONFIGS = CONFIGS_REDUCED

SEEDS             = [0, 1, 2]
MAX_EPISODE_STEPS = 512

# ── leader hyperparameters ────────────────────────────────────────────────
LEADER_COMM_RADIUS = 7
DEADLOCK_THRESHOLD = 4


def run_episode(map_name: str, num_agents: int, seed: int) -> tuple[float, float]:
    """Run one episode. Returns (avg_throughput, elapsed_seconds)."""
    env_cfg = DecMAPFConfig(
        with_animation=False,
        num_agents=num_agents,
        seed=seed,
        map_name=map_name,
        max_episode_steps=MAX_EPISODE_STEPS,
    )
    base_env = create_env_base(env_cfg)
    if map_name != "wfi_warehouse":
        base_env = LifeLongAverageThroughputMetric(base_env)

    policy = LocalLeadersPolicy(
        num_agents=num_agents,
        leader_comm_radius=LEADER_COMM_RADIUS,
        escape_threshold=DEADLOCK_THRESHOLD,
    )

    t_start = time.perf_counter()
    obs, _ = base_env.reset(seed=seed)

    # Initialise policy with obstacle grid (static for the whole episode)
    grid = base_env.get_global_obstacles()  # numpy array; convert to list-of-lists
    grid_list = [list(row) for row in grid]
    policy.reset(grid_list)

    while True:
        positions = base_env.get_global_agents_xy()
        targets   = base_env.get_global_targets_xy()
        actions   = policy.act(positions, targets)
        obs, rew, dones, trunc, infos = base_env.step(actions)
        if all(dones) or all(trunc):
            break

    elapsed = time.perf_counter() - t_start

    metrics    = infos[0].get("metrics", {})
    throughput = metrics.get("avg_throughput", None)
    return throughput, elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/benchmark_results.csv")
    parser.add_argument(
        "--seeds", type=int, default=len(SEEDS),
        help="Number of seeds per config",
    )
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seeds))

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "map_name", "num_agents", "seed",
                         "avg_throughput", "elapsed_s"])

        total = len(CONFIGS) * len(seeds)
        done  = 0
        t_bench = time.perf_counter()

        for (map_name, num_agents, label), seed in product(CONFIGS, seeds):
            done += 1
            print(f"[{done}/{total}] {label} seed={seed} ...", end=" ", flush=True)
            try:
                throughput, elapsed = run_episode(map_name, num_agents, seed)

                elapsed_total = time.perf_counter() - t_bench
                avg_per = elapsed_total / done
                eta_s   = avg_per * (total - done)
                eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_s))
                bar = "#" * int(30 * done / total) + "-" * (30 - int(30 * done / total))
                tp_str = f"{throughput:.4f}" if throughput is not None else "None"
                print(f"throughput={tp_str}  ({elapsed:.1f}s)  ETA {eta_str}  [{bar}]")

                writer.writerow([label, map_name, num_agents, seed,
                                 throughput, f"{elapsed:.2f}"])
                f.flush()
            except Exception as e:
                import traceback
                print(f"ERROR: {e}")
                traceback.print_exc()
                writer.writerow([label, map_name, num_agents, seed, "ERROR", ""])
                f.flush()

    total_time = time.strftime(
        "%H:%M:%S", time.gmtime(time.perf_counter() - t_bench)
    )
    print(f"\nDone in {total_time}. Results saved to {args.output}")


if __name__ == "__main__":
    main()
