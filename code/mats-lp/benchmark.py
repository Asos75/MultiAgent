import csv
import time
import argparse
from itertools import product
from env.create_env import create_env_base, DecMAPFConfig
from env.custom_maps import MAPS_REGISTRY
from mcts_cpp.cppmcts import mcts_preprocessor
from agents.mats_lp_agent import MATSLPAgent
from pogema.wrappers.metrics import LifeLongAverageThroughputMetric

CONFIGS_FULL = [
    # --- Random maps: all densities, all agent counts ---
    *[("pico_s.*_od0_na32", n, f"random_d0_a{n}") for n in [8, 16, 32]],
    *[("pico_s.*_od10_na32", n, f"random_d10_a{n}") for n in [8, 16, 32]],
    *[("pico_s.*_od20_na32", n, f"random_d20_a{n}") for n in [8, 16, 32]],
    *[("pico_s.*_od30_na32", n, f"random_d30_a{n}") for n in [8, 16, 32]],
    # --- Mazes: all sizes ---
    *[("mazes-s.*_10x10", n, f"maze_10x10_a{n}") for n in [2, 4, 8, 16]],
    *[("mazes-s.*_20x20", n, f"maze_20x20_a{n}") for n in [4, 8, 16, 32]],
    *[("mazes-s.*_30x30", n, f"maze_30x30_a{n}") for n in [8, 16, 32, 64]],
    # --- Warehouse ---
    *[("wfi_warehouse", n, f"warehouse_a{n}") for n in [32, 64, 96, 128, 160, 192]],
]

CONFIGS_REDUCED = [
    # --- Random maps: best case (0%) and worst case (30%) density only ---
    *[("pico_s.*_od0_na32", n, f"random_d0_a{n}") for n in [8, 16, 32]],
    *[("pico_s.*_od30_na32", n, f"random_d30_a{n}") for n in [8, 16, 32]],
    # --- Mazes: 20x20 only ---
    *[("mazes-s.*_20x20", n, f"maze_20x20_a{n}") for n in [4, 8, 16, 32]],
    # --- Warehouse: low / mid / high agent count ---
    *[("wfi_warehouse", n, f"warehouse_a{n}") for n in [32, 96, 192]],
]

# All of the experiments we want to run: (map_name_pattern, num_agents, label)
CONFIGS = CONFIGS_REDUCED

# Constants - same as the article's
SEEDS = [0, 1, 2]
MAX_EPISODE_STEPS = 512
NUM_EXPANSIONS = 250
NUM_THREADS = 4
PB_C_INIT = 4.44

# Runs one episode AKA one experiment and returns the throughput and elapsed time in seconds
def run_episode(map_name, num_agents, seed):
    env_cfg = DecMAPFConfig(
        with_animation=False,
        num_agents=num_agents,
        seed=seed,
        map_name=map_name,
        max_episode_steps=MAX_EPISODE_STEPS,
    )
    algo = MATSLPAgent(
        num_expansions=NUM_EXPANSIONS,
        num_threads=NUM_THREADS,
        pb_c_init=PB_C_INIT,
    )
    base_env = create_env_base(env_cfg)
    if map_name != 'wfi_warehouse':
        base_env = LifeLongAverageThroughputMetric(base_env)
    env = mcts_preprocessor(base_env)

    t_start = time.perf_counter()
    obs, _ = env.reset(seed=seed)
    while True:
        obs, rew, dones, tr, infos = env.step(algo.act(obs))
        if all(dones) or all(tr):
            break
    elapsed = time.perf_counter() - t_start

    metrics = infos[0].get("metrics", {})
    throughput = metrics.get("avg_throughput", None)
    return throughput, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/benchmark_results.csv")
    parser.add_argument("--seeds", type=int, default=len(SEEDS), help="Number of seeds per config")
    args = parser.parse_args()

    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    seeds = list(range(args.seeds))

    with open(args.output, "w", newline="") as f:

        # Init output csv and writer
        writer = csv.writer(f)
        writer.writerow(["label", "map_name", "num_agents", "seed", "avg_throughput", "elapsed_s"])

        total = len(CONFIGS) * len(seeds)
        done = 0
        t_bench_start = time.perf_counter()

        # We iterate over all combinations of configs and seeds
        for (map_name, num_agents, label), seed in product(CONFIGS, seeds):
            done += 1
            print(f"[{done}/{total}] {label} seed={seed} ...", end=" ", flush=True)
            try:

                # One experiment
                throughput, elapsed = run_episode(map_name, num_agents, seed)

                # Metrics and ETA calculation
                elapsed_total = time.perf_counter() - t_bench_start
                avg_per_run = elapsed_total / done
                eta_s = avg_per_run * (total - done)
                eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_s))

                bar_filled = int(30 * done / total)
                bar = "#" * bar_filled + "-" * (30 - bar_filled)
                tp_str = f"{throughput:.4f}" if throughput is not None else "None"
                print(f"throughput={tp_str}  ({elapsed:.1f}s)  ETA {eta_str}  [{bar}]")

                # Log results to csv
                writer.writerow([label, map_name, num_agents, seed, throughput, f"{elapsed:.2f}"])
                f.flush()
            except Exception as e:
                print(f"ERROR: {e}")
                writer.writerow([label, map_name, num_agents, seed, "ERROR", ""])
                f.flush()

    total_time = time.strftime("%H:%M:%S", time.gmtime(time.perf_counter() - t_bench_start))
    print(f"\nDone in {total_time}. Results saved to {args.output}")


if __name__ == "__main__":
    main()
