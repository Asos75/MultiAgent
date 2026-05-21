import csv
import sys
import time
import argparse
from datetime import datetime
from itertools import product
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_MATS = _ROOT / "code" / "mats-lp"
if str(_MATS) not in sys.path:
    sys.path.insert(0, str(_MATS))

from env.create_env import create_env_base, DecMAPFConfig
from pogema.wrappers.metrics import LifeLongAverageThroughputMetric


CONFIGS = [
    *[("pico_s.*_od0_na32",  n, f"random_d0_a{n}")  for n in [8, 16, 32]],
    *[("pico_s.*_od30_na32", n, f"random_d30_a{n}") for n in [8, 16, 32]],
    *[("mazes-s.*_20x20",    n, f"maze_20x20_a{n}") for n in [4, 8, 16, 32]],
    *[("wfi_warehouse",      n, f"warehouse_a{n}")  for n in [32, 96]],
]

SEEDS = [0, 1, 2]
MAX_EPISODE_STEPS = 512

# We load the cpp module at runtime
def load_module():
    import cppimport
    src = Path(__file__).parent / "src" / "local_leaders_v2.cpp"
    return cppimport.imp_from_filepath(str(src))

# We define the parameters for our algorithm
def make_config(module):
    cfg = module.PolicyConfig()
    cfg.agent_view       = 11
    cfg.leader_view      = 17
    cfg.escape_thresh    = 1
    cfg.regroup_interval = 5
    cfg.hint_use_desired = False
    cfg.criteria = [
        module.Criterion.ESCAPE,
        module.Criterion.LEADER,
        module.Criterion.AGENT_ID,
        module.Criterion.LEAST_STUCK,
        module.Criterion.FOLLOWER,
    ]
    return cfg


def run_episode(module, map_name, num_agents, seed):
    env_cfg = DecMAPFConfig(
        with_animation=False,
        num_agents=num_agents,
        seed=seed,
        map_name=map_name,
        max_episode_steps=MAX_EPISODE_STEPS,
    )

    # We create the test environment in POGEMA
    env = create_env_base(env_cfg)
    if map_name != "wfi_warehouse":
        env = LifeLongAverageThroughputMetric(env)

    # We set the policy based on our config
    policy = module.LocalLeadersPolicy(num_agents, seed, make_config(module))

    # Experiment start
    t0 = time.perf_counter()

    # We reset the environment and pass in the obstacles
    env.reset(seed=seed)
    policy.reset([list(map(int, row)) for row in env.get_global_obstacles()])

    # 1 iteration = 1 timestep
    while True:
        positions = [tuple(p) for p in env.get_global_agents_xy()]
        targets   = [tuple(t) for t in env.get_global_targets_xy()]
        _, _, dones, trunc, infos = env.step(policy.act(positions, targets))
        if all(dones) or all(trunc):
            # Episode done
            break

    elapsed = time.perf_counter() - t0
    throughput = infos[0].get("metrics", {}).get("avg_throughput")
    return throughput, elapsed


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Parse the algorithm parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=f"results/benchmark_{ts}.csv")
    parser.add_argument("--seeds", type=int, default=len(SEEDS))
    args = parser.parse_args()

    print("Compiling C++ module ...", flush=True)
    # We load the c++ module
    module = load_module()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seeds))
    runs = list(product(CONFIGS, seeds))

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "map_name", "num_agents", "seed", "avg_throughput", "elapsed_s"])

        # We run the entire benchmark
        t_start = time.perf_counter()
        # We iterate over all run configs
        for done, ((map_name, num_agents, label), seed) in enumerate(runs, 1):
            print(f"[{done}/{len(runs)}] {label} seed={seed} ...", end=" ", flush=True)
            try:
                # One episode
                throughput, elapsed = run_episode(module, map_name, num_agents, seed)

                # UI feedback
                eta = (time.perf_counter() - t_start) / done * (len(runs) - done)
                bar = "#" * int(30 * done / len(runs)) + "-" * (30 - int(30 * done / len(runs)))
                tp  = f"{throughput:.4f}" if throughput is not None else "None"
                print(f"throughput={tp}  ({elapsed:.2f}s)  ETA {time.strftime('%H:%M:%S', time.gmtime(eta))}  [{bar}]")

                # Save results to CSV
                writer.writerow([label, map_name, num_agents, seed, throughput, f"{elapsed:.3f}"])
            except Exception as e:
                import traceback
                print(f"ERROR: {e}")
                traceback.print_exc()
                writer.writerow([label, map_name, num_agents, seed, "ERROR", ""])
            f.flush()

    # Final elapsed time
    total = time.strftime("%H:%M:%S", time.gmtime(time.perf_counter() - t_start))
    print(f"\nDone in {total}. Results saved to {args.output}")


if __name__ == "__main__":
    main()
