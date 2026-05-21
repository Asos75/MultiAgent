import sys
import argparse
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_MATS = _ROOT / "code" / "mats-lp"
if str(_MATS) not in sys.path:
    sys.path.insert(0, str(_MATS))

from env.create_env import create_env_base, DecMAPFConfig
from pogema.wrappers.metrics import LifeLongAverageThroughputMetric


FAST_CONFIGS = [
    ("pico_s.*_od0_na32",  16, "random_d0_a16"),
    ("pico_s.*_od30_na32", 16, "random_d30_a16"),
    ("mazes-s.*_20x20",    16, "maze_20x20_a16"),
]

FULL_CONFIGS = [
    *[("pico_s.*_od0_na32",  n, f"random_d0_a{n}")  for n in [8, 16, 32]],
    *[("pico_s.*_od30_na32", n, f"random_d30_a{n}") for n in [8, 16, 32]],
    *[("mazes-s.*_20x20",    n, f"maze_20x20_a{n}") for n in [4, 8, 16, 32]],
    *[("wfi_warehouse",      n, f"warehouse_a{n}")  for n in [32, 96]],
]

FAST_SEEDS = [0]
FULL_SEEDS = [0, 1, 2]
MAX_EPISODE_STEPS = 512

CRITERION_NAMES = [
    "ESCAPE", "LEADER", "FOLLOWER",
    "PROXIMITY_CLOSEST", "PROXIMITY_FURTHEST",
    "AGENT_ID", "MOST_STUCK", "LEAST_STUCK",
]


# We load the cpp module at runtime
def load_module():
    import cppimport
    src = Path(__file__).parent / "src" / "local_leaders_online.cpp"
    return cppimport.imp_from_filepath(str(src))

# We build the criteria list from Optuna weights
def build_criteria(module, trial):
    weights = {
        name: trial.suggest_float(f"crit_{name}", 0.0, 1.0)
        for name in CRITERION_NAMES
    }
    selected = sorted(
        [(w, name) for name, w in weights.items() if w > 0.5],
        reverse=True,
    )
    if not selected:
        return [module.Criterion.AGENT_ID]
    return [getattr(module.Criterion, name) for _, name in selected]


# We run one episode and return its throughput
def run_episode(module, cfg, map_name, num_agents, seed):
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
    policy = module.LocalLeadersPolicy(num_agents, seed, cfg)

    # We reset the environment and pass in the obstacles
    env.reset(seed=seed)
    policy.reset([list(map(int, row)) for row in env.get_global_obstacles()])

    # One iteration = 1 alg. timestep
    while True:
        positions = [tuple(p) for p in env.get_global_agents_xy()]
        targets   = [tuple(t) for t in env.get_global_targets_xy()]
        _, _, dones, trunc, infos = env.step(policy.act(positions, targets))
        if all(dones) or all(trunc):
            break

    # Return avg throughput
    return float(infos[0].get("metrics", {}).get("avg_throughput") or 0.0)


# We define the Optuna objective function
def make_objective(module, configs, seeds):
    def objective(trial):
        cfg = module.PolicyConfig()
        cfg.agent_view       = trial.suggest_int("agent_view", 2, 12)
        cfg.leader_view      = cfg.agent_view + trial.suggest_int("leader_view_offset", 0, 8)
        cfg.escape_thresh    = trial.suggest_int("escape_thresh", 1, 12)
        cfg.regroup_interval = trial.suggest_int("regroup_interval", 1, 20)
        cfg.hint_use_desired = trial.suggest_categorical("hint_use_desired", [True, False])
        cfg.criteria         = build_criteria(module, trial)

        scores = []
        # We run one episode for each config and seed, and average the results
        for (map_name, num_agents, _), seed in [(c, s) for c in configs for s in seeds]:
            try:
                score = run_episode(module, cfg, map_name, num_agents, seed)
                scores.append(score)
            except Exception:
                scores.append(0.0)

        return sum(scores) / len(scores) if scores else 0.0

    return objective


# We print the best found parameters
def print_best(study):
    p = study.best_params
    leader_view = p["agent_view"] + p["leader_view_offset"]

    print(f"\nBest average throughput: {study.best_value:.4f}")
    print(f"  agent_view       = {p['agent_view']}")
    print(f"  leader_view      = {leader_view}  (agent_view + {p['leader_view_offset']})")
    print(f"  escape_thresh    = {p['escape_thresh']}")
    print(f"  regroup_interval = {p['regroup_interval']}")
    print(f"  hint_use_desired = {p['hint_use_desired']}")

    weights  = {name: p[f"crit_{name}"] for name in CRITERION_NAMES}
    selected = sorted(
        [(w, name) for name, w in weights.items() if w > 0.5],
        reverse=True,
    )
    criteria_str = (
        ", ".join(f"Criterion.{name}" for _, name in selected)
        or "Criterion.AGENT_ID"
    )
    print(f"  criteria         = [{criteria_str}]")


def main():

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--full",   action="store_true")
    parser.add_argument("--study",  default="local_leaders_opt")
    args = parser.parse_args()

    configs = FULL_CONFIGS if args.full else FAST_CONFIGS
    seeds   = FULL_SEEDS   if args.full else FAST_SEEDS

    print("Compiling C++ module ...", flush=True)
    module = load_module()

    # Create Optuna instance
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        study_name=args.study,
        storage=f"sqlite:///{args.study}.db",
        direction="maximize",
        load_if_exists=True,
    )

    n_existing = len(study.trials)
    print(f"Study '{args.study}' — {n_existing} existing trials, running {args.trials} more.")
    print(f"Eval: {len(configs)} configs x {len(seeds)} seed(s) per trial\n")

    # Run the optuna optimization
    study.optimize(
        make_objective(module, configs, seeds),
        n_trials=args.trials,
        show_progress_bar=True,
    )

    # Print best results
    print_best(study)


if __name__ == "__main__":
    main()
