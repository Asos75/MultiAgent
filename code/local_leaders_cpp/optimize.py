"""
Parameter optimizer for LocalLeadersPolicy using Optuna (TPE Bayesian optimization).

Searches over:
  - agent_view, leader_view, escape_thresh, regroup_interval  (integers)
  - criteria ordering  (each criterion gets a weight; include if > 0.5, sort descending)

Usage:
    python optimize.py                        # 50 trials, fast eval subset
    python optimize.py --trials 200 --full    # 200 trials, full benchmark suite
    python optimize.py --study my_run         # resume / extend a named study

Results are saved to <study>.db (SQLite) so runs can be interrupted and resumed.
Install dependency once:  pip install optuna
"""

import sys
import argparse
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_MATS = _ROOT / "code" / "mats-lp"
if str(_MATS) not in sys.path:
    sys.path.insert(0, str(_MATS))

from env.create_env import create_env_base, DecMAPFConfig
from pogema.wrappers.metrics import LifeLongAverageThroughputMetric


# Fast subset used by default — quick enough for many trials.
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

FAST_SEEDS        = [0]
FULL_SEEDS        = [0, 1, 2]
MAX_EPISODE_STEPS = 512

CRITERION_NAMES = [
    "ESCAPE", "LEADER", "FOLLOWER",
    "PROXIMITY_CLOSEST", "PROXIMITY_FURTHEST",
    "AGENT_ID", "MOST_STUCK", "LEAST_STUCK",
]


def load_module():
    import cppimport
    src = Path(__file__).parent / "src" / "local_leaders_online.cpp"
    return cppimport.imp_from_filepath(str(src))


def build_criteria(mod, trial):
    """
    Each criterion gets a float weight suggested in [0, 1].
    Those with weight > 0.5 are included; the list is sorted by weight descending
    so the optimizer controls both membership and relative priority in one go.
    Falls back to [AGENT_ID] if nothing clears the threshold.
    """
    weights = {
        name: trial.suggest_float(f"crit_{name}", 0.0, 1.0)
        for name in CRITERION_NAMES
    }
    selected = sorted(
        [(w, name) for name, w in weights.items() if w > 0.5],
        reverse=True,
    )
    if not selected:
        return [mod.Criterion.AGENT_ID]
    return [getattr(mod.Criterion, name) for _, name in selected]


def run_episode(mod, cfg, map_name, num_agents, seed) -> float:
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

    policy = mod.LocalLeadersPolicy(num_agents, seed, cfg)
    obs, _ = base_env.reset(seed=seed)
    grid_list = [list(map(int, row)) for row in base_env.get_global_obstacles()]
    policy.reset(grid_list)

    while True:
        positions = [tuple(p) for p in base_env.get_global_agents_xy()]
        targets   = [tuple(t) for t in base_env.get_global_targets_xy()]
        actions   = policy.act(positions, targets)
        obs, rew, dones, trunc, infos = base_env.step(actions)
        if all(dones) or all(trunc):
            break

    metrics = infos[0].get("metrics", {})
    return float(metrics.get("avg_throughput") or 0.0)


def make_objective(mod, configs, seeds):
    def objective(trial):
        cfg = mod.PolicyConfig()

        cfg.agent_view       = trial.suggest_int("agent_view",    2, 12)
        # leader_view is sampled as an offset so it is always >= agent_view
        cfg.leader_view      = cfg.agent_view + trial.suggest_int("leader_view_offset", 0, 8)
        cfg.escape_thresh    = trial.suggest_int("escape_thresh", 1, 12)
        cfg.regroup_interval = trial.suggest_int("regroup_interval", 1, 20)
        cfg.hint_use_desired = trial.suggest_categorical("hint_use_desired", [True, False])
        cfg.criteria         = build_criteria(mod, trial)

        scores = []
        for (map_name, num_agents, _), seed in [
            (conf, s) for conf in configs for s in seeds
        ]:
            try:
                scores.append(run_episode(mod, cfg, map_name, num_agents, seed))
            except Exception:
                scores.append(0.0)

        return sum(scores) / len(scores) if scores else 0.0

    return objective


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50,
                        help="Number of new trials to run (default: 50)")
    parser.add_argument("--full",   action="store_true",
                        help="Use full benchmark suite instead of fast subset")
    parser.add_argument("--study",  default="local_leaders_opt",
                        help="Study name — also used as the SQLite DB filename")
    args = parser.parse_args()

    configs = FULL_CONFIGS if args.full else FAST_CONFIGS
    seeds   = FULL_SEEDS   if args.full else FAST_SEEDS

    print("Compiling C++ module ...", flush=True)
    mod = load_module()

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    storage = f"sqlite:///{args.study}.db"
    study = optuna.create_study(
        study_name=args.study,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
    )

    n_existing = len(study.trials)
    print(f"Study '{args.study}' — {n_existing} existing trials, running {args.trials} more.")
    print(f"Eval: {len(configs)} map configs × {len(seeds)} seed(s) per trial\n")

    study.optimize(
        make_objective(mod, configs, seeds),
        n_trials=args.trials,
        show_progress_bar=True,
    )

    print_best(study)


if __name__ == "__main__":
    main()
