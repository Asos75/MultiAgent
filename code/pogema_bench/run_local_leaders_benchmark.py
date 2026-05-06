from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any, Dict

from pogema_bench.run_benchmark import build_env, run_local_leaders


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--out_csv", type=str, required=True)
    p.add_argument("--seeds", type=str, default="0,1,2")

    # Mirrors the CSV labels already present in mats-lp results.
    p.add_argument("--labels", type=str, default="random_d0_a8,random_d0_a16,random_d0_a32,random_d30_a8,random_d30_a16,random_d30_a32")

    # Default map params (we use pogema-random here).
    p.add_argument("--size", type=int, default=32)

    # Local leaders config
    p.add_argument("--ll_group_radius", type=int, default=50)
    p.add_argument("--ll_max_group_size", type=int, default=50)
    p.add_argument("--ll_leader_view_radius", type=int, default=50)
    p.add_argument("--ll_leader_election", choices=["static", "dynamic"], default="static")
    # Set <=0 to disable the solver time limit (recommended for debugging convergence).
    p.add_argument("--ll_time_limit_sec", type=float, default=0.0)

    return p.parse_args()


def _label_to_params(label: str) -> dict[str, Any]:
    # Expected labels like: random_d0_a8
    parts = label.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unsupported label: {label}")
    density_part = parts[1]  # d0 or d30
    agents_part = parts[2]   # a8

    if not density_part.startswith("d") or not agents_part.startswith("a"):
        raise ValueError(f"Unsupported label: {label}")

    density = float(density_part[1:]) / 100.0
    num_agents = int(agents_part[1:])

    # Use same map_name pattern strings as in the existing CSV (for easier join/compare).
    map_name = f"pico_s.*_od{int(density*100)}_na32"
    return {"density": density, "num_agents": num_agents, "map_name": map_name}


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
                f"(agents={lp['num_agents']} density={lp['density']}) ...",
                flush=True,
            )
            # Build a tiny args object compatible with build_env/run_local_leaders.
            ns = argparse.Namespace(
                algo="local-leaders",
                env="pogema-random",
                map_name="random",
                size=args.size,
                density=lp["density"],
                num_agents=lp["num_agents"],
                seed=seed,
                obs_radius=5,
                max_episode_steps=256,
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

            env = build_env(ns)

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
