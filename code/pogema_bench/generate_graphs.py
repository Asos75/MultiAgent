"""
Graph generation for the MAPF benchmark comparison.

Reads results from:
  - code/pogema_bench/results/movingai_benchmark.csv  (LaCAM + Local Leaders)
  - code/mats-lp/results/benchmark_results.csv        (MATS-LP throughput)
  - code/mats-lp/results/local_leaders_cpp.csv        (Local Leaders C++ throughput)
  - code/pogema_bench/results/follower_results_follower_summary.csv  (Follower)

Produces figures in code/pogema_bench/results/graphs/

Usage (from repo root):
    python -m code.pogema_bench.generate_graphs
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional
import warnings

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
RESULTS_DIR = Path(__file__).parent / "results"
GRAPHS_DIR = RESULTS_DIR / "graphs"
MATS_LP_RESULTS = CODE_ROOT / "mats-lp" / "results"

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib not found — install it to generate graphs")

ALGO_COLORS = {
    "lacam": "#2196F3",
    "local-leaders": "#4CAF50",
    "matslp": "#FF9800",
    "follower": "#9C27B0",
    "local-leaders-cpp": "#8BC34A",
}

ALGO_LABELS = {
    "lacam": "LaCAM",
    "local-leaders": "Local Leaders",
    "matslp": "MATS-LP",
    "follower": "Follower",
    "local-leaders-cpp": "Local Leaders (C++)",
}

MAP_TYPE_ORDER = ["random", "maze", "warehouse"]


# ── Data loading ─────────────────────────────────────────────────────────────

def load_movingai_results(path: Path) -> List[dict]:
    if not path.exists():
        print(f"[WARN] {path} not found — run run_full_benchmark.py first")
        return []
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["num_agents"] = int(r["num_agents"])
        r["seed"] = int(r["seed"])
        r["solved"] = r["solved"].lower() == "true"
        r["makespan"] = float(r["makespan"]) if r["makespan"] else None
        r["soc"] = float(r["soc"]) if r["soc"] else None
        r["comp_time_ms"] = float(r["comp_time_ms"]) if r["comp_time_ms"] else None
    return rows


def load_throughput_results(path: Path, algo_name: str) -> List[dict]:
    if not path.exists():
        print(f"[WARN] {path} not found")
        return []
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "algo": algo_name,
                "label": r["label"],
                "map_type": _label_to_map_type(r["label"]),
                "num_agents": int(r["num_agents"]),
                "seed": int(r["seed"]),
                "throughput": float(r["avg_throughput"]) if r.get("avg_throughput") else None,
                "elapsed_s": float(r["elapsed_s"]) if r.get("elapsed_s") else None,
                "solved": r.get("solved", "True").lower() == "true" if r.get("solved") else True,
            })
    return rows


def _agents_from_label(label: str) -> Optional[int]:
    import re
    m = re.search(r"_a(\d+)$", label)
    return int(m.group(1)) if m else None


def load_follower_summary(path: Path) -> List[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            label = r["label"]
            rows.append({
                "algo": "follower",
                "label": label,
                "map_type": _label_to_map_type(label),
                "num_agents": _agents_from_label(label),
                "throughput": float(r["mean_throughput"]) if r.get("mean_throughput") else None,
                "success_rate": float(r["success_rate"]) if r.get("success_rate") else None,
                "ISR": float(r["mean_ISR"]) if r.get("mean_ISR") else None,
                "CSR": float(r["mean_CSR"]) if r.get("mean_CSR") else None,
                "ep_length": float(r["mean_ep_length"]) if r.get("mean_ep_length") else None,
            })
    return rows


def _label_to_map_type(label: str) -> str:
    if "maze" in label:
        return "maze"
    if "warehouse" in label:
        return "warehouse"
    return "random"


# ── Aggregation helpers ───────────────────────────────────────────────────────

def agg_movingai(rows: List[dict], algo: str, metric: str) -> Dict[str, Dict[int, dict]]:
    """Returns {map_type: {n_agents: {mean, std, n_solved, n_total}}}."""
    from collections import defaultdict
    bucket: Dict[str, Dict[int, list]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["algo"] != algo:
            continue
        v = r.get(metric)
        if r["solved"] and v is not None:
            bucket[r["map_type"]][r["num_agents"]].append(v)

    result = {}
    for mt, agents_dict in bucket.items():
        result[mt] = {}
        for na, vals in agents_dict.items():
            result[mt][na] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "n": len(vals),
            }
    return result


def success_rate_movingai(rows: List[dict], algo: str) -> Dict[str, Dict[int, float]]:
    """Returns {map_type: {n_agents: success_rate}}."""
    from collections import defaultdict
    solved: Dict[str, Dict[int, list]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["algo"] != algo:
            continue
        solved[r["map_type"]][r["num_agents"]].append(1.0 if r["solved"] else 0.0)

    return {
        mt: {na: float(np.mean(vals)) for na, vals in nd.items()}
        for mt, nd in solved.items()
    }


def agg_throughput(rows: List[dict], algo: str) -> Dict[str, Dict[int, dict]]:
    """Returns {map_type: {n_agents: {mean, std}}}."""
    from collections import defaultdict
    bucket: Dict[str, Dict[int, list]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["algo"] != algo:
            continue
        if r.get("throughput") is not None:
            bucket[r["map_type"]][r["num_agents"]].append(r["throughput"])

    result = {}
    for mt, nd in bucket.items():
        result[mt] = {}
        for na, vals in nd.items():
            result[mt][na] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}
    return result


# ── Plot helpers ──────────────────────────────────────────────────────────────

def _bar_group(
    ax,
    groups: List[str],
    algo_data: Dict[str, List[Optional[float]]],
    algo_errs: Dict[str, List[Optional[float]]],
    ylabel: str,
    title: str,
    log_scale: bool = False,
) -> None:
    algos = list(algo_data.keys())
    n_algos = len(algos)
    n_groups = len(groups)
    width = 0.8 / n_algos
    x = np.arange(n_groups)

    for i, algo in enumerate(algos):
        vals = [v if v is not None else 0.0 for v in algo_data[algo]]
        errs = [e if e is not None else 0.0 for e in algo_errs.get(algo, [0.0] * n_groups)]
        offset = (i - n_algos / 2 + 0.5) * width
        bars = ax.bar(
            x + offset,
            vals,
            width,
            label=ALGO_LABELS.get(algo, algo),
            color=ALGO_COLORS.get(algo, "#999"),
            yerr=errs,
            capsize=3,
            alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    if log_scale:
        ax.set_yscale("log")
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)


# ── Figures ───────────────────────────────────────────────────────────────────

def fig_soc_comparison(movingai_rows: List[dict], out_dir: Path) -> None:
    """Figure 1: SoC comparison — LaCAM vs Local Leaders per map type."""
    if not movingai_rows:
        return

    algos = ["lacam", "local-leaders"]
    map_types = [mt for mt in MAP_TYPE_ORDER if any(r["map_type"] == mt for r in movingai_rows)]

    fig, axes = plt.subplots(1, len(map_types), figsize=(5 * len(map_types), 4), sharey=False)
    if len(map_types) == 1:
        axes = [axes]

    for ax, mt in zip(axes, map_types):
        mt_rows = [r for r in movingai_rows if r["map_type"] == mt]
        agent_counts = sorted(set(r["num_agents"] for r in mt_rows))
        groups = [str(n) for n in agent_counts]

        algo_data: Dict[str, List[Optional[float]]] = {}
        algo_errs: Dict[str, List[Optional[float]]] = {}

        for algo in algos:
            agg = agg_movingai(mt_rows, algo, "soc")
            mt_agg = agg.get(mt, {})
            vals, errs = [], []
            for na in agent_counts:
                d = mt_agg.get(na, {})
                vals.append(d.get("mean"))
                errs.append(d.get("std", 0.0))
            algo_data[algo] = vals
            algo_errs[algo] = errs

        _bar_group(ax, groups, algo_data, algo_errs,
                   ylabel="Sum of Costs", title=f"SoC — {mt} maps")
        ax.set_xlabel("Number of agents")

    fig.suptitle("Sum of Costs: LaCAM vs Local Leaders", fontweight="bold")
    fig.tight_layout()
    path = out_dir / "fig1_soc_comparison.pdf"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def fig_makespan_comparison(movingai_rows: List[dict], out_dir: Path) -> None:
    """Figure 2: Makespan comparison — LaCAM vs Local Leaders."""
    if not movingai_rows:
        return

    algos = ["lacam", "local-leaders"]
    map_types = [mt for mt in MAP_TYPE_ORDER if any(r["map_type"] == mt for r in movingai_rows)]

    fig, axes = plt.subplots(1, len(map_types), figsize=(5 * len(map_types), 4), sharey=False)
    if len(map_types) == 1:
        axes = [axes]

    for ax, mt in zip(axes, map_types):
        mt_rows = [r for r in movingai_rows if r["map_type"] == mt]
        agent_counts = sorted(set(r["num_agents"] for r in mt_rows))
        groups = [str(n) for n in agent_counts]

        algo_data, algo_errs = {}, {}
        for algo in algos:
            agg = agg_movingai(mt_rows, algo, "makespan")
            mt_agg = agg.get(mt, {})
            vals, errs = [], []
            for na in agent_counts:
                d = mt_agg.get(na, {})
                vals.append(d.get("mean"))
                errs.append(d.get("std", 0.0))
            algo_data[algo] = vals
            algo_errs[algo] = errs

        _bar_group(ax, groups, algo_data, algo_errs,
                   ylabel="Makespan (steps)", title=f"Makespan — {mt} maps")
        ax.set_xlabel("Number of agents")

    fig.suptitle("Makespan: LaCAM vs Local Leaders", fontweight="bold")
    fig.tight_layout()
    path = out_dir / "fig2_makespan_comparison.pdf"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def fig_planning_time(movingai_rows: List[dict], out_dir: Path) -> None:
    """Figure 3: Planning time — LaCAM vs Local Leaders (log scale)."""
    if not movingai_rows:
        return

    algos = ["lacam", "local-leaders"]
    map_types = [mt for mt in MAP_TYPE_ORDER if any(r["map_type"] == mt for r in movingai_rows)]

    fig, axes = plt.subplots(1, len(map_types), figsize=(5 * len(map_types), 4), sharey=True)
    if len(map_types) == 1:
        axes = [axes]

    for ax, mt in zip(axes, map_types):
        mt_rows = [r for r in movingai_rows if r["map_type"] == mt]
        agent_counts = sorted(set(r["num_agents"] for r in mt_rows))
        groups = [str(n) for n in agent_counts]

        algo_data, algo_errs = {}, {}
        for algo in algos:
            agg = agg_movingai(mt_rows, algo, "comp_time_ms")
            mt_agg = agg.get(mt, {})
            vals, errs = [], []
            for na in agent_counts:
                d = mt_agg.get(na, {})
                vals.append(d.get("mean"))
                errs.append(d.get("std", 0.0))
            algo_data[algo] = vals
            algo_errs[algo] = errs

        _bar_group(ax, groups, algo_data, algo_errs,
                   ylabel="Planning time (ms)", title=f"Planning time — {mt} maps",
                   log_scale=True)
        ax.set_xlabel("Number of agents")

    fig.suptitle("Planning Time: LaCAM vs Local Leaders (log scale)", fontweight="bold")
    fig.tight_layout()
    path = out_dir / "fig3_planning_time.pdf"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def fig_success_rate(movingai_rows: List[dict], out_dir: Path) -> None:
    """Figure 4: Success rate — LaCAM vs Local Leaders per map type."""
    if not movingai_rows:
        return

    algos = ["lacam", "local-leaders"]
    map_types = [mt for mt in MAP_TYPE_ORDER if any(r["map_type"] == mt for r in movingai_rows)]

    fig, axes = plt.subplots(1, len(map_types), figsize=(5 * len(map_types), 4), sharey=True)
    if len(map_types) == 1:
        axes = [axes]

    for ax, mt in zip(axes, map_types):
        mt_rows = [r for r in movingai_rows if r["map_type"] == mt]
        agent_counts = sorted(set(r["num_agents"] for r in mt_rows))
        groups = [str(n) for n in agent_counts]

        algo_data = {}
        for algo in algos:
            sr = success_rate_movingai(mt_rows, algo)
            mt_sr = sr.get(mt, {})
            algo_data[algo] = [mt_sr.get(na) for na in agent_counts]

        _bar_group(ax, groups, algo_data, {},
                   ylabel="Success rate", title=f"Success Rate — {mt} maps")
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Number of agents")

    fig.suptitle("Success Rate: LaCAM vs Local Leaders", fontweight="bold")
    fig.tight_layout()
    path = out_dir / "fig4_success_rate.pdf"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def fig_throughput_comparison(
    matslp_rows: List[dict],
    ll_cpp_rows: List[dict],
    follower_rows: List[dict],
    out_dir: Path,
) -> None:
    """Figure 5: Throughput — MATS-LP vs Local Leaders C++ vs Follower."""
    all_rows = (
        [(r, "matslp") for r in matslp_rows]
        + [(r, "local-leaders-cpp") for r in ll_cpp_rows]
    )
    if not all_rows and not follower_rows:
        print("[INFO] No throughput data available for Fig 5")
        return

    labels_all = sorted(set(
        r["label"] for r in matslp_rows + ll_cpp_rows
    ))
    if not labels_all:
        return

    algos_tp = [a for a in ["matslp", "local-leaders-cpp"] if any(r[1] == a for r in all_rows)]

    # Pick labels present in at least one algo
    fig, ax = plt.subplots(figsize=(max(8, len(labels_all) * 0.9), 5))
    groups = labels_all

    algo_data: Dict[str, List[Optional[float]]] = {}
    algo_errs: Dict[str, List[Optional[float]]] = {}

    for algo in algos_tp:
        src_rows = matslp_rows if algo == "matslp" else ll_cpp_rows
        agg = agg_throughput(src_rows, algo)
        # Reconstruct by label
        label_means: Dict[str, Optional[float]] = {}
        label_stds: Dict[str, Optional[float]] = {}
        for r in src_rows:
            if r["algo"] != algo:
                continue
            label = r["label"]
            if label not in label_means:
                vals = [rr["throughput"] for rr in src_rows
                        if rr["algo"] == algo and rr["label"] == label and rr.get("throughput") is not None]
                if vals:
                    label_means[label] = float(np.mean(vals))
                    label_stds[label] = float(np.std(vals))
        algo_data[algo] = [label_means.get(g) for g in groups]
        algo_errs[algo] = [label_stds.get(g, 0.0) for g in groups]

    # Add Follower from summary
    if follower_rows:
        f_means: Dict[str, Optional[float]] = {r["label"]: r.get("throughput") for r in follower_rows}
        algo_data["follower"] = [f_means.get(g) for g in groups]
        algo_errs["follower"] = [0.0] * len(groups)
        if "follower" not in algos_tp:
            algos_tp = algos_tp + ["follower"]

    _bar_group(ax, groups, {a: algo_data[a] for a in algos_tp if a in algo_data},
               {a: algo_errs[a] for a in algos_tp if a in algo_errs},
               ylabel="Avg throughput (agents/step)",
               title="Throughput: MATS-LP vs Local Leaders C++ vs Follower")
    ax.set_xlabel("Scenario")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    path = out_dir / "fig5_throughput_comparison.pdf"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def fig_scaling(movingai_rows: List[dict], out_dir: Path) -> None:
    """Figure 6: SoC ratio (local-leaders / lacam) vs n_agents — shows overhead."""
    if not movingai_rows:
        return

    map_types = [mt for mt in MAP_TYPE_ORDER if any(r["map_type"] == mt for r in movingai_rows)]
    fig, axes = plt.subplots(1, len(map_types), figsize=(5 * len(map_types), 4), sharey=True)
    if len(map_types) == 1:
        axes = [axes]

    for ax, mt in zip(axes, map_types):
        mt_rows = [r for r in movingai_rows if r["map_type"] == mt]
        agent_counts = sorted(set(r["num_agents"] for r in mt_rows))

        lacam_agg = agg_movingai(mt_rows, "lacam", "soc").get(mt, {})
        ll_agg = agg_movingai(mt_rows, "local-leaders", "soc").get(mt, {})

        ratios, xs = [], []
        for na in agent_counts:
            lc = lacam_agg.get(na, {}).get("mean")
            ll = ll_agg.get(na, {}).get("mean")
            if lc and ll and lc > 0:
                ratios.append(ll / lc)
                xs.append(na)

        if xs:
            ax.plot(xs, ratios, marker="o", color=ALGO_COLORS["local-leaders"], linewidth=2)
            ax.axhline(1.0, color=ALGO_COLORS["lacam"], linestyle="--", linewidth=1, label="LaCAM (baseline)")
            ax.set_xlabel("Number of agents")
            ax.set_ylabel("SoC ratio (LL / LaCAM)")
            ax.set_title(f"{mt} maps")
            ax.legend(fontsize=8)
            ax.yaxis.grid(True, alpha=0.3)

    fig.suptitle("SoC Overhead: Local Leaders vs LaCAM", fontweight="bold")
    fig.tight_layout()
    path = out_dir / "fig6_soc_overhead.pdf"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def print_summary_table(movingai_rows: List[dict]) -> None:
    """Print a summary table to stdout for quick inspection."""
    if not movingai_rows:
        print("No movingai results to summarize.")
        return

    algos = sorted(set(r["algo"] for r in movingai_rows))
    map_types = sorted(set(r["map_type"] for r in movingai_rows))

    print("\n=== Summary Table ===")
    print(f"{'algo':<18} {'map_type':<12} {'n_agents':<10} {'success%':>8} {'soc_mean':>10} {'mk_mean':>10} {'time_ms':>10}")
    print("-" * 80)

    for algo in algos:
        for mt in map_types:
            rows = [r for r in movingai_rows if r["algo"] == algo and r["map_type"] == mt]
            if not rows:
                continue
            agent_counts = sorted(set(r["num_agents"] for r in rows))
            for na in agent_counts:
                na_rows = [r for r in rows if r["num_agents"] == na]
                sr = np.mean([1.0 if r["solved"] else 0.0 for r in na_rows])
                soc_vals = [r["soc"] for r in na_rows if r["soc"] is not None]
                mk_vals = [r["makespan"] for r in na_rows if r["makespan"] is not None]
                t_vals = [r["comp_time_ms"] for r in na_rows if r["comp_time_ms"] is not None]
                soc_s = f"{np.mean(soc_vals):.0f}" if soc_vals else "—"
                mk_s = f"{np.mean(mk_vals):.0f}" if mk_vals else "—"
                t_s = f"{np.mean(t_vals):.0f}" if t_vals else "—"
                print(f"{algo:<18} {mt:<12} {na:<10} {sr*100:>7.0f}% {soc_s:>10} {mk_s:>10} {t_s:>10}")


def main() -> None:
    if not HAS_MPL:
        return

    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    movingai_csv = RESULTS_DIR / "movingai_benchmark.csv"
    movingai_rows = load_movingai_results(movingai_csv)

    matslp_rows = load_throughput_results(MATS_LP_RESULTS / "benchmark_results.csv", "matslp")
    ll_cpp_rows = load_throughput_results(MATS_LP_RESULTS / "local_leaders_cpp.csv", "local-leaders-cpp")
    follower_summary = load_follower_summary(RESULTS_DIR / "follower_results_follower_summary.csv")

    print_summary_table(movingai_rows)

    print("\nGenerating figures...")
    fig_soc_comparison(movingai_rows, GRAPHS_DIR)
    fig_makespan_comparison(movingai_rows, GRAPHS_DIR)
    fig_planning_time(movingai_rows, GRAPHS_DIR)
    fig_success_rate(movingai_rows, GRAPHS_DIR)
    fig_throughput_comparison(matslp_rows, ll_cpp_rows, follower_summary, GRAPHS_DIR)
    fig_scaling(movingai_rows, GRAPHS_DIR)

    print(f"\nAll figures saved to {GRAPHS_DIR}/")


if __name__ == "__main__":
    main()
