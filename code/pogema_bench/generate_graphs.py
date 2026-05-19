"""
Graph generation for the MAPF benchmark comparison.

Reads results from:
  - code/pogema_bench/results/movingai_benchmark.csv  (LaCAM + Local Leaders)
  - code/mats-lp/results/benchmark_results.csv        (MATS-LP throughput)
  - code/pogema_bench/results/follower_results_follower_summary.csv  (Follower)

Produces 6 publication-quality PDF+PNG figures in code/pogema_bench/results/graphs/

Usage (from repo root):
    python -m pogema_bench.generate_graphs
"""
from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    from matplotlib import rcParams

    # Publication-quality global style
    rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    })
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib not found — install it to generate graphs")


# ── Colour & label palette ────────────────────────────────────────────────────

COLORS = {
    "lacam":             "#2196F3",   # blue
    "local-leaders":     "#4CAF50",   # green
    "matslp":            "#FF9800",   # orange
    "follower":          "#9C27B0",   # purple
}

LABELS = {
    "lacam":         "LaCAM",
    "local-leaders": "Local Leaders",
    "matslp":        "MATS-LP",
    "follower":      "Follower",
}

MAP_TYPE_ORDER = ["random", "maze", "warehouse"]
MAP_TYPE_TITLES = {"random": "Random", "maze": "Maze", "warehouse": "Warehouse"}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_movingai(path: Path) -> List[dict]:
    if not path.exists():
        print(f"[WARN] {path} not found — run run_full_benchmark.py first")
        return []
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            r["num_agents"] = int(r["num_agents"])
            r["seed"] = int(r["seed"])
            r["solved"] = r["solved"].lower() == "true"
            r["makespan"] = float(r["makespan"]) if r["makespan"] else None
            r["soc"] = float(r["soc"]) if r["soc"] else None
            r["comp_time_ms"] = float(r["comp_time_ms"]) if r["comp_time_ms"] else None
            rows.append(r)
    return rows


def load_throughput(path: Path, algo: str) -> List[dict]:
    if not path.exists():
        print(f"[WARN] {path} not found")
        return []
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            tp = float(r["avg_throughput"]) if r.get("avg_throughput") else None
            elapsed = float(r["elapsed_s"]) if r.get("elapsed_s") else None
            # Filter obvious outliers (elapsed < 0.1s suggests measurement error)
            if elapsed is not None and elapsed < 0.1 and tp is not None and tp > 10:
                tp = None
            rows.append({
                "algo": algo,
                "label": r["label"],
                "map_type": _label_map_type(r["label"]),
                "obstacle_density": _label_density(r["label"]),
                "num_agents": int(r["num_agents"]),
                "seed": int(r["seed"]),
                "throughput": tp,
                "elapsed_s": elapsed,
            })
    return rows


def load_follower_summary(path: Path) -> List[dict]:
    """Load Follower summary, recomputing throughput as ISR*n/ep_length (goals/step)."""
    if not path.exists():
        return []
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            label = r["label"]
            na = _agents_from_label(label)
            mean_ISR = float(r["mean_ISR"]) if r.get("mean_ISR") else None
            mean_ep = float(r["mean_ep_length"]) if r.get("mean_ep_length") else None
            # Compute goals/timestep, same units as MATS-LP throughput
            if mean_ISR is not None and mean_ep and mean_ep > 0 and na:
                throughput = mean_ISR * na / mean_ep
            else:
                throughput = None
            rows.append({
                "algo": "follower",
                "label": label,
                "map_type": _label_map_type(label),
                "obstacle_density": _label_density(label),
                "num_agents": na,
                "throughput": throughput,
                "ISR": mean_ISR,
                "CSR": float(r["mean_CSR"]) if r.get("mean_CSR") else None,
            })
    return rows


def load_ll_pogema(path: Path) -> List[dict]:
    """Load Local Leaders POGEMA benchmark results."""
    if not path.exists():
        print(f"[WARN] {path} not found — run run_full_benchmark.py first")
        return []
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            tp = float(r["avg_throughput"]) if r.get("avg_throughput") else None
            ISR = float(r["ISR"]) if r.get("ISR") else None
            CSR = r.get("CSR", "False").lower() == "true"
            ep_length = float(r["ep_length"]) if r.get("ep_length") else None
            rows.append({
                "algo": "local-leaders",
                "label": r["label"],
                "map_type": _label_map_type(r["label"]),
                "obstacle_density": _label_density(r["label"]),
                "num_agents": int(r["num_agents"]),
                "seed": int(r["seed"]),
                "throughput": tp,
                "ISR": ISR,
                "CSR": CSR,
                "ep_length": ep_length,
            })
    return rows


def _label_map_type(label: str) -> str:
    if "maze" in label:
        return "maze"
    if "warehouse" in label:
        return "warehouse"
    return "random"


def _label_density(label: str) -> int:
    m = re.search(r"_d(\d+)", label)
    return int(m.group(1)) if m else 0


def _agents_from_label(label: str) -> Optional[int]:
    m = re.search(r"_a(\d+)$", label)
    return int(m.group(1)) if m else None


# ── Aggregation helpers ───────────────────────────────────────────────────────

def agg(rows: List[dict], algo: str, metric: str,
        solved_only: bool = True) -> Dict[str, Dict[int, dict]]:
    """
    Aggregate metric by (map_type, num_agents).
    Returns {map_type: {n_agents: {mean, std, n}}}.
    """
    bucket: Dict[str, Dict[int, list]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["algo"] != algo:
            continue
        if solved_only and not r.get("solved", True):
            continue
        v = r.get(metric)
        if v is not None:
            bucket[r["map_type"]][r["num_agents"]].append(float(v))
    return {
        mt: {
            na: {"mean": float(np.mean(vs)), "std": float(np.std(vs)), "n": len(vs)}
            for na, vs in nd.items()
        }
        for mt, nd in bucket.items()
    }


def success_rate(rows: List[dict], algo: str) -> Dict[str, Dict[int, float]]:
    """Returns {map_type: {n_agents: rate}}."""
    bucket: Dict[str, Dict[int, list]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["algo"] != algo:
            continue
        bucket[r["map_type"]][r["num_agents"]].append(1.0 if r["solved"] else 0.0)
    return {mt: {na: float(np.mean(vs)) for na, vs in nd.items()} for mt, nd in bucket.items()}


def agg_throughput_by_agents(
    rows: List[dict], algo: str, density: int = 0
) -> Dict[int, dict]:
    """Returns {num_agents: {mean, std}} for a given obstacle density."""
    bucket: Dict[int, list] = defaultdict(list)
    for r in rows:
        if r["algo"] != algo:
            continue
        if r.get("obstacle_density", 0) != density:
            continue
        tp = r.get("throughput")
        if tp is not None:
            bucket[r["num_agents"]].append(tp)
    return {
        na: {"mean": float(np.mean(vs)), "std": float(np.std(vs)), "n": len(vs)}
        for na, vs in bucket.items()
    }


# ── Shared plot primitives ────────────────────────────────────────────────────

def _setup_ax(ax, xlabel: str, ylabel: str, title: str, log_y: bool = False) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=6)
    if log_y:
        ax.set_yscale("log")
    ax.set_axisbelow(True)


def _grouped_bars(
    ax,
    agent_counts: List[int],
    algo_vals: Dict[str, List[Optional[float]]],
    algo_errs: Dict[str, List[float]],
    log_y: bool = False,
) -> None:
    algos = list(algo_vals.keys())
    n = len(algos)
    w = 0.75 / n
    xs = np.arange(len(agent_counts))

    for i, algo in enumerate(algos):
        vals = np.array([v if v is not None else 0.0 for v in algo_vals[algo]])
        errs = np.array(algo_errs.get(algo, [0.0] * len(agent_counts)))
        offset = (i - n / 2 + 0.5) * w
        ax.bar(
            xs + offset, vals, w,
            label=LABELS.get(algo, algo),
            color=COLORS.get(algo, "#999"),
            yerr=errs, capsize=3,
            alpha=0.87, zorder=3,
        )

    ax.set_xticks(xs)
    ax.set_xticklabels([str(n) for n in agent_counts])
    ax.legend()
    if log_y:
        # Ensure minimum > 0 so log works
        ax.set_yscale("log")


def _save(fig: "plt.Figure", stem: str, out: Path) -> None:
    fig.tight_layout()
    pdf = out / f"{stem}.pdf"
    png = out / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {pdf.name}")


# ── Figure 1: SoC comparison ──────────────────────────────────────────────────

def fig_soc(rows: List[dict], out: Path) -> None:
    """SoC — LaCAM vs Local Leaders, one panel per map type."""
    if not rows:
        return
    algos = ["lacam", "local-leaders"]
    map_types = [mt for mt in MAP_TYPE_ORDER if any(r["map_type"] == mt for r in rows)]

    fig, axes = plt.subplots(1, len(map_types), figsize=(4.5 * len(map_types), 4.2))
    if len(map_types) == 1:
        axes = [axes]

    for ax, mt in zip(axes, map_types):
        mt_rows = [r for r in rows if r["map_type"] == mt]
        agents = sorted({r["num_agents"] for r in mt_rows})
        algo_vals, algo_errs = {}, {}
        for algo in algos:
            a = agg(mt_rows, algo, "soc").get(mt, {})
            algo_vals[algo] = [a.get(n, {}).get("mean") for n in agents]
            algo_errs[algo] = [a.get(n, {}).get("std", 0.0) for n in agents]
        _grouped_bars(ax, agents, algo_vals, algo_errs)
        _setup_ax(ax, "Number of agents", "Sum of Costs", MAP_TYPE_TITLES[mt])

    fig.suptitle("Sum of Costs: LaCAM vs Local Leaders", fontweight="bold", y=1.01)
    _save(fig, "fig1_soc_comparison", out)


# ── Figure 2: Makespan ────────────────────────────────────────────────────────

def fig_makespan(rows: List[dict], out: Path) -> None:
    """Makespan — LaCAM vs Local Leaders, one panel per map type."""
    if not rows:
        return
    algos = ["lacam", "local-leaders"]
    map_types = [mt for mt in MAP_TYPE_ORDER if any(r["map_type"] == mt for r in rows)]

    fig, axes = plt.subplots(1, len(map_types), figsize=(4.5 * len(map_types), 4.2))
    if len(map_types) == 1:
        axes = [axes]

    for ax, mt in zip(axes, map_types):
        mt_rows = [r for r in rows if r["map_type"] == mt]
        agents = sorted({r["num_agents"] for r in mt_rows})
        algo_vals, algo_errs = {}, {}
        for algo in algos:
            a = agg(mt_rows, algo, "makespan").get(mt, {})
            algo_vals[algo] = [a.get(n, {}).get("mean") for n in agents]
            algo_errs[algo] = [a.get(n, {}).get("std", 0.0) for n in agents]
        _grouped_bars(ax, agents, algo_vals, algo_errs)
        _setup_ax(ax, "Number of agents", "Makespan (steps)", MAP_TYPE_TITLES[mt])

    fig.suptitle("Makespan: LaCAM vs Local Leaders", fontweight="bold", y=1.01)
    _save(fig, "fig2_makespan_comparison", out)


# ── Figure 3: Planning time (log scale, all runs) ─────────────────────────────

def fig_planning_time(rows: List[dict], out: Path) -> None:
    """Planning time on log scale — all runs (including unsolved) per map type."""
    if not rows:
        return
    algos = ["lacam", "local-leaders"]
    map_types = [mt for mt in MAP_TYPE_ORDER if any(r["map_type"] == mt for r in rows)]

    fig, axes = plt.subplots(1, len(map_types), figsize=(4.5 * len(map_types), 4.2), sharey=True)
    if len(map_types) == 1:
        axes = [axes]

    for ax, mt in zip(axes, map_types):
        mt_rows = [r for r in rows if r["map_type"] == mt]
        agents = sorted({r["num_agents"] for r in mt_rows})
        algo_vals, algo_errs = {}, {}
        for algo in algos:
            # Include ALL runs (not just solved) for honest timing comparison
            a = agg(mt_rows, algo, "comp_time_ms", solved_only=False).get(mt, {})
            means = [a.get(n, {}).get("mean") for n in agents]
            # Floor LaCAM times at 0.1 ms so log scale renders them
            if algo == "lacam":
                means = [max(v, 0.1) if v is not None else 0.1 for v in means]
            algo_vals[algo] = means
            algo_errs[algo] = [a.get(n, {}).get("std", 0.0) for n in agents]

        _grouped_bars(ax, agents, algo_vals, algo_errs, log_y=True)
        _setup_ax(ax, "Number of agents", "Planning time (ms)" if ax is axes[0] else "",
                  MAP_TYPE_TITLES[mt], log_y=True)

    fig.suptitle("Planning Time: LaCAM vs Local Leaders  (log scale, all runs)",
                 fontweight="bold", y=1.01)
    _save(fig, "fig3_planning_time", out)


# ── Figure 4: Success rate ────────────────────────────────────────────────────

def fig_success_rate(
    movingai_rows: List[dict],
    follower_rows: List[dict],
    matslp_rows: List[dict],
    out: Path,
) -> None:
    """
    Left panels: MovingAI one-shot success rate (LaCAM vs Local Leaders).
    Right panel: POGEMA ISR (Follower vs MATS-LP, random maps, density=0).
    """
    algos_mv = ["lacam", "local-leaders"]
    map_types = [mt for mt in MAP_TYPE_ORDER if any(r["map_type"] == mt for r in movingai_rows)]
    n_panels = len(map_types) + 1   # +1 for POGEMA ISR

    fig, axes = plt.subplots(1, n_panels, figsize=(4.2 * n_panels, 4.2))

    # MovingAI panels
    for ax, mt in zip(axes, map_types):
        mt_rows = [r for r in movingai_rows if r["map_type"] == mt]
        agents = sorted({r["num_agents"] for r in mt_rows})
        algo_vals = {}
        for algo in algos_mv:
            sr = success_rate(mt_rows, algo).get(mt, {})
            algo_vals[algo] = [sr.get(n, 0.0) for n in agents]
        _grouped_bars(ax, agents, algo_vals, {})
        ax.set_ylim(0, 1.1)
        ax.axhline(1.0, color="#ccc", linewidth=0.8, zorder=1)
        _setup_ax(ax, "Number of agents", "Success rate", MAP_TYPE_TITLES[mt])

    # POGEMA ISR panel (Follower vs MATS-LP, random maps, density=0)
    ax_pogema = axes[-1]
    pogema_algos = []
    all_pogema_agents: set = set()

    def _pogema_vals(src_rows, algo, density=0):
        bucket: Dict[int, list] = defaultdict(list)
        for r in src_rows:
            if r.get("obstacle_density", 0) != density:
                continue
            na = r["num_agents"]
            v = r.get("ISR") or r.get("throughput")
            if na is not None and v is not None:
                bucket[na].append(float(v))
        return {na: float(np.mean(vs)) for na, vs in bucket.items()}

    f_isr = _pogema_vals(follower_rows, "follower")
    m_isr = _pogema_vals(matslp_rows, "matslp")

    all_pogema_agents = sorted(set(f_isr) | set(m_isr))
    if all_pogema_agents:
        algo_vals_p = {
            "follower": [f_isr.get(n) for n in all_pogema_agents],
            "matslp":   [m_isr.get(n) for n in all_pogema_agents],
        }
        _grouped_bars(ax_pogema, all_pogema_agents, algo_vals_p, {})
        ax_pogema.set_ylim(0, 1.1)
        ax_pogema.axhline(1.0, color="#ccc", linewidth=0.8, zorder=1)
        _setup_ax(ax_pogema, "Number of agents", "ISR (POGEMA)", "POGEMA (random, 0% obstacles)")
    else:
        ax_pogema.text(0.5, 0.5, "No POGEMA data", ha="center", va="center",
                       transform=ax_pogema.transAxes, color="gray")
        ax_pogema.set_title("POGEMA ISR")

    fig.suptitle("Success Rate / ISR", fontweight="bold", y=1.01)
    _save(fig, "fig4_success_rate", out)


# ── Figure 5: Throughput (MATS-LP vs Local Leaders vs Follower) ──────────────

def fig_throughput(
    matslp_rows: List[dict],
    follower_rows: List[dict],
    ll_rows: List[dict],
    out: Path,
) -> None:
    """Throughput (goals/timestep) by agent count for 0% and 30% obstacle density.

    All three algorithms are shown on the same scale:
      - MATS-LP: from POGEMA LifeLong metric (goals/step)
      - Follower: ISR × n_agents / ep_length  (goals/step, recomputed)
      - Local Leaders: agents_done / ep_length  (goals/step)
    """
    densities = [0, 30]
    algos = ["matslp", "local-leaders", "follower"]
    all_rows = {
        "matslp": matslp_rows,
        "follower": follower_rows,
        "local-leaders": ll_rows,
    }

    fig, axes = plt.subplots(1, len(densities), figsize=(5.5 * len(densities), 4.2), sharey=True)
    if len(densities) == 1:
        axes = [axes]

    for ax, density in zip(axes, densities):
        agent_sets: set = set()
        for algo in algos:
            # Only random-map rows so maze/warehouse don't bleed into density panels
            rand_rows = [r for r in all_rows[algo] if r.get("map_type") == "random"]
            d = agg_throughput_by_agents(rand_rows, algo, density)
            agent_sets |= set(d.keys())
        agents = sorted(agent_sets)
        if not agents:
            ax.set_title(f"{density}% obstacles")
            continue

        algo_vals, algo_errs = {}, {}
        for algo in algos:
            rand_rows = [r for r in all_rows[algo] if r.get("map_type") == "random"]
            d = agg_throughput_by_agents(rand_rows, algo, density)
            algo_vals[algo] = [d.get(n, {}).get("mean") for n in agents]
            algo_errs[algo] = [d.get(n, {}).get("std", 0.0) for n in agents]

        _grouped_bars(ax, agents, algo_vals, algo_errs)
        _setup_ax(
            ax,
            "Number of agents",
            "Throughput (goals / timestep)" if density == 0 else "",
            f"Random maps, {density}% obstacles",
        )

    fig.suptitle(
        "Throughput: MATS-LP vs Local Leaders vs Follower  (POGEMA, random maps)",
        fontweight="bold", y=1.01,
    )
    _save(fig, "fig5_throughput_comparison", out)


# ── Figure 6: SoC overhead ratio ─────────────────────────────────────────────

def fig_soc_overhead(rows: List[dict], out: Path) -> None:
    """SoC ratio (Local Leaders / LaCAM) per map type — shows cost overhead."""
    if not rows:
        return
    map_types = [mt for mt in MAP_TYPE_ORDER if any(r["map_type"] == mt for r in rows)]

    fig, axes = plt.subplots(1, len(map_types), figsize=(4.5 * len(map_types), 4.2), sharey=True)
    if len(map_types) == 1:
        axes = [axes]

    for ax, mt in zip(axes, map_types):
        mt_rows = [r for r in rows if r["map_type"] == mt]
        agents = sorted({r["num_agents"] for r in mt_rows})
        lc_agg = agg(mt_rows, "lacam", "soc").get(mt, {})
        ll_agg = agg(mt_rows, "local-leaders", "soc").get(mt, {})

        xs, ratios = [], []
        for na in agents:
            lc_m = lc_agg.get(na, {}).get("mean")
            ll_m = ll_agg.get(na, {}).get("mean")
            if lc_m and ll_m and lc_m > 0:
                xs.append(na)
                ratios.append(ll_m / lc_m)

        if xs:
            ax.plot(xs, ratios, marker="o", color=COLORS["local-leaders"],
                    linewidth=2, markersize=6, label="Local Leaders / LaCAM", zorder=3)
        ax.axhline(1.0, color=COLORS["lacam"], linestyle="--", linewidth=1.2,
                   label="LaCAM (ratio = 1)", zorder=2)
        _setup_ax(ax, "Number of agents", "SoC ratio (LL / LaCAM)" if mt == map_types[0] else "",
                  MAP_TYPE_TITLES[mt])
        ax.set_ylim(bottom=0)
        ax.legend()

    fig.suptitle("SoC Overhead: Local Leaders vs LaCAM  (ratio > 1 = worse than LaCAM)",
                 fontweight="bold", y=1.01)
    _save(fig, "fig6_soc_overhead", out)


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary(rows: List[dict]) -> None:
    if not rows:
        return
    algos = sorted({r["algo"] for r in rows})
    map_types = sorted({r["map_type"] for r in rows})

    print("\n=== MovingAI Summary ===")
    header = f"{'algo':<18} {'map_type':<12} {'n':>4}  {'ok%':>5}  {'soc':>8}  {'mk':>7}  {'ms':>9}"
    print(header)
    print("-" * len(header))
    for algo in algos:
        for mt in map_types:
            sub = [r for r in rows if r["algo"] == algo and r["map_type"] == mt]
            if not sub:
                continue
            for na in sorted({r["num_agents"] for r in sub}):
                na_rows = [r for r in sub if r["num_agents"] == na]
                sr = np.mean([1.0 if r["solved"] else 0.0 for r in na_rows])
                soc_v = [r["soc"] for r in na_rows if r["soc"] is not None]
                mk_v  = [r["makespan"] for r in na_rows if r["makespan"] is not None]
                t_v   = [r["comp_time_ms"] for r in na_rows if r["comp_time_ms"] is not None]
                soc_s = f"{np.mean(soc_v):8.0f}" if soc_v else "       —"
                mk_s  = f"{np.mean(mk_v):7.0f}" if mk_v else "      —"
                t_s   = f"{np.mean(t_v):9.1f}" if t_v else "        —"
                print(f"{algo:<18} {mt:<12} {na:>4}  {sr*100:>4.0f}%  {soc_s}  {mk_s}  {t_s}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    if not HAS_MPL:
        return

    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    movingai_rows  = load_movingai(RESULTS_DIR / "movingai_benchmark.csv")
    matslp_rows    = load_throughput(MATS_LP_RESULTS / "benchmark_results.csv", "matslp")
    follower_rows  = load_follower_summary(RESULTS_DIR / "follower_results_follower_summary.csv")
    ll_pogema_rows = load_ll_pogema(RESULTS_DIR / "ll_pogema_results.csv")

    print_summary(movingai_rows)
    print(f"\nGenerating figures → {GRAPHS_DIR}/")

    fig_soc(movingai_rows, GRAPHS_DIR)
    fig_makespan(movingai_rows, GRAPHS_DIR)
    fig_planning_time(movingai_rows, GRAPHS_DIR)
    fig_success_rate(movingai_rows, follower_rows, matslp_rows, GRAPHS_DIR)
    fig_throughput(matslp_rows, follower_rows, ll_pogema_rows, GRAPHS_DIR)
    fig_soc_overhead(movingai_rows, GRAPHS_DIR)

    print("Done.")


if __name__ == "__main__":
    main()
