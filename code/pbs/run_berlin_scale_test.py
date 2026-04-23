import csv
import time
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt

from pbs_wrapper import solve


DEFAULT_K_VALUES = [10, 25, 50, 75, 100,150,200]
DEFAULT_TIMEOUT_SECONDS = 60
TARGET_MAPS = {"lak303d", "brc202d"}

SOLVER_METRIC_FIELDS = {
    "solver_runtime_sec": "runtime",
    "high_level_generated": "#high-level generated",
    "high_level_expanded": "#high-level expanded",
    "low_level_generated": "#low-level generated",
    "low_level_expanded": "#low-level expanded",
    "flowtime": "solution cost",
    "root_g_value": "root g value",
}


def scenario_sort_key(path: Path) -> int:
    try:
        return int(path.stem.split("-")[-1])
    except ValueError:
        return 0


def discover_datasets_even(root: Path) -> List[Dict[str, Path]]:
    map_dir = root / "maps"
    scen_root = root / "scenarios" / "scen-even"

    datasets: List[Dict[str, Path]] = []
    for map_file in sorted(map_dir.glob("*.map")):
        if map_file.stem not in TARGET_MAPS:
            continue

        scen_files = sorted(
            scen_root.glob(f"{map_file.stem}-even-*.scen"),
            key=scenario_sort_key,
        )
        if not scen_files:
            print(f"Skipping {map_file.name}: no matching scenarios found in {scen_root}")
            continue

        datasets.append({"map": map_file, "scen_root": scen_root, "name": map_file.stem, "scenarios": scen_files})

    return datasets

def discover_datasets(root: Path) -> List[Dict[str, Path]]:
    map_dir = root / "maps"
    scen_root = root / "scenarios" / "scen-random"

    datasets: List[Dict[str, Path]] = []
    for map_file in sorted(map_dir.glob("*.map")):
        if map_file.stem not in TARGET_MAPS:
            continue

        scen_files = sorted(
            scen_root.glob(f"{map_file.stem}-random-*.scen"),
            key=scenario_sort_key,
        )
        if not scen_files:
            print(f"Skipping {map_file.name}: no matching scenarios found in {scen_root}")
            continue

        datasets.append({"map": map_file, "scen_root": scen_root, "name": map_file.stem, "scenarios": scen_files})

    return datasets


def safe_name(text: str) -> str:
    return text.replace(" ", "_").replace("/", "_")


def aggregate_by_k(rows: List[Dict[str, object]], k_values: Iterable[int]) -> List[Dict[str, float]]:
    aggregate: List[Dict[str, float]] = []
    for k in k_values:
        subset = [row for row in rows if int(row["k"]) == k]
        count = len(subset)
        success_count = sum(1 for row in subset if bool(row["success"]))
        avg_elapsed = sum(float(row["elapsed_sec"]) for row in subset) / count if count else 0.0
        success_rate = (100.0 * success_count / count) if count else 0.0
        aggregate_row: Dict[str, float] = {
            "k": float(k),
            "runs": float(count),
            "successes": float(success_count),
            "success_rate_pct": round(success_rate, 2),
            "avg_elapsed_sec": round(avg_elapsed, 3),
        }

        for output_key in SOLVER_METRIC_FIELDS:
            values = [float(row[output_key]) for row in subset if output_key in row and row[output_key] != ""]
            aggregate_row[f"avg_{output_key}"] = round(sum(values) / len(values), 3) if values else 0.0

        suboptimality_values = [
            float(row["suboptimality_ratio"])
            for row in subset
            if "suboptimality_ratio" in row and row["suboptimality_ratio"] != ""
        ]
        aggregate_row["avg_suboptimality_ratio"] = (
            round(sum(suboptimality_values) / len(suboptimality_values), 4) if suboptimality_values else 0.0
        )

        aggregate.append(aggregate_row)

    return aggregate


def read_solver_metrics(csv_file: Path) -> Dict[str, float]:
    if not csv_file.exists() or not csv_file.is_file():
        return {}

    with open(csv_file, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        row = next(reader, None)
        if not row:
            return {}

    metrics: Dict[str, float] = {}
    for output_key, csv_key in SOLVER_METRIC_FIELDS.items():
        raw_value = row.get(csv_key, "")
        try:
            metrics[output_key] = float(raw_value)
        except (TypeError, ValueError):
            metrics[output_key] = 0.0

    return metrics


def append_solver_csv(source_csv: Path, dest_csv: Path) -> None:
    if not source_csv.exists() or not source_csv.is_file():
        return

    lines = source_csv.read_text(encoding="utf-8").splitlines()
    if not lines:
        source_csv.unlink(missing_ok=True)
        return

    if dest_csv.exists() and dest_csv.stat().st_size > 0:
        lines_to_append = lines[1:]
    else:
        lines_to_append = lines

    if lines_to_append:
        with open(dest_csv, "a", encoding="utf-8", newline="") as f:
            for line in lines_to_append:
                f.write(f"{line}\n")

    source_csv.unlink(missing_ok=True)


def read_map_as_image(map_file: Path) -> List[List[int]]:
    lines = map_file.read_text(encoding="utf-8").splitlines()
    grid_start = None
    for idx, line in enumerate(lines):
        if line.strip().lower() == "map":
            grid_start = idx + 1
            break

    if grid_start is None or grid_start >= len(lines):
        return []

    grid: List[List[int]] = []
    for raw in lines[grid_start:]:
        row = raw.rstrip("\n")
        if not row:
            continue
        # Free cells are 0 (light), blocked cells are 1 (dark).
        grid.append([0 if ch in {".", "G", "S"} else 1 for ch in row])

    return grid


def plot_single_metric(
    plot_file: Path,
    aggregate_rows: List[Dict[str, float]],
    metric_key: str,
    y_label: str,
    line_color: str,
    y_max_override: float = 0.0,
    map_image: List[List[int]] | None = None,
) -> None:
    if not aggregate_rows:
        return

    x = [int(row["k"]) for row in aggregate_rows]
    y_values = [float(row[metric_key]) for row in aggregate_rows]
    y_max = y_max_override if y_max_override > 0.0 else max(max(y_values), 1.0)

    fig, ax = plt.subplots(figsize=(9, 5.6), dpi=140)
    ax.plot(x, y_values, marker="o", linewidth=2.4, markersize=6, color=line_color)
    ax.set_xlabel("Agents (k)")
    ax.set_ylabel(y_label)
    ax.set_xticks(x)
    ax.set_ylim(0.0, y_max)
    ax.grid(True, which="major", axis="both", linestyle="--", linewidth=0.7, alpha=0.35)


    if map_image:
        # Upper-left map inset for runtime plots.
        inset = ax.inset_axes([0.0, 0.66, 0.34, 0.34])  # [x0, y0, width, height] in axes fraction
        inset.imshow(map_image, cmap="gray_r", interpolation="nearest", aspect="auto")
        inset.set_xticks([])
        inset.set_yticks([])
        inset.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in inset.spines.values():
            spine.set_edgecolor("#374151")
            spine.set_linewidth(1.0)

    fig.tight_layout()
    plot_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_file, bbox_inches="tight")
    plt.close(fig)


def plot_runtime(plot_file: Path, aggregate_rows: List[Dict[str, float]], map_file: Path | None = None) -> None:
    map_image = read_map_as_image(map_file) if map_file else None
    plot_single_metric(
        plot_file,
        aggregate_rows,
        metric_key="avg_elapsed_sec",
        y_label="Runtime (s)",
        line_color="#1d4ed8",
        map_image=map_image,
    )


def plot_success(plot_file: Path, aggregate_rows: List[Dict[str, float]]) -> None:
    plot_single_metric(
        plot_file,
        aggregate_rows,
        metric_key="success_rate_pct",
        y_label="Success rate (%)",
        line_color="#15803d",
        y_max_override=100.0,
    )


def plot_suboptimality(plot_file: Path, aggregate_rows: List[Dict[str, float]]) -> None:
    plot_single_metric(
        plot_file,
        aggregate_rows,
        metric_key="avg_suboptimality_ratio",
        y_label="Suboptimality ratio",
        line_color="#b91c1c",
    )


def plot_summary_table(plot_file: Path, aggregate_rows: List[Dict[str, float]]) -> None:
    if not aggregate_rows:
        return

    headers = [
        "k",
        "runs",
        "successes",
        "success rate (%)",
        "avg runtime (s)",
        "avg suboptimality",
        "avg low-level\nexpansions",
        "avg high-level\nexpansions",
        "avg flowtime",
    ]

    fig, ax = plt.subplots(figsize=(16, max(2.8, 0.55 * (len(aggregate_rows) + 1))), dpi=160)
    ax.axis("off")

    cell_text: List[List[str]] = []
    for row in aggregate_rows:
        cell_text.append(
            [
                f"{int(row['k'])}",
                f"{int(row['runs'])}",
                f"{int(row['successes'])}",
                f"{row['success_rate_pct']:.2f}",
                f"{row['avg_elapsed_sec']:.3f}",
                f"{row.get('avg_suboptimality_ratio', 0.0):.4f}",
                f"{row.get('avg_low_level_expanded', 0.0):.2f}",
                f"{row.get('avg_high_level_expanded', 0.0):.2f}",
                f"{row.get('avg_flowtime', 0.0):.2f}",
            ]
        )

    table = ax.table(
        cellText=cell_text,
        colLabels=headers,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.45)

    for (row_idx, _col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#111827")
        cell.set_linewidth(0.6)
        if row_idx == 0:
            cell.set_facecolor("#e5e7eb")
            cell.get_text().set_fontweight("bold")
        elif row_idx % 2 == 1:
            cell.set_facecolor("#f9fafb")

    fig.tight_layout(pad=0.8)
    plot_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_file, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parent

    datasets = discover_datasets(root)
    datasets_even = discover_datasets_even(root)
    if not datasets:
        raise SystemExit("No map/scenario datasets found.")

    output_dir = root / "scale_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = output_dir / "all_maps_scale_summary.csv"
    aggregate_csv = output_dir / "all_maps_scale_aggregate_by_k.csv"
    runtime_plot_file = output_dir / "all_maps_scale_runtime_plot.png"
    success_plot_file = output_dir / "all_maps_scale_success_plot.png"
    suboptimality_plot_file = output_dir / "all_maps_scale_suboptimality_plot.png"
    table_plot_file = output_dir / "all_maps_scale_table.png"

    k_values = DEFAULT_K_VALUES
    timeout_seconds = DEFAULT_TIMEOUT_SECONDS

    rows = []
    print("Running scale test for all available maps and matching scenarios...")
    print(f"Maps discovered: {len(datasets_even)}")
    print(f"k values: {k_values}")
    print(f"timeout: {timeout_seconds}s")

    for dataset_idx, dataset in enumerate(datasets_even, start=1):
        map_file = dataset["map"]
        scen_files = dataset["scenarios"]
        dataset_name = dataset["name"]
        dataset_dir = output_dir / safe_name(dataset_name)
        dataset_dir.mkdir(parents=True, exist_ok=True)

        print()
        print(f"Dataset {dataset_idx:>2}/{len(datasets_even)}: {dataset_name}")
        print(f"Map: {map_file}")
        print(f"Scenarios: {len(scen_files)} files from {dataset['scen_root']}")

        dataset_rows = []
        dataset_solver_csv = dataset_dir / f"{dataset_name}.csv"
        for scen_idx, scen_file in enumerate(scen_files, start=1):
            print(f"  Scenario {scen_idx:>2}/{len(scen_files)}: {scen_file.name}")
            for k in k_values:
                tmp_out_csv = dataset_dir / f"{scen_file.stem}_k{k}_tmp.csv"
                output_paths = dataset_dir / f"{scen_file.stem}_k{k}_paths.txt"

                t0 = time.perf_counter()
                result = solve(
                    str(map_file),
                    str(scen_file),
                    str(tmp_out_csv),
                    str(output_paths),
                    k,
                    timeout_seconds,
                )
                elapsed = time.perf_counter() - t0
                solver_metrics = read_solver_metrics(tmp_out_csv)
                append_solver_csv(tmp_out_csv, dataset_solver_csv)

                row = {
                    "map": dataset_name,
                    "scenario": scen_file.name,
                    "k": k,
                    "success": bool(result.get("success")),
                    "elapsed_sec": round(elapsed, 3),
                    "returncode": result.get("returncode"),
                    "parsed_paths": len(result.get("paths", []) or []),
                    "error": result.get("error", ""),
                    "out_csv": str(dataset_solver_csv),
                    "output_paths": str(output_paths),
                }
                row.update(solver_metrics)
                flowtime = float(row.get("flowtime", 0.0) or 0.0)
                root_g = float(row.get("root_g_value", 0.0) or 0.0)
                row["suboptimality_ratio"] = round(flowtime / root_g, 4) if root_g > 0.0 else 0.0
                rows.append(row)
                dataset_rows.append(row)

                status = "OK" if row["success"] else "FAIL"
                print(
                    f"    k={k:>3} | {status:>4} | elapsed={row['elapsed_sec']:>7}s | "
                    f"paths={row['parsed_paths']:>3} | rc={row['returncode']}"
                )
                if row["error"]:
                    print(f"      error: {row['error']}")
        dataset_summary_csv = dataset_dir / f"{dataset_name}_scale_summary.csv"
        dataset_aggregate_csv = dataset_dir / f"{dataset_name}_scale_aggregate_by_k.csv"
        dataset_runtime_plot = dataset_dir / f"{dataset_name}_scale_runtime_plot.png"
        dataset_success_plot = dataset_dir / f"{dataset_name}_scale_success_plot.png"
        dataset_suboptimality_plot = dataset_dir / f"{dataset_name}_scale_suboptimality_plot.png"
        dataset_table_plot = dataset_dir / f"{dataset_name}_scale_table.png"

        with open(dataset_summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(dataset_rows[0].keys()))
            writer.writeheader()
            writer.writerows(dataset_rows)

        dataset_aggregate = aggregate_by_k(dataset_rows, k_values)
        with open(dataset_aggregate_csv, "w", newline="", encoding="utf-8") as f:
            dataset_aggregate_fieldnames = [
                "k",
                "runs",
                "successes",
                "success_rate_pct",
                "avg_elapsed_sec",
                "avg_suboptimality_ratio",
                *[f"avg_{key}" for key in SOLVER_METRIC_FIELDS],
            ]
            writer = csv.DictWriter(
                f,
                fieldnames=dataset_aggregate_fieldnames,
            )
            writer.writeheader()
            writer.writerows(dataset_aggregate)

        plot_runtime(
            dataset_runtime_plot,
            dataset_aggregate,
            map_file,
        )
        plot_success(
            dataset_success_plot,
            dataset_aggregate,
        )
        plot_suboptimality(
            dataset_suboptimality_plot,
            dataset_aggregate,
        )
        plot_summary_table(dataset_table_plot, dataset_aggregate)

        print(f"  Wrote dataset summary: {dataset_summary_csv}")
        print(f"  Wrote dataset aggregate: {dataset_aggregate_csv}")
        print(f"  Wrote dataset runtime plot: {dataset_runtime_plot}")
        print(f"  Wrote dataset success plot: {dataset_success_plot}")
        print(f"  Wrote dataset suboptimality plot: {dataset_suboptimality_plot}")
        print(f"  Wrote dataset table: {dataset_table_plot}")

    fieldnames = [
        "map",
        "scenario",
        "k",
        "success",
        "elapsed_sec",
        "returncode",
        "parsed_paths",
        "error",
        "out_csv",
        "output_paths",
        "suboptimality_ratio",
        *SOLVER_METRIC_FIELDS.keys(),
    ]

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    aggregate = aggregate_by_k(rows, k_values)

    with open(aggregate_csv, "w", newline="", encoding="utf-8") as f:
        aggregate_fieldnames = [
            "k",
            "runs",
            "successes",
            "success_rate_pct",
            "avg_elapsed_sec",
            "avg_suboptimality_ratio",
            *[f"avg_{key}" for key in SOLVER_METRIC_FIELDS],
        ]
        writer = csv.DictWriter(
            f,
            fieldnames=aggregate_fieldnames,
        )
        writer.writeheader()
        writer.writerows(aggregate)

    plot_runtime(runtime_plot_file, aggregate)
    plot_success(success_plot_file, aggregate)
    plot_suboptimality(suboptimality_plot_file, aggregate)
    plot_summary_table(table_plot_file, aggregate)

    print()
    print(f"Detailed summary written to: {summary_csv}")
    print(f"Aggregate summary written to: {aggregate_csv}")
    print(f"Runtime plot written to: {runtime_plot_file}")
    print(f"Success plot written to: {success_plot_file}")
    print(f"Suboptimality plot written to: {suboptimality_plot_file}")
    print(f"Table plot written to: {table_plot_file}")
    print(f"Dataset outputs written under: {output_dir}")


if __name__ == "__main__":
    main()
