"""
PBS benchmark runner — same map/agent/seed configurations as LaCAM/Local Leaders.

Outputs results in movingai_benchmark.csv format so generate_graphs.py can
plot PBS alongside LaCAM and Local Leaders.
"""

import csv
import subprocess
import tempfile
import time
from pathlib import Path

PBS_DIR   = Path(__file__).resolve().parents[1] / "pbs"
PBS_BIN   = PBS_DIR / "pbs"
MAP_DIR   = PBS_DIR / "maps"
SCEN_DIR  = PBS_DIR / "scenarios" / "scen-random"

RESULTS_DIR = Path(__file__).parent / "results"
OUT_CSV     = RESULTS_DIR / "movingai_benchmark.csv"

TIMEOUT_SEC = 30  # per instance

# (map_type, map_name, num_agents, seed)  — identical to movingai_benchmark.csv
CONFIGS = [
    # random 32×32
    *[("random", "random-32-32-10", n, s) for n in [8, 16, 32]  for s in [1, 2, 3]],
    # random 64×64
    *[("random", "random-64-64-10", n, s) for n in [16, 32, 64] for s in [1, 2, 3]],
    # maze 32×32 corridor-2
    *[("maze",   "maze-32-32-2",    n, s) for n in [4, 8, 16]   for s in [1, 2, 3]],
    # maze 32×32 corridor-4
    *[("maze",   "maze-32-32-4",    n, s) for n in [8, 16]      for s in [1, 2, 3]],
    # warehouse small
    *[("warehouse", "warehouse-10-20-10-2-1", n, s) for n in [8, 16]  for s in [1, 2, 3]],
    # warehouse large
    *[("warehouse", "warehouse-20-40-10-2-1", n, s) for n in [16, 32] for s in [1, 2, 3]],
]


def makespan_from_paths(paths_file: str) -> int | None:
    """Count max path length (= makespan) from PBS --outputPaths file."""
    try:
        max_len = 0
        with open(paths_file) as f:
            for line in f:
                line = line.strip()
                if not line or ":" not in line:
                    continue
                # "Agent N: (r,c)->(r,c)->..."
                steps = line.split(":")[1].strip().split("->")
                max_len = max(max_len, len([s for s in steps if s.strip()]))
        return max_len if max_len > 0 else None
    except Exception:
        return None


def run_pbs(map_name: str, num_agents: int, seed: int) -> dict:
    map_file  = MAP_DIR  / f"{map_name}.map"
    scen_file = SCEN_DIR / f"{map_name}-random-{seed}.scen"

    if not map_file.exists():
        return {"solved": False, "error": f"map not found: {map_file}"}
    if not scen_file.exists():
        return {"solved": False, "error": f"scen not found: {scen_file}"}

    with tempfile.TemporaryDirectory() as tmp:
        out_csv   = f"{tmp}/result.csv"
        out_paths = f"{tmp}/paths.txt"

        cmd = [
            str(PBS_BIN),
            "-m", str(map_file),
            "-a", str(scen_file),
            "-o", out_csv,
            "--outputPaths", out_paths,
            "-k", str(num_agents),
            "-t", str(TIMEOUT_SEC),
        ]

        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=TIMEOUT_SEC + 15,
            )
        except subprocess.TimeoutExpired:
            return {"solved": False, "error": "timeout", "comp_time_ms": TIMEOUT_SEC * 1000}
        except Exception as e:
            return {"solved": False, "error": str(e)}

        elapsed_ms = (time.perf_counter() - t0) * 1000

        # First stdout line: "PBS with SIPP   : Succeed,<soc>,<runtime>,..."
        first_line = (proc.stdout or "").splitlines()[0] if proc.stdout else ""
        solved = "Succeed" in first_line

        soc = None
        comp_time_ms = None
        makespan = None

        if solved and Path(out_csv).exists():
            try:
                with open(out_csv) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        comp_time_ms = float(row["runtime"]) * 1000
                        soc = float(row["solution cost"])
                        break
            except Exception:
                pass

            makespan = makespan_from_paths(out_paths)

        return {
            "solved":       solved,
            "soc":          soc,
            "makespan":     makespan,
            "comp_time_ms": comp_time_ms if comp_time_ms is not None else elapsed_ms,
            "error":        "" if solved else (proc.stderr or proc.stdout or "")[:120],
        }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing rows (non-PBS) so we can append/replace PBS rows
    existing = []
    if OUT_CSV.exists():
        with open(OUT_CSV, newline="") as f:
            for row in csv.DictReader(f):
                if row["algo"] != "pbs":
                    existing.append(row)

    pbs_rows = []
    total = len(CONFIGS)
    t_bench = time.perf_counter()

    for i, (map_type, map_name, num_agents, seed) in enumerate(CONFIGS, 1):
        print(f"[{i}/{total}] PBS  {map_name}  N={num_agents}  seed={seed} ...", end=" ", flush=True)
        result = run_pbs(map_name, num_agents, seed)

        elapsed_total = time.perf_counter() - t_bench
        eta = elapsed_total / i * (total - i)
        eta_str = f"{int(eta//60):02d}:{int(eta%60):02d}"

        if result["solved"]:
            print(f"SoC={result['soc']:.0f}  mk={result['makespan']}  "
                  f"t={result['comp_time_ms']:.1f}ms  ETA {eta_str}")
        else:
            print(f"FAILED ({result.get('error', '')[:60]})  ETA {eta_str}")

        pbs_rows.append({
            "algo":        "pbs",
            "map_type":    map_type,
            "map_name":    map_name,
            "num_agents":  num_agents,
            "seed":        seed,
            "solved":      result["solved"],
            "makespan":    result.get("makespan") or "",
            "soc":         result.get("soc") or "",
            "comp_time_ms": f"{result.get('comp_time_ms', 0):.2f}",
            "error":       result.get("error", ""),
        })

    fieldnames = ["algo", "map_type", "map_name", "num_agents", "seed",
                  "solved", "makespan", "soc", "comp_time_ms", "error"]

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in existing:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
        writer.writerows(pbs_rows)

    total_time = time.perf_counter() - t_bench
    print(f"\nDone in {total_time:.1f}s. Results appended to {OUT_CSV}")


if __name__ == "__main__":
    main()
