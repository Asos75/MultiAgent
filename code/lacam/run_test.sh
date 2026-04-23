#!/bin/bash
set -euo pipefail

tests=(
  "room-32-32-4|50 100 200 300"
  "room-64-64-16|50 250 500 750 1000"
  "room-64-64-8|50 250 500 750 1000"
  "random-32-32-20|50 100 200 300 400"
  "random-64-64-20|50 250 500 750 1000"
  "maze-32-32-2|50 100 200 300"
  "warehouse-20-40-10-2-1|50 250 500 750 1000"
  "warehouse-20-40-10-2-2|50 250 500 750 1000"
  "ost003d|50 250 500 750 1000"
  "lt_gallowstemplar_n|50 250 500 750 1000"
  "brc202d|50 250 500 750 1000"
  "orz900d|50 250 500 750 1000"
)

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

MAP_DIR="${REPO_DIR}/assets/moving_ai_maps"
SCEN_DIR="${REPO_DIR}/assets/moving_ai_scen"
BIN="${SCRIPT_DIR}/build/main"

RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "$RESULTS_DIR"

CSV_FILE="${RESULTS_DIR}/results.csv"
RAW_OUTPUT="${RESULTS_DIR}/raw_output.txt"

# RESET CSV EVERY RUN (important for experiments)
echo "map,scenario,N,runtime_ms,makespan,soc,success" > "$CSV_FILE"
: > "$RAW_OUTPUT"

for test in "${tests[@]}"; do
  IFS="|" read -r map Ns <<< "$test"

  map_file="${MAP_DIR}/${map}.map"
  if [[ ! -f "$map_file" ]]; then
    echo "Missing map: $map_file"
    continue
  fi

  scen_files=("${SCEN_DIR}/${map}-"*.scen)

  if [[ ! -e "${scen_files[0]}" ]]; then
    echo "No scen files for $map"
    continue
  fi

  for scen_file in "${scen_files[@]}"; do
    scen_name=$(basename "$scen_file" .scen)

    for N in $Ns; do
      echo "Running $map | $scen_name | N=$N"

      output=$("$BIN" -m "$map_file" -i "$scen_file" -N "$N" -v 1 2>&1 || true)

      echo "===== $map | $scen_name | N=$N =====" >> "$RAW_OUTPUT"
      echo "$output" >> "$RAW_OUTPUT"
      echo "" >> "$RAW_OUTPUT"

      # runtime in ms (IMPORTANT FIX)
      runtime_ms=$(echo "$output" | grep -oP 'solved:\s*\K[0-9]+' | head -1 || echo "NA")

      makespan=$(echo "$output" | grep -i "makespan" | grep -oE '[0-9]+' | head -1 || echo "NA")
      soc=$(echo "$output" | grep -Ei "sum.?of.?costs|soc" | grep -oE '[0-9]+' | head -1 || echo "NA")

      if echo "$output" | grep -qi "failed to solve"; then
        success=0
      else
        success=1
      fi

      echo "$map,$scen_name,$N,$runtime_ms,$makespan,$soc,$success" >> "$CSV_FILE"
    done
  done
done

echo "Done. Results: $CSV_FILE"