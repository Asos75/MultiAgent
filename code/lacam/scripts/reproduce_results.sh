#!/usr/bin/env bash
# Reproduce (a subset of) LaCAM runs and export a single CSV with key metrics.
#
# This script intentionally targets small/medium instances so it finishes on a laptop.
# Adjust MAPS / AGENTS to match the paper-scale experiments.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LACAM_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ROOT_CODE_DIR="$(cd -- "${LACAM_DIR}/.." && pwd)"

BIN="${LACAM_DIR}/build/main"
MAP_DIR="${ROOT_CODE_DIR}/assets/moving_ai_maps"
SCEN_DIR="${ROOT_CODE_DIR}/assets/moving_ai_scen"

OUT_DIR="${LACAM_DIR}/repro_out"
mkdir -p "${OUT_DIR}"

CSV="${OUT_DIR}/lacam_results.csv"
echo "map,scen,N,seed,solved,soc,makespan,comp_time_ms,output_file" > "${CSV}"

MAPS=(
  "random-32-32-10"
  "maze-32-32-2"
  "room-32-32-4"
)

AGENTS=(50 100)
SEEDS=(0)
TIME_LIMIT_SEC=10

pick_scen_file() {
  local map_name="$1"
  local preferred="${SCEN_DIR}/${map_name}-random-1.scen"
  if [[ -f "$preferred" ]]; then
    echo "$preferred"
    return 0
  fi
  local first
  first=$(ls -1 "${SCEN_DIR}/${map_name}-"*.scen 2>/dev/null | head -n 1 || true)
  if [[ -n "$first" ]]; then
    echo "$first"
    return 0
  fi
  return 1
}

parse_result_kv() {
  local file="$1"
  local key="$2"
  # keys are in form "key=value" one per line
  local line
  line=$(grep -E "^${key}=" "$file" | head -n 1 || true)
  if [[ -z "$line" ]]; then
    echo ""
  else
    echo "${line#*=}"
  fi
}

if [[ ! -x "${BIN}" ]]; then
  echo "Binary not found: ${BIN}" >&2
  echo "Build first: cmake -B build && make -C build" >&2
  exit 1
fi

for map in "${MAPS[@]}"; do
  map_file="${MAP_DIR}/${map}.map"
  if [[ ! -f "$map_file" ]]; then
    echo "Map file not found: ${map_file}" >&2
    continue
  fi

  if ! scen_file="$(pick_scen_file "$map")"; then
    echo "Scenario file not found for map: ${map}" >&2
    continue
  fi

  for N in "${AGENTS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      out_file="${OUT_DIR}/${map}-N${N}-seed${seed}.txt"

      echo "[LaCAM] map=${map} N=${N} seed=${seed}"
      "${BIN}" -m "$map_file" -i "$scen_file" -N "$N" -s "$seed" -t "$TIME_LIMIT_SEC" -v 1 -o "$out_file" -l

      solved=$(parse_result_kv "$out_file" "solved")
      soc=$(parse_result_kv "$out_file" "soc")
      makespan=$(parse_result_kv "$out_file" "makespan")
      comp_time_ms=$(parse_result_kv "$out_file" "comp_time")

      echo "${map},$(basename "$scen_file"),${N},${seed},${solved},${soc},${makespan},${comp_time_ms},${out_file}" >> "$CSV"
    done
  done
done

echo "Wrote: ${CSV}"
