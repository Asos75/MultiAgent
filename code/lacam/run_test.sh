tests=(
  # Room maps
  "room-32-32-4|50 100 200 300"
  "room-64-64-16|50 250 500 750 1000"
  "room-64-64-8|50 250 500 750 1000"

  # Random maps
  "random-32-32-20|50 100 200 300 400"
  "random-64-64-20|50 250 500 750 1000"

  # Maze map
  "maze-32-32-2|50 100 200 300"

  # Warehouse / others
  "warehouse-20-40-10-2-1|50 250 500 750 1000"
  "warehouse-20-40-10-2-2|50 250 500 750 1000"
  "ost003d|50 250 500 750 1000"
  "lt-gallowstemplar-n|50 250 500 750 1000"
  "brc202d|50 250 500 750 1000"
  "orz900d|50 250 500 750 1000"
)

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_CODE_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
MAP_DIR="${REPO_CODE_DIR}/assets/moving_ai_maps"
SCEN_DIR="${REPO_CODE_DIR}/assets/moving_ai_scen"

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

# Runner script
for test in "${tests[@]}"; do
  IFS="|" read -r map Ns <<< "$test"

  map_file="${MAP_DIR}/${map}.map"
  if [[ ! -f "$map_file" ]]; then
    echo "map file not found: $map_file" >&2
    continue
  fi

  if ! scen_file="$(pick_scen_file "$map")"; then
    echo "scenario file not found for map: $map (looked in $SCEN_DIR)" >&2
    continue
  fi

  for N in $Ns; do
    echo "Running $map with N=$N"
    "${SCRIPT_DIR}/build/main" -m "$map_file" -i "$scen_file" -N "$N" -v 1
  done
done