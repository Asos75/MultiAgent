from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Tuple


@dataclass(frozen=True)
class MovingAIScenRow:
    map_file: str
    width: int
    height: int
    start_xy: Tuple[int, int]
    goal_xy: Tuple[int, int]


def read_movingai_map(map_path: Path) -> List[List[int]]:
    """Read MovingAI .map into a 0/1 obstacle grid (1=obstacle, 0=free)."""
    text = map_path.read_text(encoding="utf-8", errors="replace").splitlines()

    # Find the 'map' line.
    try:
        map_idx = next(i for i, ln in enumerate(text) if ln.strip() == "map")
    except StopIteration as e:
        raise ValueError(f"Invalid .map (missing 'map' line): {map_path}") from e

    header = text[:map_idx]
    body = text[map_idx + 1 :]

    h = None
    w = None
    for ln in header:
        ln = ln.strip()
        if ln.startswith("height"):
            h = int(ln.split()[1])
        elif ln.startswith("width"):
            w = int(ln.split()[1])

    if h is None or w is None:
        raise ValueError(f"Invalid .map (missing width/height): {map_path}")
    if len(body) < h:
        raise ValueError(f"Invalid .map (not enough rows): {map_path}")

    grid: List[List[int]] = []
    for y in range(h):
        row = body[y]
        if len(row) < w:
            raise ValueError(f"Invalid .map row width at y={y}: {map_path}")
        # In MovingAI, '@' and 'T' are obstacles.
        grid.append([1 if c in ("@", "T") else 0 for c in row[:w]])

    return grid


def iter_scen_rows(scen_path: Path) -> Iterator[MovingAIScenRow]:
    """Yield rows from MovingAI .scen."""
    for ln in scen_path.read_text(encoding="utf-8", errors="replace").splitlines():
        ln = ln.strip()
        if not ln or ln.lower().startswith("version"):
            continue
        parts = ln.split("\t")
        if len(parts) < 8:
            # Some files may be space-separated; ignore those for now.
            continue
        # parts: bucket, map, w, h, xs, ys, xg, yg, (opt) dist
        map_file = parts[1]
        w = int(parts[2])
        h = int(parts[3])
        xs, ys, xg, yg = map(int, parts[4:8])
        yield MovingAIScenRow(
            map_file=map_file,
            width=w,
            height=h,
            start_xy=(xs, ys),
            goal_xy=(xg, yg),
        )


def pick_n_agents(scen_path: Path, n: int) -> Tuple[Path, List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Pick the first N scen rows and return (map_file_in_scen, starts, goals)."""
    rows = []
    for row in iter_scen_rows(scen_path):
        rows.append(row)
        if len(rows) >= n:
            break
    if len(rows) < n:
        raise ValueError(f"Not enough scen rows in {scen_path} for N={n}")

    map_file = rows[0].map_file
    starts = [r.start_xy for r in rows]
    goals = [r.goal_xy for r in rows]
    return Path(map_file), starts, goals
