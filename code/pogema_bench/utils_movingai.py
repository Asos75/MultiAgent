from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple, List


@dataclass(frozen=True)
class MovingAIMap:
    height: int
    width: int
    grid: List[List[int]]  # 1 obstacle, 0 free


def grid_to_movingai_map_text(grid: List[List[int]]) -> str:
    """Convert a 0/1 obstacle grid to a MovingAI .map content.

    MovingAI convention:
    - '.' traversable
    - '@' obstacle

    LaCAM's Graph loader supports MovingAI maps.
    """
    height = len(grid)
    width = len(grid[0]) if height else 0

    lines = [
        "type octile",
        f"height {height}",
        f"width {width}",
        "map",
    ]
    for r in range(height):
        row = "".join("@" if grid[r][c] else "." for c in range(width))
        lines.append(row)
    return "\n".join(lines) + "\n"


def scen_text(
    map_file_name: str,
    instances: Iterable[Tuple[int, int, int, int]],
    *,
    width: int,
    height: int,
) -> str:
    """Create a MovingAI .scen file content.

    LaCAM parses lines matching:
    \d+\t.+\.map\t\d+\t\d+\t(x_s)\t(y_s)\t(x_g)\t(y_g)\t.+

    We'll output version 1 header and tab-separated rows.
    """
    lines = ["version 1"]
    for idx, (xs, ys, xg, yg) in enumerate(instances):
        # LaCAM's regex expects: <id>\t<map>.map\t<map_w>\t<map_h>\t<xs>\t<ys>\t<xg>\t<yg>\t...
        # Use explicit tab-join to avoid accidental space conversions.
        lines.append(
            "\t".join(
                [
                    str(idx),
                    map_file_name,
                    str(width),
                    str(height),
                    str(xs),
                    str(ys),
                    str(xg),
                    str(yg),
                    "1",
                ]
            )
        )
    return "\n".join(lines) + "\n"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
