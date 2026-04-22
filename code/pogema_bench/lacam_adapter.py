from __future__ import annotations

import json
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

from .utils_movingai import grid_to_movingai_map_text, scen_text, ensure_dir


# NOTE: POGEMA's Grid uses obstacles[x, y] where x=row and y=col.
# When we build a MovingAI-backed environment we swap MovingAI (col,row)
# into POGEMA (row,col). Therefore the action deltas must follow:
# - up:    (row-1, col)
# - down:  (row+1, col)
# - left:  (row, col-1)
# - right: (row, col+1)
MOVE_DELTAS = {
    (0, 0): 0,  # wait
    (-1, 0): 1,  # up
    (1, 0): 2,  # down
    (0, -1): 3,  # left
    (0, 1): 4,  # right
}


@dataclass
class LaCAMRunResult:
    solved: bool
    soc: int
    makespan: int
    comp_time_ms: float
    solution_xy: List[List[Tuple[int, int]]]


def _parse_lacam_result_txt(text: str) -> LaCAMRunResult:
    solved = False
    soc = 0
    makespan = 0
    comp_time_ms = 0.0

    solution_lines: List[str] = []
    in_solution = False

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("solved="):
            solved = line.split("=", 1)[1].strip() == "1"
        elif line.startswith("soc="):
            soc = int(float(line.split("=", 1)[1].strip()))
        elif line.startswith("makespan="):
            makespan = int(float(line.split("=", 1)[1].strip()))
        elif line.startswith("comp_time="):
            comp_time_ms = float(line.split("=", 1)[1].strip())
        elif line.startswith("solution="):
            in_solution = True
        elif in_solution:
            solution_lines.append(line)

    # Each solution line: t:(x,y),(x,y),...
    step_re = re.compile(r"^(\d+):(.*)$")
    xy_re = re.compile(r"\((\d+),(\d+)\)")
    steps: List[List[Tuple[int, int]]] = []
    for ln in solution_lines:
        m = step_re.match(ln)
        if not m:
            continue
        coords_part = m.group(2)
        coords = [(int(x), int(y)) for x, y in xy_re.findall(coords_part)]
        if coords:
            steps.append(coords)

    return LaCAMRunResult(
        solved=solved and bool(steps),
        soc=soc,
        makespan=makespan,
        comp_time_ms=comp_time_ms,
        solution_xy=steps,
    )


def plan_with_lacam(
    *,
    lacam_binary: Path,
    obstacles: List[List[int]],
    starts_xy: List[Tuple[int, int]],
    goals_xy: List[Tuple[int, int]],
    seed: int,
    time_limit_sec: int = 10,
) -> LaCAMRunResult:
    if len(starts_xy) != len(goals_xy):
        raise ValueError("starts/goals size mismatch")

    n_agents = len(starts_xy)

    with tempfile.TemporaryDirectory(prefix="pogema_lacam_") as td:
        td_path = Path(td)
        ensure_dir(td_path)

        map_name = "instance.map"
        scen_name = "instance.scen"
        out_name = "result.txt"

        # POGEMA's `get_obstacles()` typically returns a padded grid (size + 2*obs_radius).
        # `get_agents_xy()` / `get_targets_xy()` are in the *same global padded coordinates*.
        # LaCAM expects coordinates within [0,width) x [0,height) of the written map.
        height = len(obstacles)
        width = len(obstacles[0]) if height else 0

        def _in_bounds_xy(x: int, y: int) -> bool:
            return 0 <= x < width and 0 <= y < height

        for (sx, sy), (gx, gy) in zip(starts_xy, goals_xy):
            if not _in_bounds_xy(sx, sy) or not _in_bounds_xy(gx, gy):
                raise ValueError(
                    f"POGEMA start/goal out of bounds for obstacles grid: "
                    f"start={(sx, sy)} goal={(gx, gy)} grid={(width, height)}"
                )

        (td_path / map_name).write_text(grid_to_movingai_map_text(obstacles), encoding="utf-8")
        (td_path / scen_name).write_text(
            scen_text(
                map_name,
                [(sx, sy, gx, gy) for (sx, sy), (gx, gy) in zip(starts_xy, goals_xy)],
                width=width,
                height=height,
            ),
            encoding="utf-8",
        )

        cmd = [
            str(lacam_binary),
            "-m",
            map_name,
            "-i",
            scen_name,
            "-N",
            str(n_agents),
            "-s",
            str(seed),
            "-t",
            str(time_limit_sec),
            "-o",
            out_name,
            "-v",
            "0",
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(td_path))
        if not (td_path / out_name).exists():
            # LaCAM didn't write an output file. Report failure but keep diagnostics accessible.
            return LaCAMRunResult(
                solved=False,
                soc=0,
                makespan=0,
                comp_time_ms=0.0,
                solution_xy=[],
            )

        result_text = (td_path / out_name).read_text(encoding="utf-8")
        parsed = _parse_lacam_result_txt(result_text)
        if proc.returncode != 0:
            parsed.solved = False
        return parsed


def solution_to_actions(solution_xy: List[List[Tuple[int, int]]]) -> List[List[int]]:
    """Convert LaCAM solution (per timestep config) to actions per timestep.

    Returns list of [actions_for_all_agents] for each transition t->t+1.
    """
    if not solution_xy or len(solution_xy) < 2:
        return []

    actions: List[List[int]] = []
    for t in range(len(solution_xy) - 1):
        cur = solution_xy[t]
        nxt = solution_xy[t + 1]
        step_actions: List[int] = []
        for (x0, y0), (x1, y1) in zip(cur, nxt):
            dx, dy = x1 - x0, y1 - y0
            if (dx, dy) not in MOVE_DELTAS:
                raise ValueError(f"Unsupported move delta {(dx, dy)} at t={t}")
            step_actions.append(MOVE_DELTAS[(dx, dy)])
        actions.append(step_actions)
    return actions
