from __future__ import annotations

import random
from typing import Optional

from ..core.types import Grid, Position


# All functions return Grid = list[list[int]] where 0=free, 1=obstacle,

# Return grid with no obstacles
def empty_map(height: int, width: int) -> Grid:
    return [[0] * width for _ in range(height)]

# Return completely random map (uniform randomness)
def random_map(
    height: int,
    width: int,
    obstacle_density: float = 0.2,
    seed: int = 0,
) -> Grid:

    if not 0.0 <= obstacle_density < 1.0:
        raise ValueError(f"obstacle_density must be in [0, 1), got {obstacle_density}")

    rng = random.Random(seed)
    grid = [[0] * width for _ in range(height)]

    for x in range(height):
        for y in range(width):
            if rng.random() < obstacle_density:
                grid[x][y] = 1

    return grid

# Generate random grid divided into rectangular rooms connected by narrow doorways.
def room_map(
    height: int,
    width: int,
    room_size: int = 5,
    door_width: int = 1,
    seed: int = 0,
) -> Grid:

    rng = random.Random(seed)
    grid = [[0] * width for _ in range(height)]

    # Horizontal walls
    for wall_x in range(room_size, height, room_size + 1):
        for y in range(width):
            grid[wall_x][y] = 1
        # Cut one door per room-width span
        y = 0
        while y < width:
            span_end = min(y + room_size, width)
            door_start = rng.randint(y, max(y, span_end - door_width))
            for dy in range(door_width):
                if door_start + dy < width:
                    grid[wall_x][door_start + dy] = 0
            y = span_end + 1

    # Vertical walls
    for wall_y in range(room_size, width, room_size + 1):
        for x in range(height):
            grid[x][wall_y] = 1
        # Cut one door per room-height span
        x = 0
        while x < height:
            span_end = min(x + room_size, height)
            door_start = rng.randint(x, max(x, span_end - door_width))
            for dx in range(door_width):
                if door_start + dx < height:
                    grid[door_start + dx][wall_y] = 0
            x = span_end + 1

    return grid

# Generate random maze (no loops)
def maze_map(height: int, width: int, seed: int = 0) -> Grid:

    # Work on an odd-sized grid so every cell can be a passage cell
    h = height if height % 2 == 1 else height + 1
    w = width if width % 2 == 1 else width + 1

    # Start with all walls
    grid = [[1] * w for _ in range(h)]
    rng = random.Random(seed)

    # Carve maze with recursion
    def carve(x: int, y: int) -> None:
        grid[x][y] = 0
        directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]
        rng.shuffle(directions)
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < h and 0 <= ny < w and grid[nx][ny] == 1:
                # Knock down the wall between (x,y) and (nx,ny)
                grid[x + dx // 2][y + dy // 2] = 0
                carve(nx, ny)

    carve(1, 1)

    # Crop back to requested size
    return [row[:width] for row in grid[:height]]


# Assigns every agent a start and a goal position (random)
def place_agents(
    grid: Grid,
    num_agents: int,
    seed: int = 0,
    allow_overlapping_goals: bool = False, # TODO ???
) -> tuple[list[Position], list[Position]]:

    height = len(grid)
    width = len(grid[0]) if grid else 0
    free: list[Position] = [
        (x, y) for x in range(height) for y in range(width) if grid[x][y] == 0
    ]

    required = 2 * num_agents
    if len(free) < required:
        raise ValueError(
            f"Grid has only {len(free)} free cells but {required} are needed "
            f"for {num_agents} agents."
        )

    rng = random.Random(seed)
    chosen = rng.sample(free, required)
    starts = chosen[:num_agents]
    goals = chosen[num_agents:]
    return starts, goals

# Assigns every agent a start and a goal position (random)
# Agent starts are assigned in group clusters
def place_agents_clustered(
    grid: Grid,
    num_agents: int,
    num_clusters: int,
    cluster_radius: int = 3,
    seed: int = 0,
) -> tuple[list[Position], list[Position]]:

    height = len(grid)
    width = len(grid[0]) if grid else 0
    free: list[Position] = [
        (x, y) for x in range(height) for y in range(width) if grid[x][y] == 0
    ]

    if not free:
        raise ValueError("Grid has no free cells.")

    rng = random.Random(seed)
    centers = rng.sample(free, min(num_clusters, len(free)))

    starts: list[Position] = []
    goals: list[Position] = []
    used: set[Position] = set()

    agents_per_cluster = (num_agents + num_clusters - 1) // num_clusters

    for center in centers:
        cx, cy = center
        nearby = [
            (x, y)
            for x in range(cx - cluster_radius, cx + cluster_radius + 1)
            for y in range(cy - cluster_radius, cy + cluster_radius + 1)
            if (x, y) not in used and 0 <= x < height and 0 <= y < width and grid[x][y] == 0
        ]
        if not nearby:
            continue
        for pos in rng.sample(nearby, min(agents_per_cluster, len(nearby))):
            if len(starts) >= num_agents:
                break
            starts.append(pos)
            used.add(pos)
        if len(starts) >= num_agents:
            break

    remaining_free = [p for p in free if p not in used]
    goal_sample = rng.sample(remaining_free, min(num_agents, len(remaining_free)))
    goals = goal_sample[:num_agents]

    if len(starts) < num_agents or len(goals) < num_agents:
        raise ValueError("Not enough free cells for clustered placement.")

    return starts, goals


# draws current map state (goals, starts, obsticles)
def map_to_str(grid: Grid, agents: Optional[list] = None) -> str:
    char = [['#' if cell else '.' for cell in row] for row in grid]

    if agents:
        for a in agents:
            x, y = a.start
            if 0 <= x < len(char) and 0 <= y < len(char[0]):
                char[x][y] = 'S'
            x, y = a.goal
            if 0 <= x < len(char) and 0 <= y < len(char[0]):
                char[x][y] = 'G'

    return '\n'.join(''.join(row) for row in char)


def get_map_info(grid: Grid) -> dict:
    height = len(grid)
    width = len(grid[0]) if grid else 0
    total = height * width
    obstacles = sum(cell for row in grid for cell in row)
    return {
        "height": height,
        "width": width,
        "total_cells": total,
        "obstacle_cells": obstacles,
        "free_cells": total - obstacles,
        "obstacle_density": round(obstacles / total, 4) if total else 0.0,
    }
