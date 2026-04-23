
from __future__ import annotations
import sys
from pathlib import Path as FilePath

from .types import Agent, Grid, Position


# holds the grid and all agent start/goal positions.
class MAPFInstance:

    # Default constructor
    def __init__(self, grid: Grid, agents: list[Agent]) -> None:
        self.grid = grid
        self.agents = agents


    # Alternative constructor: build problem from pre-defined arrays
    @staticmethod
    def from_arrays(
        grid: Grid,
        starts: list[Position],
        goals: list[Position],
    ) -> MAPFInstance:
        if len(starts) != len(goals):
            raise ValueError("starts and goals must have the same length")
        agents = [Agent(id=i, start=s, goal=g) for i, (s, g) in enumerate(zip(starts, goals))]
        return MAPFInstance(grid=grid, agents=agents)

    # Alternative constructor: build problem from files
    @staticmethod
    def from_movingai(map_path: str, scen_path: str, num_agents: int) -> MAPFInstance:

        bench_dir = str(FilePath(__file__).parent.parent.parent / "pogema_bench")
        if bench_dir not in sys.path:
            sys.path.insert(0, bench_dir)

        from movingai_io import read_movingai_map, iter_scen_rows, pick_n_agents  # type: ignore

        grid: Grid = read_movingai_map(map_path)
        scen_rows = list(iter_scen_rows(scen_path))
        picked = pick_n_agents(scen_rows, num_agents)

        agents = []
        for i, row in enumerate(picked):
            # MovingAI (x, y) -> internal (row, col) = (y, x)
            agents.append(Agent(
                id=i,
                start=(row.start_y, row.start_x),
                goal=(row.goal_y, row.goal_x),
            ))

        return MAPFInstance(grid=grid, agents=agents)


    @property
    def height(self) -> int:
        return len(self.grid)

    @property
    def width(self) -> int:
        return len(self.grid[0]) if self.grid else 0

    @property
    def num_agents(self) -> int:
        return len(self.agents)

    
    # Helpers
    def in_bounds(self, pos: Position) -> bool:
        x, y = pos
        return 0 <= x < self.height and 0 <= y < self.width

    def is_obstacle(self, pos: Position) -> bool:
        if not self.in_bounds(pos):
            return True
        x, y = pos
        return self.grid[x][y] == 1

    def is_free(self, pos: Position) -> bool:
        return not self.is_obstacle(pos)

    # up, down, left, right
    def get_neighbours(self, pos: Position) -> list[Position]:
        x, y = pos
        candidates = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        return [p for p in candidates if self.is_free(p)]

    def get_non_obstacle_cells(self) -> list[Position]:
        return [
            (x, y)
            for x in range(self.height)
            for y in range(self.width)
            if self.grid[x][y] == 0
        ]


    # Checks if the problem is solvable 
    def validate(self) -> list[str]:
        errors: list[str] = []
        seen_starts: set[Position] = set()
        seen_goals: set[Position] = set()

        for agent in self.agents:
            if self.is_obstacle(agent.start):
                errors.append(f"Agent {agent.id}: start {agent.start} is an obstacle or out of bounds")
            if self.is_obstacle(agent.goal):
                errors.append(f"Agent {agent.id}: goal {agent.goal} is an obstacle or out of bounds")
            if agent.start in seen_starts:
                errors.append(f"Agent {agent.id}: start {agent.start} already used by another agent")
            if agent.goal in seen_goals:
                errors.append(f"Agent {agent.id}: goal {agent.goal} conflicts with another agent's goal")
            seen_starts.add(agent.start)
            seen_goals.add(agent.goal)

        return errors

