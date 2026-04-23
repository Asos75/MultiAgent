from __future__ import annotations
import time
from typing import Optional

from .mapf_instance import MAPFInstance
from .types import (
    LocalGroup,
    LocalLeadersConfig,
    Path,
    Position,
    Plan,
    SolveResult,
)


class LocalLeadersMAPF:

    def __init__(
        self,
        instance: MAPFInstance,
        config: Optional[LocalLeadersConfig] = None,
    ) -> None:
        self.instance = instance
        self.config = config or LocalLeadersConfig()
        self._groups: list[LocalGroup] = []


    # Run the algorithm and return a SolveResult.
    def solve(self) -> SolveResult:

        t0 = time.perf_counter()

        groups = self._form_groups()
        self._groups = groups

        partial_plans: dict[int, Path] = {}
        for group in groups:
            plan = self._compute_local_plan(group)
            partial_plans.update(plan)

        solution = self._resolve_inter_group_conflicts(groups, partial_plans)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        if solution is None:
            return SolveResult(solved=False, comp_time_ms=elapsed_ms)

        conflicts = self._detect_conflicts(solution)
        return SolveResult(
            solved=len(conflicts) == 0,
            soc=self._compute_soc(solution),
            makespan=self._compute_makespan(solution),
            comp_time_ms=elapsed_ms,
            num_groups=len(groups),
            avg_group_size=(
                sum(len(g.member_ids) for g in groups) / len(groups) if groups else 0.0
            ),
            num_conflicts_resolved=0,  # TODO: count inside _resolve_inter_group_conflicts
            solution=solution,
        )

    # TODO: form groups
    def _form_groups(self) -> list[LocalGroup]:
        raise NotImplementedError("Group formation not yet implemented")

    # TODO: elect a leader in the group
    def _elect_leader(self, group: LocalGroup) -> int:
        raise NotImplementedError("Leader election not yet implemented")

    # TODO: leader needs to update plans of agents
    def _compute_local_plan(self, group: LocalGroup) -> dict[int, Path]:
        raise NotImplementedError("Local planning not yet implemented")

    # TODO: Globally resolve agents conflicts after updating plans
    def _resolve_inter_group_conflicts(
        self,
        groups: list[LocalGroup],
        partial_plans: dict[int, Path],
    ) -> Optional[Plan]:
        raise NotImplementedError("Inter-group conflict resolution not yet implemented")


    # Get local view from position
    def _compute_local_view(self, center: Position, radius: int) -> set[Position]:
        x0, y0 = center
        return {
            (x, y)
            for x in range(x0 - radius, x0 + radius + 1)
            for y in range(y0 - radius, y0 + radius + 1)
            if self.instance.is_free((x, y))
        }

    # Find conflicts in agent's plans
    # 2 possible issues: agents want to occupy same vertex at same time or they want to pass through each other
    def _detect_conflicts(self, solution: Plan) -> list[tuple]:

        conflicts: list[tuple] = []
        agent_ids = list(solution.keys())
        if not agent_ids:
            return conflicts

        makespan = max(len(p) for p in solution.values())

        def pos_at(aid: int, t: int) -> Position:
            path = solution[aid]
            # Agent stays at goal after its path ends
            return path[min(t, len(path) - 1)]

        for t in range(makespan):
            for i, ai in enumerate(agent_ids):
                for aj in agent_ids[i + 1:]:
                    pi_t = pos_at(ai, t)
                    pj_t = pos_at(aj, t)
                    pi_t1 = pos_at(ai, t + 1)
                    pj_t1 = pos_at(aj, t + 1)

                    if pi_t == pj_t:
                        conflicts.append((ai, aj, t, "vertex"))

                    if pi_t == pj_t1 and pj_t == pi_t1:
                        conflicts.append((ai, aj, t, "swap"))

        return conflicts

    # SOC
    def _compute_soc(self, solution: Plan) -> int:
        return sum(len(path) - 1 for path in solution.values())

    # Makespan
    def _compute_makespan(self, solution: Plan) -> int:
        return max(len(path) - 1 for path in solution.values())

    # Get manhattan distance between 2 points
    @staticmethod
    def _manhattan(a: Position, b: Position) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # Get Chebyshev distance between 2 points
    @staticmethod
    def _chebyshev(a: Position, b: Position) -> int:
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))
