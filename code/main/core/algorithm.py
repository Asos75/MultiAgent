from __future__ import annotations
import time
from typing import Optional
from collections import deque
import random

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

        solution, num_fixed = self._resolve_inter_group_conflicts(groups, partial_plans)

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
            num_conflicts_resolved=num_fixed,
            solution=solution,
        )

    # TODO: form groups
    def _form_groups(self) -> list[LocalGroup]:
        agents = self.instance.agents
        if not agents:
            return []

        remaining: set[int] = {a.id for a in agents}
        pos_by_id: dict[int, Position] = {a.id: a.start for a in agents}

        groups: list[LocalGroup] = []
        r = int(self.config.group_radius)
        max_size = int(self.config.max_group_size)

        while remaining:
            seed_id = next(iter(remaining))
            seed_pos = pos_by_id[seed_id]

            # Collect candidates in radius and clamp by max_group_size.
            members = [
                aid
                for aid in remaining
                if self._chebyshev(pos_by_id[aid], seed_pos) <= r
            ]

            # Prefer closer agents if we need to trim.
            if len(members) > max_size:
                members.sort(key=lambda aid: self._chebyshev(pos_by_id[aid], seed_pos))
                members = members[:max_size]

            leader_id = members[0]
            group = LocalGroup(leader_id=leader_id, member_ids=list(members))
            group.leader_id = self._elect_leader(group)

            # Local view is around leader and all members.
            view: set[Position] = set()
            for aid in group.member_ids:
                view |= self._compute_local_view(pos_by_id[aid], self.config.leader_view_radius)
            view |= self._compute_local_view(pos_by_id[group.leader_id], self.config.leader_view_radius)
            group.local_view = view

            groups.append(group)
            remaining -= set(group.member_ids)

        return groups

    # TODO: elect a leader in the group
    def _elect_leader(self, group: LocalGroup) -> int:
        # Static: pick the member with minimum avg Manhattan distance to others.
        # Dynamic: add a simple local-density term (how many group members are within radius 2).
        member_ids = group.member_ids
        if not member_ids:
            raise ValueError("LocalGroup has no members")

        pos_by_id: dict[int, Position] = {a.id: a.start for a in self.instance.agents}

        def avg_dist(aid: int) -> float:
            pa = pos_by_id[aid]
            if len(member_ids) == 1:
                return 0.0
            return sum(self._manhattan(pa, pos_by_id[bid]) for bid in member_ids if bid != aid) / (
                len(member_ids) - 1
            )

        def density(aid: int) -> int:
            pa = pos_by_id[aid]
            return sum(1 for bid in member_ids if bid != aid and self._chebyshev(pa, pos_by_id[bid]) <= 2)

        mode = (self.config.leader_election or "static").lower()

        if mode == "dynamic":
            # Prefer high density (more coordination needed) and then centrality.
            # We maximize density, minimize avg_dist.
            best = max(member_ids, key=lambda aid: (density(aid), -avg_dist(aid), -aid))
            return best

        best = min(member_ids, key=lambda aid: (avg_dist(aid), aid))
        return best

    # TODO: leader needs to update plans of agents
    def _compute_local_plan(self, group: LocalGroup) -> dict[int, Path]:
        # Simple local planner:
        # - plan each agent independently with BFS (ignoring other agents)
        # - then do an in-group deconfliction by inserting waits when vertex conflicts appear
        pos_by_id: dict[int, Position] = {a.id: a.start for a in self.instance.agents}
        goal_by_id: dict[int, Position] = {a.id: a.goal for a in self.instance.agents}

        # Optional dynamic leader re-selection (within group only). Here we do a single reselection
        # based on current starts; doing it each timestep would require simulation.
        if (self.config.leader_election or "static").lower() == "dynamic" and self.config.dynamic_reselect_every:
            group.leader_id = self._elect_leader(group)

        view = group.local_view or self._compute_local_view(pos_by_id[group.leader_id], self.config.leader_view_radius)

        def bfs(start: Position, goal: Position) -> Optional[Path]:
            if start == goal:
                return [start]
            q = deque([start])
            prev: dict[Position, Position] = {}
            seen = {start}
            while q:
                cur = q.popleft()
                for nb in self.instance.get_neighbours(cur):
                    if nb not in view:
                        continue
                    if nb in seen:
                        continue
                    seen.add(nb)
                    prev[nb] = cur
                    if nb == goal:
                        # reconstruct
                        path = [goal]
                        while path[-1] != start:
                            path.append(prev[path[-1]])
                        path.reverse()
                        return path
                    q.append(nb)
            return None

        plans: dict[int, Path] = {}
        for aid in group.member_ids:
            p = bfs(pos_by_id[aid], goal_by_id[aid])
            if p is None:
                # fallback: stay in place
                plans[aid] = [pos_by_id[aid]]
            else:
                plans[aid] = p

        # In-group conflict smoothing: iteratively resolve vertex conflicts by adding waits
        # to the non-leader agent involved.
        def pos_at(path: Path, t: int) -> Position:
            return path[min(t, len(path) - 1)]

        leader = group.leader_id
        max_iters = 200
        for _ in range(max_iters):
            agent_ids = list(plans.keys())
            makespan = max(len(p) for p in plans.values())
            conflict_found = False
            for t in range(makespan):
                occ: dict[Position, int] = {}
                for aid in agent_ids:
                    p = pos_at(plans[aid], t)
                    if p in occ:
                        other = occ[p]
                        # decide who waits
                        waiter = aid
                        if aid == leader and other != leader:
                            waiter = other
                        elif other == leader and aid != leader:
                            waiter = aid
                        else:
                            waiter = max(aid, other)

                        wpos = pos_at(plans[waiter], max(0, t - 1))
                        plans[waiter].insert(t, wpos)
                        conflict_found = True
                        break
                    occ[p] = aid
                if conflict_found:
                    break
            if not conflict_found:
                break

        return plans

    # TODO: Globally resolve agents conflicts after updating plans
    def _resolve_inter_group_conflicts(
        self,
        groups: list[LocalGroup],
        partial_plans: dict[int, Path],
    ) -> tuple[Optional[Plan], int]:
        # Leader-mediated resolution: when conflicts appear between different groups,
        # make the non-leader side wait, or if both are leaders, pick one deterministically.
        if not groups:
            return dict(partial_plans), 0

        plan: Plan = dict(partial_plans)
        group_by_agent: dict[int, int] = {}
        leaders: set[int] = set()
        for gi, g in enumerate(groups):
            leaders.add(g.leader_id)
            for aid in g.member_ids:
                group_by_agent[aid] = gi

        def pos_at(aid: int, t: int) -> Position:
            path = plan[aid]
            return path[min(t, len(path) - 1)]

        fixed = 0
        start_t = time.perf_counter()
        max_rounds = 500

        rnd = random.Random(self.config.seed)

        for _ in range(max_rounds):
            if (time.perf_counter() - start_t) > float(self.config.time_limit_sec):
                return None, fixed

            conflicts = self._detect_conflicts(plan)
            if not conflicts:
                return plan, fixed

            # pick one conflict to fix
            ai, aj, t, ctype = rnd.choice(conflicts)
            if ai not in plan or aj not in plan:
                continue
            gi = group_by_agent.get(ai, -1)
            gj = group_by_agent.get(aj, -1)
            if gi == gj and gi != -1:
                # already handled locally, but still shows up; resolve by waiting non-leader
                pass

            ai_is_leader = ai in leaders
            aj_is_leader = aj in leaders

            # Decide who should wait.
            if ai_is_leader and not aj_is_leader:
                waiter = aj
            elif aj_is_leader and not ai_is_leader:
                waiter = ai
            else:
                # both leaders or both non-leaders: deterministic tie-break
                waiter = max(ai, aj)

            # Insert a wait at time t for the waiter (stay at previous position).
            wait_pos = pos_at(waiter, max(0, t))
            if t > 0:
                wait_pos = pos_at(waiter, t - 1)
            plan[waiter].insert(t, wait_pos)
            fixed += 1

        # if not resolved within rounds
        return plan if self._detect_conflicts(plan) == [] else None, fixed


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
