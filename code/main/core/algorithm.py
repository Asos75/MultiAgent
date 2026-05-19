from __future__ import annotations
import heapq
import time
from collections import deque
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

    def _form_groups(self) -> list[LocalGroup]:
        agents = self.instance.agents
        if not agents:
            return []

        remaining: set[int] = {a.id for a in agents}
        pos_by_id: dict[int, Position] = {a.id: a.start for a in agents}
        goal_by_id: dict[int, Position] = {a.id: a.goal for a in agents}

        groups: list[LocalGroup] = []
        r = int(self.config.group_radius)
        max_size = int(self.config.max_group_size)

        while remaining:
            seed_id = next(iter(remaining))
            seed_pos = pos_by_id[seed_id]

            members = [
                aid
                for aid in remaining
                if self._chebyshev(pos_by_id[aid], seed_pos) <= r
            ]

            if len(members) > max_size:
                members.sort(key=lambda aid: self._chebyshev(pos_by_id[aid], seed_pos))
                members = members[:max_size]

            leader_id = members[0]
            group = LocalGroup(leader_id=leader_id, member_ids=list(members))
            group.leader_id = self._elect_leader(group)

            view: set[Position] = set()
            for aid in group.member_ids:
                view |= self._compute_local_view(pos_by_id[aid], self.config.leader_view_radius)
                view |= self._compute_local_view(goal_by_id[aid], self.config.leader_view_radius)
            view |= self._compute_local_view(pos_by_id[group.leader_id], self.config.leader_view_radius)
            view |= self._compute_local_view(goal_by_id[group.leader_id], self.config.leader_view_radius)
            group.local_view = view

            groups.append(group)
            remaining -= set(group.member_ids)

        return groups

    def _elect_leader(self, group: LocalGroup) -> int:
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
            best = max(member_ids, key=lambda aid: (density(aid), -avg_dist(aid), -aid))
            return best

        best = min(member_ids, key=lambda aid: (avg_dist(aid), aid))
        return best

    def _compute_local_plan(self, group: LocalGroup) -> dict[int, Path]:
        """Plan agents within a group using prioritised space-time A*."""
        pos_by_id: dict[int, Position] = {a.id: a.start for a in self.instance.agents}
        goal_by_id: dict[int, Position] = {a.id: a.goal for a in self.instance.agents}

        if (self.config.leader_election or "static").lower() == "dynamic" and self.config.dynamic_reselect_every:
            group.leader_id = self._elect_leader(group)

        view = group.local_view or self._compute_local_view(
            pos_by_id[group.leader_id], self.config.leader_view_radius
        )

        # Leader first, then others sorted by distance to goal (shortest first)
        leader = group.leader_id
        others = sorted(
            [aid for aid in group.member_ids if aid != leader],
            key=lambda aid: self._manhattan(pos_by_id[aid], goal_by_id[aid]),
        )
        priority_order = [leader] + others

        max_dist = max(
            (self._manhattan(pos_by_id[aid], goal_by_id[aid]) for aid in group.member_ids),
            default=1,
        )
        t_max = max(max_dist * 4, 200)

        reserved_vertex: dict = {}
        reserved_edge: dict = {}
        plans: dict[int, Path] = {}

        for aid in priority_order:
            start = pos_by_id[aid]
            goal = goal_by_id[aid]

            # Precompute heuristic from goal (reverse BFS within view, then global fallback)
            h_local = self._reverse_bfs(goal, allowed=view)
            h_global = self._reverse_bfs(goal)

            # Try within local view first, fall back to full grid
            path = self._spacetime_astar(start, goal, reserved_vertex, reserved_edge, t_max, h_local, allowed=view)
            if path is None:
                path = self._spacetime_astar(start, goal, reserved_vertex, reserved_edge, t_max, h_global)
            if path is None:
                path = [start]

            plans[aid] = path
            self._reserve_path(path, aid, reserved_vertex, reserved_edge, t_max)

        return plans

    def _resolve_inter_group_conflicts(
        self,
        groups: list[LocalGroup],
        partial_plans: dict[int, Path],
    ) -> tuple[Optional[Plan], int]:
        """Resolve inter-group conflicts via global prioritised space-time A*.

        Agents with shorter optimal paths are planned first so they can claim
        their goal positions before longer-path agents transit through them.
        Uses A* with a precomputed heuristic for each agent — dramatically
        faster than BFS on large maps (warehouse-20-40 etc.).
        """
        if not groups:
            return dict(partial_plans), 0

        goal_by_id = {a.id: a.goal for a in self.instance.agents}
        start_by_id = {a.id: a.start for a in self.instance.agents}

        all_agent_ids = [a.id for a in self.instance.agents]

        # Precompute reverse-BFS heuristic maps and clean-map distances per unique goal.
        # Multiple agents may share a goal only in degenerate inputs; cache by goal cell.
        h_map_cache: dict[Position, dict[Position, int]] = {}
        dist_cache: dict[Position, int] = {}

        def get_h_map(goal: Position) -> dict[Position, int]:
            if goal not in h_map_cache:
                h_map_cache[goal] = self._reverse_bfs(goal)
            return h_map_cache[goal]

        def get_dist(start: Position, goal: Position) -> int:
            key = (start, goal)
            if key not in dist_cache:
                h = get_h_map(goal)
                dist_cache[key] = h.get(start, 10**9)
            return dist_cache[key]

        # Sort by clean-map distance: shorter-path agents plan first so they
        # claim their goals before longer-path agents transit through them.
        priority_order = sorted(
            all_agent_ids,
            key=lambda aid: (get_dist(start_by_id[aid], goal_by_id[aid]), aid),
        )

        max_partial = max((len(p) for p in partial_plans.values()), default=1)
        t_max = max(max_partial * 3, 300)

        reserved_vertex: dict = {}
        reserved_edge: dict = {}
        plan: Plan = {}
        fixed = 0

        for aid in priority_order:
            start = start_by_id[aid]
            goal = goal_by_id[aid]
            h_map = get_h_map(goal)

            new_path = self._spacetime_astar(start, goal, reserved_vertex, reserved_edge, t_max, h_map)

            old_path = partial_plans.get(aid, [start])
            if new_path is None:
                new_path = old_path
            elif new_path != old_path:
                fixed += 1

            plan[aid] = new_path
            self._reserve_path(new_path, aid, reserved_vertex, reserved_edge, t_max)

        conflicts = self._detect_conflicts(plan)
        return (plan, fixed) if not conflicts else (None, fixed)

    def _reverse_bfs(
        self,
        goal: Position,
        allowed: Optional[set[Position]] = None,
    ) -> dict[Position, int]:
        """BFS from goal backwards to compute h(pos) = distance to goal.

        Returns a dict mapping reachable positions to their distance.
        Positions not in the dict are unreachable (treat as infinity).
        """
        h: dict[Position, int] = {goal: 0}
        q: deque[Position] = deque([goal])
        while q:
            pos = q.popleft()
            for nb in self.instance.get_neighbours(pos):
                if nb in h:
                    continue
                if allowed is not None and nb not in allowed:
                    continue
                h[nb] = h[pos] + 1
                q.append(nb)
        return h

    def _spacetime_astar(
        self,
        start: Position,
        goal: Position,
        reserved_vertex: dict,
        reserved_edge: dict,
        t_max: int,
        h_map: dict[Position, int],
        allowed: Optional[set[Position]] = None,
    ) -> Optional[Path]:
        """Space-time A* avoiding reserved vertices and swap conflicts.

        h_map: precomputed reverse-BFS heuristic (distance from each cell to goal).
        allowed: optional local-view cell filter.
        Returns the shortest conflict-free path, or None if not found within t_max.
        """
        if allowed is not None and start not in allowed:
            return None
        if (start, 0) in reserved_vertex:
            return None
        if start == goal:
            return [start]
        if start not in h_map:
            return None  # goal unreachable from start (disconnected)

        # Priority queue entries: (f, g, pos, t)
        h0 = h_map.get(start, 10**9)
        pq: list[tuple] = [(h0, 0, start, 0)]
        g_best: dict[tuple[Position, int], int] = {(start, 0): 0}
        prev: dict[tuple, Optional[tuple]] = {(start, 0): None}

        while pq:
            f, g, cur_pos, t = heapq.heappop(pq)

            # Stale entry check
            if g_best.get((cur_pos, t), 10**9) < g:
                continue

            if cur_pos == goal:
                path: list[Position] = []
                state: Optional[tuple] = (cur_pos, t)
                while state is not None:
                    path.append(state[0])
                    state = prev[state]
                path.reverse()
                return path

            if t >= t_max:
                continue

            nt = t + 1
            for next_pos in self.instance.get_neighbours(cur_pos) + [cur_pos]:
                if allowed is not None and next_pos not in allowed:
                    continue
                if (next_pos, nt) in reserved_vertex:
                    continue
                if next_pos != cur_pos and (next_pos, cur_pos, t) in reserved_edge:
                    continue

                new_g = g + 1
                state_key = (next_pos, nt)
                if g_best.get(state_key, 10**9) <= new_g:
                    continue

                g_best[state_key] = new_g
                prev[state_key] = (cur_pos, t)
                h = h_map.get(next_pos, 10**9)
                heapq.heappush(pq, (new_g + h, new_g, next_pos, nt))

        return None

    def _reserve_path(
        self,
        path: Path,
        aid: int,
        reserved_vertex: dict,
        reserved_edge: dict,
        t_max: int,
    ) -> None:
        """Mark all space-time cells of path as reserved by aid."""
        for t, pos in enumerate(path):
            reserved_vertex[(pos, t)] = aid
            if t > 0:
                reserved_edge[(path[t - 1], pos, t - 1)] = aid
        # Agent stays at its goal after the path ends
        for t in range(len(path) - 1, t_max + 1):
            reserved_vertex[(path[-1], t)] = aid

    def _compute_local_view(self, center: Position, radius: int) -> set[Position]:
        x0, y0 = center
        return {
            (x, y)
            for x in range(x0 - radius, x0 + radius + 1)
            for y in range(y0 - radius, y0 + radius + 1)
            if self.instance.is_free((x, y))
        }

    def _detect_conflicts(self, solution: Plan) -> list[tuple]:
        conflicts: list[tuple] = []
        agent_ids = list(solution.keys())
        if not agent_ids:
            return conflicts

        makespan = max(len(p) for p in solution.values())

        def pos_at(aid: int, t: int) -> Position:
            path = solution[aid]
            return path[min(t, len(path) - 1)]

        for t in range(makespan):
            # Vertex conflicts: O(n) with hash map
            occ: dict[Position, int] = {}
            for ai in agent_ids:
                p = pos_at(ai, t)
                if p in occ:
                    conflicts.append((occ[p], ai, t, "vertex"))
                else:
                    occ[p] = ai

            # Swap conflicts: O(n) with edge hash map
            edges: dict[tuple[Position, Position], int] = {}
            for ai in agent_ids:
                edge = (pos_at(ai, t), pos_at(ai, t + 1))
                edges[edge] = ai
            for ai in agent_ids:
                reverse = (pos_at(ai, t + 1), pos_at(ai, t))
                if reverse in edges and edges[reverse] != ai:
                    aj = edges[reverse]
                    if ai < aj:
                        conflicts.append((ai, aj, t, "swap"))

        return conflicts

    def _compute_soc(self, solution: Plan) -> int:
        return sum(len(path) - 1 for path in solution.values())

    def _compute_makespan(self, solution: Plan) -> int:
        return max(len(path) - 1 for path in solution.values())

    def _bfs_dist(self, start: Position, goal: Position) -> int:
        """Shortest path distance on the clean grid (no other agents)."""
        if start == goal:
            return 0
        q: deque[tuple[Position, int]] = deque([(start, 0)])
        seen: set[Position] = {start}
        while q:
            pos, d = q.popleft()
            for nb in self.instance.get_neighbours(pos):
                if nb == goal:
                    return d + 1
                if nb not in seen:
                    seen.add(nb)
                    q.append((nb, d + 1))
        return 10**9  # unreachable

    @staticmethod
    def _manhattan(a: Position, b: Position) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    @staticmethod
    def _chebyshev(a: Position, b: Position) -> int:
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))
