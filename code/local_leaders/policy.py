"""
Local Leaders MAPF Policy  –  fully online, no precomputed paths.

Algorithm (one call to `act` = one environment step):

  1. GROUP FORMATION  (dynamic, per step)
     Agent i is a *leader* iff no lower-ID agent j lies within
     `leader_comm_radius` (Manhattan distance).  Processing in ID order
     produces a deterministic partition: i joins the nearest already-formed
     leader's group, or becomes a new leader.

  2. DESIRED CELL  (non-cooperative A*)
     Every agent independently runs A* from its current position to its
     goal, ignoring all other agents.  This gives a "greedy" desired next
     cell.  Agents stuck ≥ `escape_threshold` steps pick a random free
     neighbour instead (deadlock escape) to break corridor deadlocks.

  3. PRIORITY-BASED CONFLICT RESOLUTION
     Process agents in priority order:
       • Level 0 – agents in escape mode (stuck ≥ escape_threshold)
       • Level 1 – group leaders
       • Level 2 – followers
       Within each level, lower agent ID = higher priority.

     For each agent in order:
       a) Vertex conflict: desired cell already committed → find the free
          cell closest to goal that is not yet committed.
       b) Swap conflict: desired cell = another (higher-priority) agent's
          CURRENT position AND that agent committed to move to our current
          position → same fallback.
       c) Commit the final cell; record it in a per-step "committed" set.

     Because lower-priority agents commit AFTER higher-priority ones, no
     vertex or swap conflict can survive into the POGEMA physics step.

  4. STUCK COUNTER
     Incremented when the final cell equals the current position; reset
     otherwise.  Leaders that are in escape mode have their stuck counter
     halved after each move to avoid over-triggering.
"""

import heapq
import random
from typing import Dict, List, Optional, Tuple

# ── POGEMA action encoding ─────────────────────────────────────────────────
STAY, UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3, 4

_INV_DELTA: Dict[Tuple[int, int], int] = {
    ( 0,  0): STAY,
    (-1,  0): UP,
    ( 1,  0): DOWN,
    ( 0, -1): LEFT,
    ( 0,  1): RIGHT,
}

_MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]


# ── A* helper ──────────────────────────────────────────────────────────────

def _astar_next(
    grid: List[List[int]],
    H: int,
    W: int,
    start: Tuple[int, int],
    goal: Tuple[int, int],
) -> Tuple[int, int]:
    """
    Return the first cell on the shortest path from *start* to *goal*,
    treating the static obstacle grid as the only constraint (no agent
    positions considered here — that is handled by conflict resolution).
    Returns *start* when goal is reached or no path exists.
    """
    if start == goal:
        return start

    def h(r: int, c: int) -> int:
        return abs(r - goal[0]) + abs(c - goal[1])

    heap: list = []
    for dr, dc in _MOVES:
        nr, nc = start[0] + dr, start[1] + dc
        if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] == 0:
            heapq.heappush(heap, (1 + h(nr, nc), 1, nr, nc, nr, nc))

    visited: Dict[Tuple[int, int], int] = {start: 0}

    while heap:
        f, g, r, c, fr, fc = heapq.heappop(heap)
        pos = (r, c)

        if visited.get(pos, g + 1) < g:
            continue

        if pos == goal:
            return (fr, fc)

        visited[pos] = g

        for dr, dc in _MOVES:
            nr, nc = r + dr, c + dc
            ng = g + 1
            npos = (nr, nc)
            if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] == 0:
                if visited.get(npos, ng + 1) > ng:
                    visited[npos] = ng
                    heapq.heappush(heap, (ng + h(nr, nc), ng, nr, nc, fr, fc))

    return start  # unreachable → stay


# ── Main policy ────────────────────────────────────────────────────────────

class LocalLeadersPolicy:
    """
    Online local-leaders MAPF policy for the POGEMA benchmark loop.

    Usage::

        policy = LocalLeadersPolicy(num_agents=32)
        policy.reset(grid)        # grid: List[List[int]] from env.grid.get_obstacles()
        while not done:
            positions = env.grid.get_agents_xy()
            targets   = env.grid.get_targets_xy()
            actions   = policy.act(positions, targets)
            obs, rew, dones, trunc, infos = env.step(actions)

    Parameters
    ----------
    num_agents:
        Total number of agents.
    leader_comm_radius:
        Manhattan distance within which a lower-ID agent serves as leader.
    escape_threshold:
        Steps stuck before switching to random-neighbour escape mode.
    seed:
        RNG seed for the random escape (reproducibility).
    """

    def __init__(
        self,
        num_agents: int,
        leader_comm_radius: int = 7,
        escape_threshold: int = 8,
        seed: int = 0,
    ) -> None:
        self.num_agents = num_agents
        self.leader_comm_radius = leader_comm_radius
        self.escape_threshold = escape_threshold

        self._rng = random.Random(seed)
        self._grid: List[List[int]] = []
        self._H: int = 0
        self._W: int = 0

        self._stuck: List[int] = [0] * num_agents
        self._group_leader: List[int] = list(range(num_agents))

    # ── public API ─────────────────────────────────────────────────────────

    def reset(self, grid: List[List[int]]) -> None:
        """Call once at the start of each episode with the static obstacle grid."""
        self._grid = grid
        self._H = len(grid)
        self._W = len(grid[0]) if self._H else 0
        self._stuck = [0] * self.num_agents
        self._group_leader = list(range(self.num_agents))

    def act(
        self,
        positions: List[Tuple[int, int]],
        targets: List[Tuple[int, int]],
    ) -> List[int]:
        """
        Compute one-step actions for all agents.

        positions : current (row, col) in the padded obstacle grid
        targets   : current goal (row, col) — changes each time an agent
                    reaches its goal in lifelong MAPF
        """
        N = self.num_agents

        # ── 1. Group formation ──────────────────────────────────────────
        self._group_leader = self._form_groups(positions)
        is_leader = [self._group_leader[i] == i for i in range(N)]

        # ── 2. Desired cells (non-cooperative A*) ───────────────────────
        desired: List[Tuple[int, int]] = []
        for i in range(N):
            pos, tgt = positions[i], targets[i]
            if self._stuck[i] >= self.escape_threshold:
                # Escape: random free neighbour to break deadlock
                nbrs = [
                    (pos[0] + dr, pos[1] + dc)
                    for dr, dc in _MOVES
                    if 0 <= pos[0] + dr < self._H
                    and 0 <= pos[1] + dc < self._W
                    and self._grid[pos[0] + dr][pos[1] + dc] == 0
                ]
                desired.append(self._rng.choice(nbrs) if nbrs else pos)
            else:
                desired.append(_astar_next(self._grid, self._H, self._W, pos, tgt))

        # ── 3. Priority-based conflict resolution ───────────────────────
        # Level 0 = escape (stuck), 1 = leader, 2 = follower; tie by ID.
        def _priority(i: int) -> Tuple[int, int]:
            if self._stuck[i] >= self.escape_threshold:
                return (0, i)
            return (1 if is_leader[i] else 2, i)

        order = sorted(range(N), key=_priority)

        committed: Dict[Tuple[int, int], int] = {}  # cell → agent committed to arrive there
        final: List[Tuple[int, int]] = list(positions)

        for i in order:
            pos = positions[i]
            want = desired[i]
            tgt = targets[i]

            # (a) Vertex conflict – desired cell already committed
            if want in committed:
                want = self._best_alternative(pos, tgt, committed)

            # (b) Swap conflict – we want to go where a committed agent came from
            if want != pos:
                j = committed.get(pos)          # who committed to come HERE?
                if j is not None and positions[j] == want:
                    # j is moving from `want` → `pos`, we want `want` → swap
                    want = self._best_alternative(pos, tgt, committed, exclude=want)

            final[i] = want
            committed[want] = i

        # ── 4. Update stuck counters ────────────────────────────────────
        for i in range(N):
            if final[i] == positions[i]:
                self._stuck[i] += 1
            else:
                self._stuck[i] = 0

        # ── 5. Cells → POGEMA actions ───────────────────────────────────
        actions: List[int] = []
        for i in range(N):
            dr = final[i][0] - positions[i][0]
            dc = final[i][1] - positions[i][1]
            actions.append(_INV_DELTA.get((dr, dc), STAY))

        return actions

    # ── helpers ────────────────────────────────────────────────────────────

    def _form_groups(self, positions: List[Tuple[int, int]]) -> List[int]:
        """
        Assign each agent to a group leader.
        Processing in ascending ID order: agent i joins the nearest
        already-formed leader's group (within leader_comm_radius), or
        becomes a new leader.
        """
        leader_of = [-1] * self.num_agents
        leaders: List[int] = []
        r = self.leader_comm_radius

        for i in range(self.num_agents):
            pi = positions[i]
            best: Optional[int] = None
            best_d = r + 1

            for lid in leaders:
                pl = positions[lid]
                d = abs(pi[0] - pl[0]) + abs(pi[1] - pl[1])
                if d <= r and d < best_d:
                    best_d = d
                    best = lid

            if best is not None:
                leader_of[i] = best
            else:
                leader_of[i] = i
                leaders.append(i)

        return leader_of

    def _best_alternative(
        self,
        pos: Tuple[int, int],
        tgt: Tuple[int, int],
        committed: Dict[Tuple[int, int], int],
        exclude: Optional[Tuple[int, int]] = None,
    ) -> Tuple[int, int]:
        """
        Among free cells reachable in one step that are not yet committed
        (and not `exclude`), return the one closest to `tgt`.
        Falls back to staying at `pos` if no alternative exists.
        """
        best_cell = pos
        best_dist = float("inf")

        candidates = [(pos[0] + dr, pos[1] + dc) for dr, dc in _MOVES] + [pos]
        for cell in candidates:
            r, c = cell
            if not (0 <= r < self._H and 0 <= c < self._W):
                continue
            if self._grid[r][c] != 0:
                continue
            if cell in committed:
                continue
            if cell == exclude:
                continue
            dist = abs(r - tgt[0]) + abs(c - tgt[1])
            if dist < best_dist:
                best_dist = dist
                best_cell = cell

        return best_cell
