#include "policy.hpp"

#include <algorithm>
#include <climits>
#include <functional>
#include <numeric>
#include <queue>
#include <tuple>

// ── A* ────────────────────────────────────────────────────────────────────────

Pos astar_next(const Grid& grid, int H, int W, Pos start, Pos goal) {
    if (start == goal) return start;

    auto h = [&](int r, int c) noexcept {
        return std::abs(r - goal.first) + std::abs(c - goal.second);
    };

    // heap entry: (f, g, r, c, first_r, first_c)
    using Entry = std::tuple<int, int, int, int, int, int>;
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> heap;

    for (int d = 0; d < 4; ++d) {
        int nr = start.first + DR[d], nc = start.second + DC[d];
        if (nr >= 0 && nr < H && nc >= 0 && nc < W && grid[nr][nc] == 0)
            heap.emplace(1 + h(nr, nc), 1, nr, nc, nr, nc);
    }

    std::unordered_map<int, int> visited;
    visited[cell_encode(start.first, start.second, W)] = 0;

    while (!heap.empty()) {
        auto [f, g, r, c, fr, fc] = heap.top();
        heap.pop();

        int enc = cell_encode(r, c, W);
        auto it = visited.find(enc);
        if (it != visited.end() && it->second < g) continue;

        if (r == goal.first && c == goal.second) return {fr, fc};

        visited[enc] = g;

        for (int d = 0; d < 4; ++d) {
            int nr = r + DR[d], nc = c + DC[d];
            if (nr < 0 || nr >= H || nc < 0 || nc >= W || grid[nr][nc] != 0) continue;
            int ng   = g + 1;
            int nenc = cell_encode(nr, nc, W);
            auto nit = visited.find(nenc);
            if (nit == visited.end() || nit->second > ng) {
                visited[nenc] = ng;
                heap.emplace(ng + h(nr, nc), ng, nr, nc, fr, fc);
            }
        }
    }
    return start; // unreachable → stay
}

// ── Policy ────────────────────────────────────────────────────────────────────

LocalLeadersPolicy::LocalLeadersPolicy(int num_agents, unsigned seed)
    : N_(num_agents), H_(0), W_(0), stuck_(num_agents, 0), rng_(seed) {}

void LocalLeadersPolicy::reset(const Grid& grid) {
    grid_ = grid;
    H_    = static_cast<int>(grid.size());
    W_    = H_ ? static_cast<int>(grid[0].size()) : 0;
    stuck_.assign(N_, 0);
}

// ── Group formation ───────────────────────────────────────────────────────────
// Processing agents in ascending ID order guarantees a deterministic partition.
// Agent i becomes a leader if no already-formed leader lies within
// LEADER_VIEW_RADIUS (Manhattan distance); otherwise it joins the nearest one.

IntVec LocalLeadersPolicy::form_groups(const PosVec& positions) const {
    IntVec leader_of(N_, -1);
    std::vector<int> leaders;
    leaders.reserve(N_);

    for (int i = 0; i < N_; ++i) {
        Pos pi      = positions[i];
        int best    = -1;
        int best_d  = LEADER_VIEW_RADIUS + 1;

        for (int lid : leaders) {
            int d = manh(pi, positions[lid]);
            if (d <= LEADER_VIEW_RADIUS && d < best_d) {
                best_d = d;
                best   = lid;
            }
        }

        leader_of[i] = (best == -1) ? i : best;
        if (leader_of[i] == i) leaders.push_back(i);
    }
    return leader_of;
}

// ── View-radius check ─────────────────────────────────────────────────────────
// Leaders use LEADER_VIEW_RADIUS; followers use AGENT_VIEW_RADIUS.
// An agent only performs conflict resolution with peers it can see.

bool LocalLeadersPolicy::in_view(bool i_is_leader, Pos pi, Pos pj) const noexcept {
    int radius = i_is_leader ? LEADER_VIEW_RADIUS : AGENT_VIEW_RADIUS;
    return manh(pi, pj) <= radius;
}

// ── Best alternative cell ──────────────────────────────────────────────────────
// Among cells reachable in one step (4 neighbours + stay) that are free in
// the static grid, not yet committed, and not equal to exclude_enc, return
// the one closest (Manhattan) to tgt. Falls back to pos (stay) if none exist.

Pos LocalLeadersPolicy::best_alternative(
    Pos pos, Pos tgt, const Committed& committed, int exclude_enc) const
{
    Pos best_cell = pos;
    int best_dist = INT_MAX;

    // Candidates: 4 neighbours + current cell (stay)
    const int cand_r[5] = {pos.first-1, pos.first+1, pos.first,   pos.first,   pos.first};
    const int cand_c[5] = {pos.second,  pos.second,  pos.second-1, pos.second+1, pos.second};

    for (int k = 0; k < 5; ++k) {
        int r = cand_r[k], c = cand_c[k];
        if (r < 0 || r >= H_ || c < 0 || c >= W_) continue;
        if (grid_[r][c] != 0) continue;
        int enc = cell_encode(r, c, W_);
        if (committed.count(enc)) continue;
        if (enc == exclude_enc) continue;
        int dist = std::abs(r - tgt.first) + std::abs(c - tgt.second);
        if (dist < best_dist) {
            best_dist = dist;
            best_cell = {r, c};
        }
    }
    return best_cell;
}

// ── Main act() ────────────────────────────────────────────────────────────────

IntVec LocalLeadersPolicy::act(const PosVec& positions, const PosVec& targets) {
    // 1. Group formation (uses LEADER_VIEW_RADIUS)
    IntVec leader_of = form_groups(positions);
    std::vector<bool> is_leader(N_);
    for (int i = 0; i < N_; ++i) is_leader[i] = (leader_of[i] == i);

    // 2. Desired cells: A* or random escape
    PosVec desired(N_);
    for (int i = 0; i < N_; ++i) {
        Pos pos = positions[i];
        if (stuck_[i] >= ESCAPE_THRESHOLD) {
            // Random free neighbour — breaks corridor deadlocks
            std::vector<Pos> nbrs;
            for (int d = 0; d < 4; ++d) {
                int nr = pos.first + DR[d], nc = pos.second + DC[d];
                if (nr >= 0 && nr < H_ && nc >= 0 && nc < W_ && grid_[nr][nc] == 0)
                    nbrs.push_back({nr, nc});
            }
            desired[i] = nbrs.empty()
                ? pos
                : nbrs[std::uniform_int_distribution<int>(0, (int)nbrs.size() - 1)(rng_)];
        } else {
            desired[i] = astar_next(grid_, H_, W_, pos, targets[i]);
        }
    }

    // 3. Priority order
    //    Level 0 (highest): escape mode (stuck >= ESCAPE_THRESHOLD)
    //    Level 1:           leaders
    //    Level 2:           followers
    //    Tie within level:  lower agent ID wins
    IntVec order(N_);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        int pa = (stuck_[a] >= ESCAPE_THRESHOLD) ? 0 : (is_leader[a] ? 1 : 2);
        int pb = (stuck_[b] >= ESCAPE_THRESHOLD) ? 0 : (is_leader[b] ? 1 : 2);
        return (pa != pb) ? (pa < pb) : (a < b);
    });

    // 4. Conflict resolution — committed maps encoded cell → agent index
    Committed committed;
    committed.reserve(N_ * 2);
    PosVec final_pos = positions;

    for (int i : order) {
        Pos pos  = positions[i];
        Pos want = desired[i];
        Pos tgt  = targets[i];

        int want_enc = cell_encode(want.first, want.second, W_);
        int pos_enc  = cell_encode(pos.first,  pos.second,  W_);

        // (a) Vertex conflict: target cell already committed
        if (committed.count(want_enc)) {
            // Only redirect if the conflicting agent is in view; otherwise stay
            int owner = committed.at(want_enc);
            if (in_view(is_leader[i], pos, positions[owner])) {
                want     = best_alternative(pos, tgt, committed);
                want_enc = cell_encode(want.first, want.second, W_);
            } else {
                // Out-of-view conflict: stay put, let POGEMA physics handle it
                want     = pos;
                want_enc = pos_enc;
            }
        }

        // (b) Swap conflict: we want a cell that a committed agent is leaving
        if (want != pos) {
            auto jt = committed.find(pos_enc);
            if (jt != committed.end()) {
                int j = jt->second;
                if (positions[j] == want && in_view(is_leader[i], pos, positions[j])) {
                    want     = best_alternative(pos, tgt, committed, want_enc);
                    want_enc = cell_encode(want.first, want.second, W_);
                }
            }
        }

        final_pos[i] = want;
        committed[want_enc] = i;
    }

    // 5. Update stuck counters
    for (int i = 0; i < N_; ++i)
        stuck_[i] = (final_pos[i] == positions[i]) ? stuck_[i] + 1 : 0;

    // 6. Convert final cells to POGEMA action integers
    IntVec actions(N_);
    for (int i = 0; i < N_; ++i) {
        int dr = final_pos[i].first  - positions[i].first;
        int dc = final_pos[i].second - positions[i].second;
        if      (dr == -1 && dc ==  0) actions[i] = ACTION_UP;
        else if (dr ==  1 && dc ==  0) actions[i] = ACTION_DOWN;
        else if (dr ==  0 && dc == -1) actions[i] = ACTION_LEFT;
        else if (dr ==  0 && dc ==  1) actions[i] = ACTION_RIGHT;
        else                           actions[i] = ACTION_STAY;
    }
    return actions;
}
