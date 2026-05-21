#include "policy_v2.hpp"

#include <algorithm>
#include <climits>
#include <numeric>
#include <queue>


LocalLeadersPolicy::LocalLeadersPolicy(int num_agents, unsigned seed, PolicyConfig config)
    : N_(num_agents), H_(0), W_(0),
      config_(std::move(config)),
      stuck_(num_agents, 0), rng_(seed) {}


void LocalLeadersPolicy::reset(const Grid& grid) {
    H_ = static_cast<int>(grid.size());
    W_ = H_ ? static_cast<int>(grid[0].size()) : 0;

    grid_flat_.resize(H_ * W_);
    for (int r = 0; r < H_; ++r)
        for (int c = 0; c < W_; ++c)
            grid_flat_[r * W_ + c] = grid[r][c];

    astar_dist_.assign(H_ * W_, INT_MAX);
    astar_dirty_.clear();
    astar_dirty_.reserve(H_ * W_ / 4);

    stuck_.assign(N_, 0);
    step_count_ = 0;
    leader_of_.resize(N_);
    is_leader_.resize(N_);
    desired_.resize(N_);
    provisional_.resize(N_);
    final_pos_.resize(N_);
    committed_.reserve(N_ * 2);
    leaders_.reserve(N_);
    group_members_.reserve(N_);
}


Pos LocalLeadersPolicy::astar_next(Pos start, Pos goal) const {
    if (start == goal) return start;

    for (int enc : astar_dirty_) astar_dist_[enc] = INT_MAX;
    astar_dirty_.clear();

    auto h = [&](int r, int c) noexcept {
        return std::abs(r - goal.first) + std::abs(c - goal.second);
    };

    std::priority_queue<AStarNode, std::vector<AStarNode>, std::greater<AStarNode>> heap;

    int start_enc = cell_encode(start.first, start.second, W_);
    astar_dist_[start_enc] = 0;
    astar_dirty_.push_back(start_enc);

    for (int d = 0; d < 4; ++d) {
        int nr = start.first + DR[d], nc = start.second + DC[d];
        if (is_passable(nr, nc)) {
            int nenc = cell_encode(nr, nc, W_);
            astar_dist_[nenc] = 1;
            astar_dirty_.push_back(nenc);
            heap.push({1 + h(nr, nc), 1, nr, nc, nr, nc});
        }
    }

    while (!heap.empty()) {
        auto [f, g, r, c, fr, fc] = heap.top();
        heap.pop();

        int enc = cell_encode(r, c, W_);
        if (astar_dist_[enc] < g) continue;

        if (r == goal.first && c == goal.second) return {fr, fc};

        for (int d = 0; d < 4; ++d) {
            int nr = r + DR[d], nc = c + DC[d];
            if (!is_passable(nr, nc)) continue;
            int ng   = g + 1;
            int nenc = cell_encode(nr, nc, W_);
            if (astar_dist_[nenc] > ng) {
                if (astar_dist_[nenc] == INT_MAX) astar_dirty_.push_back(nenc);
                astar_dist_[nenc] = ng;
                heap.push({ng + h(nr, nc), ng, nr, nc, fr, fc});
            }
        }
    }
    return start;
}


// Build leader_of_, is_leader_, leaders_, group_members_ from current positions.
void LocalLeadersPolicy::form_groups(const PosVec& positions) {
    leaders_.clear();
    group_members_.clear();

    for (int i = 0; i < N_; ++i) {
        Pos pi = positions[i];
        int closestLeader = -1;
        int minDistance   = config_.leader_view + 1;

        for (int leader : leaders_) {
            int d = manh(pi, positions[leader]);
            if (d <= config_.leader_view && d < minDistance) {
                minDistance   = d;
                closestLeader = leader;
            }
        }

        leader_of_[i] = (closestLeader == -1) ? i : closestLeader;
        if (leader_of_[i] == i) {
            leaders_.push_back(i);
            group_members_[i] = {i};
        } else {
            group_members_[leader_of_[i]].push_back(i);
        }
    }
}


inline bool LocalLeadersPolicy::is_passable(int r, int c) const noexcept {
    return r >= 0 && r < H_ && c >= 0 && c < W_ && grid_flat_[r * W_ + c] == 0;
}


Pos LocalLeadersPolicy::best_alternative(
    Pos pos, Pos hint_target, const Committed& committed, int exclude_enc) const
{
    Pos best_cell = pos;
    int best_dist = INT_MAX;

    for (int k = 0; k < 5; ++k) {
        int r = (k < 4) ? pos.first  + DR[k] : pos.first;
        int c = (k < 4) ? pos.second + DC[k] : pos.second;
        if (!is_passable(r, c)) continue;
        int enc = cell_encode(r, c, W_);
        if (committed.count(enc)) continue;
        if (enc == exclude_enc)  continue;
        int dist = std::abs(r - hint_target.first) + std::abs(c - hint_target.second);
        if (dist < best_dist) {
            best_dist = dist;
            best_cell = {r, c};
        }
    }
    return best_cell;
}


// Returns the processing order for a group: leader first, then members sorted
// by (stuck desc, dist-to-target asc, agent-ID asc).
std::vector<int> LocalLeadersPolicy::group_order(
    int leader,
    const PosVec& positions,
    const PosVec& targets) const
{
    const auto& members = group_members_.at(leader);
    std::vector<int> order;
    order.reserve(members.size());
    order.push_back(leader);
    for (int m : members)
        if (m != leader) order.push_back(m);

    std::sort(order.begin() + 1, order.end(), [&](int a, int b) {
        // Stuck agents get priority within the group so the leader ensures
        // their escape move can actually commit (mirrors ESCAPE criterion).
        bool ea = stuck_[a] >= config_.escape_thresh;
        bool eb = stuck_[b] >= config_.escape_thresh;
        if (ea != eb) return ea > eb;   // escape = first
        // Among equals: closer to target = more urgent.
        int da = manh(positions[a], targets[a]);
        int db = manh(positions[b], targets[b]);
        if (da != db) return da < db;
        return a < b;
    });
    return order;
}


// Phase A: leader L resolves vertex and swap conflicts among its own group,
// writing the result into provisional_[].
void LocalLeadersPolicy::resolve_intra_group(
    int leader,
    const PosVec& positions,
    const PosVec& targets)
{
    std::vector<int> order = group_order(leader, positions, targets);

    // Per-group committed map: cell_enc -> agent that will occupy it.
    Committed grp;
    grp.reserve(order.size() * 2);

    for (int i : order) {
        Pos pos    = positions[i];
        Pos want   = desired_[i];
        int want_enc = cell_encode(want.first, want.second, W_);
        int pos_enc  = cell_encode(pos.first,  pos.second,  W_);

        // --- Vertex conflict (within group) ---
        if (grp.count(want_enc)) {
            // Another group-member already claimed this cell; re-route.
            want     = best_alternative(pos, targets[i], grp);
            want_enc = cell_encode(want.first, want.second, W_);
        }

        // --- Swap conflict (within group) ---
        // j is committed to move to pos (i's current cell); if i wants j's
        // current position, they would swap — re-route i.
        {
            auto jt = grp.find(pos_enc);
            if (jt != grp.end() && want != pos) {
                int j = jt->second;
                if (positions[j] == want) {  // i and j are trying to swap
                    want     = best_alternative(pos, targets[i], grp, want_enc);
                    want_enc = cell_encode(want.first, want.second, W_);
                }
            }
        }

        provisional_[i] = want;
        grp[want_enc]   = i;
    }
}


// Main per-timestep function.
IntVec LocalLeadersPolicy::act(const PosVec& positions, const PosVec& targets) {

    // 1. Reform groups if scheduled.
    if (step_count_ % config_.regroup_interval == 0) {
        form_groups(positions);
        for (int i = 0; i < N_; ++i)
            is_leader_[i] = (leader_of_[i] == i);
    }
    ++step_count_;

    // 2. Every agent independently computes its desired next cell via A*.
    //    Stuck agents pick a random free neighbour (escape mode).
    for (int i = 0; i < N_; ++i) {
        Pos pos = positions[i];

        if (stuck_[i] >= config_.escape_thresh) {
            Pos nbrs[4]; int vn = 0;
            for (int d = 0; d < 4; ++d) {
                int nr = pos.first + DR[d], nc = pos.second + DC[d];
                if (is_passable(nr, nc)) nbrs[vn++] = {nr, nc};
            }
            desired_[i] = (vn == 0)
                ? pos
                : nbrs[std::uniform_int_distribution<int>(0, vn - 1)(rng_)];
        } else {
            desired_[i] = astar_next(pos, targets[i]);
        }
    }

    // 3. Phase A — intra-group resolution.
    //    Each leader mediates conflicts among its own followers independently.
    //    Result stored in provisional_[].
    for (int leader : leaders_)
        resolve_intra_group(leader, positions, targets);

    // 4. Phase B — inter-group resolution (leader negotiation).
    //
    //    Two sub-passes:
    //      B1. Escape agents (stuck >= escape_thresh) commit first globally,
    //          ordered by leader priority then agent ID.  This mirrors the
    //          original ESCAPE criterion so stuck agents can always break free.
    //      B2. Normal agents commit in group-priority order.  Groups whose
    //          leader is closest to its target go first; within each group the
    //          intra-group order from phase A is preserved.  When a conflict is
    //          detected, the lower-priority group's agent re-routes — this is the
    //          inter-leader "negotiation".

    // Sort leaders for inter-group priority:
    //   - groups with more stuck members go first (keeps escape priority at group level)
    //   - tiebreak: leader closer to its own target = more urgent
    //   - final tiebreak: lower leader ID
    std::vector<int> sorted_leaders = leaders_;
    std::sort(sorted_leaders.begin(), sorted_leaders.end(), [&](int a, int b) {
        // Max stuck count in each group (include leader).
        int ms_a = stuck_[a], ms_b = stuck_[b];
        for (int m : group_members_[a]) ms_a = std::max(ms_a, stuck_[m]);
        for (int m : group_members_[b]) ms_b = std::max(ms_b, stuck_[m]);
        if (ms_a != ms_b) return ms_a > ms_b;   // more stuck = first
        int da = manh(positions[a], targets[a]);
        int db = manh(positions[b], targets[b]);
        if (da != db) return da < db;
        return a < b;
    });

    Committed global;
    global.reserve(N_ * 2);
    final_pos_ = positions;   // default: stay

    // --- B1: escape agents commit first (overriding group order) ---
    {
        std::vector<int> escape_agents;
        escape_agents.reserve(N_);
        for (int i = 0; i < N_; ++i)
            if (stuck_[i] >= config_.escape_thresh) escape_agents.push_back(i);

        std::sort(escape_agents.begin(), escape_agents.end(), [&](int a, int b) {
            int la = leader_of_[a], lb = leader_of_[b];
            int da = manh(positions[la], targets[la]);
            int db = manh(positions[lb], targets[lb]);
            if (da != db) return da < db;
            return a < b;
        });

        for (int i : escape_agents) {
            Pos pos      = positions[i];
            Pos want     = provisional_[i];
            int want_enc = cell_encode(want.first, want.second, W_);

            if (global.count(want_enc)) {
                want     = best_alternative(pos, targets[i], global);
                want_enc = cell_encode(want.first, want.second, W_);
            }
            final_pos_[i]    = want;
            global[want_enc] = i;
        }
    }

    // --- B2: normal agents in group-priority order (leader negotiation) ---
    for (int leader : sorted_leaders) {
        std::vector<int> order = group_order(leader, positions, targets);

        for (int i : order) {
            if (stuck_[i] >= config_.escape_thresh) continue;  // already committed

            Pos pos      = positions[i];
            Pos want     = provisional_[i];
            int want_enc = cell_encode(want.first, want.second, W_);
            int pos_enc  = cell_encode(pos.first,  pos.second,  W_);

            // Inter-group vertex conflict: higher-priority group already claimed cell.
            if (global.count(want_enc)) {
                want     = best_alternative(pos, targets[i], global);
                want_enc = cell_encode(want.first, want.second, W_);
            }

            // Inter-group swap conflict: higher-priority agent j is moving to i's
            // current cell and i wants j's current cell.
            {
                auto jt = global.find(pos_enc);
                if (jt != global.end() && want != pos) {
                    int j = jt->second;
                    if (positions[j] == want) {
                        want     = best_alternative(pos, targets[i], global, want_enc);
                        want_enc = cell_encode(want.first, want.second, W_);
                    }
                }
            }

            final_pos_[i]    = want;
            global[want_enc] = i;
        }
    }

    // 5. Update stuck counters.
    for (int i = 0; i < N_; ++i)
        stuck_[i] = (final_pos_[i] == positions[i]) ? stuck_[i] + 1 : 0;

    // 6. Convert final cells to POGEMA action integers.
    IntVec actions(N_);
    for (int i = 0; i < N_; ++i) {
        int dr = final_pos_[i].first  - positions[i].first;
        int dc = final_pos_[i].second - positions[i].second;
        if      (dr == -1) actions[i] = ACTION_UP;
        else if (dr ==  1) actions[i] = ACTION_DOWN;
        else if (dc == -1) actions[i] = ACTION_LEFT;
        else if (dc ==  1) actions[i] = ACTION_RIGHT;
        else               actions[i] = ACTION_STAY;
    }
    return actions;
}
