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
    desired_.resize(N_);
    agent_wants_.resize(N_);
    final_pos_.resize(N_);
    leaders_.reserve(N_);
    group_members_.reserve(N_);
}


// A* algorithm to find the next cell on the path from start to goal, ignoring other agents
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
        int minDistance = config_.leader_view + 1;

        // Iterate over existing leaders to find the closest one within leader_view
        for (int leader : leaders_) {
            int d = manh(pi, positions[leader]);
            if (d <= config_.leader_view && d < minDistance) {
                minDistance   = d;
                closestLeader = leader;
            }
        }

        // If no leader found, agent becomes the leader
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

// Best alternative cell if desired cell found by A* is blocked 
// Picks cell closes to target
Pos LocalLeadersPolicy::best_alternative(
    Pos pos, Pos target, const Committed& committed, int exclude_enc) const {
    Pos best_cell = pos;
    int best_dist = INT_MAX;

    // We check the desired cell first and then its 4 neighbors
    for (int k = 0; k < 5; ++k) {
        int row = (k < 4) ? pos.first  + DR[k] : pos.first;
        int col = (k < 4) ? pos.second + DC[k] : pos.second;
        if (!is_passable(row, col)) continue;
        int enc = cell_encode(row, col, W_);

        // We skip cells already occupied and the one in conflict
        if (committed.count(enc)) continue;
        if (enc == exclude_enc) continue;

        int dist = std::abs(row - target.first) + std::abs(col - target.second);
        if (dist < best_dist) {
            best_dist = dist;
            best_cell = {row, col};
        }
    }
    return best_cell;
}


// Returns a sorted list of agents for a particular group
std::vector<int> LocalLeadersPolicy::group_order(
    int leader,
    const PosVec& positions,
    const PosVec& targets) const
{

    // We fetch all members of the group with a particular leader
    const auto& members = group_members_.at(leader);

    std::vector<int> order;
    order.reserve(members.size());
    order.push_back(leader);
    for (int m : members)
        if (m != leader) order.push_back(m);

    std::sort(order.begin() + 1, order.end(), [&](int a, int b) {
        // Stuck agents get priority within the group
        bool ea = stuck_[a] >= config_.escape_thresh;
        bool eb = stuck_[b] >= config_.escape_thresh;
        if (ea != eb) return ea > eb; 
        // Among equals: closer to target = higher priority
        int da = manh(positions[a], targets[a]);
        int db = manh(positions[b], targets[b]);
        if (da != db) return da < db;
        return a < b;
    });
    return order;
}


// Leader resolves vertex and swap conflicts among its own group,
void LocalLeadersPolicy::resolve_intra_group(
    int leader,
    const PosVec& positions,
    const PosVec& targets)
{
    std::vector<int> ordered_agents = group_order(leader, positions, targets);

    // Per-group committed map: cell_enc -> agent that will occupy it.
    Committed cell_claims;
    cell_claims.reserve(ordered_agents.size() * 2);

    for (int agent_idx : ordered_agents) {
        Pos pos = positions[agent_idx];
        Pos want = desired_[agent_idx];
        int want_enc = cell_encode(want.first, want.second, W_);
        int pos_enc  = cell_encode(pos.first,  pos.second,  W_);

        // Vertex conflict within group
        if (cell_claims.count(want_enc)) {
            // Another group-member already claimed this cell, we try an alternative
            want = best_alternative(pos, targets[agent_idx], cell_claims);
            want_enc = cell_encode(want.first, want.second, W_);
        }

        // Swap conflict within group
        {
            auto jt = cell_claims.find(pos_enc);
            if (jt != cell_claims.end() && want != pos) {
                int j = jt->second;
                if (positions[j] == want) {  // i and j are trying to swap
                    want = best_alternative(pos, targets[agent_idx], cell_claims, want_enc);
                    want_enc = cell_encode(want.first, want.second, W_);
                }
            }
        }

        agent_wants_[agent_idx] = want;
        cell_claims[want_enc] = agent_idx;
    }
}


// Main per-timestep function.
IntVec LocalLeadersPolicy::act(const PosVec& positions, const PosVec& targets) {

    // 1. Reform groups if scheduled.
    if (step_count_ % config_.regroup_interval == 0) {
        form_groups(positions);
    }
    ++step_count_;

    // 2. Every agent independently computes its desired next cell via A*.
    for (int i = 0; i < N_; ++i) {
        Pos pos = positions[i];

        // Stuck agents instead pick a random free neighbour (escape mode).
        if (stuck_[i] >= config_.escape_thresh) {
            Pos nbrs[4]; int vn = 0;
            for (int d = 0; d < 4; ++d) {
                int nr = pos.first + DR[d], nc = pos.second + DC[d];
                if (is_passable(nr, nc)) {
                    nbrs[vn++] = {nr, nc};
                }
            }
            // Pick random among valid neighbours
            desired_[i] = (vn == 0)
                ? pos
                : nbrs[std::uniform_int_distribution<int>(0, vn - 1)(rng_)];
        } else {
            // Compute next move for agent i
            desired_[i] = astar_next(pos, targets[i]);
        }
    }

    // 3. Phase A - intra-group resolution.
    for (int leader : leaders_){
        resolve_intra_group(leader, positions, targets);
    }

    // 4. Inter-group resolution 

    // Sort leaders by priority
    std::vector<int> sorted_leaders = leaders_;
    std::sort(sorted_leaders.begin(), sorted_leaders.end(), [&](int a, int b) {
        // Primary: leader of group with more stuck agents wins
        int ms_a = stuck_[a], ms_b = stuck_[b];
        for (int m : group_members_[a]) ms_a = std::max(ms_a, stuck_[m]);
        for (int m : group_members_[b]) ms_b = std::max(ms_b, stuck_[m]);
        if (ms_a != ms_b) return ms_a > ms_b;   // more stuck = first
        int da = manh(positions[a], targets[a]);
        int db = manh(positions[b], targets[b]);
        // Seconddary: closer to target wins
        if (da != db) return da < db;
        // Tertiary: lower agent ID wins
        return a < b;
    });

    Committed cell_claims;
    cell_claims.reserve(N_ * 2);
    final_pos_ = positions;   // default: stay

    // Deal with escape-mode agents seperately first
    {
        std::vector<int> escape_agents;
        escape_agents.reserve(N_);
        // Fetch all agents that are in escape mode
        for (int i = 0; i < N_; ++i)
            if (stuck_[i] >= config_.escape_thresh) {
                escape_agents.push_back(i);
            }

        // Sort escape agents by priority
        std::sort(escape_agents.begin(), escape_agents.end(), [&](int a, int b) {
            int la = leader_of_[a], lb = leader_of_[b];
            int da = manh(positions[la], targets[la]);
            int db = manh(positions[lb], targets[lb]);
            // Primary: agent with leader of group closer to its target wins
            if (da != db) return da < db;
            // Secondary: lower agent ID wins
            return a < b;
        });

        for (int i : escape_agents) {
            Pos pos = positions[i];
            Pos want = agent_wants_[i];
            int want_enc = cell_encode(want.first, want.second, W_);

            // If desired cell is taken we settle for the best alternative
            if (cell_claims.count(want_enc)) {
                want = best_alternative(pos, targets[i], cell_claims);
                want_enc = cell_encode(want.first, want.second, W_);
            }
            final_pos_[i] = want;
            cell_claims[want_enc] = i;
        }
    }

    // Leader negotiation in priority order
    for (int leader : sorted_leaders) {

        // We fetch the intra-group resolution order for this leader and its group
        std::vector<int> ordered_agents = group_order(leader, positions, targets);

        for (int i : ordered_agents) {
            if (stuck_[i] >= config_.escape_thresh) continue;  // already committed in prev step

            Pos pos = positions[i];
            Pos want  = agent_wants_[i];
            int want_enc = cell_encode(want.first, want.second, W_);
            int pos_enc = cell_encode(pos.first,  pos.second,  W_);

            // Inter-group vertex conflict: higher-priority group already claimed cell so we settle for an alternative
            if (cell_claims.count(want_enc)) {
                want = best_alternative(pos, targets[i], cell_claims);
                want_enc = cell_encode(want.first, want.second, W_);
            }

            // Inter-group swap conflict: higher-priority agent j is moving to where i is
            {
                auto jt = cell_claims.find(pos_enc);
                if (jt != cell_claims.end() && want != pos) {
                    int j = jt->second;

                    // If agent j is trying to move to our location we settle for an alternative 
                    if (positions[j] == want) {
                        want = best_alternative(pos, targets[i], cell_claims, want_enc);
                        want_enc = cell_encode(want.first, want.second, W_);
                    }
                }
            }

            final_pos_[i] = want;
            cell_claims[want_enc] = i;
        }
    }

    // 5. Update stuck counters.
    for (int i = 0; i < N_; ++i){
        stuck_[i] = (final_pos_[i] == positions[i]) ? stuck_[i] + 1 : 0;
    }

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
