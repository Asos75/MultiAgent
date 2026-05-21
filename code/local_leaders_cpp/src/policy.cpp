#include "policy.hpp"

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

    // Flatten the 2-D grid for cache-friendly access in A* inner loops.
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
    ordered_agents_.resize(N_);
    final_pos_.resize(N_);
    committed_.reserve(N_ * 2);
    leaders_.reserve(N_);
}

// A* algorithm
Pos LocalLeadersPolicy::astar_next(Pos start, Pos goal) const {
    if (start == goal) return start;

    // Lazily reset cells touched by the previous call.
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
    return start; // unreachable -> stay
}


// Group formation
void LocalLeadersPolicy::form_groups(const PosVec& positions) {
    leaders_.clear();

    // For every agent we look for the closest leader and assign him there
    // If no such leader is found, he becomes one
    for (int i = 0; i < N_; ++i) {
        Pos pi = positions[i];
        int closestLeader = -1;
        int minDistance = config_.leader_view + 1;

        for (int leader : leaders_) {
            int d = manh(pi, positions[leader]);
            if (d <= config_.leader_view && d < minDistance) {
                minDistance = d;
                closestLeader = leader;
            }
        }

        leader_of_[i] = (closestLeader == -1) ? i : closestLeader;
        if (leader_of_[i] == i) {
            leaders_.push_back(i);
        }
    }
}

// Grid helpers 

inline bool LocalLeadersPolicy::is_passable(int r, int c) const noexcept {
    return r >= 0 && r < H_ && c >= 0 && c < W_ && grid_flat_[r * W_ + c] == 0;
}

inline bool LocalLeadersPolicy::in_view(bool i_is_leader, Pos pi, Pos pj) const noexcept {
    return manh(pi, pj) <= (i_is_leader ? config_.leader_view : config_.agent_view);
}


// When the desired cell is taken we look for the best alternative move
Pos LocalLeadersPolicy::best_alternative(
    Pos pos, Pos desire, const Committed& committed, int exclude_enc) const {
    Pos best_cell = pos;
    int best_dist = INT_MAX;

    for (int k = 0; k < 5; ++k) {
        int r = (k < 4) ? pos.first  + DR[k] : pos.first;
        int c = (k < 4) ? pos.second + DC[k] : pos.second;
        if (!is_passable(r, c)) continue;
        int enc = cell_encode(r, c, W_);
        if (committed.count(enc)) continue;
        if (enc == exclude_enc) continue;
        int dist = std::abs(r - desire.first) + std::abs(c - desire.second);
        if (dist < best_dist) {
            best_dist = dist;
            best_cell = {r, c};
        }
    }
    return best_cell;
}

// Function for resolving vertex conflicts
Pos LocalLeadersPolicy::resolve_vertex(
    int i, Pos pos, Pos want, int& want_enc, Pos desire, int pos_enc,
    const PosVec& positions) const {
    auto it = committed_.find(want_enc);
    if (it == committed_.end()) return want;

    int owner = it->second;
    if (in_view(is_leader_[i], pos, positions[owner])) {
        want = best_alternative(pos, desire, committed_);
        want_enc = cell_encode(want.first, want.second, W_);
    } else {
        want = pos;
        want_enc = pos_enc;
    }
    return want;
}

// Function for resolving swap conflicts
Pos LocalLeadersPolicy::resolve_swap(
    int i, Pos pos, Pos want, int& want_enc, Pos desire, int pos_enc,
    const PosVec& positions) const {
    if (want == pos) return want;

    auto jt = committed_.find(pos_enc);
    if (jt == committed_.end()) return want;

    int j = jt->second;
    if (positions[j] == want && in_view(is_leader_[i], pos, positions[j])) {
        want = best_alternative(pos, desire, committed_, want_enc);
        want_enc = cell_encode(want.first, want.second, W_);
    }
    return want;
}

// Main action function
IntVec LocalLeadersPolicy::act(const PosVec& positions, const PosVec& targets) {
    
    // 1. Group formation - recomputed every regroup_interval steps.
    if (step_count_ % config_.regroup_interval == 0) {
        form_groups(positions);
        for (int i = 0; i < N_; ++i)
            is_leader_[i] = (leader_of_[i] == i);
    }
    ++step_count_;

    // 2. For each agent we compute the next move with A*
    // Alternatively, if the agent is stuck we move randomly
    for (int i = 0; i < N_; ++i) {
        // Agents' current position
        Pos pos = positions[i];

        if (stuck_[i] >= config_.escape_thresh) {
            // Agent is stuck
            Pos nbrs[4];
            int valid_neighbours = 0;
            // We check each of the 4 neighbourts and store the valid ones in nbrs
            for (int d = 0; d < 4; ++d) {
                int nbr_row = pos.first + DR[d];
                int nbr_col = pos.second + DC[d];
                if (is_passable(nbr_row, nbr_col)){
                    nbrs[valid_neighbours] = {nbr_row, nbr_col};
                    valid_neighbours++;
                }
            }
            // If there are no valid neighbours, we stay in place, otherwise we pick a random valid neighbour
            if (valid_neighbours == 0)
                desired_[i] = pos;
            else
                desired_[i] = nbrs[std::uniform_int_distribution<int>(0, valid_neighbours - 1)(rng_)];
        } else {
            // We compute the desired next position with A*
            desired_[i] = astar_next(pos, targets[i]);
        }
    }

    // 3. We sort agents by their priority as defined by the config criteria.
    std::iota(ordered_agents_.begin(), ordered_agents_.end(), 0);
    std::sort(ordered_agents_.begin(), ordered_agents_.end(), [&](int x, int y) {
        for (Criterion cr : config_.criteria) {
            int kx, ky;
            switch (cr) {
                case Criterion::ESCAPE:
                    kx = (stuck_[x] >= config_.escape_thresh) ? 0 : 1;
                    ky = (stuck_[y] >= config_.escape_thresh) ? 0 : 1;
                    break;
                case Criterion::LEADER:
                    kx = is_leader_[x] ? 0 : 1;
                    ky = is_leader_[y] ? 0 : 1;
                    break;
                case Criterion::FOLLOWER:
                    kx = is_leader_[x] ? 1 : 0;
                    ky = is_leader_[y] ? 1 : 0;
                    break;
                case Criterion::PROXIMITY_CLOSEST:
                    kx = manh(positions[x], targets[x]);
                    ky = manh(positions[y], targets[y]);
                    break;
                case Criterion::PROXIMITY_FURTHEST:
                    kx = -manh(positions[x], targets[x]);
                    ky = -manh(positions[y], targets[y]);
                    break;
                case Criterion::AGENT_ID:
                    kx = x; ky = y;
                    break;
                case Criterion::MOST_STUCK:
                    kx = -stuck_[x]; ky = -stuck_[y];
                    break;
                case Criterion::LEAST_STUCK:
                    kx = stuck_[x]; ky = stuck_[y];
                    break;
                default:
                    kx = ky = 0;
            }
            if (kx != ky) return kx < ky;
        }
        return false;
    });

    // 4. Conflict resolution - committed maps encoded cell -> owning agent index.
    committed_.clear();
    final_pos_ = positions;

    for (int i : ordered_agents_) {
        Pos pos    = positions[i];
        Pos want   = desired_[i];
        Pos desire = config_.hint_use_desired ? desired_[i] : targets[i];

        // Flat encodings of current and desired cells for hashmap
        int want_enc = cell_encode(want.first, want.second, W_);
        int pos_enc  = cell_encode(pos.first,  pos.second,  W_);

        // Resolve conflicts with previously committed agents, in priority order.
        // Want can change here
        want = resolve_vertex(i, pos, want, want_enc, desire, pos_enc, positions);
        want = resolve_swap(i, pos, want, want_enc, desire, pos_enc, positions);

        // Register where the agent will actually go
        final_pos_[i] = want;
        committed_[want_enc] = i;
    }

    // 5. Update stuck counters.
    for (int i = 0; i < N_; ++i){
        // If the agent didn't move, increase its stuck counter, otherwise reset it.
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
