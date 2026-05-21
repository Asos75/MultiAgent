#pragma once
#include <climits>
#include <cstdlib>
#include <random>
#include <unordered_map>
#include <utility>
#include <vector>

static constexpr int AGENT_VIEW_RADIUS  = 5;
static constexpr int LEADER_VIEW_RADIUS = 7;
static constexpr int ESCAPE_THRESHOLD   = 4;

static constexpr int ACTION_STAY  = 0;
static constexpr int ACTION_UP    = 1;
static constexpr int ACTION_DOWN  = 2;
static constexpr int ACTION_LEFT  = 3;
static constexpr int ACTION_RIGHT = 4;

using Grid      = std::vector<std::vector<int>>;
using Pos       = std::pair<int, int>;
using PosVec    = std::vector<Pos>;
using IntVec    = std::vector<int>;
using Committed = std::unordered_map<int, int>;

inline int cell_encode(int r, int c, int W) noexcept { return r * W + c; }
inline Pos cell_decode(int code, int W)     noexcept { return {code / W, code % W}; }

inline int manh(Pos a, Pos b) noexcept {
    return std::abs(a.first - b.first) + std::abs(a.second - b.second);
}

static constexpr int DR[4] = {-1,  1,  0,  0};
static constexpr int DC[4] = { 0,  0, -1,  1};

struct AStarNode {
    int f, g, r, c, fr, fc;
    bool operator>(const AStarNode& o) const noexcept { return f > o.f; }
};

enum class Criterion {
    ESCAPE,
    LEADER,
    FOLLOWER,
    PROXIMITY_CLOSEST,
    PROXIMITY_FURTHEST,
    AGENT_ID,
    MOST_STUCK,
    LEAST_STUCK,
};

struct PolicyConfig {
    int  agent_view        = AGENT_VIEW_RADIUS;
    int  leader_view       = LEADER_VIEW_RADIUS;
    int  escape_thresh     = ESCAPE_THRESHOLD;
    int  regroup_interval  = 1;
    bool hint_use_desired  = true;

    std::vector<Criterion> criteria = {
        Criterion::ESCAPE,
        Criterion::LEADER,
        Criterion::AGENT_ID,
    };
};

// V2: leaders genuinely mediate conflicts.
//
// Two-phase resolution each timestep:
//   Phase A — intra-group: each leader resolves vertex/swap conflicts among its
//             own followers before any inter-group negotiation happens.
//   Phase B — inter-group: groups commit to the global map in leader-priority
//             order; when a conflict is detected, the lower-priority group's
//             agent re-routes (the two leaders have "negotiated").
class LocalLeadersPolicy {
public:
    explicit LocalLeadersPolicy(int num_agents, unsigned seed = 0,
                                PolicyConfig config = {});

    void   reset(const Grid& grid);
    IntVec act(const PosVec& positions, const PosVec& targets);

private:
    int          N_, H_, W_;
    PolicyConfig config_;

    std::vector<int> grid_flat_;
    IntVec           stuck_;
    std::mt19937     rng_;
    int              step_count_ = 0;

    mutable std::vector<int> astar_dist_;
    mutable std::vector<int> astar_dirty_;

    // Per-step scratch
    IntVec             leader_of_;
    std::vector<bool>  is_leader_;
    PosVec             desired_;
    PosVec             provisional_;   // after intra-group resolution
    PosVec             final_pos_;
    Committed          committed_;     // reused for intra-group passes
    std::vector<int>   leaders_;

    // Leader -> all members (including leader itself), built each regroup.
    std::unordered_map<int, std::vector<int>> group_members_;

    Pos  astar_next(Pos start, Pos goal) const;
    void form_groups(const PosVec& positions);

    inline bool is_passable(int r, int c) const noexcept;

    Pos best_alternative(Pos pos, Pos hint_target, const Committed& committed,
                         int exclude_enc = -1) const;

    // Phase A: leader L resolves all intra-group conflicts, writes provisional_.
    void resolve_intra_group(int leader,
                             const PosVec& positions,
                             const PosVec& targets);

    // Returns intra-group priority order for a group (leader first).
    std::vector<int> group_order(int leader,
                                 const PosVec& positions,
                                 const PosVec& targets) const;
};
