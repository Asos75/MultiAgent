#pragma once
#include <climits>
#include <cstdlib>
#include <random>
#include <unordered_map>
#include <utility>
#include <vector>

// Default tuning constants
static constexpr int AGENT_VIEW_RADIUS  = 5;
static constexpr int LEADER_VIEW_RADIUS = 7;
static constexpr int ESCAPE_THRESHOLD   = 4;

// POGEMA action encoding
static constexpr int ACTION_STAY  = 0;
static constexpr int ACTION_UP    = 1;
static constexpr int ACTION_DOWN  = 2;
static constexpr int ACTION_LEFT  = 3;
static constexpr int ACTION_RIGHT = 4;

// Convenience aliases
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

// Neighbour deltas (up, down, left, right)
static constexpr int DR[4] = {-1,  1,  0,  0};
static constexpr int DC[4] = { 0,  0, -1,  1};

// A* heap node
struct AStarNode {
    int f, g, r, c, fr, fc;
    bool operator>(const AStarNode& o) const noexcept { return f > o.f; }
};

// Priority criteria
enum class Criterion {
    ESCAPE,             // stuck (>= escape_thresh) agents go first
    LEADER,             // leaders go first
    FOLLOWER,           // followers go first (inverse of LEADER)
    PROXIMITY_CLOSEST,  // agent closest to its target goes first
    PROXIMITY_FURTHEST, // agent furthest from its target goes first
    AGENT_ID,           // lower agent index goes first
    MOST_STUCK,         // agent with more stuck steps goes first
    LEAST_STUCK,        // agent with fewer stuck steps goes first
};

// Policy configuration
struct PolicyConfig {
    int agent_view       = AGENT_VIEW_RADIUS;
    int leader_view      = LEADER_VIEW_RADIUS;
    int escape_thresh    = ESCAPE_THRESHOLD;
    int  regroup_interval    = 1;    // recompute groups every N steps; 1 = every step
    bool hint_use_desired    = true; // fallback hint: true = A* next step, false = final target

    // Priority ordering: evaluated left-to-right, first difference wins.
    std::vector<Criterion> criteria = {
        Criterion::ESCAPE,
        Criterion::LEADER,
        Criterion::AGENT_ID,
    };
};

// Main policy class
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

    // Per-step scratch buffers - allocated once in reset(), reused every act() call.
    IntVec             leader_of_;
    std::vector<bool>  is_leader_;
    PosVec             desired_;
    IntVec             ordered_agents_;
    PosVec             final_pos_;
    Committed          committed_;
    std::vector<int>   leaders_;

    // A* algorithm
    Pos  astar_next(Pos start, Pos goal) const;

    // Group formation
    void form_groups(const PosVec& positions);

    // Grid helpers
    inline bool is_passable(int r, int c) const noexcept;
    inline bool in_view(bool i_is_leader, Pos pi, Pos pj) const noexcept;

    // When the desired cell is taken we look for the best alternative move
    Pos best_alternative(Pos pos, Pos desire, const Committed& committed,
                         int exclude_enc = -1) const;

    // Function for resolving vertex conflicts
    Pos resolve_vertex(int i, Pos pos, Pos want, int& want_enc, Pos desire,
                       int pos_enc, const PosVec& positions) const;

    // Function for resolving swap conflicts
    Pos resolve_swap(int i, Pos pos, Pos want, int& want_enc, Pos desire,
                     int pos_enc, const PosVec& positions) const;
};
