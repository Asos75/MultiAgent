#pragma once
#include <random>
#include <unordered_map>
#include <utility>
#include <vector>

// ── Visibility constants — change here to tune the algorithm ──────────────────
//
//  AGENT_VIEW_RADIUS : observation radius for regular (follower) agents.
//                      Followers resolve conflicts only with agents within
//                      this Manhattan distance; others are treated as static.
//  LEADER_VIEW_RADIUS: observation / communication radius for leaders.
//                      Used for group-formation: agent i joins the nearest
//                      already-formed leader within this distance.
//  ESCAPE_THRESHOLD  : consecutive stuck steps before an agent switches to
//                      random-neighbour escape mode.
//
static constexpr int AGENT_VIEW_RADIUS  = 5;
static constexpr int LEADER_VIEW_RADIUS = 7;
static constexpr int ESCAPE_THRESHOLD   = 4;

// ── POGEMA action encoding ────────────────────────────────────────────────────
static constexpr int ACTION_STAY  = 0;
static constexpr int ACTION_UP    = 1;
static constexpr int ACTION_DOWN  = 2;
static constexpr int ACTION_LEFT  = 3;
static constexpr int ACTION_RIGHT = 4;

// ── Convenience aliases ───────────────────────────────────────────────────────
using Grid   = std::vector<std::vector<int>>;
using Pos    = std::pair<int, int>;
using PosVec = std::vector<Pos>;
using IntVec = std::vector<int>;
// committed map: encoded cell → agent index that owns it
using Committed = std::unordered_map<int, int>;

// Flat encoding for fast hashing
inline int  cell_encode(int r, int c, int W) noexcept { return r * W + c; }
inline Pos  cell_decode(int code, int W)    noexcept { return {code / W, code % W}; }

// Manhattan distance
inline int manh(Pos a, Pos b) noexcept {
    return std::abs(a.first - b.first) + std::abs(a.second - b.second);
}

// ── Neighbour deltas (up, down, left, right) ──────────────────────────────────
static constexpr int DR[4] = {-1,  1,  0,  0};
static constexpr int DC[4] = { 0,  0, -1,  1};

// ── A* helper ─────────────────────────────────────────────────────────────────
// Returns the first cell on the shortest static-obstacle-only path from
// `start` to `goal`. Returns `start` if goal is already reached or unreachable.
Pos astar_next(const Grid& grid, int H, int W, Pos start, Pos goal);

// ── Main policy class ─────────────────────────────────────────────────────────
class LocalLeadersPolicy {
public:
    explicit LocalLeadersPolicy(int num_agents, unsigned seed = 0);

    // Call once per episode with the static obstacle grid (0 = free, 1 = wall).
    void reset(const Grid& grid);

    // Compute one-step POGEMA actions for all agents.
    // positions, targets: (row, col) in the padded grid coordinate system.
    IntVec act(const PosVec& positions, const PosVec& targets);

private:
    int          N_;
    Grid         grid_;
    int          H_, W_;
    IntVec       stuck_;       // consecutive stuck steps per agent
    std::mt19937 rng_;

    // Assign each agent to a leader (returns leader index per agent).
    // Uses LEADER_VIEW_RADIUS for group formation.
    IntVec form_groups(const PosVec& positions) const;

    // Best un-committed cell reachable in one step that is closest to tgt.
    // exclude_enc: additional encoded cell to skip (-1 = none).
    Pos best_alternative(Pos pos, Pos tgt, const Committed& committed,
                         int exclude_enc = -1) const;

    // Whether agent i can "see" agent j given their roles.
    // Followers use AGENT_VIEW_RADIUS; leaders use LEADER_VIEW_RADIUS.
    bool in_view(bool i_is_leader, Pos pi, Pos pj) const noexcept;
};
