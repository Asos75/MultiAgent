#include "solver.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <limits>
#include <queue>
#include <stdexcept>

namespace llcpp {

using Pos = std::pair<int, int>; // (r,c)

static inline int manhattan(const Pos& a, const Pos& b) {
  return std::abs(a.first - b.first) + std::abs(a.second - b.second);
}
static inline int chebyshev(const Pos& a, const Pos& b) {
  return std::max(std::abs(a.first - b.first), std::abs(a.second - b.second));
}

struct Group {
  int leader_id;
  std::vector<int> members;
  std::vector<uint8_t> view_mask; // H*W bool
};

static inline bool in_bounds(int r, int c, int H, int W) {
  return r >= 0 && r < H && c >= 0 && c < W;
}

static inline int idx(int r, int c, int W) { return r * W + c; }

static std::vector<Pos> neighbours4(const Pos& p) {
  return {{p.first - 1, p.second}, {p.first + 1, p.second}, {p.first, p.second - 1}, {p.first, p.second + 1}};
}

static int elect_leader(const std::vector<Pos>& starts_by_id, const std::vector<int>& member_ids, const Config& cfg) {
  if (member_ids.empty()) throw std::runtime_error("Group has no members");

  auto avg_dist = [&](int aid) {
    if (member_ids.size() == 1) return 0.0;
    double sum = 0;
    for (int bid : member_ids) {
      if (bid == aid) continue;
      sum += manhattan(starts_by_id[aid], starts_by_id[bid]);
    }
    return sum / double(member_ids.size() - 1);
  };

  auto density = [&](int aid) {
    int d = 0;
    for (int bid : member_ids) {
      if (bid == aid) continue;
      if (chebyshev(starts_by_id[aid], starts_by_id[bid]) <= 2) d++;
    }
    return d;
  };

  std::string mode = cfg.leader_election;
  std::transform(mode.begin(), mode.end(), mode.begin(), ::tolower);

  int best = member_ids[0];
  if (mode == "dynamic") {
    for (int aid : member_ids) {
      auto cur = std::make_tuple(density(aid), -avg_dist(aid), -aid);
      auto bst = std::make_tuple(density(best), -avg_dist(best), -best);
      if (cur > bst) best = aid;
    }
    return best;
  }

  for (int aid : member_ids) {
    auto cur = std::make_tuple(avg_dist(aid), aid);
    auto bst = std::make_tuple(avg_dist(best), best);
    if (cur < bst) best = aid;
  }
  return best;
}

static void add_view(const std::vector<std::vector<uint8_t>>& grid, int r0, int c0, int radius, std::vector<uint8_t>& mask) {
  int H = (int)grid.size();
  int W = H ? (int)grid[0].size() : 0;
  for (int r = r0 - radius; r <= r0 + radius; r++) {
    for (int c = c0 - radius; c <= c0 + radius; c++) {
      if (!in_bounds(r, c, H, W)) continue;
      if (grid[r][c] != 0) continue;
      mask[idx(r, c, W)] = 1;
    }
  }
}

static std::vector<int> bfs_path(const std::vector<std::vector<uint8_t>>& grid,
                                 const std::vector<uint8_t>& view_mask,
                                 const Pos& start,
                                 const Pos& goal) {
  if (start == goal) {
    return {idx(start.first, start.second, (int)grid[0].size())};
  }
  int H = (int)grid.size();
  int W = H ? (int)grid[0].size() : 0;

  const int N = H * W;
  std::deque<int> q;
  std::vector<int> prev(N, -1);
  std::vector<uint8_t> seen(N, 0);

  int s = idx(start.first, start.second, W);
  int g = idx(goal.first, goal.second, W);
  q.push_back(s);
  seen[s] = 1;

  while (!q.empty()) {
    int cur = q.front();
    q.pop_front();
    int r = cur / W;
    int c = cur % W;
    for (const auto& nb : neighbours4({r, c})) {
      if (!in_bounds(nb.first, nb.second, H, W)) continue;
      if (grid[nb.first][nb.second] != 0) continue;
      int ni = idx(nb.first, nb.second, W);
      if (!view_mask.empty() && !view_mask[ni]) continue;
      if (seen[ni]) continue;
      seen[ni] = 1;
      prev[ni] = cur;
      if (ni == g) {
        std::vector<int> path;
        path.push_back(g);
        while (path.back() != s) path.push_back(prev[path.back()]);
        std::reverse(path.begin(), path.end());
        return path;
      }
      q.push_back(ni);
    }
  }
  return {}; // fail
}

static inline Pos pos_at(const std::vector<int>& path_idx, int t, int W) {
  int it = path_idx[std::min<int>(t, (int)path_idx.size() - 1)];
  return {it / W, it % W};
}

struct Conflict { int ai; int aj; int t; int type; }; // type:0 vertex,1 swap

static bool find_earliest_conflict(const std::vector<std::vector<int>>& paths,
                                   int W,
                                   Conflict& out,
                                   int* total_conflicts,
                                   int max_conflicts) {
  int A = (int)paths.size();
  if (A == 0) {
    if (total_conflicts) *total_conflicts = 0;
    return false;
  }
  int makespan = 0;
  for (auto& p : paths) makespan = std::max(makespan, (int)p.size());

  bool found = false;
  int total = 0;

  std::unordered_map<int, int> occ;
  std::unordered_map<std::uint64_t, int> edge;
  occ.reserve(A * 2);
  edge.reserve(A * 2);

  auto ekey = [](int u, int v) -> std::uint64_t {
    return (std::uint64_t(u) << 32) ^ std::uint64_t(std::uint32_t(v));
  };

  for (int t = 0; t < makespan; t++) {
    occ.clear();
    edge.clear();
    for (int i = 0; i < A; i++) {
      int u = paths[i][std::min<int>(t, (int)paths[i].size() - 1)];
      int v = paths[i][std::min<int>(t + 1, (int)paths[i].size() - 1)];

      auto it = occ.find(u);
      if (it != occ.end()) {
        total++;
        if (!found) {
          out = {i, it->second, t, 0};
          found = true;
        }
        if (max_conflicts > 0 && total >= max_conflicts) {
          if (total_conflicts) *total_conflicts = total;
          return found;
        }
      } else {
        occ.emplace(u, i);
      }

      auto rit = edge.find(ekey(v, u));
      if (rit != edge.end()) {
        total++;
        if (!found) {
          out = {i, rit->second, t, 1};
          found = true;
        }
        if (max_conflicts > 0 && total >= max_conflicts) {
          if (total_conflicts) *total_conflicts = total;
          return found;
        }
      }
      edge.emplace(ekey(u, v), i);
    }
  }

  if (total_conflicts) *total_conflicts = total;
  return found;
}

// --- reservation-table planning (prioritized planning) ---
// Reserve vertex at time t, and reserve directed edge (u->v) at time t for swap avoidance.
struct ReservationTable {
  int W;
  // Use 64-bit key (nodeIdx, t) for vertex reservations.
  std::unordered_set<std::uint64_t> vertex;
  // Use 64-bit key (u, v, t) for edge reservations.
  std::unordered_set<std::uint64_t> edge;

  explicit ReservationTable(int W_) : W(W_) {
    vertex.reserve(1 << 18);
    edge.reserve(1 << 18);
  }

  static inline std::uint64_t vkey(int node, int t) {
    return (std::uint64_t(node) << 32) ^ std::uint64_t(std::uint32_t(t));
  }
  static inline std::uint64_t ekey(int u, int v, int t) {
    // pack u(21b) v(21b) t(22b) is enough for our sizes; keep simple bit packing
    return (std::uint64_t(u) << 42) ^ (std::uint64_t(v) << 21) ^ std::uint64_t(std::uint32_t(t));
  }

  bool is_vertex_reserved(int node, int t) const { return vertex.find(vkey(node, t)) != vertex.end(); }
  bool is_edge_reserved(int u, int v, int t) const { return edge.find(ekey(u, v, t)) != edge.end(); }

  void reserve_path(const std::vector<int>& path, int hold_goal_for) {
    if (path.empty()) return;
    for (int t = 0; t < (int)path.size(); t++) {
      vertex.insert(vkey(path[t], t));
      if (t > 0) {
        edge.insert(ekey(path[t - 1], path[t], t - 1));
      }
    }
    // Hold goal for a while to reduce late collisions.
    int goal = path.back();
    int start_t = (int)path.size();
    for (int dt = 0; dt < hold_goal_for; dt++) {
      vertex.insert(vkey(goal, start_t + dt));
    }
  }
};

struct STNode {
  int node;
  int t;
};

static std::vector<int> reconstruct_st(int goal_k, const std::vector<int>& prev) {
  std::vector<int> out;
  int k = goal_k;
  while (k != -1) {
    out.push_back(k);
    k = prev[k];
  }
  std::reverse(out.begin(), out.end());
  return out;
}

static std::vector<int> plan_with_reservations(const std::vector<std::vector<uint8_t>>& grid,
                                               const std::vector<uint8_t>& view_mask,
                                               const Pos& start,
                                               const Pos& goal,
                                               const ReservationTable& rt,
                                               int max_time) {
  int H = (int)grid.size();
  int W = H ? (int)grid[0].size() : 0;
  if (start == goal) {
    int s = idx(start.first, start.second, W);
    if (!rt.is_vertex_reserved(s, 0)) return {s};
    // Otherwise: still search for a safe wait.
  }

  // BFS in space-time (uniform cost) with implicit waiting.
  // State encoding: k = t*(H*W) + node
  int N = H * W;
  int K = (max_time + 1) * N;
  std::deque<int> q;
  std::vector<int> prev(K, -1);
  std::vector<uint8_t> seen(K, 0);

  int s = idx(start.first, start.second, W);
  int g = idx(goal.first, goal.second, W);

  auto push = [&](int node, int t, int from_k) {
    int k = t * N + node;
    if (k < 0 || k >= K) return;
    if (seen[k]) return;
    seen[k] = 1;
    prev[k] = from_k;
    q.push_back(k);
  };

  // start
  if (!view_mask.empty() && !view_mask[s]) return {}; // can't see start -> shouldn't happen
  if (rt.is_vertex_reserved(s, 0)) {
    // blocked at t=0
    return {};
  }
  push(s, 0, -1);

  while (!q.empty()) {
    int cur_k = q.front();
    q.pop_front();
    int t = cur_k / N;
    int node = cur_k % N;
    if (node == g) {
      // found earliest arrival
      auto ks = reconstruct_st(cur_k, prev);
      std::vector<int> path;
      path.reserve(ks.size());
      for (int k : ks) path.push_back(k % N);
      return path;
    }
    if (t == max_time) continue;

    int r = node / W;
    int c = node % W;

    // 4-neigh + wait
    std::vector<Pos> nbs = neighbours4({r, c});
    nbs.push_back({r, c});
    for (const auto& nb : nbs) {
      if (!in_bounds(nb.first, nb.second, H, W)) continue;
      if (grid[nb.first][nb.second] != 0) continue;
      int nnode = idx(nb.first, nb.second, W);
      if (!view_mask.empty() && !view_mask[nnode]) continue;

      // vertex at t+1 must be free
      if (rt.is_vertex_reserved(nnode, t + 1)) continue;
      // swap: can't traverse edge if reverse is reserved at t
      if (rt.is_edge_reserved(nnode, node, t)) continue;

      push(nnode, t + 1, cur_k);
    }
  }
  return {};
}

Result solve(const std::vector<std::vector<uint8_t>>& grid,
             const std::vector<Pos>& starts,
             const std::vector<Pos>& goals,
             const Config& cfg) {
  auto t0 = std::chrono::steady_clock::now();

  int H = (int)grid.size();
  int W = H ? (int)grid[0].size() : 0;
  int A = (int)starts.size();
  if ((int)goals.size() != A) throw std::runtime_error("starts/goals size mismatch");

  std::vector<Pos> starts_by_id = starts;

  auto time_exceeded = [&]() -> bool {
    if (cfg.time_limit_sec <= 0.0) return false;
    double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return elapsed > cfg.time_limit_sec;
  };

  // --- form groups (same greedy as python) ---
  std::vector<int> remaining(A);
  for (int i = 0; i < A; i++) remaining[i] = i;

  std::vector<uint8_t> is_remaining(A, 1);
  std::vector<Group> groups;

  while (true) {
    int seed_id = -1;
    for (int i = 0; i < A; i++) {
      if (is_remaining[i]) { seed_id = i; break; }
    }
    if (seed_id == -1) break;
    Pos seed_pos = starts_by_id[seed_id];

    std::vector<int> members;
    members.reserve(cfg.max_group_size);
    for (int i = 0; i < A; i++) {
      if (!is_remaining[i]) continue;
      if (chebyshev(starts_by_id[i], seed_pos) <= cfg.group_radius) members.push_back(i);
    }
    if ((int)members.size() > cfg.max_group_size) {
      std::sort(members.begin(), members.end(), [&](int a, int b) {
        return chebyshev(starts_by_id[a], seed_pos) < chebyshev(starts_by_id[b], seed_pos);
      });
      members.resize(cfg.max_group_size);
    }

    int leader = elect_leader(starts_by_id, members, cfg);

    std::vector<uint8_t> view_mask(H * W, 0);
    for (int aid : members) {
      add_view(grid, starts_by_id[aid].first, starts_by_id[aid].second, cfg.leader_view_radius, view_mask);
      add_view(grid, goals[aid].first, goals[aid].second, cfg.leader_view_radius, view_mask);
    }
    add_view(grid, starts_by_id[leader].first, starts_by_id[leader].second, cfg.leader_view_radius, view_mask);
    add_view(grid, goals[leader].first, goals[leader].second, cfg.leader_view_radius, view_mask);

    groups.push_back(Group{leader, members, std::move(view_mask)});
    for (int aid : members) is_remaining[aid] = 0;
  }

  // --- local plan per group ---
  std::vector<std::vector<int>> paths(A);

  // Time horizon for space-time planning.
  // Bound it, but give enough room for detours.
  int max_time = std::max(64, (H * W));
  max_time = std::min(max_time, 4096);

  // Plan each group with prioritized planning (leader first).
  // Within group: reservation-table planning guarantees no in-group conflicts.
  std::string election_mode = cfg.leader_election;
  std::transform(election_mode.begin(), election_mode.end(), election_mode.begin(), ::tolower);

  for (size_t gi = 0; gi < groups.size(); gi++) {
    auto& g = groups[gi];
    if (cfg.dynamic_reselect_every > 0 && election_mode == "dynamic") {
      // One-time reselection before planning (mirrors Python behavior).
      g.leader_id = elect_leader(starts_by_id, g.members, cfg);
    }

    ReservationTable rt(W);
    std::vector<int> order = g.members;
    // Put leader first.
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
      if (a == g.leader_id) return true;
      if (b == g.leader_id) return false;
      return a < b;
    });

    for (int aid : order) {
      int manh = manhattan(starts[aid], goals[aid]);
      int agent_max_time = std::max(16, manh * 4 + 32);
      agent_max_time = std::min(agent_max_time, max_time);
      auto p = plan_with_reservations(grid, g.view_mask, starts[aid], goals[aid], rt, agent_max_time);
      if (p.empty()) {
        // fallback: ignore reservations (still better than failing)
        p = bfs_path(grid, g.view_mask, starts[aid], goals[aid]);
      }
      if (p.empty()) {
        // fallback: full-grid BFS if local view is too restrictive
        p = bfs_path(grid, std::vector<uint8_t>(), starts[aid], goals[aid]);
      }
      if (p.empty()) {
        p = {idx(starts[aid].first, starts[aid].second, W)};
      }
      paths[aid] = p;
      rt.reserve_path(paths[aid], /*hold_goal_for*/ 8);
    }
  }

  // Resolve inter-group conflicts by inserting waits (fast, bounded).
  int fixed = 0;
  Conflict first_conflict{0, 0, 0, 0};
  int initial_conflicts = 0;
  bool has_conflict = find_earliest_conflict(
      paths, W, first_conflict, &initial_conflicts, cfg.max_initial_conflicts + 1);
  if (initial_conflicts > cfg.max_initial_conflicts) {
    Result r;
    r.solved = false;
    r.num_conflicts_resolved = 0;
    r.num_groups = (int)groups.size();
    double avg = 0.0;
    for (auto& g : groups) avg += (double)g.members.size();
    if (!groups.empty()) avg /= (double)groups.size();
    r.avg_group_size = avg;
    r.comp_time_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return r;
  }

  if (has_conflict) {
    std::vector<uint8_t> is_leader(A, 0);
    std::vector<int> group_by_agent(A, -1);
    for (int gi = 0; gi < (int)groups.size(); gi++) {
      const auto& g = groups[gi];
      is_leader[g.leader_id] = 1;
      for (int aid : g.members) group_by_agent[aid] = gi;
    }

    int max_rounds = std::max(200, std::max(1, cfg.max_rounds_factor) * A);
    for (int rounds = 0; rounds < max_rounds; rounds++) {
      if (time_exceeded()) break;
      Conflict c{0, 0, 0, 0};
      int dummy = 0;
      if (!find_earliest_conflict(paths, W, c, &dummy, 0)) break;

      int ai = c.ai;
      int aj = c.aj;
      int t = c.t;

      int waiter = ai;
      if (is_leader[ai] && !is_leader[aj]) {
        waiter = aj;
      } else if (is_leader[aj] && !is_leader[ai]) {
        waiter = ai;
      } else {
        waiter = std::max(ai, aj);
      }

      int wait_pos = paths[waiter][std::max(0, t - 1)];
      int insert_at = std::min<int>(t, (int)paths[waiter].size());
      paths[waiter].insert(paths[waiter].begin() + insert_at, wait_pos);
      fixed++;
    }
  }

  // verify
  Conflict dummy{0, 0, 0, 0};
  bool ok = !find_earliest_conflict(paths, W, dummy, nullptr, 0);

  Result r;
  r.solved = ok;
  r.num_conflicts_resolved = fixed;
  r.num_groups = (int)groups.size();
  double avg = 0.0;
  for (auto& g : groups) avg += (double)g.members.size();
  if (!groups.empty()) avg /= (double)groups.size();
  r.avg_group_size = avg;

  if (ok) {
    int soc = 0;
    int makespan = 0;
    for (auto& p : paths) {
      soc += (int)p.size() - 1;
      makespan = std::max(makespan, (int)p.size() - 1);
    }
    r.soc = soc;
    r.makespan = makespan;
    // convert idx paths into (r,c)
    std::vector<std::vector<Pos>> out_paths(A);
    for (int aid = 0; aid < A; aid++) {
      out_paths[aid].reserve(paths[aid].size());
      for (int id : paths[aid]) out_paths[aid].push_back({id / W, id % W});
    }
    r.paths = std::move(out_paths);
  }

  r.comp_time_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  return r;
}

}  // namespace llcpp
