#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace llcpp {

struct Config {
  int group_radius = 5;
  int max_group_size = 10;
  int leader_view_radius = 6;
  std::string leader_election = "static";  // static|dynamic
  int dynamic_reselect_every = 0;
  double time_limit_sec = 0.0;             // <=0 => disabled
  int seed = 0;
  int max_rounds_factor = 500;             // safety cap: max_rounds = max(200, factor*num_agents)
  int max_initial_conflicts = 2000;        // fail-fast
};

struct Result {
  bool solved = false;
  std::optional<int> soc;
  std::optional<int> makespan;
  double comp_time_ms = 0.0;
  std::optional<int> num_groups;
  std::optional<double> avg_group_size;
  int num_conflicts_resolved = 0;
  std::optional<std::vector<std::vector<std::pair<int, int>>>> paths; // paths[agent][t]=(r,c)
};

Result solve(const std::vector<std::vector<uint8_t>>& grid,
             const std::vector<std::pair<int, int>>& starts,
             const std::vector<std::pair<int, int>>& goals,
             const Config& cfg);

}  // namespace llcpp
