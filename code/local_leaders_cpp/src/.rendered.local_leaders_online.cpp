/*

*/

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "policy.hpp"

namespace py = pybind11;

PYBIND11_MODULE(local_leaders_online, m) {
    m.doc() = "Online Local Leaders MAPF policy - C++ backend";

    m.attr("AGENT_VIEW_RADIUS")  = AGENT_VIEW_RADIUS;
    m.attr("LEADER_VIEW_RADIUS") = LEADER_VIEW_RADIUS;
    m.attr("ESCAPE_THRESHOLD")   = ESCAPE_THRESHOLD;

    py::enum_<Criterion>(m, "Criterion")
        .value("ESCAPE",             Criterion::ESCAPE)
        .value("LEADER",             Criterion::LEADER)
        .value("FOLLOWER",           Criterion::FOLLOWER)
        .value("PROXIMITY_CLOSEST",  Criterion::PROXIMITY_CLOSEST)
        .value("PROXIMITY_FURTHEST", Criterion::PROXIMITY_FURTHEST)
        .value("AGENT_ID",           Criterion::AGENT_ID)
        .value("MOST_STUCK",         Criterion::MOST_STUCK)
        .value("LEAST_STUCK",        Criterion::LEAST_STUCK)
        .export_values();

    py::class_<PolicyConfig>(m, "PolicyConfig",
        "All tuning knobs for LocalLeadersPolicy.\n\n"
        "Example::\n\n"
        "    cfg = PolicyConfig()\n"
        "    cfg.criteria = [Criterion.LEADER, Criterion.PROXIMITY_CLOSEST]\n"
        "    policy = LocalLeadersPolicy(32, config=cfg)\n")
        .def(py::init<>())
        .def_readwrite("agent_view",       &PolicyConfig::agent_view)
        .def_readwrite("leader_view",      &PolicyConfig::leader_view)
        .def_readwrite("escape_thresh",    &PolicyConfig::escape_thresh)
        .def_readwrite("regroup_interval",  &PolicyConfig::regroup_interval)
        .def_readwrite("hint_use_desired",  &PolicyConfig::hint_use_desired)
        .def_readwrite("criteria",          &PolicyConfig::criteria);

    py::class_<LocalLeadersPolicy>(m, "LocalLeadersPolicy",
        "Online, fully decentralised Local Leaders MAPF policy.\n\n"
        "Usage::\n\n"
        "    policy = LocalLeadersPolicy(num_agents)\n"
        "    policy.reset(grid)          # grid: list[list[int]], 0=free 1=wall\n"
        "    while not done:\n"
        "        actions = policy.act(positions, targets)\n"
        "        obs, rew, dones, trunc, infos = env.step(actions)\n")
        .def(py::init<int, unsigned, PolicyConfig>(),
             py::arg("num_agents"),
             py::arg("seed")   = 0u,
             py::arg("config") = PolicyConfig{},
             "Construct the policy. Optional: RNG seed and PolicyConfig.")
        .def("reset", &LocalLeadersPolicy::reset,
             py::arg("grid"),
             "Initialise for a new episode. Call once with the static obstacle grid.")
        .def("act", &LocalLeadersPolicy::act,
             py::arg("positions"),
             py::arg("targets"),
             "Return a list of POGEMA action ints for one timestep.\n\n"
             "positions / targets: list of (row, col) tuples in padded grid coords.");
}
