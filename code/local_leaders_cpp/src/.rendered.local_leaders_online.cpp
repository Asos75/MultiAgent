/*

*/

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "policy.hpp"

namespace py = pybind11;

PYBIND11_MODULE(local_leaders_online, m) {
    m.doc() = "Online Local Leaders MAPF policy — C++ backend";

    // Expose the compile-time constants so Python can inspect them.
    m.attr("AGENT_VIEW_RADIUS")  = AGENT_VIEW_RADIUS;
    m.attr("LEADER_VIEW_RADIUS") = LEADER_VIEW_RADIUS;
    m.attr("ESCAPE_THRESHOLD")   = ESCAPE_THRESHOLD;

    py::class_<LocalLeadersPolicy>(m, "LocalLeadersPolicy",
        "Online, fully decentralised Local Leaders MAPF policy.\n\n"
        "Usage::\n\n"
        "    policy = LocalLeadersPolicy(num_agents)\n"
        "    policy.reset(grid)          # grid: list[list[int]], 0=free 1=wall\n"
        "    while not done:\n"
        "        actions = policy.act(positions, targets)\n"
        "        obs, rew, dones, trunc, infos = env.step(actions)\n")
        .def(py::init<int, unsigned>(),
             py::arg("num_agents"),
             py::arg("seed") = 0u,
             "Construct the policy for `num_agents` agents with optional RNG seed.")
        .def("reset", &LocalLeadersPolicy::reset,
             py::arg("grid"),
             "Initialise for a new episode. Call once with the static obstacle grid.")
        .def("act", &LocalLeadersPolicy::act,
             py::arg("positions"),
             py::arg("targets"),
             "Return a list of POGEMA action ints for one timestep.\n\n"
             "positions / targets: list of (row, col) tuples in padded grid coords.");
}
