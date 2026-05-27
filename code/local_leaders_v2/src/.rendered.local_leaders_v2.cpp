/*

*/

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "policy_v2.hpp"

namespace py = pybind11;

PYBIND11_MODULE(local_leaders_v2, m) {
    m.doc() = "Local Leaders V2 - leader-mediated conflict resolution";

    m.attr("LEADER_VIEW_RADIUS") = LEADER_VIEW_RADIUS;
    m.attr("ESCAPE_THRESHOLD")   = ESCAPE_THRESHOLD;

    py::class_<PolicyConfig>(m, "PolicyConfig")
        .def(py::init<>())
        .def_readwrite("leader_view",     &PolicyConfig::leader_view)
        .def_readwrite("escape_thresh",   &PolicyConfig::escape_thresh)
        .def_readwrite("regroup_interval",&PolicyConfig::regroup_interval);

    py::class_<LocalLeadersPolicy>(m, "LocalLeadersPolicy")
        .def(py::init<int, unsigned, PolicyConfig>(),
             py::arg("num_agents"),
             py::arg("seed")   = 0u,
             py::arg("config") = PolicyConfig{})
        .def("reset", &LocalLeadersPolicy::reset, py::arg("grid"))
        .def("act",   &LocalLeadersPolicy::act,
             py::arg("positions"), py::arg("targets"));
}
