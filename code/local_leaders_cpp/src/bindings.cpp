#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "solver.hpp"

namespace py = pybind11;

PYBIND11_MODULE(local_leaders_cpp, m) {
  m.doc() = "Local Leaders MAPF solver (C++/pybind11)";

  py::class_<llcpp::Config>(m, "Config")
      .def(py::init<>())
      .def_readwrite("group_radius", &llcpp::Config::group_radius)
      .def_readwrite("max_group_size", &llcpp::Config::max_group_size)
      .def_readwrite("leader_view_radius", &llcpp::Config::leader_view_radius)
      .def_readwrite("leader_election", &llcpp::Config::leader_election)
      .def_readwrite("time_limit_sec", &llcpp::Config::time_limit_sec)
      .def_readwrite("seed", &llcpp::Config::seed)
      .def_readwrite("max_rounds_factor", &llcpp::Config::max_rounds_factor)
      .def_readwrite("max_initial_conflicts", &llcpp::Config::max_initial_conflicts);

  py::class_<llcpp::Result>(m, "Result")
      .def(py::init<>())
      .def_readonly("solved", &llcpp::Result::solved)
      .def_readonly("soc", &llcpp::Result::soc)
      .def_readonly("makespan", &llcpp::Result::makespan)
      .def_readonly("comp_time_ms", &llcpp::Result::comp_time_ms)
      .def_readonly("num_groups", &llcpp::Result::num_groups)
      .def_readonly("avg_group_size", &llcpp::Result::avg_group_size)
      .def_readonly("num_conflicts_resolved", &llcpp::Result::num_conflicts_resolved)
      .def_readonly("paths", &llcpp::Result::paths);

  m.def("solve", &llcpp::solve, py::arg("grid"), py::arg("starts"), py::arg("goals"), py::arg("cfg"));
}
