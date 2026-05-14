cmake_minimum_required(VERSION 3.16)
project(local_leaders_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build a Python extension module via pybind11.
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(local_leaders_cpp MODULE
    src/bindings.cpp
    src/solver.cpp
)

target_include_directories(local_leaders_cpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
