/*
cppimport
<%
setup_pybind11(cfg)
cfg['compiler_args'] = ['-std=c++17']

# Build multiple translation units.
cfg['sources'] = ['bindings.cpp', 'solver.cpp']
%>
*/

// Intentionally empty. cppimport will compile the sources declared above.
