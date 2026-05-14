# local_leaders_cpp

C++ implementation of the project’s **Local Leaders** MAPF baseline, exposed to Python via **pybind11**.

This module is used by `code/pogema_bench/run_benchmark.py` when `--algo local-leaders` is selected.

## Notes

- This is a *baseline* implementation (mirrors the current Python logic) but runs much faster.
- It’s a global one-shot planner: plan on the full grid, then replay actions in POGEMA.

