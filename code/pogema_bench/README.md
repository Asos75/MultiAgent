# POGEMA benchmark integration (LaCAM + MATS-LP)

This folder provides a small, unified benchmark runner based on **POGEMA** that lets you evaluate:

- **LaCAM** (centralized planner, C++ binary in `code/lacam/build/main`)
- **MATS-LP** (decentralized MCTS + NN, Python in `code/mats-lp`) *(currently disabled in this harness)*

The goal is to run both algorithms with the *same* environment generator and compute comparable metrics.

## What’s implemented

- A runner: `run_benchmark.py`
- A LaCAM adapter that:
  - generates a temporary `.map` and `.scen` compatible with LaCAM
  - runs the LaCAM binary
  - parses its output `result.txt` and replays the planned actions inside POGEMA
- A `matslp_adapter.py` placeholder.

## Recommended mode (reproducible): MovingAI → POGEMA

For LaCAM we **don’t** export POGEMA-random instances anymore. Instead we build a POGEMA environment from a concrete MovingAI `.map` + `.scen`, and run LaCAM on the same instance.

This is controlled with:

- `--env movingai` (default)
- `--movingai_map .../something.map`
- `--movingai_scen .../something.scen`

### Coordinate conventions (why this matters)

- MovingAI / LaCAM: `(x, y) = (col, row)`
- POGEMA 1.x uses numpy-style indexing internally (effectively `(row, col)`) when validating obstacles.

The runner swaps coordinates as needed to keep LaCAM planning and POGEMA replay consistent.

## Install (Python)

POGEMA is already a dependency of `code/mats-lp` (`pogema==1.2.2`).

If you don’t have a venv yet, install the MATS-LP requirements (from repo root):

```bash
pip install -r code/mats-lp/docker/requirements.txt

You’ll also typically want to run with:

```bash
export PYTHONPATH="$PWD/code"
```
```

## Build LaCAM

LaCAM is a CMake project in `code/lacam`. The binary should be:

- `code/lacam/build/main`

If it doesn’t exist yet:

```bash
cmake -S code/lacam -B code/lacam/build
cmake --build code/lacam/build -j

## Run (LaCAM)

From repo root:

```bash
PYTHONPATH=$PWD/code .venv/bin/python -m pogema_bench.run_benchmark \
  --algo lacam \
  --env movingai \
  --num_agents 8 \
  --movingai_map code/assets/moving_ai_maps/random-32-32-10.map \
  --movingai_scen code/assets/moving_ai_scen/random-32-32-10-random-1.scen
```
```

## Run

Example run on a built-in POGEMA random map:

```bash
python code/pogema_bench/run_benchmark.py --algo lacam --num_agents 16 --map_name random --seed 0 --max_episode_steps 128
python code/pogema_bench/run_benchmark.py --algo matslp --num_agents 16 --map_name random --seed 0 --max_episode_steps 128
```

## Notes

- **Fairness:** LaCAM is a centralized *fixed-target* MAPF solver. MATS-LP is designed for lifelong MAPF, but here we run it in a comparable step-by-step POGEMA setting.
- For stronger scientific claims, keep seeds/maps identical and report confidence intervals.
