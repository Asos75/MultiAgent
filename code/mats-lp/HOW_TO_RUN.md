# WINDOWS


# Usual program

## SETUP

Open Docker Desktop

Run these in the project root:

docker pull tviskaron/mats-lp
docker tag tviskaron/mats-lp mats-lp

## RUN

docker run --rm -ti -v ${PWD}:/code -w /code mats-lp python3 main.py

After the program finishes you a simulation of it should be created in /rendersž


# Benchmarks - paper reproduction

Reproduces the throughput results from the MATS-LP paper (Figures 3, 4, 5) across random maps, maze maps, and the warehouse map.

## Verify available map names (do this first)

```bash
docker run --rm -ti -v ${PWD}:/code -w /code mats-lp python3 main.py --show_map_names
```

Check that names like `pico_s24_od0_na32`, `mazes-s0_wc8_od55`, etc. appear. If the naming pattern differs, update the `CONFIGS` list at the top of `benchmark.py` accordingly.

## Quick sanity check (1 seed, fast)

```bash
docker run --rm -ti -v ${PWD}:/code -w /code mats-lp python3 benchmark.py --seeds 1
```

## Full benchmark 

```bash
docker run --rm -ti -v ${PWD}:/code -w /code mats-lp python3 benchmark.py
```

Results are saved to `results/benchmark_results.csv` with columns:
`label, map_name, num_agents, seed, throughput, elapsed_s`

Each row is one episode. Average `throughput` over seeds per label to compare against the paper figures.
