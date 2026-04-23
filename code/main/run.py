#!/usr/bin/env python3
"""
CLI entry point for the Local Leaders MAPF algorithm.

Usage examples
--------------
  # Random 32x32 map, 20 agents, show map, save JSON
  python -m local_leaders.run --gen-random 32 32 --num-agents 20 --show-map --output-json result.json

  # Room map, clustered start placement
  python -m local_leaders.run --gen-rooms 40 40 --room-size 6 --num-agents 30 --placement clustered

  # MovingAI scenario
  python -m local_leaders.run --movingai-map path/to/map.map --movingai-scen path/to/scen.scen --num-agents 50
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path as FilePath

from .core.algorithm import LocalLeadersMAPF
from .core.mapf_instance import MAPFInstance
from .generation.map_generator import (
    empty_map,
    get_map_info,
    map_to_str,
    maze_map,
    place_agents,
    place_agents_clustered,
    random_map,
    room_map,
)
from .core.types import LocalLeadersConfig

# Define CLI flags
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m local_leaders.run",
        description="Local Leaders MAPF solver",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Map source (mutually exclusive options):
    # .map file, empty, random, random room, random maze
    map_src = parser.add_mutually_exclusive_group(required=True)
    map_src.add_argument(
        "--movingai-map", metavar="PATH",
        help="Path to a MovingAI .map file",
    )
    map_src.add_argument(
        "--gen-empty", nargs=2, type=int, metavar=("H", "W"),
        help="Generate an empty H×W map",
    )
    map_src.add_argument(
        "--gen-random", nargs=2, type=int, metavar=("H", "W"),
        help="Generate a random H×W map",
    )
    map_src.add_argument(
        "--gen-rooms", nargs=2, type=int, metavar=("H", "W"),
        help="Generate a room-based H×W map",
    )
    map_src.add_argument(
        "--gen-maze", nargs=2, type=int, metavar=("H", "W"),
        help="Generate a maze H×W map (recursive backtracking)",
    )

    # .scen that pairs with .map file (MovingAI)
    parser.add_argument(
        "--movingai-scen", metavar="PATH",
        help="Path to a MovingAI .scen file (required with --movingai-map)",
    )

    # Total agent count
    parser.add_argument("--num-agents", type=int, default=10, metavar="N",
                   help="Number of agents")
    
    # Agent positions
    parser.add_argument(
        "--placement", choices=["random", "clustered"], default="random",
        help="Agent placement strategy for generated maps",
    )

    # Clustering parameters for agent placement = clustered
    parser.add_argument("--num-clusters", type=int, default=4, metavar="K",
                   help="Number of clusters for --placement clustered")
    parser.add_argument("--cluster-radius", type=int, default=3,
                   help="Cell radius per cluster for clustered placement")

    # Map generation parameters --
    parser.add_argument("--density", type=float, default=0.2,
                   help="Obstacle density for --gen-random (0.0–1.0)")
    parser.add_argument("--room-size", type=int, default=5,
                   help="Room size for --gen-rooms")
    parser.add_argument("--seed", type=int, default=0,
                   help="Random seed")

    # Algorithm config
    parser.add_argument("--group-radius", type=int, default=5,
                   help="Chebyshev radius for clustering agents into groups")
    parser.add_argument("--max-group-size", type=int, default=10,
                   help="Maximum agents per group")
    parser.add_argument("--leader-view-radius", type=int, default=3,
                   help="Extra cells the leader can see beyond its agents")
    parser.add_argument("--time-limit", type=float, default=60.0,
                   help="Solve time limit in seconds")

    # Output
    parser.add_argument("--show-map", action="store_true",
                   help="Print ASCII map before solving")
    parser.add_argument("--output-json", metavar="PATH",
                   help="Write result JSON to this file")

    return parser

# Builds an object containing the problem from CLI flags
def build_instance(args: argparse.Namespace) -> MAPFInstance:
    if args.movingai_map:
        if not args.movingai_scen:
            print("ERROR: --movingai-map requires --movingai-scen", file=sys.stderr)
            sys.exit(1)
        return MAPFInstance.from_movingai(args.movingai_map, args.movingai_scen, args.num_agents)

    if args.gen_empty:
        h, w = args.gen_empty
        grid = empty_map(h, w)
    elif args.gen_random:
        h, w = args.gen_random
        grid = random_map(h, w, obstacle_density=args.density, seed=args.seed)
    elif args.gen_rooms:
        h, w = args.gen_rooms
        grid = room_map(h, w, room_size=args.room_size, seed=args.seed)
    elif args.gen_maze:
        h, w = args.gen_maze
        grid = maze_map(h, w, seed=args.seed)
    else:
        print("ERROR: no map source specified", file=sys.stderr)
        sys.exit(1)

    if args.placement == "clustered":
        starts, goals = place_agents_clustered(
            grid,
            args.num_agents,
            num_clusters=args.num_clusters,
            cluster_radius=args.cluster_radius,
            seed=args.seed,
        )
    else:
        starts, goals = place_agents(grid, args.num_agents, seed=args.seed)

    return MAPFInstance.from_arrays(grid, starts, goals)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    instance = build_instance(args)

    if args.show_map:
        print(map_to_str(instance.grid, agents=instance.agents))
        stats = get_map_info(instance.grid)
        print(
            f"\nGrid: {stats['height']}×{stats['width']}  "
            f"free={stats['free_cells']}  obstacles={stats['obstacle_cells']}  "
            f"density={stats['obstacle_density']}  agents={instance.num_agents}"
        )
        print()

    errors = instance.validate()
    if errors:
        for e in errors:
            print(f"Validation error: {e}", file=sys.stderr)
        sys.exit(1)

    config = LocalLeadersConfig(
        group_radius=args.group_radius,
        max_group_size=args.max_group_size,
        leader_view_radius=args.leader_view_radius,
        time_limit_sec=args.time_limit,
        seed=args.seed,
    )

    print(
        f"Solving: {instance.num_agents} agents on "
        f"{instance.height}×{instance.width} grid  "
        f"[group_radius={config.group_radius}, max_group_size={config.max_group_size}]"
    )

    try:
        solver = LocalLeadersMAPF(instance, config)
        result = solver.solve()
    except NotImplementedError as exc:
        print(f"\nNot yet implemented: {exc}", file=sys.stderr)
        sys.exit(2)

    output = {
        **result.to_dict(),
        "num_agents": instance.num_agents,
        "map_size": f"{instance.height}x{instance.width}",
        "config": {
            "group_radius": config.group_radius,
            "max_group_size": config.max_group_size,
            "leader_view_radius": config.leader_view_radius,
            "time_limit_sec": config.time_limit_sec,
            "seed": config.seed,
        },
    }

    print(json.dumps(output, indent=2))

    if args.output_json:
        out_path = FilePath(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, indent=2))
        print(f"\nResult saved to {args.output_json}")


if __name__ == "__main__":
    main()
