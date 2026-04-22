from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class MatsLPConfig:
    repo_root: Path
    num_expansions: int = 250
    num_threads: int = 4
    pb_c_init: float = 4.44
    obs_radius: int = 5
    collision_system: str = "soft"
    weights_path: Optional[Path] = None


class MatsLPPolicy:
    """Thin wrapper around `code/mats-lp/mcts_cpp/cppmcts.py` MCTSInference.

    Code goes here.
    """

    def __init__(self, cfg: MatsLPConfig):
        raise NotImplementedError(
            "MATS-LP integration is intentionally not implemented in `pogema_bench`."
        )

    def act(self, observations: List[Dict[str, Any]]) -> List[int]:
        raise NotImplementedError(
            "MATS-LP integration is intentionally not implemented in `pogema_bench`."
        )
