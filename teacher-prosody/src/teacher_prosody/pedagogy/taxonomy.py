from __future__ import annotations

from functools import lru_cache
from importlib import resources

import yaml


@lru_cache(maxsize=1)
def load_taxonomy() -> dict:
    with resources.files("teacher_prosody").joinpath("data/taxonomy.yaml").open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def beat_names() -> list[str]:
    return list(load_taxonomy()["beats"].keys())
