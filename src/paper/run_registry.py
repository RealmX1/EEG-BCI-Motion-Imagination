"""Load the paper run/path registry from ``paper/run_registry.yaml``."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = PROJECT_ROOT / "paper" / "run_registry.yaml"


@lru_cache(maxsize=1)
def _load_registry_data() -> Dict:
    with open(REGISTRY_PATH, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    runs = data.get("runs")
    if not isinstance(runs, dict):
        raise ValueError(f"Invalid run registry format: missing 'runs' mapping in {REGISTRY_PATH}")
    return data


def load_run_entries() -> Dict[str, Dict[str, str]]:
    """Return the full run metadata mapping keyed by logical run name."""
    runs = _load_registry_data()["runs"]
    return {key: dict(value) for key, value in runs.items()}


def load_run_registry() -> Dict[str, str]:
    """Return a key -> relative result path mapping for convenience."""
    return {key: value["path"] for key, value in load_run_entries().items()}


def get_run_entry(key: str) -> Dict[str, str]:
    """Return one registry entry by key."""
    runs = load_run_entries()
    if key not in runs:
        raise KeyError(f"Unknown paper run key: {key}")
    return runs[key]


def get_run_path(key_or_path: str) -> str:
    """Resolve either a registry key or an already-explicit path string."""
    runs = load_run_registry()
    return runs.get(key_or_path, key_or_path)


def format_run_source(key: str) -> str:
    """Format a registry entry for user-facing provenance text."""
    entry = get_run_entry(key)
    return f"run `{entry['run_tag']}`: `{entry['path']}`"


def resolve_project_path(key_or_path: str) -> Path:
    """Return an absolute path from either a registry key or a repo-relative path."""
    path = Path(get_run_path(key_or_path))
    return path if path.is_absolute() else PROJECT_ROOT / path

