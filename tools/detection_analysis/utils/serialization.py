"""Serialization helpers for dataclass-heavy analysis payloads."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def to_plain(value: Any) -> Any:
    """Convert dataclasses, numpy scalars, and paths to JSON-compatible values."""

    if is_dataclass(value):
        return to_plain(asdict(value))
    if isinstance(value, dict):
        return {str(k): to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def dump_json(path: str | Path, value: Any) -> None:
    """Write a JSON file with stable formatting."""

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(to_plain(value), f, indent=2, sort_keys=True)


def load_json(path: str | Path) -> Any:
    """Read a JSON file."""

    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)

