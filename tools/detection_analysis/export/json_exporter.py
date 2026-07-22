"""JSON export helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tools.detection_analysis.utils.serialization import dump_json


def export_json(path: str | Path, value: Any) -> None:
    """Export any JSON-compatible analysis value."""

    dump_json(path, value)

