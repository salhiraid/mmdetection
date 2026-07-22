"""Validation helpers and explicit exceptions."""

from __future__ import annotations


class DetectionAnalysisError(RuntimeError):
    """Base exception for this tool."""


class ResultFormatError(DetectionAnalysisError):
    """Raised when a detector output cannot be normalized."""


class DatasetLoadError(DetectionAnalysisError):
    """Raised when dataset metadata or annotations cannot be loaded."""


def require(condition: bool, message: str) -> None:
    """Raise a user-facing validation error when ``condition`` is false."""

    if not condition:
        raise DetectionAnalysisError(message)

