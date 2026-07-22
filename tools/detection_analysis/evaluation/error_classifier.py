"""Error category documentation helpers.

The actual classification is implemented in ``matcher.py`` so matching and
status assignment remain one deterministic pass.
"""

from __future__ import annotations

from tools.detection_analysis.constants import PREDICTION_ERROR_ORDER


def matching_priority() -> list[str]:
    """Return the primary prediction status decision order."""

    return list(PREDICTION_ERROR_ORDER)

