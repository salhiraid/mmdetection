"""Plotly chart factories."""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
import plotly.express as px


def metric_bar(rows: List[Dict[str, Any]], metric: str):
    """Create a detector metric bar chart."""

    return px.bar(pd.DataFrame(rows), x="detector", y=metric, color="detector")


def status_counts_bar(counts: Dict[str, int], title: str):
    """Create an error/status count chart."""

    df = pd.DataFrame({"status": list(counts.keys()), "count": list(counts.values())})
    return px.bar(df, x="status", y="count", title=title)

