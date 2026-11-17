"""Real-data payloads for the interactive DL-4-HL widgets."""

from __future__ import annotations

import json
from functools import lru_cache

from .data import artifact


@lru_cache(maxsize=1)
def _all_samples() -> dict:
    return json.loads(artifact("widget_samples.json").read_text())


def widget_sample(name: str) -> dict:
    """Return the precomputed real-data payload for widget ``name``."""
    data = _all_samples()
    if name not in data:
        raise KeyError(f"No precomputed sample for {name!r}. Available: {', '.join(sorted(data))}")
    return data[name]


def gallery_samples() -> dict:
    """Return all precomputed real-data widget payloads for the local widget gallery."""
    return dict(_all_samples())
