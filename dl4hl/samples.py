"""Real-data payloads for the interactive DL-4-HL widgets."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from .data import artifact

ASSETS = Path(__file__).absolute().parent / "assets"  # not resolve(): Hub snapshots are symlinks into blobs/

_GALLERY_LABELS = {
    "activation_splines": "Wordbank vocabulary",
    "llm_explainer": "GPT-2 (12 heads)",
    "sae_superposition": "Sarcasm DistilBERT SAE",
    "relative_reps": "Kiki-Bouba OpenCLIP",
    "layerwise_readout": "OASIS + SimpleShapes ViT",
}


@lru_cache(maxsize=1)
def _all_samples() -> dict:
    data = dict(json.loads(artifact("widget_samples.json").read_text()))
    llm_local = ASSETS / "llm_gpt2_cases.json"
    if llm_local.exists():
        data["llm_explainer"] = json.loads(llm_local.read_text())
    return data


def widget_sample(name: str) -> dict:
    """Return the precomputed real-data payload for widget ``name``."""
    data = _all_samples()
    if name not in data:
        raise KeyError(f"No precomputed sample for {name!r}. Available: {', '.join(sorted(data))}")
    return data[name]


def gallery_samples() -> dict:
    """Return labeled real-data widget payloads for ``DL4HL-Widgets.html``."""
    raw = _all_samples()
    return {
        name: {"label": _GALLERY_LABELS.get(name, name), "data": payload}
        for name, payload in raw.items()
    }
