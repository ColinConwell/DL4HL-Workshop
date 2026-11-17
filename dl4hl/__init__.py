"""DL-4-HL: interactive widgets, simulators, and dataset loaders for the
Deep Learning for Human Learners tutorial."""

__version__ = "0.1.0"

from .plotting import COLORS, PALETTE, set_style  # noqa: E402
from .theming import enable_theme_plots  # noqa: E402
from .widgets import list_widgets, manifest, reset_assets, show, widget_html  # noqa: E402
from .data import (  # noqa: E402
    ensure_deepdive,
    load_dollar_street,
    load_household_survey,
    load_kiki_bouba,
    load_music_emotions,
    load_musiclab,
    load_oasis,
    load_perinatal_dev,
    load_sarcasm,
    load_simple_shapes,
    load_wordbank,
)

__all__ = [
    "show",
    "widget_html",
    "list_widgets",
    "manifest",
    "reset_assets",
    "set_style",
    "enable_theme_plots",
    "COLORS",
    "PALETTE",
    "__version__",
    "ensure_deepdive",
    "load_simple_shapes",
    "load_oasis",
    "load_kiki_bouba",
    "load_sarcasm",
    "load_dollar_street",
    "load_household_survey",
    "load_perinatal_dev",
    "load_music_emotions",
    "load_musiclab",
    "load_wordbank",
]
