"""pyrop_energy — multi-year electricity-consumption analysis of an ore-preparation plant.

This package contains the analysis pipeline accompanying the manuscript
"Multi-year electricity-consumption analysis and machine-learning forecasting
of an ore-preparation plant integrated with on-site cogeneration".

Modules
-------
parsers     : SCADA Excel + SpreadsheetML 2003 + power-quality file loaders.
stats       : descriptive statistics, goodness-of-fit, stationarity.
features    : calendar / lag / rolling / panel features.
ml          : 18-model rolling-origin benchmark with optional GPU.
asymmetry   : Granger causality + sign-conditional cross-correlation.
reserves    : the three energy-saving reserves (pf, VFD idle, transformers).
plots       : Matplotlib figures for the manuscript.

Public API
----------
The package is intended to be used either through the helper scripts in
`scripts/` or by calling the functions directly from a notebook. See the
README for a one-page reproduction recipe.
"""

__version__ = "1.0.0"
__all__ = [
    "parsers", "stats", "features", "ml",
    "asymmetry", "reserves", "plots",
]
