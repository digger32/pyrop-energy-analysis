"""Smoke tests: validate that the package imports and runs on synthetic data.

These tests do not validate scientific correctness; they ensure that the
public API does not break under typical CI conditions.
"""

import numpy as np
import pandas as pd
import pytest


def test_imports():
    """Every module must import without errors."""
    from pyrop_energy import parsers, stats, features, ml, asymmetry, reserves
    assert parsers is not None
    assert stats is not None
    assert features is not None
    assert ml is not None
    assert asymmetry is not None
    assert reserves is not None


def test_basic_stats_on_synthetic_data():
    np.random.seed(0)
    df = pd.DataFrame({
        "P": np.random.normal(2000, 500, 100),
        "Q": np.random.normal(400, 200, 100),
    })
    from pyrop_energy.stats import basic_stats
    s = basic_stats(df)
    assert len(s) == 2
    assert "mean" in s.columns
    assert s.loc[s["var"] == "P", "mean"].iloc[0] == pytest.approx(2000, abs=200)


def test_feature_matrix_on_synthetic_data():
    """build_feature_matrix should produce a non-empty matrix when the panel
    has full coverage on the target index."""
    idx = pd.date_range("2024-01-01", periods=500, freq="h")
    df = pd.DataFrame({"P": np.random.normal(2000, 500, 500),
                       "Q": np.random.normal(400, 200, 500)}, index=idx)
    df.attrs["source"] = "test_source"
    panel_p = pd.DataFrame({"test_source": df["P"], "other": df["P"] * 0.8}, index=idx)
    panel_q = pd.DataFrame({"test_source": df["Q"], "other": df["Q"] * 0.8}, index=idx)

    from pyrop_energy.features import build_feature_matrix
    fm = build_feature_matrix(df, "hourly", "P", panel_p, panel_q)
    assert len(fm) > 200
    assert "y" in fm.columns


def test_orphan_panel_source_is_filtered():
    """A panel source with <50% coverage on the target index should be filtered."""
    idx = pd.date_range("2024-01-01", periods=500, freq="h")
    short_idx = pd.date_range("2024-01-15", periods=20, freq="h")  # 20/500 = 4%
    df = pd.DataFrame({"P": np.random.normal(2000, 500, 500),
                       "Q": np.random.normal(400, 200, 500)}, index=idx)
    df.attrs["source"] = "main_source"

    panel_p = pd.concat({
        "main_source": df["P"],
        "orphan":      pd.Series(np.random.normal(0, 1, 20), index=short_idx),
    }, axis=1)
    panel_q = panel_p.copy()  # same shape

    from pyrop_energy.features import build_feature_matrix
    fm = build_feature_matrix(df, "hourly", "P", panel_p, panel_q)
    # Orphan source columns should NOT appear in the feature matrix
    assert not any("orphan" in c for c in fm.columns), \
        "orphan source with <50% coverage was not filtered out"
    assert len(fm) > 200, f"feature matrix too small after filtering: {len(fm)} rows"


def test_reserves_total():
    from pyrop_energy.reserves import compute_reserves
    res = compute_reserves()
    total_row = res.iloc[-1]
    assert total_row["reserve"] == "TOTAL"
    # Total saving is approximately 4 GWh / year as reported in the manuscript.
    assert 3.0e6 < total_row["saving_kwh_per_year"] < 5.0e6
    # CO2 avoidance approximately 1500 t / year.
    assert 1200 < total_row["co2_avoided_t_per_year"] < 2000


def test_gpu_detection_returns_str_or_none():
    """detect_gpu must not raise; it returns either the device name or None."""
    from pyrop_energy.ml import detect_gpu
    result = detect_gpu()
    assert result is None or isinstance(result, str)
