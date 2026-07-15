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


def _synthetic_panels(n_hours: int = 2000):
    """Two-incomer + two-generator synthetic hourly panels."""
    idx = pd.date_range("2023-01-01", periods=n_hours, freq="h")
    rng = np.random.default_rng(0)
    p = pd.DataFrame({
        "bus_1_july":      rng.normal(2000, 300, n_hours).clip(min=100),
        "bus_2_july":      rng.normal(1700, 300, n_hours).clip(min=100),
        "gen_cell_2_july": rng.normal(5000, 400, n_hours).clip(min=100),
        "gen_cell_23_july": rng.normal(5500, 400, n_hours).clip(min=100),
    }, index=idx)
    # bus_2 at a deep-lagging power factor, everything else near unity
    q = pd.DataFrame({
        "bus_1_july":      p["bus_1_july"] * 0.35,
        "bus_2_july":      p["bus_2_july"] * 2.2,
        "gen_cell_2_july": p["gen_cell_2_july"] * 0.08,
        "gen_cell_23_july": p["gen_cell_23_july"] * 0.10,
    }, index=idx)
    return p, q


def test_reserves_total():
    from pyrop_energy.reserves import compute_reserves
    panel_p, panel_q = _synthetic_panels()
    res = compute_reserves(panel_p, panel_q)
    total_row = res.iloc[-1]
    assert total_row["reserve"] == "TOTAL"
    # Structural invariants (values are synthetic-data dependent):
    assert res.shape[0] == 4                       # three reserves + TOTAL
    assert total_row["saving_kwh_per_year"] == pytest.approx(
        res.iloc[:-1]["saving_kwh_per_year"].sum())
    assert total_row["saving_kwh_per_year"] > 0
    assert "annual_consumption_kwh" in res.attrs
    # The deep-lagging incomer must dominate the power-factor reserve:
    assert "bus_2" in res.iloc[0]["reserve"]
    # CO2 follows the emission factor identically:
    assert total_row["co2_avoided_t_per_year"] == pytest.approx(
        total_row["saving_kwh_per_year"] * 0.4 / 1000.0)


def test_reserves_reproduce_manuscript_headline():
    """On the deposited 2023 panels the totals must equal the manuscript.

    Skipped when the deposited panel CSVs are not present (e.g. bare CI).
    """
    from pathlib import Path
    from pyrop_energy.reserves import compute_reserves
    tables = Path(__file__).resolve().parents[1] / "output" / "tables"
    p_csv = tables / "panel_P_hourly.csv"
    q_csv = tables / "panel_Q_hourly.csv"
    if not (p_csv.exists() and q_csv.exists()):
        pytest.skip("deposited hourly panels not present")
    panel_p = pd.read_csv(p_csv, index_col=0, parse_dates=True)
    panel_q = pd.read_csv(q_csv, index_col=0, parse_dates=True)
    res = compute_reserves(panel_p, panel_q)
    total = res.iloc[-1]
    assert res.attrs["annual_consumption_kwh"] / 1e6 == pytest.approx(126.7, abs=0.05)
    assert total["saving_kwh_per_year"] / 1e6 == pytest.approx(1.22, abs=0.01)
    assert total["co2_avoided_t_per_year"] == pytest.approx(487, abs=1)


def test_feature_matrix_excludes_contemporaneous_target_quantities():
    """Leakage policy: same-hour Q/S/pf of the target must not be features."""
    from pyrop_energy.features import build_feature_matrix
    idx = pd.date_range("2023-01-01", periods=600, freq="h")
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"P": rng.normal(2000, 300, 600),
                       "Q": rng.normal(700, 100, 600)}, index=idx)
    df.attrs["source"] = "bus_x_july"
    fm = build_feature_matrix(df, "hourly", "P")
    assert "Q" not in fm.columns
    assert "S" not in fm.columns
    assert "pf" not in fm.columns
    assert any(c.startswith("Q_lag_") for c in fm.columns)  # lags stay
    # Nowcasting variant keeps the raw contemporaneous Q only:
    fm_now = build_feature_matrix(df, "hourly", "P",
                                  include_contemporaneous_q=True)
    assert "Q" in fm_now.columns
    assert "S" not in fm_now.columns and "pf" not in fm_now.columns


def test_gpu_detection_returns_str_or_none():
    """detect_gpu must not raise; it returns either the device name or None."""
    from pyrop_energy.ml import detect_gpu
    result = detect_gpu()
    assert result is None or isinstance(result, str)
