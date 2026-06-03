"""Feature engineering for the supervised-learning task.

The feature matrix combines:
1. Calendar features (year, quarter, month, day-of-month, day-of-year, weekday,
   weekend indicator, hour-of-day for hourly series).
2. Cyclic sine-cosine encodings of hour, weekday, and month.
3. Lagged values of the target and of all panel sources at multiple lags.
4. Rolling means and standard deviations (one-step lagged to avoid leakage).
5. Panel-aggregate features (P_all, Q_all, S_all, pf_all) summed across all
   sources at each timestamp.

Important: panel sources with hourly coverage below 50% on the target index
are filtered out before lag construction. This prevents spurious all-NaN
columns that would eliminate the target series after the final dropna() step.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_LAGS_HOURLY = (1, 6, 12, 24, 48, 72, 96, 120, 144, 168)
DEFAULT_LAGS_MONTHLY = (1, 2, 3, 6, 12)
DEFAULT_ROLL_HOURLY  = (3, 6, 24, 168)
DEFAULT_ROLL_MONTHLY = (3, 6, 12)


def add_calendar_features(df: pd.DataFrame, granularity: str) -> pd.DataFrame:
    """Add calendar columns and cyclic sine/cosine encodings."""
    out = df.copy()
    idx = out.index
    out["year"]       = idx.year
    out["quarter"]    = idx.quarter
    out["month"]      = idx.month
    out["day"]        = idx.day
    out["day_of_year"]= idx.dayofyear
    out["weekday"]    = idx.weekday
    out["is_weekend"] = (idx.weekday >= 5).astype(int)
    if granularity == "hourly":
        out["hour"] = idx.hour
        out["sin_h"] = np.sin(2 * np.pi * idx.hour / 24)
        out["cos_h"] = np.cos(2 * np.pi * idx.hour / 24)
    out["sin_d"] = np.sin(2 * np.pi * idx.weekday / 7)
    out["cos_d"] = np.cos(2 * np.pi * idx.weekday / 7)
    out["sin_m"] = np.sin(2 * np.pi * (idx.month - 1) / 12)
    out["cos_m"] = np.cos(2 * np.pi * (idx.month - 1) / 12)
    return out


def add_lags(df: pd.DataFrame, cols=("P", "Q"), lags=None, granularity="hourly") -> pd.DataFrame:
    """Add lagged versions of the named columns."""
    if lags is None:
        lags = DEFAULT_LAGS_HOURLY if granularity == "hourly" else DEFAULT_LAGS_MONTHLY
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            continue
        for L in lags:
            out[f"{col}_lag_{L}"] = out[col].shift(L)
    return out


def add_rolling(df: pd.DataFrame, cols=("P", "Q"), windows=None, granularity="hourly") -> pd.DataFrame:
    """Add rolling means and stds (one-step lagged to avoid leakage)."""
    if windows is None:
        windows = DEFAULT_ROLL_HOURLY if granularity == "hourly" else DEFAULT_ROLL_MONTHLY
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            continue
        shifted = out[col].shift(1)
        for w in windows:
            out[f"{col}_roll_mean_{w}"] = shifted.rolling(w, min_periods=max(2, w // 2)).mean()
            out[f"{col}_roll_std_{w}"]  = shifted.rolling(w, min_periods=max(2, w // 2)).std()
    return out


def assemble_panel(loaded: dict, granularity: str, var: str) -> pd.DataFrame:
    """Stack one variable (P or Q) from all sources into a wide panel.

    Parameters
    ----------
    loaded : dict
        Mapping (entity, period, granularity) -> DataFrame with P, Q columns.
    granularity : str
        'hourly' or 'monthly' — selects sources with the matching resolution.
    var : str
        Variable to assemble ('P' or 'Q').

    Returns
    -------
    pd.DataFrame
        Wide panel with one column per source, indexed by timestamp (outer-joined).
    """
    cols = {}
    for (ent, per, gran), df in loaded.items():
        if gran != granularity or var not in df.columns:
            continue
        label = f"{ent}_{per}"
        cols[label] = df[var]
    return pd.concat(cols, axis=1)


def build_feature_matrix(df: pd.DataFrame,
                         granularity: str = "hourly",
                         target_col: str = "P",
                         panel_p: pd.DataFrame | None = None,
                         panel_q: pd.DataFrame | None = None,
                         lags=None,
                         add_other_buses: bool = True,
                         drop_na: bool = True,
                         min_panel_coverage: float = 0.5,
                         verbose: bool = False) -> pd.DataFrame:
    """Build the supervised-learning feature matrix for one source.

    Strategy
    --------
    1. Calendar features and cyclic encodings.
    2. Lagged values of the target's own P and Q.
    3. Rolling means and stds (one-step lagged).
    4. Panel sources are reindexed onto the target's timestamp index.
    5. Panel sources with hourly coverage below `min_panel_coverage` on the
       target index are filtered out as 'orphans' (this avoids spurious
       all-NaN columns after lag construction).
    6. Lags of the surviving panel sources are added.
    7. Final dropna() removes rows with any missing value.

    Returns
    -------
    pd.DataFrame
        Feature matrix with the target column renamed to "y".
    """
    fm = add_calendar_features(df, granularity)
    fm = add_lags(fm, ("P", "Q"), lags, granularity)
    fm = add_rolling(fm, ("P", "Q"), None, granularity)

    chosen_lags = lags or (DEFAULT_LAGS_HOURLY if granularity == "hourly" else DEFAULT_LAGS_MONTHLY)

    n_panel_kept = 0
    n_panel_dropped = 0

    if add_other_buses and panel_p is not None:
        own = df.attrs.get("source")
        panel_p_aligned = panel_p.reindex(df.index)
        panel_q_aligned = panel_q.reindex(df.index) if panel_q is not None else None

        good_p = [c for c in panel_p_aligned.columns
                  if c != own and panel_p_aligned[c].notna().mean() >= min_panel_coverage]
        n_panel_kept += len(good_p)
        n_panel_dropped += len([c for c in panel_p_aligned.columns
                                if c != own and c not in good_p])

        for col in good_p:
            for L in chosen_lags:
                fm[f"{col}_P_lag_{L}"] = panel_p_aligned[col].shift(L)

        if panel_q_aligned is not None:
            good_q = [c for c in panel_q_aligned.columns
                      if c != own and panel_q_aligned[c].notna().mean() >= min_panel_coverage]
            n_panel_kept += len(good_q)
            n_panel_dropped += len([c for c in panel_q_aligned.columns
                                    if c != own and c not in good_q])

            for col in good_q:
                for L in chosen_lags:
                    fm[f"{col}_Q_lag_{L}"] = panel_q_aligned[col].shift(L)

    if verbose:
        print(f"  panel cols kept: {n_panel_kept}, dropped (low coverage): {n_panel_dropped}")

    fm = fm.rename(columns={target_col: "y"})
    return fm.dropna() if drop_na else fm
