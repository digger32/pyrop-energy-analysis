"""Descriptive statistics, goodness-of-fit, and stationarity tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


def basic_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute descriptive statistics for each column of df.

    Returns a DataFrame indexed by variable name with columns: n, mean, std,
    median, p05, p95, min, max, skew, kurt, cv (coefficient of variation).

    Pearson skewness and excess kurtosis follow SciPy conventions
    (`scipy.stats.skew` with bias=False, `scipy.stats.kurtosis` with
    fisher=True).
    """
    out = []
    for col in df.columns:
        s = df[col].dropna()
        if len(s) == 0:
            continue
        out.append({
            "var":    col,
            "n":      len(s),
            "mean":   float(s.mean()),
            "std":    float(s.std()),
            "median": float(s.median()),
            "p05":    float(s.quantile(0.05)),
            "p95":    float(s.quantile(0.95)),
            "min":    float(s.min()),
            "max":    float(s.max()),
            "skew":   float(scipy_stats.skew(s, bias=False)),
            "kurt":   float(scipy_stats.kurtosis(s, fisher=True, bias=False)),
            "cv":     float(s.std() / s.mean()) if s.mean() != 0 else np.nan,
        })
    return pd.DataFrame(out)


def confidence_interval(s: pd.Series, alpha: float = 0.05) -> tuple[float, float]:
    """Two-sided confidence interval of the mean (Student's t).

    Parameters
    ----------
    s : pd.Series
        Sample.
    alpha : float
        Significance level (default 0.05 for 95% CI).
    """
    s = s.dropna()
    n = len(s)
    if n < 2:
        return (float("nan"), float("nan"))
    mean = float(s.mean())
    se = float(s.std(ddof=1)) / np.sqrt(n)
    t_crit = scipy_stats.t.ppf(1 - alpha / 2, n - 1)
    return (mean - t_crit * se, mean + t_crit * se)


def goodness_of_fit_tests(s: pd.Series) -> pd.DataFrame:
    """Apply four goodness-of-fit tests against a fitted normal distribution.

    Tests: Shapiro-Wilk, Kolmogorov-Smirnov (against fitted normal),
    Anderson-Darling, D'Agostino-Pearson (omnibus).
    """
    s = s.dropna()
    rows = []
    if len(s) >= 3:
        try:
            sw = scipy_stats.shapiro(s)
            rows.append({"test": "Shapiro-Wilk", "stat": float(sw.statistic), "pvalue": float(sw.pvalue)})
        except Exception:
            pass
        try:
            ks = scipy_stats.kstest(s, "norm", args=(s.mean(), s.std(ddof=1)))
            rows.append({"test": "Kolmogorov-Smirnov", "stat": float(ks.statistic), "pvalue": float(ks.pvalue)})
        except Exception:
            pass
        try:
            ad = scipy_stats.anderson(s, dist="norm")
            rows.append({"test": "Anderson-Darling", "stat": float(ad.statistic),
                         "pvalue_5pct": float(ad.critical_values[2])})
        except Exception:
            pass
        try:
            dp = scipy_stats.normaltest(s)
            rows.append({"test": "D'Agostino-Pearson", "stat": float(dp.statistic), "pvalue": float(dp.pvalue)})
        except Exception:
            pass
    return pd.DataFrame(rows)


def stationarity_tests(s: pd.Series) -> pd.DataFrame:
    """Augmented Dickey-Fuller and KPSS stationarity tests.

    The two tests have opposite null hypotheses: ADF tests for a unit root
    (rejection means stationary); KPSS tests for trend-stationarity (rejection
    means non-stationary). Reporting both is the standard practice.
    """
    from statsmodels.tsa.stattools import adfuller, kpss

    s = s.dropna()
    rows = []
    if len(s) >= 24:
        try:
            adf = adfuller(s, autolag="AIC")
            rows.append({"test": "ADF", "stat": float(adf[0]), "pvalue": float(adf[1]),
                         "crit_5pct": float(adf[4]["5%"]), "stationary": adf[1] < 0.05})
        except Exception:
            pass
        try:
            kp = kpss(s, regression="c", nlags="auto")
            rows.append({"test": "KPSS", "stat": float(kp[0]), "pvalue": float(kp[1]),
                         "crit_5pct": float(kp[3]["5%"]), "stationary": kp[1] >= 0.05})
        except Exception:
            pass
    return pd.DataFrame(rows)
