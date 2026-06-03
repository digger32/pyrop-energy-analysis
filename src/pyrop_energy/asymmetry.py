"""Asymmetric coupling between site load and on-site cogeneration.

Two complementary tests are used:

1. Bivariate Granger-causality (load -> cogeneration) at lags 1, 3, 6, 12, 24
   hours. The F-statistic of the SSR test is reported, as implemented in
   `statsmodels.tsa.stattools.grangercausalitytests`.

2. Sign-conditional lagged cross-correlation: the Pearson correlation between
   $\\Delta P_\\mathrm{load}(t)$ restricted to $\\{\\Delta P_\\mathrm{load} > 0\\}$
   or $\\{\\Delta P_\\mathrm{load} < 0\\}$, and $\\Delta P_\\mathrm{gen}(t + L)$
   for $L \\in \\{0, 1, ..., 24\\}$.

Combined, the two tests detect both linear-predictability asymmetry (Granger)
and sign-conditional response asymmetry (cross-correlation), the latter being
the more direct empirical observable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def aggregate_load_and_gen(panel_p: pd.DataFrame,
                           load_cols: list[str],
                           gen_cols: list[str]) -> tuple[pd.Series, pd.Series]:
    """Sum the active power of the named source columns into load and gen."""
    load = panel_p[load_cols].sum(axis=1, min_count=1)
    gen  = panel_p[gen_cols].sum(axis=1, min_count=1)
    return load.rename("P_load"), gen.rename("P_gen")


def granger_load_to_gen(load: pd.Series, gen: pd.Series,
                        lags: list[int] = (1, 3, 6, 12, 24)) -> pd.DataFrame:
    """Bivariate Granger-causality (load -> gen) at the given lags.

    Returns a DataFrame with columns: lag, F, pvalue.
    """
    from statsmodels.tsa.stattools import grangercausalitytests

    aligned = pd.concat([gen.diff(), load.diff()], axis=1).dropna()
    aligned.columns = ["dY", "dX"]
    rows = []
    for lag in lags:
        # statsmodels expects [Y, X] for testing whether X Granger-causes Y
        try:
            res = grangercausalitytests(aligned[["dY", "dX"]], maxlag=lag, verbose=False)
            ssr_F, ssr_p = res[lag][0]["ssr_ftest"][0:2]
            rows.append({"lag": lag, "F": float(ssr_F), "pvalue": float(ssr_p)})
        except Exception as exc:
            rows.append({"lag": lag, "F": float("nan"), "pvalue": float("nan")})
    return pd.DataFrame(rows)


def sign_conditional_cross_correlation(load: pd.Series, gen: pd.Series,
                                       max_lag: int = 24,
                                       min_pairs: int = 10) -> pd.DataFrame:
    """Sign-conditional lagged cross-correlation between dload and dgen.

    For each lag L in [0, max_lag] and each sign condition (dload > 0 vs
    dload < 0), compute the Pearson correlation between dload(t) and
    dgen(t + L). Only timestamps with at least `min_pairs` valid pairs are
    retained.

    Returns a DataFrame with columns: lag, corr_pos_increment,
    corr_neg_increment, n_pos, n_neg.
    """
    dload = load.diff()
    dgen  = gen.diff()
    rows = []
    for L in range(max_lag + 1):
        gen_lag = dgen.shift(-L)
        df = pd.concat([dload, gen_lag], axis=1).dropna()
        df.columns = ["dload", "dgen"]

        pos = df[df["dload"] > 0]
        neg = df[df["dload"] < 0]

        cor_pos = float(pos.corr().iloc[0, 1]) if len(pos) >= min_pairs else float("nan")
        cor_neg = float(neg.corr().iloc[0, 1]) if len(neg) >= min_pairs else float("nan")

        rows.append({"lag": L, "corr_pos_increment": cor_pos, "corr_neg_increment": cor_neg,
                     "n_pos": len(pos), "n_neg": len(neg)})
    return pd.DataFrame(rows)
