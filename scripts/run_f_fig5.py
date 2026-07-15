#!/usr/bin/env python3
"""Run (f): rebuild Fig5 (ML benchmark) for the revision + honest SHAP.

The published Fig5 told the leakage-era story (R^2 in [0.993,1.000]) and
its panel (d) hardcoded shap_summary_GradBoosting.png for cell_307_july,
which no longer exists (top-1 is now the Persistence_lag1 baseline, which
SHAP cannot analyse). This script rebuilds the figure around the
corrected narrative:

  (a) best learned model vs Persistence_lag1 RMSE, all 12 combos
  (b) predicted-vs-actual, last CV fold, representative combo
  (c) R^2 of best learned model vs Persistence_lag1 (diagonal plot)
  (d) SHAP beeswarm for the best LEARNED (non-baseline) model of the
      representative combo

Usage (repo root, after the v5 run):
    python runs/run_f_fig5.py [--repo ~/Documents/pyrop-analysis] \
        [--combo cell_307_july]
Output: output/figures_final/Fig5_ml_benchmark.png (300 dpi)
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402
import pandas as pd              # noqa: E402
from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor,
                              RandomForestRegressor)  # noqa: E402
from sklearn.linear_model import BayesianRidge, HuberRegressor, Ridge  # noqa: E402
from sklearn.metrics import mean_squared_error, r2_score  # noqa: E402
from sklearn.model_selection import TimeSeriesSplit  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

warnings.filterwarnings("ignore")
SEED = 42
DEFAULT_REPO = Path.home() / "Documents/pyrop-analysis"

LEARNED = {
    "ExtraTrees":    lambda: ExtraTreesRegressor(n_estimators=300, random_state=SEED, n_jobs=-1),
    "RandomForest":  lambda: RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1),
    "GradBoosting":  lambda: GradientBoostingRegressor(random_state=SEED),
    "BayesianRidge": lambda: make_pipeline(StandardScaler(), BayesianRidge()),
    "Huber":         lambda: make_pipeline(StandardScaler(), HuberRegressor(max_iter=2000)),
    "Ridge":         lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=SEED)),
}


def load_leaderboards(ml_dir: Path) -> pd.DataFrame:
    rows = []
    for d in sorted(ml_dir.iterdir()):
        f = d / "leaderboard.csv"
        if not f.exists():
            continue
        lb = pd.read_csv(f)
        lb = lb[lb["name"] != "LinearReg"]
        p1 = lb[lb["name"] == "Persistence_lag1"].iloc[0]
        learned = lb[lb["family"] != "baseline"].iloc[0]
        rows.append({"label": d.name,
                     "learned_name": learned["name"],
                     "learned_RMSE": learned["RMSE_mean"],
                     "learned_R2": learned["R2_mean"],
                     "pers_RMSE": p1["RMSE_mean"],
                     "pers_R2": p1["R2_mean"]})
    return pd.DataFrame(rows).sort_values("pers_RMSE").reset_index(drop=True)


def build_fm(repo: Path, combo: str):
    sys.path.insert(0, str(repo / "src"))
    from pyrop_energy.features import build_feature_matrix
    tables = repo / "output/tables"
    panel_p = pd.read_csv(tables / "panel_P_hourly.csv", index_col=0, parse_dates=True)
    panel_q = pd.read_csv(tables / "panel_Q_hourly.csv", index_col=0, parse_dates=True)
    df = pd.DataFrame({"P": panel_p[combo]})
    if combo in panel_q.columns:
        df["Q"] = panel_q[combo]
    df = df.dropna(subset=["P"])
    df.attrs["source"] = combo
    fm = build_feature_matrix(df, "hourly", "P", panel_p, panel_q)
    y = fm["y"]
    X = fm.drop(columns=["y", "Q", "S", "pf"], errors="ignore")  # v5 feature set
    return X, y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    ap.add_argument("--combo", default="cell_307_july")
    args = ap.parse_args()

    ml_dir = args.repo / "output/ml"
    out_dir = args.repo / "output/figures_final"
    out_dir.mkdir(parents=True, exist_ok=True)

    t = load_leaderboards(ml_dir)
    combo_row = t[t["label"] == args.combo].iloc[0]
    model_name = combo_row["learned_name"]
    print(f"Representative combo: {args.combo}, best learned: {model_name}")

    # refit best learned model on the representative combo (last CV fold)
    X, y = build_fm(args.repo, args.combo)
    tr, te = list(TimeSeriesSplit(n_splits=5).split(X))[-1]
    mdl = LEARNED[model_name]()
    mdl.fit(X.iloc[tr], y.iloc[tr])
    pred = mdl.predict(X.iloc[te])
    fold_r2 = r2_score(y.iloc[te], pred)
    fold_rmse = mean_squared_error(y.iloc[te], pred) ** 0.5

    fig = plt.figure(figsize=(13, 9.5))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.32, height_ratios=[1, 1.15])

    # (a) paired bars: learned vs persistence
    ax = fig.add_subplot(gs[0, 0])
    ypos = np.arange(len(t))
    ax.barh(ypos - 0.2, t["pers_RMSE"], height=0.4, color="#888",
            label="Persistence (lag 1)")
    ax.barh(ypos + 0.2, t["learned_RMSE"], height=0.4, color="#2ca02c",
            label="Best learned model")
    ax.set_yticks(ypos)
    ax.set_yticklabels([s.replace("_", " ") for s in t["label"]], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("RMSE, kW (mean across 5 folds)")
    ax.set_title("(a) Best learned model vs lag-1 persistence")
    ax.legend(fontsize=8, loc="lower right")

    # (b) pred vs actual
    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(y.iloc[te], pred, s=4, alpha=0.35, color="#2ca02c")
    lims = [min(y.iloc[te].min(), pred.min()), max(y.iloc[te].max(), pred.max())]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set_xlabel("Actual P, kW")
    ax.set_ylabel("Predicted P, kW")
    ax.set_title(f"(b) {args.combo.replace('_',' ')}: {model_name}, last fold\n"
                 f"R\u00b2={fold_r2:.3f}, RMSE={fold_rmse:.0f} kW")

    # (c) R2 diagonal: persistence vs learned
    ax = fig.add_subplot(gs[1, 0])
    ax.scatter(t["pers_R2"], t["learned_R2"], color="#1f77b4", s=45, zorder=3)
    lims = [0.2, 0.85]
    ax.plot(lims, lims, "k--", lw=1)
    for _, r in t.iterrows():
        ax.annotate(r["label"].replace("_december", " D").replace("_july", " J")
                    .replace("gen_cell", "gen").replace("cell_", "c").replace("bus_", "b"),
                    (r["pers_R2"], r["learned_R2"]), fontsize=6.5,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("R\u00b2, lag-1 persistence")
    ax.set_ylabel("R\u00b2, best learned model")
    ax.set_title("(c) Hour-ahead skill clusters on the naive diagonal")

    # (d) SHAP beeswarm for the refit learned model
    ax = fig.add_subplot(gs[1, 1])
    try:
        import shap
        sample = X.iloc[te].sample(min(1000, len(te)), random_state=SEED)
        base = mdl[-1] if hasattr(mdl, "steps") else mdl
        if hasattr(base, "estimators_"):
            expl = shap.TreeExplainer(base)
            sv = expl.shap_values(sample)
        else:
            Xs = mdl[:-1].transform(sample) if hasattr(mdl, "steps") else sample
            expl = shap.LinearExplainer(base, Xs)
            sv = expl.shap_values(Xs)
        plt.sca(ax)
        shap.summary_plot(sv, sample, max_display=12, show=False, plot_size=None)
        ax.set_title(f"(d) SHAP, {model_name} on {args.combo.replace('_',' ')}",
                     fontsize=10)
    except Exception as exc:  # pragma: no cover
        ax.text(0.5, 0.5, f"SHAP unavailable: {exc}", ha="center", va="center",
                fontsize=8, wrap=True)
        ax.axis("off")

    fig.savefig(out_dir / "Fig5_ml_benchmark.png", dpi=300, bbox_inches="tight")
    print("saved:", out_dir / "Fig5_ml_benchmark.png")


if __name__ == "__main__":
    main()
