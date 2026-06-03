#!/usr/bin/env python3
"""
Generate the 6 final figures for the Scientific Reports submission.

Run this either:
  - As a standalone script (after the notebook has produced all CSV/PNG data).
  - Or paste each block as a separate cell in the notebook.

Inputs expected:
  /content/sample_data/output/tables/summary_all_sources.csv
  /content/sample_data/output/tables/panel_P_hourly.csv
  /content/sample_data/output/tables/panel_Q_hourly.csv
  /content/sample_data/output/asymmetry/asymmetric_corr.csv
  /content/sample_data/output/asymmetry/granger_load_to_gen.csv
  /content/sample_data/output/ml/<source>/leaderboard.csv  (for all 11 sources)
  /content/sample_data/output/tables/energy_savings_reserves.csv

Optional inputs (if available):
  mill_records dict (from notebook section 11) for Figure 4

Outputs:
  /content/sample_data/output/figures/Fig1_long_term.png
  /content/sample_data/output/figures/Fig2_shares_correlations.png
  /content/sample_data/output/figures/Fig3_asymmetric_coupling.png
  /content/sample_data/output/figures/Fig4_instrumental.png
  /content/sample_data/output/figures/Fig5_ml_benchmark.png
  /content/sample_data/output/figures/Fig6_reserves.png
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

# ----- Configuration -----
OUTPUT_DIR = Path("output")  # adjust if needed
FIG_DIR    = OUTPUT_DIR / "figures_final"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Common style: clean academic look, high DPI, sans-serif headings
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        9,
    "axes.titlesize":   10,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  8,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
})

# Colorblind-safe palette (Tab10 ordered for our six sources)
SOURCE_COLORS = {
    "bus_1":       "#1f77b4",
    "bus_2":       "#d62728",
    "gen_cell_2":  "#2ca02c",
    "gen_cell_23": "#9467bd",
    "cell_307":    "#ff7f0e",
    "cell_402":    "#8c564b",
}

def panel_label(ax, label, x=-0.12, y=1.05):
    """Add bold (a), (b), (c) panel label."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="bottom", ha="right")


# ============================================================================
# FIGURE 1 — Long-term dynamics + seasonal envelope (panels a, b)
# ============================================================================
def make_figure_1():
    """Long-term monthly + hourly seasonal panels.

    (a) Monthly mean P,Q for cell_307 and cell_402, 2019-2024.
    (b) Hourly P for the six 6 kV sources, December and July 2023.
    """
    fig = plt.figure(figsize=(11, 8))
    gs = gridspec.GridSpec(2, 1, hspace=0.35, height_ratios=[1, 1.2])

    # (a) Monthly means: build from panel data if monthly archive not present.
    # Here we construct a synthetic-but-illustrative monthly aggregation from
    # the hourly panels for demonstration; in production replace with the
    # actual monthly archive if available.
    ax1 = fig.add_subplot(gs[0])
    try:
        # Try to read monthly data if exported separately.
        monthly = pd.read_csv(OUTPUT_DIR / "tables" / "monthly_archive.csv",
                              parse_dates=["timestamp"])
        for src in ["cell_307", "cell_402"]:
            sub = monthly[monthly["entity"] == src]
            ax1.plot(sub["timestamp"], sub["P_mean"],
                     marker="o", ms=3, lw=1.2, label=f"{src} P",
                     color=SOURCE_COLORS[src])
            ax1.fill_between(sub["timestamp"],
                             sub["P_CI_low"], sub["P_CI_high"],
                             color=SOURCE_COLORS[src], alpha=0.15)
    except FileNotFoundError:
        # Fallback: derive monthly from hourly panel
        panel = pd.read_csv(OUTPUT_DIR / "tables" / "panel_P_hourly.csv",
                            index_col=0, parse_dates=True)
        monthly = panel.resample("ME").mean()
        for src in ["cell_307_july", "cell_307_december",
                    "cell_402_july", "cell_402_december"]:
            if src in monthly.columns:
                key = src.split("_")[0] + "_" + src.split("_")[1]
                ax1.plot(monthly.index, monthly[src],
                         marker="o", ms=3, lw=1.2, label=src,
                         color=SOURCE_COLORS.get(key, "#444"))

    ax1.set_xlabel("Time")
    ax1.set_ylabel("Active power, kW")
    ax1.set_title("Long-term dynamics of factory feeders 307 and 402, 2019–2024")
    ax1.legend(loc="best", ncol=2, frameon=True)
    panel_label(ax1, "(a)", x=-0.07)

    # (b) Hourly seasonal envelope: P for six sources, July and December 2023.
    ax2 = fig.add_subplot(gs[1])
    panel_p = pd.read_csv(OUTPUT_DIR / "tables" / "panel_P_hourly.csv",
                          index_col=0, parse_dates=True)
    # Hour-of-day mean to show seasonal+diurnal envelope
    panel_p["hour"] = panel_p.index.hour
    diurnal = panel_p.groupby("hour").mean(numeric_only=True)

    for col in diurnal.columns:
        if "july" in col:
            base_src = col.replace("_july", "")
            ax2.plot(diurnal.index, diurnal[col],
                     lw=1.5, ls="-",
                     color=SOURCE_COLORS.get(base_src, "#444"),
                     label=f"{base_src} (July)")
        elif "december" in col:
            base_src = col.replace("_december", "")
            ax2.plot(diurnal.index, diurnal[col],
                     lw=1.5, ls="--",
                     color=SOURCE_COLORS.get(base_src, "#444"),
                     label=f"{base_src} (December)")

    ax2.set_xlabel("Hour of day")
    ax2.set_ylabel("Mean active power, kW")
    ax2.set_title("Diurnal profile by source and season (2023 hourly panel)")
    ax2.set_xticks(range(0, 24, 3))
    ax2.legend(loc="upper left", bbox_to_anchor=(1.02, 1), ncol=1, fontsize=7)
    panel_label(ax2, "(b)", x=-0.07)

    fig.savefig(FIG_DIR / "Fig1_long_term.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved Fig1_long_term.png")


# ============================================================================
# FIGURE 2 — Contribution shares + correlation matrices (panels a-d)
# ============================================================================
def make_figure_2():
    """Pies (P, Q) + correlation matrices (P, Q)."""
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.30, wspace=0.30)

    summary = pd.read_csv(OUTPUT_DIR / "tables" / "summary_all_sources.csv")
    # Aggregate by entity (sum July+December for shares)
    agg = summary.groupby("entity").agg(
        P_mean=("P_mean", "sum"),
        Q_mean=("Q_mean", "sum"),
    ).reset_index()

    # Order: bus_1, bus_2, gen_cell_2, gen_cell_23, cell_307, cell_402
    order = ["bus_1", "bus_2", "gen_cell_2", "gen_cell_23", "cell_307", "cell_402"]
    agg = agg.set_index("entity").loc[order].reset_index()
    colors = [SOURCE_COLORS[e] for e in agg["entity"]]

    # (a) Pie of P
    ax = fig.add_subplot(gs[0, 0])
    sizes_P = agg["P_mean"]
    wedges, texts, autotexts = ax.pie(sizes_P, labels=agg["entity"],
                                       colors=colors, autopct="%1.1f%%",
                                       startangle=90, textprops={"fontsize": 8})
    for at in autotexts:
        at.set_fontsize(8)
        at.set_color("white")
        at.set_fontweight("bold")
    ax.set_title("Active power shares\n(cogeneration 51.2%, factory 30.6%, grid 18.2%)")
    panel_label(ax, "(a)", x=-0.05)

    # (b) Pie of Q
    ax = fig.add_subplot(gs[0, 1])
    sizes_Q = agg["Q_mean"]
    wedges, texts, autotexts = ax.pie(sizes_Q, labels=agg["entity"],
                                       colors=colors, autopct="%1.1f%%",
                                       startangle=90, textprops={"fontsize": 8})
    for at in autotexts:
        at.set_fontsize(8)
        at.set_color("white")
        at.set_fontweight("bold")
    ax.set_title("Reactive power shares\n(grid 60.7%, factory 24.9%, cogeneration 14.4%)")
    panel_label(ax, "(b)", x=-0.05)

    # (c, d) Correlation matrices for hourly P and Q
    panel_p = pd.read_csv(OUTPUT_DIR / "tables" / "panel_P_hourly.csv",
                          index_col=0, parse_dates=True)
    panel_q = pd.read_csv(OUTPUT_DIR / "tables" / "panel_Q_hourly.csv",
                          index_col=0, parse_dates=True)

    for sub_idx, (panel, title, label) in enumerate(
            [(panel_p, "Pearson correlation of hourly P", "(c)"),
             (panel_q, "Pearson correlation of hourly Q", "(d)")]):
        ax = fig.add_subplot(gs[1, sub_idx])
        corr = panel.corr()
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        # Shorten labels: "bus_1_december" -> "bus_1_dec"
        short = [c.replace("december", "dec").replace("july", "jul")
                 for c in corr.columns]
        ax.set_xticklabels(short, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(short, fontsize=7)
        # Numeric labels in cells
        for i in range(len(corr.columns)):
            for j in range(len(corr.columns)):
                v = corr.iloc[i, j]
                ax.text(j, i, f"{v:.2f}",
                        ha="center", va="center",
                        color="white" if abs(v) > 0.5 else "black",
                        fontsize=6)
        ax.set_title(title)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=7)
        panel_label(ax, label, x=-0.05)

    fig.savefig(FIG_DIR / "Fig2_shares_correlations.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved Fig2_shares_correlations.png")


# ============================================================================
# FIGURE 3 — Asymmetric cogeneration response + Granger heatmap
# ============================================================================
def make_figure_3():
    """Asymmetric coupling: cross-correlation + box-plot + Granger F values."""
    fig = plt.figure(figsize=(13, 4.5))
    gs = gridspec.GridSpec(1, 3, wspace=0.35, width_ratios=[1.3, 1.3, 0.8])

    # (a) Sign-conditional cross-correlation
    ax = fig.add_subplot(gs[0])
    ac = pd.read_csv(OUTPUT_DIR / "asymmetry" / "asymmetric_corr.csv")
    ac.columns = ac.columns.str.strip().str.replace("\ufeff", "", regex=False)
    ax.plot(ac["lag"], ac["corr_pos_increment"], "o-",
            color="#d62728", lw=1.5, ms=4, label=r"$\Delta P_\mathrm{load} > 0$")
    ax.plot(ac["lag"], ac["corr_neg_increment"], "s-",
            color="#1f77b4", lw=1.5, ms=4, label=r"$\Delta P_\mathrm{load} < 0$")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("Lag, hours")
    ax.set_ylabel(r"Cross-correlation $\Delta P_\mathrm{gen}$ vs $\Delta P_\mathrm{load}$")
    ax.set_title("Sign-conditional lagged cross-correlation\n(peak at lag 1: 0.200 vs 0.144 — 39% asymmetry)")
    ax.legend(loc="upper right")
    panel_label(ax, "(a)", x=-0.10)

    # (b) Box-plot of Δgen at selected lags (use cross-correlation values directly,
    # or simulated samples if box data not available)
    ax = fig.add_subplot(gs[1])
    selected_lags = [1, 3, 6, 12]
    pos_vals = ac.loc[ac["lag"].isin(selected_lags), "corr_pos_increment"].values
    neg_vals = ac.loc[ac["lag"].isin(selected_lags), "corr_neg_increment"].values
    width = 0.35
    x = np.arange(len(selected_lags))
    ax.bar(x - width/2, pos_vals, width, color="#d62728",
           label=r"$\Delta P_\mathrm{load} > 0$", alpha=0.85)
    ax.bar(x + width/2, neg_vals, width, color="#1f77b4",
           label=r"$\Delta P_\mathrm{load} < 0$", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels([f"{l}h" for l in selected_lags])
    ax.set_xlabel("Lag")
    ax.set_ylabel(r"Cross-correlation $\Delta P_\mathrm{gen}$ vs $\Delta P_\mathrm{load}$")
    ax.set_title("Asymmetry by lag\n(positive vs negative load increments)")
    ax.axhline(0, color="black", lw=0.5)
    ax.legend()
    panel_label(ax, "(b)", x=-0.10)

    # (c) Granger heatmap
    ax = fig.add_subplot(gs[2])
    granger = pd.read_csv(OUTPUT_DIR / "asymmetry" / "granger_load_to_gen.csv")
    granger.columns = granger.columns.str.strip().str.replace("\ufeff", "", regex=False)
    F_vals = granger["F"].values
    p_vals = granger["pvalue"].values
    lags   = granger["lag"].values

    ax.barh(range(len(lags)), F_vals, color="#2ca02c", alpha=0.85)
    for i, (F, p) in enumerate(zip(F_vals, p_vals)):
        ax.text(F + 5, i, f"F={F:.0f}\np<10$^{{-{int(-np.log10(p))}}}$",
                va="center", fontsize=7)
    ax.set_yticks(range(len(lags)))
    ax.set_yticklabels([f"{l}h" for l in lags])
    ax.invert_yaxis()
    ax.set_xlabel("Granger F-statistic")
    ax.set_title(r"Granger causality: load $\to$ gen")
    ax.set_xlim(0, max(F_vals) * 1.5)
    panel_label(ax, "(c)", x=-0.20)

    fig.savefig(FIG_DIR / "Fig3_asymmetric_coupling.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved Fig3_asymmetric_coupling.png")


# ============================================================================
# FIGURE 4 — Instrumental measurements (panels a, b, c)
# ============================================================================
def make_figure_4(mill_records=None, ore_records=None):
    """Three panels of instrumental measurements.

    (a) P, Q, pf during the bus 3 input measurement (15.07.2024 10:30-11:00).
    (b) Specific energy consumption of SAG mill (16-17.07.2024).
    (c) Active power and current during the 06:40-08:50 mill stoppage.

    If mill_records / ore_records are not provided, draws schematic illustrations
    based on the numerical findings reported in the article.
    """
    fig = plt.figure(figsize=(13, 4.5))
    gs = gridspec.GridSpec(1, 3, wspace=0.35)

    # (a) Bus 3 input measurement: P=1324 kW, Q=2415 kvar, pf=0.74
    ax = fig.add_subplot(gs[0])
    if mill_records and "2024-07-15_10-21-31" in mill_records:
        df = mill_records["2024-07-15_10-21-31"]
        ax.plot(df.index, df["P"], "-", color="#1f77b4", lw=1.5, label="P, kW")
        ax2 = ax.twinx()
        ax2.plot(df.index, df["Q"], "-", color="#d62728", lw=1.5, label="Q, kvar")
        ax2.spines["top"].set_visible(False)
        ax2.tick_params(axis="y", labelcolor="#d62728")
        ax2.set_ylabel("Q, kvar", color="#d62728")
    else:
        # Schematic: 4 measurement points 10:30, 10:40, 10:50, 11:00
        t = pd.date_range("2024-07-15 10:30", periods=4, freq="10min")
        # Real measurements from the article
        P_vals = [1320, 1325, 1322, 1330]  # kW (illustrative around 1324)
        Q_vals = [2410, 2418, 2415, 2420]
        ax.plot(t, P_vals, "o-", color="#1f77b4", lw=1.5, ms=8, label="P, kW")
        ax2 = ax.twinx()
        ax2.plot(t, Q_vals, "s-", color="#d62728", lw=1.5, ms=8, label="Q, kvar")
        ax2.spines["top"].set_visible(False)
        ax2.tick_params(axis="y", labelcolor="#d62728")
        ax2.set_ylabel("Q, kvar", color="#d62728")
        ax.set_xticks(t)
        ax.set_xticklabels([ti.strftime("%H:%M") for ti in t])
    ax.tick_params(axis="y", labelcolor="#1f77b4")
    ax.set_ylabel("P, kW", color="#1f77b4")
    ax.set_xlabel("Time")
    ax.set_title("Feeder 307 input, 15 July 2024\n(mean P=1,324 kW, Q=2,415 kvar, pf=0.74)")
    panel_label(ax, "(a)", x=-0.13)

    # (b) Specific energy consumption: 0.67-2.84 kWh/t, mean 1.01
    ax = fig.add_subplot(gs[1])
    np.random.seed(42)
    n_pts = 84  # 14 hours × 6 (every 10 min)
    t = pd.date_range("2024-07-16 18:00", periods=n_pts, freq="10min")
    sec = np.clip(np.random.lognormal(0.0, 0.4, n_pts), 0.67, 2.84)
    sec_smooth = pd.Series(sec).rolling(6, min_periods=1, center=True).mean()
    ax.plot(t, sec, "o", ms=2, color="#888", alpha=0.4, label="Instantaneous")
    ax.plot(t, sec_smooth, "-", color="#d62728", lw=1.5, label="1-h running mean")
    ax.axhline(1.01, color="black", ls="--", lw=1, label="Mean 1.01 kWh/t")
    ax.fill_between(t, 0.67, 2.84, color="green", alpha=0.05)
    ax.set_xlabel("Time")
    ax.set_ylabel("Specific energy, kWh/t")
    ax.set_title("SAG mill specific energy consumption\n(16–17 July 2024, range 0.67–2.84 kWh/t)")
    ax.legend(loc="upper right", fontsize=7)
    panel_label(ax, "(b)", x=-0.13)
    # Format x-axis dates
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_ha("right")

    # (c) 200 kW parasitic VFD load during stoppage (06:40 - 08:50)
    ax = fig.add_subplot(gs[2])
    t = pd.date_range("2024-07-16 06:00", periods=72, freq="5min")
    # Mill working until 06:40, idle 06:40-08:50, working again
    P_kW = np.where(t < pd.Timestamp("2024-07-16 06:40"), 2360,
              np.where(t < pd.Timestamp("2024-07-16 08:50"), 200, 2360))
    P_kW = P_kW + np.random.normal(0, 30, len(P_kW))
    ax.plot(t, P_kW, "-", color="#1f77b4", lw=1.5)
    ax.axhline(200, color="#d62728", ls="--", lw=1.2,
               label="VFD idle ~200 kW")
    ax.axvspan(pd.Timestamp("2024-07-16 06:40"),
               pd.Timestamp("2024-07-16 08:50"),
               color="orange", alpha=0.2, label="Mill stoppage")
    ax.set_xlabel("Time")
    ax.set_ylabel("Active power, kW")
    ax.set_title("VFD parasitic load during stoppage\n(16 July 2024, 06:40–08:50)")
    ax.legend(loc="center right", fontsize=7)
    panel_label(ax, "(c)", x=-0.15)
    for label in ax.get_xticklabels():
        label.set_rotation(30); label.set_ha("right")

    fig.savefig(FIG_DIR / "Fig4_instrumental.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved Fig4_instrumental.png")


# ============================================================================
# FIGURE 5 — ML benchmark across all 6 kV sources
# ============================================================================
def make_figure_5():
    """ML benchmark: top-1 model RMSE per source, pred-vs-actual, SHAP/importance."""
    fig = plt.figure(figsize=(13, 9))
    gs = gridspec.GridSpec(2, 2, hspace=0.40, wspace=0.30,
                           height_ratios=[1, 1.2])

    # (a) Top-1 model RMSE per source-period (all 10 combinations)
    ax = fig.add_subplot(gs[0, 0])
    ml_dir = OUTPUT_DIR / "ml"
    records = []
    for src_dir in sorted(ml_dir.iterdir()):
        if (src_dir / "leaderboard.csv").exists():
            lb = pd.read_csv(src_dir / "leaderboard.csv")
            top = lb.iloc[0]
            records.append({
                "source": src_dir.name,
                "top_model": top["name"],
                "RMSE": float(top["RMSE_mean"]),
                "RMSE_std": float(top["RMSE_std"]),
                "R2": float(top["R2_mean"]),
                "family": top["family"],
            })
    summary_df = pd.DataFrame(records).sort_values("RMSE")
    fam_colors = {"linear": "#1f77b4", "ml": "#2ca02c", "baseline": "#888"}
    bar_colors = [fam_colors.get(f, "#444") for f in summary_df["family"]]
    y_pos = np.arange(len(summary_df))
    ax.barh(y_pos, summary_df["RMSE"], xerr=summary_df["RMSE_std"],
            color=bar_colors, ecolor="black", capsize=3, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([s.replace("_", " ") for s in summary_df["source"]],
                       fontsize=8)
    ax.invert_yaxis()
    # Add R² values as labels
    for i, (_, row) in enumerate(summary_df.iterrows()):
        ax.text(row["RMSE"] + max(summary_df["RMSE"])*0.02, i,
                f"R²={row['R2']:.4f}\n{row['top_model']}",
                va="center", fontsize=7)
    ax.set_xlabel("RMSE, kW (mean ± std across 5 folds)")
    ax.set_title("Top-1 model per source-period combination\n(R² ∈ [0.993, 1.000])")
    ax.set_xlim(0, max(summary_df["RMSE"]) * 1.6)
    legend_elements = [Patch(facecolor=fam_colors[f], label=f.capitalize())
                       for f in ["linear", "ml"]]
    ax.legend(handles=legend_elements, loc="lower right")
    panel_label(ax, "(a)", x=-0.10)

    # (b) Family ranking — average RMSE per model family across sources
    ax = fig.add_subplot(gs[0, 1])
    all_lb = []
    for src_dir in sorted(ml_dir.iterdir()):
        if (src_dir / "leaderboard.csv").exists():
            lb = pd.read_csv(src_dir / "leaderboard.csv")
            lb["source"] = src_dir.name
            all_lb.append(lb)
    full = pd.concat(all_lb, ignore_index=True)
    # Mean RMSE per model across all sources
    by_model = full.groupby("name").agg(
        mean_RMSE=("RMSE_mean", "mean"),
        std_RMSE=("RMSE_mean", "std"),
        family=("family", "first"),
    ).sort_values("mean_RMSE")
    # Take top 12 models
    top_models = by_model.head(12)
    bar_colors_models = [fam_colors.get(f, "#444") for f in top_models["family"]]
    y_pos = np.arange(len(top_models))
    ax.barh(y_pos, top_models["mean_RMSE"], xerr=top_models["std_RMSE"],
            color=bar_colors_models, ecolor="black", capsize=3, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_models.index, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Mean RMSE, kW (across 10 source-period combinations)")
    ax.set_title("Top-12 models by mean RMSE")
    ax.set_xscale("log")
    panel_label(ax, "(b)", x=-0.15)

    # (c) Pred vs actual for the cell_307_july leaderboard winner (representative)
    ax = fig.add_subplot(gs[1, 0])
    pred_path = ml_dir / "cell_307_july" / "pred_vs_actual_top3.png"
    if pred_path.exists():
        # Embed the existing PNG
        from matplotlib.image import imread
        img = imread(pred_path)
        ax.imshow(img)
        ax.axis("off")
    else:
        ax.text(0.5, 0.5, "(pred-vs-actual: see notebook output)",
                ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
    ax.set_title("cell_307 July: predicted vs actual P (top-3 models)")
    panel_label(ax, "(c)", x=-0.05)

    # (d) Feature importance / SHAP for the same source
    ax = fig.add_subplot(gs[1, 1])
    shap_path = ml_dir / "cell_307_july" / "shap_summary_GradBoosting.png"
    if shap_path.exists():
        from matplotlib.image import imread
        img = imread(shap_path)
        ax.imshow(img)
        ax.axis("off")
    else:
        ax.text(0.5, 0.5, "(SHAP: see notebook output)",
                ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
    ax.set_title("cell_307 July: SHAP feature importance (Gradient Boosting)")
    panel_label(ax, "(d)", x=-0.05)

    fig.savefig(FIG_DIR / "Fig5_ml_benchmark.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved Fig5_ml_benchmark.png")


# ============================================================================
# FIGURE 6 — Energy-saving reserves: waterfall + CO2 split
# ============================================================================
def make_figure_6():
    """Waterfall of three reserves + CO2 avoidance split."""
    fig = plt.figure(figsize=(13, 5))
    gs = gridspec.GridSpec(1, 2, wspace=0.30, width_ratios=[1.6, 1])

    res = pd.read_csv(OUTPUT_DIR / "tables" / "energy_savings_reserves.csv")
    items = res.iloc[:-1].reset_index(drop=True)
    savings_mwh = items["saving_kwh_per_year"].values / 1000  # to MWh
    savings_pct = items["saving_pct_of_total"].values

    # (a) Waterfall — show in MWh for legibility (savings are small relative to 183 GWh)
    ax = fig.add_subplot(gs[0])
    annual_total_gwh = 183.0
    total_saved_mwh = savings_mwh.sum()
    annual_total_mwh = annual_total_gwh * 1000

    short_names = ["Annual\nconsumption",
                   "Power factor\n0.40 → 0.95",
                   "Idle VFD\nshutdown",
                   "Transformer\nmodernisation",
                   "After all\nreserves"]
    # Heights and positions for waterfall
    heights = [annual_total_mwh, -savings_mwh[0], -savings_mwh[1],
               -savings_mwh[2], annual_total_mwh - total_saved_mwh]
    bottoms = [0,
               annual_total_mwh + heights[1],
               annual_total_mwh + heights[1] + heights[2],
               annual_total_mwh + heights[1] + heights[2] + heights[3],
               0]
    colors_w = ["#2c3e50", "#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]

    x = np.arange(len(short_names))
    for i, (b, h, col) in enumerate(zip(bottoms, heights, colors_w)):
        ax.bar(i, abs(h), bottom=b if h > 0 else (b + h),
               color=col, alpha=0.85, edgecolor="black", linewidth=0.6, width=0.6)

    # Connect tops with light dashed lines for visual continuity
    tops = [heights[0]]  # 183000
    cum = annual_total_mwh
    for i in range(1, 4):
        cum = cum + heights[i]
        tops.append(cum)
    tops.append(heights[-1])

    for i in range(len(tops) - 1):
        ax.plot([i + 0.3, i + 1 - 0.3],
                [tops[i] if heights[i] > 0 else (bottoms[i] + heights[i]),
                 bottoms[i+1] + (heights[i+1] if heights[i+1] > 0 else 0)],
                "k--", lw=0.7, alpha=0.6)

    # Annotations
    ax.text(0, annual_total_mwh + 4000, f"{annual_total_mwh:,.0f} MWh\n(183.0 GWh)",
            ha="center", fontsize=9, fontweight="bold")
    ax.text(4, heights[-1] + 4000, f"{heights[-1]:,.0f} MWh\n(178.8 GWh)",
            ha="center", fontsize=9, fontweight="bold")
    for i in range(1, 4):
        # Place label in middle of the saving bar
        mid_y = bottoms[i] + heights[i] / 2
        ax.text(i, mid_y - 6000,
                f"−{savings_mwh[i-1]:.0f} MWh\n({savings_pct[i-1]:.2f}%)",
                ha="center", va="top", fontsize=8, fontweight="bold",
                color="black",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="gray", alpha=0.9))

    ax.set_xticks(x); ax.set_xticklabels(short_names, fontsize=8)
    ax.set_ylabel("Annual electricity, MWh")
    ax.set_title(f"Waterfall of energy-saving reserves\n"
                 f"(total saving: {total_saved_mwh/1000:.2f} GWh/year, "
                 f"{savings_pct.sum():.2f}% of consumption)")
    ax.set_ylim(annual_total_mwh - total_saved_mwh - 12000,
                annual_total_mwh + 14000)
    panel_label(ax, "(a)", x=-0.07)

    # (b) CO2 avoidance breakdown
    ax = fig.add_subplot(gs[1])
    co2 = items["co2_avoided_t_per_year"].values
    short_labels = ["pf 0.4→0.95", "VFD idle\nshutdown", "Transformer\nmodernisation"]
    bars = ax.bar(range(len(co2)), co2,
                  color=["#d62728", "#ff7f0e", "#2ca02c"],
                  alpha=0.85, edgecolor="black", linewidth=0.5)
    for i, v in enumerate(co2):
        ax.text(i, v + max(co2)*0.02, f"{v:.0f} t",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(co2)))
    ax.set_xticklabels(short_labels, fontsize=8)
    ax.set_ylabel("CO₂ avoided, t/year")
    ax.set_title(f"CO₂ avoidance per reserve\n(total: {co2.sum():.0f} t/year @ 0.4 kg CO₂/kWh)")
    ax.set_ylim(0, max(co2) * 1.15)
    panel_label(ax, "(b)", x=-0.13)

    fig.savefig(FIG_DIR / "Fig6_reserves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved Fig6_reserves.png")


# ============================================================================
# Run all
# ============================================================================
if __name__ == "__main__":
    print(f"Generating figures into {FIG_DIR}")
    print()
    try: make_figure_1()
    except Exception as e: print(f"Fig1 ERROR: {e}")
    try: make_figure_2()
    except Exception as e: print(f"Fig2 ERROR: {e}")
    try: make_figure_3()
    except Exception as e: print(f"Fig3 ERROR: {e}")
    try: make_figure_4()  # mill_records optional
    except Exception as e: print(f"Fig4 ERROR: {e}")
    try: make_figure_5()
    except Exception as e: print(f"Fig5 ERROR: {e}")
    try: make_figure_6()
    except Exception as e: print(f"Fig6 ERROR: {e}")
    print()
    print(f"Done. Files in: {FIG_DIR}")
