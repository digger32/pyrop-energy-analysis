#!/usr/bin/env python3
"""End-to-end analysis pipeline.

Reads raw operational data from --input, runs the full analysis (parsing,
descriptive statistics, GoF/stationarity tests, feature engineering, ML
benchmark, asymmetric-coupling tests, reserves), and writes all CSV/JSON
outputs to --output.

Usage
-----
    python scripts/run_full_analysis.py --input data/raw --output output
    python scripts/run_full_analysis.py --skip-ml          # quick run
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pyrop_energy import parsers, stats, features, ml, asymmetry, reserves


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input",  type=Path, default=Path("data/raw"),
                   help="Directory containing raw SCADA exports and instrumental files.")
    p.add_argument("--output", type=Path, default=Path("output"),
                   help="Output directory for CSV/JSON results and figures.")
    p.add_argument("--skip-ml", action="store_true",
                   help="Skip the ML benchmark (fastest path through the pipeline).")
    p.add_argument("--cpu-only", action="store_true",
                   help="Disable GPU acceleration even if CUDA is available.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "tables").mkdir(exist_ok=True)
    (args.output / "ml").mkdir(exist_ok=True)
    (args.output / "asymmetry").mkdir(exist_ok=True)

    # 1) Parse all input files
    print("=== 1. Parsing raw input files ===")
    loaded: dict = {}
    for path in sorted(args.input.glob("**/*")):
        if not path.is_file():
            continue
        # Heuristic dispatch by filename — adapt to your naming convention.
        # See README and DATA_DICTIONARY for expected file layout.
        df = parsers.load_ore_file(path)
        if df is not None and len(df) > 0:
            entity, period = path.stem.split("__", 1) if "__" in path.stem else (path.stem, "unknown")
            loaded[(entity, period, "hourly")] = df
            print(f"  loaded {entity}/{period}: {len(df)} rows")

    if not loaded:
        print("No source files were parsed. Check --input and the README.")
        return 1

    # 2) Per-source descriptive statistics, GoF, stationarity
    print("\n=== 2. Statistics, GoF, stationarity ===")
    summary_rows = []
    for (ent, per, gran), df in loaded.items():
        label = f"{ent}_{per}_{gran}"
        stats.basic_stats(df).to_csv(args.output / f"tables/stats_{label}.csv", index=False)
        stats.goodness_of_fit_tests(df["P"]).to_csv(
            args.output / f"tables/gof_{label}.csv", index=False)
        stats.stationarity_tests(df["P"]).to_csv(
            args.output / f"tables/stationarity_{label}.csv", index=False)
        ci = stats.confidence_interval(df["P"])
        summary_rows.append({
            "entity": ent, "period": per, "n": len(df),
            "P_mean": df["P"].mean(), "P_CI_low": ci[0], "P_CI_high": ci[1],
            "Q_mean": df["Q"].mean(),
            "pf_mean": (df["P"] / (df["P"] ** 2 + df["Q"] ** 2).pow(0.5)).mean(),
        })
    pd.DataFrame(summary_rows).to_csv(args.output / "tables/summary_all_sources.csv", index=False)

    # 3) Panel construction
    print("\n=== 3. Panel construction ===")
    panel_p = features.assemble_panel(loaded, "hourly", "P")
    panel_q = features.assemble_panel(loaded, "hourly", "Q")
    panel_p.to_csv(args.output / "tables/panel_P_hourly.csv")
    panel_q.to_csv(args.output / "tables/panel_Q_hourly.csv")

    # 4) Asymmetric-coupling tests
    print("\n=== 4. Granger and sign-conditional cross-correlation ===")
    load_cols = [c for c in panel_p.columns if "bus_" in c or c.startswith("cell_3") or c.startswith("cell_4")]
    gen_cols  = [c for c in panel_p.columns if "gen_cell" in c]
    if load_cols and gen_cols:
        load, gen = asymmetry.aggregate_load_and_gen(panel_p, load_cols, gen_cols)
        granger = asymmetry.granger_load_to_gen(load, gen)
        granger.to_csv(args.output / "asymmetry/granger_load_to_gen.csv", index=False)
        cc = asymmetry.sign_conditional_cross_correlation(load, gen)
        cc.to_csv(args.output / "asymmetry/asymmetric_corr.csv", index=False)
        print(f"  Granger F at lag 1: {granger.iloc[0]['F']:.1f} (p = {granger.iloc[0]['pvalue']:.2e})")

    # 5) ML benchmark
    if not args.skip_ml:
        print("\n=== 5. ML benchmark ===")
        for (ent, per, gran), df in loaded.items():
            if gran != "hourly" or len(df) < 400:
                print(f"  [skip] {ent}_{per}: too short ({len(df)} rows)")
                continue
            label = f"{ent}_{per}"
            ml_dir = args.output / f"ml/{label}"
            ml_dir.mkdir(parents=True, exist_ok=True)
            df.attrs["source"] = label
            fm = features.build_feature_matrix(df, "hourly", "P", panel_p, panel_q, verbose=True)
            if len(fm) < 200:
                continue
            print(f"  benchmarking {label} ({len(fm)} rows × {fm.shape[1]-1} features)")
            lb, _, _ = ml.run_benchmark(fm.drop(columns=["y"]), fm["y"])
            lb.to_csv(ml_dir / "leaderboard.csv", index=False, float_format="%.5f")
            print(f"    top-1: {lb.iloc[0]['name']}  RMSE={lb.iloc[0]['RMSE_mean']:.2f}  "
                  f"R2={lb.iloc[0]['R2_mean']:.4f}")

    # 6) Reserves (supply-side balance boundary: incomers + cogeneration,
    #    per-incomer power-factor loss reduction from the hourly panels)
    print("\n=== 6. Energy-saving reserves ===")
    res = reserves.compute_reserves(panel_p, panel_q)
    res.to_csv(args.output / "tables/energy_savings_reserves.csv", index=False)
    total = res.iloc[-1]
    print(f"  annual supply-side consumption: "
          f"{res.attrs['annual_consumption_kwh']/1e6:.1f} GWh/year")
    print(f"  total saving: {total['saving_kwh_per_year']/1e6:.2f} GWh/year, "
          f"{total['co2_avoided_t_per_year']:.0f} t CO2/year")

    print(f"\nDone. Outputs in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
