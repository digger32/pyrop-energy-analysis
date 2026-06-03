#!/usr/bin/env python3
"""Run only the 18-model ML benchmark (assuming raw parsing has been done).

Usage
-----
    python scripts/run_ml_benchmark.py --panel-dir output/tables \
                                       --output    output/ml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pyrop_energy import features, ml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel-dir", type=Path, default=Path("output/tables"),
                   help="Directory with panel_P_hourly.csv and panel_Q_hourly.csv.")
    p.add_argument("--output", type=Path, default=Path("output/ml"),
                   help="Where to write per-source leaderboards.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    panel_p = pd.read_csv(args.panel_dir / "panel_P_hourly.csv", index_col=0, parse_dates=True)
    panel_q = pd.read_csv(args.panel_dir / "panel_Q_hourly.csv", index_col=0, parse_dates=True)

    print(f"Panel P: {panel_p.shape[0]} timestamps × {panel_p.shape[1]} sources")
    print(f"Coverage of sources in panel P (fraction of valid values):")
    for col in panel_p.columns:
        cov = panel_p[col].notna().mean()
        flag = " ✓" if cov >= 0.5 else " ⚠ (will be filtered as orphan)"
        print(f"   {col:35s}  {cov*100:5.1f}%{flag}")

    for col in panel_p.columns:
        df = pd.DataFrame({"P": panel_p[col], "Q": panel_q[col]}).dropna()
        if len(df) < 400:
            print(f"\n[skip] {col}: too short ({len(df)} rows)")
            continue
        df.attrs["source"] = col
        fm = features.build_feature_matrix(df, "hourly", "P", panel_p, panel_q, verbose=True)
        if len(fm) < 200:
            print(f"[skip] {col}: feature matrix too small ({len(fm)} rows)")
            continue
        print(f"\n=== ML benchmark: {col}  (rows={len(fm)}, features={fm.shape[1]-1}) ===")
        out = args.output / col
        out.mkdir(parents=True, exist_ok=True)
        lb, _, _ = ml.run_benchmark(fm.drop(columns=["y"]), fm["y"])
        lb.to_csv(out / "leaderboard.csv", index=False, float_format="%.5f")
        top = lb.iloc[0]
        print(f"  top-1: {top['name']}  RMSE={top['RMSE_mean']:.2f}  R2={top['R2_mean']:.4f}")

    print(f"\nDone. Leaderboards in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
