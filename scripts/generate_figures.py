#!/usr/bin/env python3
"""Generate the six manuscript figures from the analysis outputs.

Usage
-----
    python scripts/generate_figures.py --input output --output output/figures_final
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pyrop_energy import plots


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",  type=Path, default=Path("output"),
                   help="Directory containing tables/, asymmetry/, ml/ from run_full_analysis.")
    p.add_argument("--output", type=Path, default=Path("output/figures_final"),
                   help="Where to write Fig1...Fig6.png at 300 DPI.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plots.OUTPUT_DIR = args.input
    plots.FIG_DIR = args.output

    for name, fn in [
        ("Fig1", plots.make_figure_1),
        ("Fig2", plots.make_figure_2),
        ("Fig3", plots.make_figure_3),
        ("Fig4", plots.make_figure_4),
        ("Fig5", plots.make_figure_5),
        ("Fig6", plots.make_figure_6),
    ]:
        try:
            fn()
        except Exception as exc:
            print(f"  {name} ERROR: {exc}")

    print(f"\nDone. Figures in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
