# Reproducibility

This document is a step-by-step recipe for reproducing every figure and table of the manuscript.

## 1. Software environment

The pipeline was developed and tested with:

- Python 3.10–3.12
- pandas 2.2, NumPy 1.26, SciPy 1.13
- scikit-learn 1.5, statsmodels 0.14
- XGBoost 2.0, LightGBM 4.3, CatBoost 1.2 (optional but recommended)
- SHAP 0.45 (optional, for feature-importance plots)
- Matplotlib 3.9, seaborn 0.13

The exact versions are pinned in `requirements.txt`. Use the Conda environment for guaranteed reproducibility:

```bash
conda env create -f environment.yml
conda activate pyrop
```

Random seeds are fixed at `SEED = 42` throughout; results are deterministic given the same input data and the same library versions.

## 2. Data

See `data/README.md`. Two pathways are available:

1. **Reproducing the figures from the deposited dataset** (CC-BY 4.0 on Zenodo): place the contents of the Zenodo archive in `data/processed/` and skip directly to step 4.

2. **Re-running from raw data** (NDA required): place the raw SCADA exports and instrumental files in `data/raw/`. Expected file layout is described in `docs/DATA_DICTIONARY.md`.

## 3. Run the full analysis pipeline (only if you have raw data)

```bash
python scripts/run_full_analysis.py --input data/raw --output output
```

This executes parsing, statistics, feature engineering, ML benchmark, asymmetric-coupling tests, and reserves. Approximate run times:

| Stage                      | CPU (8 cores) | GPU (T4) |
|----------------------------|---------------|----------|
| Parsing all source files   | 2 min         | 2 min    |
| Statistics + GoF + STL     | 5 min         | 5 min    |
| ML benchmark, 10 sources   | 90 min        | 25 min   |
| Asymmetric coupling tests  | 1 min         | 1 min    |
| **Total**                  | **~100 min**  | **~35 min** |

For a quick smoke test, use `--skip-ml` (omits the ML benchmark, ~10 min total).

## 4. Generate the manuscript figures

```bash
python scripts/generate_figures.py --input output --output output/figures_final
```

The script produces six PNG files at 300 DPI:

- `Fig1_long_term.png`
- `Fig2_shares_correlations.png`
- `Fig3_asymmetric_coupling.png`
- `Fig4_instrumental.png`
- `Fig5_ml_benchmark.png`
- `Fig6_reserves.png`

These are the exact files used in the submitted manuscript.

## 5. Tables

The tables of the manuscript are derived from the CSVs in `output/tables/`:

| Table in the paper                        | Source CSV                                           |
|-------------------------------------------|------------------------------------------------------|
| Table 1 (multi-year statistics)           | `output/tables/summary_all_sources.csv`              |
| Table 2 (top-1 ML model per source)       | `output/ml/<source>/leaderboard.csv` (top row)       |
| Table S1 (full descriptive statistics, P) | `output/tables/stats_<source>.csv`                   |
| Tables S5–S14 (per-source full leaderboard)| `output/ml/<source>/leaderboard.csv` (top 10 rows)  |

## 6. Continuous integration

Smoke tests run on synthetic data and validate that the package imports and the public API does not break:

```bash
pytest tests/
```

These tests do not validate scientific correctness, only the structural integrity of the codebase.

## 7. Troubleshooting

**ML benchmark is very slow on Colab.** Confirm the runtime: *Runtime → Change runtime type → T4 GPU*. The pipeline auto-detects CUDA; check the console output of the ML stage — you should see `GPU: Tesla T4 → XGBoost(device='cuda'), CatBoost(task_type='GPU')`.

**Some sources produce 0 rows after feature engineering.** A panel source with <50% coverage on the target index is filtered out as an "orphan". This is intentional and prevents spurious all-NaN columns. See `features.build_feature_matrix` for details.

**SpreadsheetML 2003 files do not parse.** These are XML files exported by some legacy SCADA tools. The parser is in `parsers._parse_spreadsheetml_2003`. If the format differs slightly from the one in our data, please open an issue with a minimal example.
