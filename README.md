# Pyrop Energy Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org)

Multi-year electricity-consumption analysis and machine-learning forecasting of an ore-preparation plant integrated with on-site cogeneration.

This repository contains the analysis pipeline accompanying the manuscript:

> *Multi-year electricity-consumption analysis and machine-learning forecasting of an ore-preparation plant integrated with on-site cogeneration*. Submitted to **Scientific Reports**, 2026.

The pipeline reproduces every figure and table of the manuscript from the deposited dataset.

---

## What is in this repository

```
pyrop-energy-analysis/
├── README.md                            ← this file
├── LICENSE                              ← MIT
├── pyproject.toml                       ← package metadata, dependencies
├── requirements.txt                     ← pinned runtime requirements
├── environment.yml                      ← Conda environment specification
├── .gitignore
├── src/
│   └── pyrop_energy/                    ← installable Python package
│       ├── __init__.py
│       ├── parsers.py                   ← SCADA + SpreadsheetML 2003 readers
│       ├── stats.py                     ← descriptive stats, GoF, stationarity
│       ├── features.py                  ← calendar, lag, rolling, panel features
│       ├── ml.py                        ← 18-model rolling-origin benchmark
│       ├── asymmetry.py                 ← Granger + sign-conditional CC
│       ├── reserves.py                  ← three energy-saving reserves
│       └── plots.py                     ← Matplotlib figures
├── notebooks/
│   └── Pyrop_analysis.ipynb             ← end-to-end Colab-ready notebook
├── scripts/
│   ├── run_full_analysis.py             ← reproduce all CSV/JSON outputs
│   ├── run_ml_benchmark.py              ← only the ML benchmark
│   └── generate_figures.py              ← only the six manuscript figures
├── tests/
│   └── test_smoke.py                    ← smoke-test on synthetic data
├── data/
│   └── README.md                        ← data access policy (see below)
└── docs/
    ├── REPRODUCIBILITY.md               ← step-by-step reproduction recipe
    └── DATA_DICTIONARY.md               ← column-by-column description
```

## Quick start

### Option A: Google Colab (recommended for first-time users)

1. Open `notebooks/Pyrop_analysis.ipynb` in Colab.
2. **Runtime → Change runtime type → T4 GPU** (recommended; the gradient-boosting models run roughly 5× faster on GPU).
3. Run all cells. The notebook installs dependencies, parses the data, runs the full benchmark, and produces every figure of the manuscript.

### Option B: local installation

```bash
git clone https://github.com/digger32/pyrop-energy-analysis.git
cd pyrop-energy-analysis

# Conda (recommended)
conda env create -f environment.yml
conda activate pyrop

# Or pip
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
```

Then:

```bash
python scripts/run_full_analysis.py --input data/raw/ --output output/
python scripts/generate_figures.py  --input output/   --output output/figures_final/
```

GPU acceleration is auto-detected: if CUDA is available, XGBoost and CatBoost will run on the GPU. No configuration is required.

## Data availability

**The raw operational records are not included in this repository.** They contain commercially sensitive information from an active mining facility and are governed by a non-disclosure agreement with the operating company.

Two pathways are available:

1. **Reproducing the figures from the deposited dataset.** A processed and anonymised version of the dataset, together with the intermediate analysis outputs (CSVs, tables, leaderboards), is deposited on Zenodo at `https://doi.org/10.5281/zenodo.20529770` under a CC-BY 4.0 licence. Place the contents of the Zenodo archive in `data/processed/` and run the figure-generation script.

2. **Access to the raw operational records.** The raw monthly and hourly SCADA exports and the instrumental measurement files are available from the corresponding author upon reasonable request, subject to a Non-Disclosure Agreement (NDA) with the operating company. Requests should outline the intended use and the institutional context. Please contact `scorpion_ser@mail.ru`.

The SpreadsheetML 2003 XML format used by the instrumental records, the multi-sheet structure of the analyser exports, and the non-trivial timestamp reconstruction required by these files are described in `docs/DATA_DICTIONARY.md`. Independent groups will be able to apply this pipeline to their own data without modifications, provided that the data are exported in the same format families (SCADA hourly Excel, instrumental SpreadsheetML 2003).

## Reproducibility

Random seeds are fixed at `SEED = 42` throughout. The full benchmark of 18 models on 10 source-period combinations completes in approximately:

- 30 minutes on a Google Colab T4 GPU instance,
- 2 hours on a CPU-only laptop with 8 cores.

Detailed reproduction steps are in `docs/REPRODUCIBILITY.md`.

## Citation

If you use this code or the deposited dataset in your work, please cite:

```bibtex
@article{<lastname>2026pyrop,
  author  = {<Yury V. Dmitrak, Roman V. Klyuev, Nikita V. Martyushev,
  Boris V. Malozyomov, Sergei O. Kurashkin, Vadim S. Tynchenko, Aleksei S. Borodulin,
  Ahmad Hammoud, Shohel Sayeed>},
  title   = {Multi-year electricity-consumption analysis and machine-learning
             forecasting of an ore-preparation plant integrated with on-site
             cogeneration},
  journal = {Scientific Reports},
  year    = {2026},
  doi     = {<DOI to be inserted at acceptance>}
}
```

## License

This code is released under the MIT License (see `LICENSE`). The deposited dataset is released separately under CC-BY 4.0.

## Acknowledgements

We thank the operating company for granting access to the operational records under NDA, and the energy department of the facility for assistance with the on-site instrumental measurements. Funding sources, where applicable, are listed in the manuscript.

## Contact

For questions about the code, please open an issue on the GitHub tracker. For questions about the manuscript or the data access policy, contact the corresponding author.
