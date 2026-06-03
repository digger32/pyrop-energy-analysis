# Data access policy

The raw operational records analysed in the manuscript **are not included in this repository**. They contain commercially sensitive information from the operating facility and are governed by a non-disclosure agreement.

## What is available

### 1. Anonymised processed dataset (CC-BY 4.0)

A processed and anonymised version of the dataset, sufficient to reproduce all figures and tables of the manuscript, is deposited on **Zenodo** at:

> DOI: `<to be inserted at acceptance>`

The dataset contains:

- Wide hourly panels of P and Q for the six 6 kV sources, two seasonal blocks of 2023.
- Monthly-aggregated archive of feeders 307 and 402 (Sept 2019 – June 2024).
- Instrumental measurement sessions of July 2024.
- Pre-built feature matrices used for the ML benchmark.

To reproduce the figures, download the Zenodo archive and place its contents in `data/processed/`. Then run:

```bash
python scripts/generate_figures.py --input data/processed --output output/figures_final
```

### 2. Raw data (NDA required)

The raw monthly and hourly SCADA exports and the instrumental measurement files (Microsoft SpreadsheetML 2003 XML) are not deposited publicly.

Access can be granted to qualified researchers upon reasonable request to the corresponding author, subject to a Non-Disclosure Agreement (NDA) with the operating company. Requests should outline the intended use and the institutional context.

Contact: `<corresponding author email>`.

## Why the data are restricted

The operational records contain detailed information about the electricity-consumption profile, the operational regimes, the production schedule, and the technological configuration of an active mining facility. Public release of this information would put the operating company at a competitive disadvantage.

The deposited anonymised dataset has been processed to remove identifying information (geographical location, company name, equipment serial numbers) and aggregated to a temporal resolution that preserves the scientific findings of the manuscript while removing operationally sensitive details (such as short-term production decisions).

## Layout of the directory

When raw data are obtained under NDA, the expected layout is:

```
data/raw/
├── monthly/
│   ├── monthly_feeder_307__2019-09_2024-06.xlsx
│   └── monthly_feeder_402__2019-09_2024-06.xlsx
├── hourly_2023_december/
│   ├── bus_1__december_hourly.xlsx
│   ├── bus_2__december_hourly.xlsx
│   ├── gen_cell_2__december_hourly.xlsx
│   ├── gen_cell_23__december_hourly.xlsx
│   ├── cell_307__december_hourly.xlsx
│   └── cell_402__december_hourly.xlsx
├── hourly_2023_july/
│   └── (same six files for July)
└── instrumental_2024_july/
    ├── 2024-07-15_10-21-31_feeder307.xls
    ├── 2024-07-16_06-40-00_sag_vfd.xls
    └── 2024-07-17_08-00-00_ball_mill.xls
```

See `docs/DATA_DICTIONARY.md` for the column-level description of each file.
