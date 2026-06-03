# Data dictionary

This document describes the structure of the source data files, the variables they contain, and the conventions used by the parsers.

## 1. Source types

The study draws on three independent sources, distinguished by temporal resolution and recording technology.

### 1.1. Monthly SCADA archive (Source 1)

- **Period covered:** September 2019 – June 2024 (~57 months)
- **Sources:** feeder 307, feeder 402 (the two main 6 kV factory feeders)
- **Format:** Microsoft Excel (`.xlsx`)
- **Resolution:** monthly mean of the active and reactive power
- **Columns:** timestamp, P_mean (kW), Q_mean (kvar)
- **Naming convention:** `monthly_<feeder>__<start>_<end>.xlsx`

### 1.2. Hourly SCADA archive (Source 2)

- **Periods covered:** December 2023 (3,648 h) and July 2023 (5,064 h)
- **Sources:** bus 1, bus 2 (grid inputs); feeder 307, feeder 402 (factory); cell 2, cell 23 (cogeneration)
- **Format:** Microsoft Excel (`.xlsx`) — six files per period (one per source)
- **Resolution:** 1 hour
- **Columns:** timestamp, P (kW), Q (kvar)
- **Naming convention:** `<source>__<period>_hourly.xlsx`, where `<period>` ∈ `{december, july}`

### 1.3. Instrumental measurements (Source 3)

- **Period:** 15–17 July 2024 (six measurement sessions)
- **Sources:** feeder 307 input, SAG-mill VFD cell, ball-mill cell
- **Format:** Microsoft SpreadsheetML 2003 (`.xls` with XML internals)
- **Resolution:** 0.2–10 seconds (analyser-dependent)
- **Worksheets** (8 per session):
  1. Currents and voltages — `I_A`, `I_B`, `I_C`, `U_AB`, `U_BC`, `U_CA`
  2. Powers — `P_A`, `P_B`, `P_C`, `P_total`, `Q_A`, `Q_B`, `Q_C`, `Q_total`, `S_total`, `cos_phi`
  3. Angles — power-factor angles per phase
  4. Power-quality indices — k_U, k_I, voltage-unbalance coefficients
  5. Current harmonics — k_I,h for h in {2, 3, …, 40}
  6. Line-voltage harmonics — k_U,h
  7. Current intergroup harmonics
  8. Line-voltage intergroup harmonics

The instrumental files contain only the time-of-day in the timestamp column; the date is implicit and is reconstructed by `parsers.load_mill_session` from the filename pattern. When a session crosses midnight, the parser detects the wrap and increments the date.

## 2. Conventions

- **Decimal separator.** SCADA exports use the European decimal comma (`,`). The parsers convert to dot.
- **Missing values.** Empty cells are treated as NaN. Aggregate footer rows ("итого", "сумма", "TOTAL") are filtered out at parse time.
- **Time zone.** All timestamps are in local time of the operating facility (Yekaterinburg, UTC+5). No daylight-saving correction is applied (Russia does not observe DST).
- **Units.** P in kW, Q in kvar. The instrumental analyser reports W and var in the source files; the parser converts.
- **Sign convention.** P and Q are positive when consumed by the load and negative when delivered to the grid. The cogeneration cells therefore have negative values when exporting power to the bus.

## 3. Anonymised dataset (Zenodo)

The processed and anonymised dataset deposited on Zenodo contains:

```
processed/
├── panel_P_hourly.csv                 wide panel of P, all 12 source-period combinations
├── panel_Q_hourly.csv                 wide panel of Q
├── monthly_archive.csv                monthly-aggregated feeder 307 / 402
├── instrumental_<session>.csv         per-session instrumental measurements
├── feature_matrix_<source>.csv        feature matrices used for the ML benchmark
└── README.md                          dataset-level README
```

Names of the operating facility, the geographical location, and any company-internal identifiers are removed or replaced with neutral labels (`bus_1`, `bus_2`, `gen_cell_2`, etc.).

## 4. Data quality notes

- `cell_307_december` and `cell_402_december` have only 52 hourly observations (a 2-day window). They are present in the panel for context but are filtered as "orphans" by `features.build_feature_matrix` when used as panel-aggregate features for other sources, and are skipped by the ML benchmark because they are below the minimum-length threshold of 400 hours.
- The instrumental files contain occasional zero-rows during instrument startup. The parser drops rows with all-zero P, Q values.
- Time gaps of less than 2 hours are forward-filled in the hourly panel to maintain a regular index. Larger gaps are left as NaN and propagate through the dropna() steps.
