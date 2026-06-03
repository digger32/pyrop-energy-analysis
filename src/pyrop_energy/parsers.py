"""Data loaders for the three source types used in the study.

Source 1: monthly SCADA Excel archive of feeders 307 / 402 (Sept 2019 – June 2024).
Source 2: hourly SCADA Excel archive of the six 6 kV sources (December and July 2023).
Source 3: instrumental measurements in Microsoft SpreadsheetML 2003 XML format,
          recorded by a class-A power-quality analyser during 15-17 July 2024.

The instrumental files contain 8 worksheets per session ("Currents and voltages",
"Powers", "Angles", "Power-quality indices", "Current harmonics", "Voltage
harmonics", "Current intergroup harmonics", "Voltage intergroup harmonics") at
sub-minute resolution. Timestamps in the source contain only time-of-day; the
loader reconstructs full timestamps using the session date extracted from the
filename via a deterministic regular expression.

Decimal commas are converted to decimal points; aggregate rows ("итого", "сумма")
present in the source are filtered out at parse time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import re


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def detect_format(path: Path) -> str:
    """Detect the actual file format by inspecting the first bytes.

    Microsoft Excel and Microsoft SpreadsheetML 2003 share the .xls / .xlsx
    extension in some exports of legacy SCADA tools, but their byte signatures
    are different. This helper returns one of: 'xlsx', 'xls', 'spreadsheetml',
    'csv', 'unknown'.
    """
    with open(path, "rb") as f:
        head = f.read(2048)
    if head.startswith(b"PK\x03\x04"):
        return "xlsx"
    if head.startswith(b"\xD0\xCF\x11\xE0"):
        return "xls"
    if b"<?xml" in head[:200] or b"urn:schemas-microsoft-com:office:spreadsheet" in head:
        return "spreadsheetml"
    try:
        head.decode("utf-8")
        if b"," in head or b";" in head:
            return "csv"
    except UnicodeDecodeError:
        pass
    return "unknown"


# ---------------------------------------------------------------------------
# Hourly / monthly SCADA Excel
# ---------------------------------------------------------------------------

def load_ore_file(path: Path) -> Optional[pd.DataFrame]:
    """Load the hourly archive of one 6 kV source from a SCADA Excel file.

    The archive contains datetime, active power, and reactive power columns.
    Returns a DataFrame indexed by timestamp with columns P (kW) and Q (kvar),
    or None if the file cannot be parsed.

    Notes
    -----
    The loader tolerates several variants seen in the field: comma decimal
    separators, mixed-language column headers ("Активная мощность", "P, кВт",
    "Active power", etc.), aggregate footer rows, and partially-filled sheets
    where only a contiguous block contains valid dates.
    """
    fmt = detect_format(path)
    if fmt == "xlsx":
        df_raw = pd.read_excel(path, header=None, engine="openpyxl")
    elif fmt == "xls":
        df_raw = pd.read_excel(path, header=None, engine="xlrd")
    elif fmt == "spreadsheetml":
        df_raw = _parse_spreadsheetml_2003(path)
    else:
        return None

    # Find the longest contiguous block of rows where the first column parses
    # as a date in 1990-2100. This skips header rows, footers, and stray cells.
    parsed = pd.to_datetime(df_raw.iloc[:, 0], errors="coerce")
    valid = parsed.between(pd.Timestamp("1990-01-01"), pd.Timestamp("2100-01-01"))
    if not valid.any():
        return None

    block = _longest_contiguous(valid)
    if block is None:
        return None
    start, end = block

    sub = df_raw.iloc[start:end + 1].copy()
    sub.iloc[:, 0] = pd.to_datetime(sub.iloc[:, 0], errors="coerce")
    sub = sub.dropna(subset=[sub.columns[0]])

    # Heuristic column mapping: P in column with header containing "акт"/"P",
    # Q in column with header containing "реакт"/"Q".
    p_idx, q_idx = _detect_pq_columns(df_raw, header_search_rows=start)
    if p_idx is None or q_idx is None:
        # Fallback: assume column order is timestamp / P / Q.
        p_idx, q_idx = 1, 2

    out = pd.DataFrame({
        "P": pd.to_numeric(_clean_decimal(sub.iloc[:, p_idx]), errors="coerce"),
        "Q": pd.to_numeric(_clean_decimal(sub.iloc[:, q_idx]), errors="coerce"),
    }, index=pd.DatetimeIndex(sub.iloc[:, 0].values))

    out = out.dropna()
    out.index.name = "timestamp"
    return out


def _clean_decimal(series: pd.Series) -> pd.Series:
    """Convert European decimal commas to dots and strip thousands separators."""
    return (series.astype(str)
                  .str.replace("\u00A0", "", regex=False)
                  .str.replace(" ", "", regex=False)
                  .str.replace(",", ".", regex=False))


def _detect_pq_columns(df_raw: pd.DataFrame, header_search_rows: int) -> tuple[Optional[int], Optional[int]]:
    """Find P and Q columns by header text in the rows above the data block."""
    p_idx = q_idx = None
    for r in range(min(header_search_rows, len(df_raw))):
        for c in range(df_raw.shape[1]):
            cell = str(df_raw.iat[r, c]).lower()
            if p_idx is None and (re.search(r"\bакт", cell) or re.search(r"\bp\b", cell)):
                p_idx = c
            if q_idx is None and (re.search(r"\bреакт", cell) or re.search(r"\bq\b", cell)):
                q_idx = c
    return p_idx, q_idx


def _longest_contiguous(mask: pd.Series) -> Optional[tuple[int, int]]:
    """Return (start, end) indices of the longest run of True values in mask."""
    best = None
    best_len = 0
    cur_start = None
    cur_len = 0
    for i, v in enumerate(mask.values):
        if v:
            if cur_start is None:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_len = cur_len
                best = (cur_start, i)
        else:
            cur_start = None
            cur_len = 0
    return best


# ---------------------------------------------------------------------------
# SpreadsheetML 2003 XML
# ---------------------------------------------------------------------------

def _parse_spreadsheetml_2003(path: Path) -> pd.DataFrame:
    """Parse a Microsoft SpreadsheetML 2003 XML file into a flat DataFrame.

    SpreadsheetML 2003 is the legacy XML format used by older Excel exports
    and some SCADA tools. We parse the first worksheet, treating empty cells
    as NaN and converting numeric strings.
    """
    import xml.etree.ElementTree as ET
    ns = {"ss": "urn:schemas-microsoft-com:office:spreadsheet"}
    tree = ET.parse(path)
    root = tree.getroot()
    rows = []
    for ws in root.findall("ss:Worksheet", ns):
        for row in ws.find("ss:Table", ns).findall("ss:Row", ns):
            cells = []
            for cell in row.findall("ss:Cell", ns):
                idx_attr = cell.get(f"{{{ns['ss']}}}Index")
                if idx_attr is not None:
                    while len(cells) < int(idx_attr) - 1:
                        cells.append(None)
                data = cell.find("ss:Data", ns)
                cells.append(data.text if data is not None else None)
            rows.append(cells)
        break  # Only the first worksheet is used for ore-file parsing.
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Mill instrumental sessions (8-sheet xlsx, July 2024)
# ---------------------------------------------------------------------------

def load_mill_session(path: Path) -> Optional[pd.DataFrame]:
    """Load the "Powers" sheet of an instrumental mill-session file.

    Returns a DataFrame with columns P (kW) and Q (kvar) at the native
    sub-minute resolution of the analyser. The session date is reconstructed
    from the filename pattern YYYY-MM-DD or DD-MM-YYYY.

    The instrument records P_sigma and Q_sigma (suite-aggregated three-phase
    quantities) in W and var; the loader converts to kW and kvar.
    """
    fmt = detect_format(path)
    sheets = (pd.read_excel(path, sheet_name=None, header=None, engine="openpyxl")
              if fmt == "xlsx" else pd.read_excel(path, sheet_name=None, header=None))

    # Find the powers sheet (usually named "Мощности" or "Powers")
    powers_sheet = None
    for name, df in sheets.items():
        if "мощност" in name.lower() or "power" in name.lower():
            powers_sheet = df
            break
    if powers_sheet is None:
        return None

    base_date = _extract_session_date(path.name)

    header_row = _find_header_row(powers_sheet)
    if header_row is None:
        return None
    headers = powers_sheet.iloc[header_row].astype(str).tolist()

    p_col = q_col = None
    for i, h in enumerate(headers):
        h_low = h.lower()
        if p_col is None and ("сумм" in h_low and "p" in h_low or "p_sum" in h_low):
            p_col = i
        if q_col is None and ("сумм" in h_low and "q" in h_low or "q_sum" in h_low):
            q_col = i
    # Fallback to fixed column positions used by the analyser default export.
    if p_col is None: p_col = 13
    if q_col is None: q_col = 14

    data = powers_sheet.iloc[header_row + 1:].copy()
    time_series = pd.to_datetime(data.iloc[:, 0], errors="coerce")
    timestamps = _build_full_datetimes(time_series, base_date)
    if timestamps is None:
        return None

    p_kW   = pd.to_numeric(_clean_decimal(data.iloc[:, p_col]), errors="coerce") / 1000.0
    q_kvar = pd.to_numeric(_clean_decimal(data.iloc[:, q_col]), errors="coerce") / 1000.0

    out = pd.DataFrame({"P": p_kW.values, "Q": q_kvar.values}, index=timestamps)
    out = out.dropna()
    out.index.name = "timestamp"
    return out


def _extract_session_date(filename: str) -> Optional[pd.Timestamp]:
    """Extract a date from the filename pattern YYYY-MM-DD or DD-MM-YYYY."""
    m = re.search(r"(\d{4})[-_](\d{2})[-_](\d{2})", filename)
    if m:
        return pd.Timestamp(year=int(m.group(1)), month=int(m.group(2)), day=int(m.group(3)))
    m = re.search(r"(\d{2})[-_](\d{2})[-_](\d{4})", filename)
    if m:
        return pd.Timestamp(year=int(m.group(3)), month=int(m.group(2)), day=int(m.group(1)))
    return None


def _find_header_row(df: pd.DataFrame, max_rows: int = 8) -> Optional[int]:
    """Locate the header row by looking for cells matching power-related labels."""
    for r in range(min(max_rows, len(df))):
        cells = [str(c).lower() for c in df.iloc[r].values]
        if any("p_сумм" in c or "p sum" in c or "p_total" in c or "сумм" in c for c in cells):
            return r
    return None


def _build_full_datetimes(time_series: pd.Series, base_date: Optional[pd.Timestamp]):
    """Combine time-of-day with a session date to form full timestamps.

    The instrument output contains only times like '08:30:00'. The session
    date comes from the filename. When the time wraps past midnight, the
    next day's date is used.
    """
    if base_date is None:
        return None
    out = []
    prev_t = None
    cur_date = base_date
    for t in time_series:
        if pd.isna(t):
            out.append(pd.NaT)
            continue
        # Take only the time-of-day component
        tod = t.time()
        full = pd.Timestamp.combine(cur_date.date(), tod)
        if prev_t is not None and full < prev_t:
            cur_date = cur_date + pd.Timedelta(days=1)
            full = pd.Timestamp.combine(cur_date.date(), tod)
        out.append(full)
        prev_t = full
    return pd.DatetimeIndex(out)
