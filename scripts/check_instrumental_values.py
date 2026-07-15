#!/usr/bin/env python3
"""Spot-check the instrument-derived manuscript numbers against the raw export.

Background
----------
The class-A analyser export "15.07.2024 10.21.31-..." stacks SEVERAL
sub-tables on one sheet. An earlier version of the pipeline read only the
first sub-table, which produced an erroneous mean active power of 1,324 kW
instead of the correct 2,647 kW. Two further manuscript numbers come from
the same export family and must be checked for the same defect:

1. The 200 kW parasitic VFD load during the mill stoppage
   (06:40--08:50 window; feeds the 0.30 GWh/year reserve).
2. The 1.01 kWh/t mean instantaneous specific energy consumption of the
   SAG mill (16--17 July 2024; range 0.67--2.84 kWh/t).

What this script does
---------------------
For every sheet of every given file it:
  * locates ALL header rows (one per stacked sub-table) and reports the
    block structure;
  * parses each block separately;
  * for power blocks: reports per-block means of P_sigma, Q_sigma, S_sigma,
    Kp_sigma and validates the invariant P = S * Kp per block;
  * for the stoppage window (06:40--08:50): reports the mean P per block;
  * for specific-consumption columns (kWh/t): reports mean and range per
    block.

Run it locally on the raw exports (they are NDA-restricted and not in this
repository). Arguments may be individual files, glob patterns, or a
DIRECTORY — a directory is scanned for every readable export inside:

    python scripts/check_instrumental_values.py input/
    python scripts/check_instrumental_values.py "input/15.07.2024*.xml" "input/16.07*"

The file format is detected by byte signature, not extension: classic .xls,
.xlsx, and Microsoft SpreadsheetML 2003 (.xml, the analyser's native
export) are all supported.

Then compare the printed per-block values with the manuscript:
  * stoppage-window mean P should be approximately 200 kW,
  * specific consumption mean approximately 1.01 kWh/t (range 0.67--2.84).
If a value only matches when blocks are MIXED, or matches in one block but
the manuscript used another, the manuscript number inherits the sub-table
defect and must be corrected.
"""

from __future__ import annotations

import argparse
import glob
import sys
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")


class _Tee:
    """Duplicate stdout into a report file."""

    def __init__(self, path: Path):
        self._file = open(path, "w", encoding="utf-8")
        self._stdout = sys.stdout

    def write(self, s):
        self._stdout.write(s)
        self._file.write(s)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        self._file.close()


def detect_format(path: Path) -> str:
    """Detect the file format by byte signature (extension is unreliable)."""
    with open(path, "rb") as f:
        head = f.read(2048)
    if head.startswith(b"PK\x03\x04"):
        return "xlsx"
    if head.startswith(b"\xD0\xCF\x11\xE0"):
        return "xls"
    if b"<?xml" in head[:200] or b"urn:schemas-microsoft-com:office:spreadsheet" in head:
        return "spreadsheetml"
    return "unknown"


def read_spreadsheetml_all_sheets(path: Path) -> dict[str, pd.DataFrame]:
    """Parse ALL worksheets of a Microsoft SpreadsheetML 2003 XML file."""
    ns = {"ss": "urn:schemas-microsoft-com:office:spreadsheet"}
    root = ET.parse(path).getroot()
    sheets: dict[str, pd.DataFrame] = {}
    for ws in root.findall("ss:Worksheet", ns):
        name = ws.get(f"{{{ns['ss']}}}Name") or f"sheet{len(sheets)}"
        table = ws.find("ss:Table", ns)
        if table is None:
            continue
        rows = []
        for row in table.findall("ss:Row", ns):
            cells: list = []
            for cell in row.findall("ss:Cell", ns):
                idx_attr = cell.get(f"{{{ns['ss']}}}Index")
                if idx_attr is not None:
                    while len(cells) < int(idx_attr) - 1:
                        cells.append(None)
                data = cell.find("ss:Data", ns)
                cells.append(data.text if data is not None else None)
            rows.append(cells)
        sheets[name] = pd.DataFrame(rows)
    return sheets


def read_any(path: Path) -> dict[str, pd.DataFrame]:
    """Read every sheet of an export, whatever its real format is."""
    fmt = detect_format(path)
    if fmt == "spreadsheetml":
        return read_spreadsheetml_all_sheets(path)
    if fmt == "xlsx":
        return pd.read_excel(path, sheet_name=None, header=None, engine="openpyxl")
    if fmt == "xls":
        return pd.read_excel(path, sheet_name=None, header=None)
    raise ValueError(f"unrecognised format (signature: {fmt})")


import re as _re

_SIGMA = _re.compile(r"\b(p|q|s|kp)\s*[σς]")  # PΣ/QΣ/SΣ/KpΣ after lower()
# NB: str.lower() maps a word-final capital Σ to FINAL sigma ς, not σ.


def _is_header_cell(c: str) -> bool:
    return ("сумм" in c or "p sum" in c or "p_total" in c
            or _SIGMA.search(c) is not None
            or "квт·ч/т" in c or "квтч/т" in c or "квт*ч/т" in c
            or "kwh/t" in c or "удельн" in c or "удел." in c)


def find_header_rows(df: pd.DataFrame) -> list[int]:
    rows = []
    for r in range(len(df)):
        cells = [str(c).lower() for c in df.iloc[r].values]
        if any(_is_header_cell(c) for c in cells):
            rows.append(r)
    return rows


def numeric(col: pd.Series) -> pd.Series:
    return pd.to_numeric(
        col.astype(str).str.replace(",", ".", regex=False).str.replace("\xa0", ""),
        errors="coerce").dropna()


def report_block(sheet: pd.DataFrame, start: int, end: int, label: str) -> None:
    headers = [str(h) for h in sheet.iloc[start].values]
    data = sheet.iloc[start + 1:end]
    print(f"    block rows {start + 1}..{end} ({end - start - 1} data rows)")
    hdr_line = " | ".join(h[:24] for h in headers if h not in ("None", "nan") and h.strip())
    print(f"      headers: {hdr_line[:220]}")

    def _agg(hl: str) -> bool:
        return "сумм" in hl or "σ" in hl or "ς" in hl

    cols = {}
    for i, h in enumerate(headers):
        hl = h.lower().strip()
        if "Kp" not in cols and (hl.startswith("kp") and _agg(hl) or "kp_sum" in hl):
            cols["Kp"] = i
        elif "P" not in cols and (hl.startswith("p") and _agg(hl) or "p_sum" in hl):
            cols["P"] = i
        elif "Q" not in cols and (hl.startswith("q") and _agg(hl) or "q_sum" in hl):
            cols["Q"] = i
        elif "S" not in cols and (hl.startswith("s") and _agg(hl) or "s_sum" in hl):
            cols["S"] = i
        elif "spec" not in cols and ("квт·ч/т" in hl or "квтч/т" in hl
                                     or "квт*ч/т" in hl or "kwh/t" in hl
                                     or "удельн" in hl or "удел." in hl):
            cols["spec"] = i
        elif "ton" not in cols and ("т/ч" in hl or "руда" in hl or "тонн" in hl
                                    or "произв" in hl or "t/h" in hl):
            cols["ton"] = i

    vals = {}
    for name, i in cols.items():
        v = numeric(data.iloc[:, i])
        if len(v) == 0:
            continue
        vals[name] = v
        unit = {"P": "W", "Q": "var", "S": "VA", "Kp": "-", "spec": "kWh/t",
                "ton": "t/h"}[name]
        print(f"      {name:4s} [{headers[i][:30]}]: mean={v.mean():,.2f} {unit}, "
              f"min={v.min():,.2f}, max={v.max():,.2f}, n={len(v)}")

    if {"P", "S", "Kp"} <= vals.keys():
        if vals["Kp"].abs().mean() < 0.05:
            print("      Kp invariant: колонка Kp пуста/нулевая — инвариант "
                  "неприменим (не признак смешения блоков)")
        else:
            p_mean = vals["P"].mean()
            skp_mean = (vals["S"] * vals["Kp"].reindex(vals["S"].index)).dropna().mean()
            rel = abs(p_mean - skp_mean) / max(p_mean, 1e-9)
            status = "OK" if rel <= 0.02 else "VIOLATED — блоки перепутаны"
            print(f"      Kp invariant P vs S*Kp: {p_mean:,.0f} vs {skp_mean:,.0f} "
                  f"({100 * rel:.1f}%) -> {status}")

    if {"P", "ton"} <= vals.keys() and "spec" not in vals:
        p_kw = vals["P"] / 1000.0
        ton = vals["ton"].reindex(p_kw.index)
        spec = (p_kw / ton).replace([float("inf"), -float("inf")], pd.NA).dropna()
        spec = spec[(ton.reindex(spec.index) > 0)]
        if len(spec) > 0:
            print(f"      spec = P/тоннаж: mean={spec.mean():.2f} kWh/t, "
                  f"min={spec.min():.2f}, max={spec.max():.2f}, n={len(spec)} "
                  f"(рукопись: mean 1.01, диапазон 0.67--2.84)")

    # Stoppage window, if the first column parses as a time or datetime
    t = pd.to_datetime(data.iloc[:, 0], errors="coerce", dayfirst=True)
    if t.notna().sum() > 0 and "P" in cols:
        tod = t.dt.time
        in_win = data[(tod >= pd.Timestamp("06:40").time())
                      & (tod <= pd.Timestamp("08:50").time())]
        pwin = numeric(in_win.iloc[:, cols["P"]]) if len(in_win) > 0 else pd.Series(dtype=float)
        if len(pwin) > 0:
            print(f"      stoppage window 06:40--08:50: mean P = "
                  f"{pwin.mean() / 1000:,.1f} kW over {len(pwin)} rows "
                  f"(manuscript: ~200 kW)")
        else:
            print("      stoppage window 06:40--08:50: строк в окне нет")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Audit analyser exports for the stacked sub-table defect.")
    ap.add_argument("inputs", nargs="+", help="files, glob patterns, or a directory")
    ap.add_argument("-o", "--out", type=Path, default=Path("instrument_check_report.txt"),
                    help="write the full report to this file as well (default: instrument_check_report.txt)")
    args = ap.parse_args(argv)

    tee = _Tee(args.out)
    sys.stdout = tee
    try:
        return _run(args.inputs, args.out)
    finally:
        sys.stdout = tee._stdout
        tee.close()


def _run(patterns: list[str], out_path: Path) -> int:
    files: list[Path] = []
    for pat in patterns:
        p = Path(pat)
        if p.is_dir():
            files += sorted(f for f in p.iterdir() if f.is_file())
        elif p.is_file():
            files.append(p)
        else:
            files += sorted(Path(m) for m in glob.glob(pat))
    if not files:
        print("No files matched. Pass files, glob patterns, or a directory.")
        return 1
    print(f"Found {len(files)} file(s):")
    for f in files:
        print(f"  {f.name}  [{detect_format(f)}]")

    for path in files:
        print(f"\n=== {path.name} ===")
        try:
            sheets = read_any(path)
        except Exception as exc:
            print(f"  cannot read: {exc}")
            continue
        matched_any = False
        for name, sheet in sheets.items():
            hdrs = find_header_rows(sheet)
            if not hdrs:
                continue
            matched_any = True
            print(f"  sheet '{name}': {len(hdrs)} sub-table(s) "
                  f"{'<-- BLOCK STRUCTURE, проверить какой блок попал в статью' if len(hdrs) > 1 else ''}")
            bounds = hdrs + [len(sheet)]
            for k in range(len(hdrs)):
                report_block(sheet, bounds[k], bounds[k + 1], name)
        if not matched_any:
            print("  [диагностика] знакомые заголовки не найдены; "
                  "первые строки каждого листа:")
            for name, sheet in sheets.items():
                print(f"    лист '{name}' ({sheet.shape[0]}x{sheet.shape[1]}):")
                shown = 0
                for r in range(min(len(sheet), 40)):
                    vals = [str(v)[:18] for v in sheet.iloc[r].values[:10]
                            if v is not None and str(v) != "nan" and str(v).strip()]
                    if not vals:
                        continue
                    print("      row %-3d | %s" % (r, " | ".join(vals)))
                    shown += 1
                    if shown >= 5:
                        break
    print("\nСравните значения по блокам с числами в статье: 200 kW (VFD, окно "
          "остановки) и 1.01 kWh/t (среднее удельное, диапазон 0.67--2.84). "
          "Если значение статьи совпадает только при смешении блоков — оно "
          "унаследовало дефект и требует замены.")
    print(f"\nПолный отчёт записан в: {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
