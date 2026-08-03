"""
make_table.py
----------------
Generates a manuscript-ready demographic summary (age, sex/gender,
race, ethnicity)
Runs entirely locally -- no patient data leaves your machine.

USAGE
-----
    python make_table.py "/path/to/your/data.xlsx"

Optional: stratify by a column (e.g. Site or Cancer Type):
    python make_table.py "/path/to/your/data.xlsx" --groupby "Cancer Type"

OUTPUT
------
Writes, next to the input file:
    table_demographics.csv
    table_demographics.docx   (manuscript-ready Word table)

REQUIREMENTS (install once)
----------------------------
    pip install tableone openpyxl python-docx pandas
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from tableone import TableOne
from docx import Document
from docx.shared import Pt


def compute_age(df: pd.DataFrame) -> pd.Series:
    """Age at encounter, in years, from Patient DOB and Date of Encounter."""
    dob = pd.to_datetime(df["Patient DOB"], errors="coerce")
    encounter = pd.to_datetime(df["Date of Encounter"], errors="coerce")
    age_years = (encounter - dob).dt.days / 365.25
    return age_years


def clean_race(series: pd.Series, other_series: pd.Series) -> pd.Series:
    """Fold free-text 'Other (Specify)' responses into the Race field as 'Other'."""
    race = series.astype("string").str.strip()
    has_other_text = other_series.astype("string").str.strip().replace("", pd.NA).notna()
    race = race.mask(race.str.lower().eq("other") & has_other_text, "Other")
    return race


def build_table(df: pd.DataFrame, groupby: str | None) -> TableOne:
    work = pd.DataFrame()
    work["Age, y"] = compute_age(df)
    work["Sex/Gender"] = df["Patient Gender"].astype("string").str.strip()
    work["Race"] = clean_race(df["What Is Your Race?"], df.get("Other (Specify)", pd.Series(dtype="string")))
    work["Ethnicity"] = df["Are You Hispanic Or Latino?"].astype("string").str.strip()

    columns = ["Age, y", "Sex/Gender", "Race", "Ethnicity"]
    categorical = ["Sex/Gender", "Race", "Ethnicity"]

    if groupby:
        work[groupby] = df[groupby]
        columns.append(groupby)

    table = TableOne(
        work,
        columns=columns,
        categorical=categorical,
        groupby=groupby,
        pval=bool(groupby),
        missing=True,
    )
    return table


def write_docx(table: TableOne, out_path: Path, title: str) -> None:
    doc = Document()
    doc.add_heading(title, level=1)

    df = table.tableone
    # tableone returns a MultiIndex frame; flatten for a clean Word table
    flat = df.reset_index()
    flat.columns = [" ".join(str(c) for c in col if c not in ("", None)).strip()
                    for col in flat.columns.to_flat_index()]

    n_rows, n_cols = flat.shape
    word_table = doc.add_table(rows=n_rows + 1, cols=n_cols)
    word_table.style = "Light Grid Accent 1"

    for j, col_name in enumerate(flat.columns):
        cell = word_table.cell(0, j)
        cell.text = str(col_name)
        for p in cell.paragraphs:
            for r in p.runs:
                r.bold = True
                r.font.size = Pt(9)

    for i in range(n_rows):
        for j in range(n_cols):
            val = flat.iat[i, j]
            word_table.cell(i + 1, j).text = "" if pd.isna(val) else str(val)
            for p in word_table.cell(i + 1, j).paragraphs:
                for r in p.runs:
                    r.font.size = Pt(9)

    doc.save(out_path)


def main():
    parser = argparse.ArgumentParser(description="Build a Table 1 demographic summary.")
    parser.add_argument("input_path", help="Path to the REDCap export .xlsx file")
    parser.add_argument(
        "--groupby",
        default=None,
        help="Optional column to stratify by, e.g. 'Site' or 'Cancer Type'",
    )
    args = parser.parse_args()

    in_path = Path(args.input_path)
    if not in_path.exists():
        sys.exit(f"File not found: {in_path}")

    df = pd.read_excel(in_path)

    required_cols = [
        "Patient DOB",
        "Date of Encounter",
        "Patient Gender",
        "What Is Your Race?",
        "Are You Hispanic Or Latino?",
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        sys.exit(f"Missing expected column(s) in the file: {missing_cols}")

    table = build_table(df, args.groupby)

    out_dir = in_path.parent
    csv_path = out_dir / "table_demographics.csv"
    docx_path = out_dir / "table_demographics.docx"

    table.tableone.to_csv(csv_path)
    write_docx(table, docx_path, title="Table 1. Demographic Characteristics of Patient Sample")

    print(table.tableone)
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {docx_path}")


if __name__ == "__main__":
    main()