"""
make_table.py
----------------
Generates a manuscript-ready Table.

USAGE
-----
    python make_table.py "/path/to/your/data.xlsx"

If the file is password-protected, just run it the same way -- you'll be
prompted securely for the password (input is hidden, nothing is logged or
written to disk):
    python make_table.py "/path/to/your/protected_data.xlsx"

You can also pass the password directly (less safe -- ends up in shell
history), or via an environment variable (safer, for scripting/automation):
    python make_table.py "/path/to/your/data.xlsx" --password "secret"
    set REDCAP_XLSX_PASSWORD=secret   &&  python make_table.py "/path/to/data.xlsx"

Optional: stratify by a column (e.g. Site or Cancer Type):
    python make_table.py "/path/to/your/data.xlsx" --groupby "Cancer Type"

Optional: stratify by MULTIPLE columns, each getting its own row-block
(e.g. Gender's categories, then Race's categories, then Ethnicity's
categories -- shown separately, not merged into one cross-tab):
    python make_table.py "/path/to/your/data.xlsx" --groupby "Patient Gender,What Is Your Race?,Are You Hispanic Or Latino?"

WHAT'S INCLUDED
---------------
Every column in the file is included EXCEPT:
    - Record ID, Audio File Name, Physician  (identifiers/labels, not
      summarizable as a table row)
    - Patient DOB, Date of Encounter, Date of Survey  (used to compute Age,
      not shown as raw dates)
    - Other (Specify)  (folded into the Race field as "Other")

Categorical vs. continuous is auto-detected: any column with 10 or fewer
unique values is treated as categorical (counts/percentages); anything with
more unique numeric values is treated as continuous (mean/SD). If this
guesses wrong for a particular column, use --force-categorical or
--force-continuous (comma-separated column names) to override.

OUTPUT
------
Writes, next to the input file:
    table1_full.csv

REQUIREMENTS (install once)
----------------------------
    pip install tableone openpyxl pandas msoffcrypto-tool
"""

import argparse
import difflib
import getpass
import io
import os
import re
import sys
from pathlib import Path

import msoffcrypto
import pandas as pd
from tableone import TableOne


EXCLUDE_ALWAYS = [
    "Record ID",
    "Audio File Name",
    "Physician",
    "Patient DOB",
    "Date of Encounter",
    "Date of Survey",
    "Other (Specify)",
]

CATEGORICAL_MAX_UNIQUE = 10


def load_excel(in_path: Path, password: str | None) -> pd.DataFrame:
    """
    Load an Excel file, transparently handling password-protected workbooks.
    Decryption happens in memory only -- no unencrypted copy is ever written
    to disk.
    """
    # First, try opening normally (handles unprotected .xlsx files).
    try:
        return pd.read_excel(in_path, engine="openpyxl")
    except Exception as first_error:
        # Check whether the file is actually encrypted before assuming so.
        with open(in_path, "rb") as f:
            office_file = msoffcrypto.OfficeFile(f)
            is_encrypted = office_file.is_encrypted()

        if not is_encrypted:
            # Not a password issue -- surface the original error.
            raise first_error

        pw = password or os.environ.get("REDCAP_XLSX_PASSWORD")
        if not pw:
            pw = getpass.getpass(
                f"'{in_path.name}' is password-protected. Enter password: "
            )

        with open(in_path, "rb") as f:
            office_file = msoffcrypto.OfficeFile(f)
            office_file.load_key(password=pw)
            decrypted = io.BytesIO()
            office_file.decrypt(decrypted)

        try:
            return pd.read_excel(decrypted, engine="openpyxl")
        except Exception:
            sys.exit(
                "Could not open the file after decryption -- the password "
                "was likely incorrect. Please try again."
            )


def _normalize(name: str) -> str:
    """Collapse whitespace/newlines and lowercase, for fuzzy column matching."""
    return re.sub(r"\s+", " ", str(name)).strip().lower()


def resolve_columns(df: pd.DataFrame, expected: list[str]) -> dict[str, str]:
    """
    Map each expected column name to whatever column actually exists in df,
    tolerating things like extra whitespace, line breaks in the header cell,
    or minor character differences (common in REDCap exports).

    Returns a dict {expected_name: actual_column_name_in_df}.
    Raises with a helpful message (including close-match suggestions) if a
    column truly can't be found.
    """
    actual_cols = list(df.columns)
    normalized_lookup = {_normalize(c): c for c in actual_cols}

    resolved = {}
    unresolved = []

    for exp in expected:
        # 1. Exact match
        if exp in actual_cols:
            resolved[exp] = exp
            continue
        # 2. Whitespace/case-insensitive match
        norm_exp = _normalize(exp)
        if norm_exp in normalized_lookup:
            resolved[exp] = normalized_lookup[norm_exp]
            continue
        # 3. Fuzzy match as a last resort
        close = difflib.get_close_matches(norm_exp, normalized_lookup.keys(), n=1, cutoff=0.8)
        if close:
            resolved[exp] = normalized_lookup[close[0]]
            continue
        unresolved.append(exp)

    if unresolved:
        lines = ["Could not find the following expected column(s) in your file:"]
        for exp in unresolved:
            suggestions = difflib.get_close_matches(
                _normalize(exp), normalized_lookup.keys(), n=3, cutoff=0.4
            )
            suggestion_text = (
                ", ".join(f'"{normalized_lookup[s]}"' for s in suggestions)
                if suggestions
                else "(no close matches found)"
            )
            lines.append(f'  - Expected "{exp}" -> closest columns in your file: {suggestion_text}')
        lines.append("\nAll columns found in your file:")
        for c in actual_cols:
            lines.append(f"  - {c!r}")
        sys.exit("\n".join(lines))

    return resolved


def compute_age(df: pd.DataFrame, dob_col: str, encounter_col: str) -> pd.Series:
    """Age at encounter, in years, from Patient DOB and Date of Encounter."""
    dob = pd.to_datetime(df[dob_col], errors="coerce")
    encounter = pd.to_datetime(df[encounter_col], errors="coerce")
    age_years = (encounter - dob).dt.days / 365.25
    return age_years


def clean_race(series: pd.Series, other_series: pd.Series) -> pd.Series:
    """Fold free-text 'Other (Specify)' responses into the Race field as 'Other'."""
    race = series.astype("string").str.strip()
    has_other_text = other_series.astype("string").str.strip().replace("", pd.NA).notna()
    race = race.mask(race.str.lower().eq("other") & has_other_text, "Other")
    return race


def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, set]:
    """
    Clean/derive fields (Age, Race) once, and figure out which raw columns
    should never appear as their own rows (IDs, free text, raw dates).
    Returns (work_df, excluded_column_names).
    """
    core_required = [
        "Patient DOB",
        "Date of Encounter",
        "What Is Your Race?",
        "Are You Hispanic Or Latino?",
    ]
    cols = resolve_columns(df, core_required)

    other_col = None
    for candidate in df.columns:
        if _normalize(candidate) == _normalize("Other (Specify)"):
            other_col = candidate
            break

    work = df.copy()
    work["Age, y"] = compute_age(df, cols["Patient DOB"], cols["Date of Encounter"])
    work[cols["What Is Your Race?"]] = clean_race(
        df[cols["What Is Your Race?"]],
        df[other_col] if other_col is not None else pd.Series(dtype="string", index=df.index),
    )

    exclude_resolved = set()
    for name in EXCLUDE_ALWAYS:
        norm = _normalize(name)
        match = next((c for c in df.columns if _normalize(c) == norm), None)
        if match:
            exclude_resolved.add(match)

    return work, exclude_resolved


def build_table(
    work: pd.DataFrame,
    exclude_resolved: set,
    groupby: str | None,
    force_categorical: list[str] | None,
    force_continuous: list[str] | None,
) -> TableOne:
    columns = [c for c in work.columns if c not in exclude_resolved]
    if groupby and groupby not in columns:
        columns.append(groupby)

    # Auto-detect categorical vs. continuous.
    force_cat = set(force_categorical or [])
    force_cont = set(force_continuous or [])
    categorical = []
    for c in columns:
        if c in (groupby, "Age, y"):
            continue  # Age is always continuous; groupby handled separately.
        if c in force_cat:
            categorical.append(c)
            continue
        if c in force_cont:
            continue
        nunique = work[c].nunique(dropna=True)
        is_numeric = pd.api.types.is_numeric_dtype(work[c])
        if not is_numeric or nunique <= CATEGORICAL_MAX_UNIQUE:
            categorical.append(c)

    table = TableOne(
        work,
        columns=columns,
        categorical=categorical,
        groupby=groupby,
        pval=bool(groupby),
        missing=False,
    )
    return table


def transpose_table(table: TableOne) -> pd.DataFrame:
    """
    Reshape a TableOne result so groups are ROWS and each field/category is
    its own COLUMN (transposed from tableone's default layout).
    """
    flat = table.tableone.copy()

    # Flatten the row index: (variable, category) -> "Variable - Category"
    flat.index = [
        " - ".join(str(x) for x in idx if x not in ("", None))
        for idx in flat.index.to_flat_index()
    ]

    # Flatten the column index if it's a MultiIndex (happens when grouped,
    # since tableone adds a "Grouped by X" header level above group names).
    if isinstance(flat.columns, pd.MultiIndex):
        flat.columns = [
            " ".join(str(x) for x in col if x not in ("", None)).strip()
            for col in flat.columns.to_flat_index()
        ]

    transposed = flat.T
    transposed.index.name = "Group"
    return transposed


def main():
    parser = argparse.ArgumentParser(description="Build a full Table 1 summary (every field).")
    parser.add_argument("input_path", help="Path to the REDCap export .xlsx file")
    parser.add_argument(
        "--groupby",
        default=None,
        help=(
            "Column(s) to stratify by, e.g. 'Site' or 'Cancer Type'. "
            "Comma-separate multiple columns (e.g. 'Patient Gender,What Is "
            "Your Race?,Are You Hispanic Or Latino?') to get a separate "
            "row-block of results for each one (Gender's categories, then "
            "Race's categories, then Ethnicity's categories -- each on "
            "their own rows, not merged together)."
        ),
    )
    parser.add_argument(
        "--password",
        default=None,
        help=(
            "Password for a protected workbook. If omitted and the file is "
            "encrypted, you'll be prompted securely (input hidden)."
        ),
    )
    parser.add_argument(
        "--force-categorical",
        default=None,
        help="Comma-separated column names to force as categorical.",
    )
    parser.add_argument(
        "--force-continuous",
        default=None,
        help="Comma-separated column names to force as continuous.",
    )
    args = parser.parse_args()

    in_path = Path(args.input_path)
    if not in_path.exists():
        sys.exit(f"File not found: {in_path}")

    df = load_excel(in_path, args.password)

    groupby_cols = None
    if args.groupby:
        requested = [c.strip() for c in args.groupby.split(",")]
        resolved = resolve_columns(df, requested)
        groupby_cols = [resolved[c] for c in requested]

    force_categorical = (
        [c.strip() for c in args.force_categorical.split(",")] if args.force_categorical else None
    )
    force_continuous = (
        [c.strip() for c in args.force_continuous.split(",")] if args.force_continuous else None
    )

    work, exclude_resolved = prepare_data(df)

    out_dir = in_path.parent
    csv_path = out_dir / "table1_full.csv"

    if not groupby_cols:
        table = build_table(work, exclude_resolved, None, force_categorical, force_continuous)
        transposed = transpose_table(table)
        print(table.tableone)
    elif len(groupby_cols) == 1:
        table = build_table(work, exclude_resolved, groupby_cols[0], force_categorical, force_continuous)
        transposed = transpose_table(table)
        print(table.tableone)
    else:
        # Build one table PER grouping column, so each variable's categories
        # get their own row-block (e.g. Gender's rows, then Race's rows,
        # then Ethnicity's rows) instead of a single merged cross-tab.
        blocks = []
        for col in groupby_cols:
            table = build_table(work, exclude_resolved, col, force_categorical, force_continuous)
            block = transpose_table(table)
            blocks.append(block)
            print(f"\n--- Grouped by {col} ---")
            print(table.tableone)
        transposed = pd.concat(blocks, axis=0)

    transposed.to_csv(csv_path)
    print(f"\nSaved: {csv_path} (groups as rows, fields as columns)")


if __name__ == "__main__":
    main()