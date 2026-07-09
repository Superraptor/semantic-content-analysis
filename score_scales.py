"""
score_scales.py

Computes scored scale variables from a REDCap CSV export (exported with field
labels as column headers) for the "Improving Advanced Cancer Patient-Centered
Care by Enabling Goals of Care Discussions" study.

Usage:
    # Extract only the relevant fields (no scoring yet — good first step):
    python score_scales.py --input full_export.csv --output extracted.csv --extract-only

    # Score all patients:
    python score_scales.py --input full_export.csv --output scored.csv

    # Score a specific subset of patient IDs:
    python score_scales.py --input full_export.csv --output scored.csv --patients 101 102 103

Requirements: pandas
    pip install pandas
"""

import argparse
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# COLUMN LABEL MAP
# Maps from the CSV column header (as REDCap exports with field labels) to
# the internal variable name used in the scoring functions below.
#
# DUPLICATE COLUMN HANDLING:
# When pandas reads a CSV with duplicate column names it adds .1, .2 ...
# suffixes automatically. The skill assessment items appear TWICE in the
# export (Reviewer 1 then Reviewer 2). Reviewer 1 keeps the raw label
# (no suffix); Reviewer 2 gets a .1 suffix.
# Items whose labels differ slightly between R1 and R2 (typos in the
# original instrument) appear only once each and need no suffix.
#
# PHQ-9 items appear in both the baseline and 6-month follow-up surveys.
# We use the first occurrence (no suffix) = baseline.
# ---------------------------------------------------------------------------

COLUMN_LABEL_MAP = {

    # -----------------------------------------------------------------------
    # PHQ-9 — English (baseline survey, first occurrence = no suffix)
    # -----------------------------------------------------------------------
    "3. Little interest or pleasure in doing things":
        "phq21",
    "4. Feeling down, depressed, or hopeless":
        "phq22",
    "5. Trouble falling asleep or staying asleep or sleeping too much":
        "phq23",
    "6. Feeling tired or having little energy":
        "phq24",
    "7. Poor appetite or overeating":
        "phq25",
    "8. Feeling bad about yourself-or that you are a failure or have let yourself or your family down":
        "phq26",
    "9. Trouble concentrating on things, such as reading the newspaper or watching television":
        "phq27",
    "10. Moving or speaking so slowly that other people could have noticed? Or the opposite-being so fidgety or restless that you have been moving around a lot more than usual":
        "phq28",
    "11. Thoughts that you would be better off dead or hurting yourself in some way":
        "phq29",

    # -----------------------------------------------------------------------
    # PHQ-9 — Spanish (baseline Spanish survey, first occurrence = no suffix)
    # Two variants per label: proper UTF-8 and the mojibake fallback that
    # appears when the file is read with the wrong encoding.
    # -----------------------------------------------------------------------
    "3. Poco interés o placer en hacer cosas":                             "phq21span",
    "3. Poco interï¿½s o placer en hacer cosas":                           "phq21span",
    "4. Se sintió triste, deprimido o sin esperanzas":                     "phq22span",
    "4. Se sintiï¿½ triste, deprimido o sin esperanzas":                   "phq22span",
    "5. Tuvo dificultades para dormirse o permanecer dormido, o durmió demasiado": "phq23span",
    "5. Tuvo dificultades para dormirse o permanecer dormido, o durmiï¿½ demasiado": "phq23span",
    "6. Se sintió cansado o con poca energía":                             "phq24span",
    "6. Se sintiï¿½ cansado o con poca energï¿½a":                         "phq24span",
    "7. Tuvo poco apetito o comió demasiado":                              "phq25span",
    "7. Tuvo poco apetito o comiï¿½ demasiado":                            "phq25span",
    "8. Se sintió mal con respecto a sí mismo, que es un fracaso o que se ha fallado a sí mismo o a su familia": "phq26span",
    "8. Se sintiï¿½ mal con respecto a sï¿½ mismo, que es un fracaso o que se ha fallado a sï¿½ mismo o a su familia": "phq26span",
    "9. Tuvo problemas para concentrase en las cosas, como leer el periódico o ver televisión": "phq27span",
    "9. Tuvo problemas para concentrase en las cosas, como leer el periï¿½dico o ver televisiï¿½n": "phq27span",
    "10. Se movía o hablaba tan lento que otras personas podrían haberse dado cuenta. O al contrario, estuvo tan inquieto o agitado que se estuvo moviendo mucho más de lo habitual": "phq28span",
    "10. Se movï¿½a o hablaba tan lento que otras personas podrï¿½an haberse dado cuenta. O al contrario, estuvo tan inquieto o agitado que se estuvo moviendo mucho mï¿½s de lo habitual": "phq28span",
    "11. Pensó que estaría mejor muerto o en lastimarse de algún modo":    "phq29span",
    "11. Pensï¿½ que estarï¿½a mejor muerto o en lastimarse de algï¿½n modo": "phq29span",

    # -----------------------------------------------------------------------
    # GAD-7 — English (only in baseline survey; labels are unique, no suffix)
    # -----------------------------------------------------------------------
    "13. Feeling nervous, anxious, or on edge":          "f",
    "14. Not being able to stop or control worrying":    "w",
    "15. Worrying too much about different things":      "gad71",
    "16. Trouble relaxing":                              "gad72",
    "17. Being so restless that it is hard to sit still": "gad73",
    "18. Becoming easily annoyed or irritable":          "gad74",
    "19. Feeling afraid as if something awful might happen": "gad75",

    # GAD-7 — Spanish (both encoding variants)
    "13. Se sintió nervioso, ansioso o irritable":       "f_span",
    "13. Se sintiï¿½ nervioso, ansioso o irritable":     "f_span",
    "14. No podía dejar de preocuparse ni controlar su preocupación": "w_span",
    "14. No podï¿½a dejar de preocuparse ni controlar su preocupaciï¿½n": "w_span",
    "15. Se preocupó mucho por cosas diferentes":        "gad71span",
    "15. Se preocupï¿½ mucho por cosas diferentes":      "gad71span",
    "16. Tuvo problemas para relajarse":                 "gad72span",
    "17. Se sintió tan intranquilo que le costaba quedarse sentado": "gad73span",
    "17. Se sintiï¿½ tan intranquilo que le costaba quedarse sentado": "gad73span",
    "18. Se enojaba o se irritaba fácilmente":           "gad74span",
    "18. Se enojaba o se irritaba fï¿½cilmente":         "gad74span",
    "19. Tenía miedo de que algo malo fuera a suceder":  "gad75span",
    "19. Tenï¿½a miedo de que algo malo fuera a suceder": "gad75span",

    # -----------------------------------------------------------------------
    # Engelberg QOC items 52–67 + GoC quality item 68
    # Only in baseline English survey; labels are unique, no suffix needed.
    # -----------------------------------------------------------------------
    "52. Include your loved ones in decisions about your treatment":
        "qoc1",
    "53. Asking what you understand about your cancer & its treatments":
        "qoc2",
    "54. Asking how much information you want to know about your cancer?":
        "qoc3",
    "55. Asking about your cultural beliefs and values":
        "qoc4",
    "56. Asking about the things in life that are important to you":
        "qoc5",
    "57. Asking about your religious or spiritual beliefs":
        "qoc6",
    "58. Using words that you can understand":
        "qoc7",
    "59. Answering all your questions about your treatment":
        "qoc8",
    "60. Involving you in decisions about treatments that you want should you get too sick to speak for yourself":
        "qoc9",
    "61. Listening to what you have to say":
        "qoc10",
    "62. Helping you understand what to expect":
        "qoc11",
    "63. Addressing emotional issues":
        "qoc12",
    "64. Caring about you as a person":
        "qoc13",
    "65. Talking with you about your feelings concerning the possibility that you might get sicker":
        "qoc14",
    "66. Working to make sure you get the care that you want":
        "qoc15",
    "67. Reviewing what you discussed and the plans for the next steps in treatment":
        "qoc16",
    "68. Goals of care discussions can be difficult because they include sensitive topics. Setting aside WHAT you and your doctor discussed, HOW WELL did your doctor talk with you about your goals?":
        "qoc17",

    # -----------------------------------------------------------------------
    # Communication Skills — Reviewer 1 (first occurrence, no pandas suffix)
    # Items that appear identically in both R1 and R2 keep the raw label for R1.
    # Items unique to R1 (slightly different wording from R2) have no suffix.
    # -----------------------------------------------------------------------
    "Greeting":                                             "greeting1",
    "Assessed patient's/family's understanding":            "ptunderstand1",
    "Asked what the patient/family wants to know":          "askpt1",
    "Gave a 'warning' shot":                                "warning1",   # R1 label (apostrophe after 'warning')
    "Gave information (about current medical condition)":   "info1",
    "Avoided use of medical jargon":                        "medjargon1",
    "Responded to emotions (NURSE)":                        "nurse1",
    "*Wish statements (for unrealistic tx goals)":          "wish1",      # R1 label (correct spelling)
    "Check-in before moving on":                            "checkin1",
    "Check for understanding":                              "understand1",
    "Summary":                                              "summary1",
    "Plan":                                                 "plan1",
    "Used empathic continuers (NURSE-at least one)":        "continuers1",
    "Used empathic terminators":                            "terminators1",
    "Prognosis (delivered as range)":                       "prognosis1",
    "Used silence appropriately":                           "silence1",
    "Elicited values (eg What's most important to you?)":   "elicitvalues1",  # R1 label (no period after eg)
    "*Goal setting (in context of ongoing or future care)": "goalset1",        # R1 label ('or')
    "Explored patient identity/family support":             "support1",

    # -----------------------------------------------------------------------
    # Communication Skills — Reviewer 2
    # Items with the same label as R1 get a .1 suffix from pandas.
    # Items with unique R2 wording (typos) use their exact label.
    # -----------------------------------------------------------------------
    "Greeting.1":                                              "greeting2",
    "Assessed patient's/family's understanding.1":             "ptunderstand2",
    "Asked what the patient/family wants to know.1":           "askpt2",
    "Gave a 'warning shot'":                                   "warning2",   # R2 label (no apostrophe after 'warning')
    "Gave information (about current medical condition).1":    "info2",
    "Avoided use of medical jargon.1":                         "medjargon2",
    "Responded to emotions (NURSE).1":                         "nurse2",
    "*Wish statemtns (for unrealistic tx goals)":              "wish2",      # R2 label (typo: statemtns)
    "Check-in before moving on.1":                             "checkin2",
    "Check for understanding.1":                               "understand2",
    "Summary.1":                                               "summary2",
    "Plan.1":                                                  "plan2",
    "Used empathic continuers (NURSE-at least one).1":         "continuers2",
    "Used empathic terminators.1":                             "terminators2",
    "Prognosis (delivered as range).1":                        "prognosis2",
    "Used silence appropriately.1":                            "silence2",
    "Elicited values (eg. What's most important to you?)":     "elicitvalues2",  # R2 label (period after eg)
    "*Goal setting (in context of ongoing of future care)":    "goalset2",        # R2 label (typo: 'of')
    "Explored patient identity/family support.1":              "support2",
}


def apply_column_label_map(df):
    """
    Rename columns using COLUMN_LABEL_MAP.
    Logs which target variables were successfully mapped and which are missing.
    """
    rename = {old: new for old, new in COLUMN_LABEL_MAP.items() if old in df.columns}
    df = df.rename(columns=rename)

    mapped_targets = set(rename.values())
    all_targets = set(COLUMN_LABEL_MAP.values())
    missing = all_targets - mapped_targets - {"f_span", "w_span",
                                               "gad71span", "gad72span", "gad73span",
                                               "gad74span", "gad75span",
                                               "phq21span", "phq22span", "phq23span",
                                               "phq24span", "phq25span", "phq26span",
                                               "phq27span", "phq28span", "phq29span"}
    if missing:
        print(f"  WARNING: {len(missing)} expected columns not found in CSV: {sorted(missing)}")

    return df


# ---------------------------------------------------------------------------
# SCORING VARIABLE LISTS
# ---------------------------------------------------------------------------

COMM_SKILLS_BASELINE = [
    "greeting1", "ptunderstand1", "askpt1", "warning1", "info1",
    "medjargon1", "nurse1", "wish1", "checkin1", "understand1",
    "summary1", "plan1", "continuers1", "terminators1",
    "prognosis1", "silence1", "elicitvalues1", "goalset1", "support1",
]

COMM_SKILLS_POST = [
    "greeting2", "ptunderstand2", "askpt2", "warning2", "info2",
    "medjargon2", "nurse2", "wish2", "checkin2", "understand2",
    "summary2", "plan2", "continuers2", "terminators2",
    "prognosis2", "silence2", "elicitvalues2", "goalset2", "support2",
]

PHQ9_ITEMS       = ["phq21", "phq22", "phq23", "phq24", "phq25", "phq26", "phq27", "phq28", "phq29"]
PHQ9_GATEWAY     = ["phq21", "phq22"]
PHQ9_CONDITIONAL = ["phq23", "phq24", "phq25", "phq26", "phq27", "phq28", "phq29"]

PHQ9_ITEMS_SPAN       = [f"{v}span" for v in PHQ9_ITEMS]
PHQ9_GATEWAY_SPAN     = [f"{v}span" for v in PHQ9_GATEWAY]
PHQ9_CONDITIONAL_SPAN = [f"{v}span" for v in PHQ9_CONDITIONAL]

GAD7_GATEWAY_ENG  = ["f", "w"]
GAD7_GATEWAY_SPAN = ["f_span", "w_span"]
GAD7_COND_ENG     = ["gad71", "gad72", "gad73", "gad74", "gad75"]
GAD7_COND_SPAN    = ["gad71span", "gad72span", "gad73span", "gad74span", "gad75span"]

QOC_GENERAL = [f"qoc{i}" for i in range(1, 10)]    # qoc1–qoc9
QOC_EOL     = [f"qoc{i}" for i in range(10, 17)]   # qoc10–qoc16
QOC_ALL     = QOC_GENERAL + QOC_EOL
GOC_QUALITY_ITEM = "qoc17"


# ---------------------------------------------------------------------------
# SCORING FUNCTIONS
# ---------------------------------------------------------------------------

def score_comm_skills(df, items):
    available = [c for c in items if c in df.columns]
    if not available:
        return pd.Series([np.nan] * len(df), index=df.index), \
               pd.Series([0] * len(df), index=df.index)
    subset = df[available].replace(777, np.nan)
    score = subset.eq(1).sum(axis=1).where(subset.notna().any(axis=1))
    n_valid = subset.notna().sum(axis=1)
    return score, n_valid


def _score_phq9_single(df, items, gateway, conditional):
    available_gateway = [c for c in gateway if c in df.columns]
    available_all     = [c for c in items if c in df.columns]
    if not available_gateway:
        return pd.Series([np.nan] * len(df), index=df.index)
    scored = df[available_all].copy().apply(pd.to_numeric, errors="coerce")
    gateway_sum       = scored[available_gateway].sum(axis=1, min_count=1)
    screened_negative = gateway_sum <= len(available_gateway)
    for col in [c for c in conditional if c in df.columns]:
        scored[col] = scored[col].fillna(
            scored[col].where(~screened_negative, other=1)
        )
    return (scored - 1).sum(axis=1, min_count=len(available_gateway))


def score_phq9(df):
    """English or Spanish — whichever gateway items are filled for that patient."""
    eng  = _score_phq9_single(df, PHQ9_ITEMS,      PHQ9_GATEWAY,      PHQ9_CONDITIONAL)
    span = _score_phq9_single(df, PHQ9_ITEMS_SPAN,  PHQ9_GATEWAY_SPAN, PHQ9_CONDITIONAL_SPAN)
    return eng.combine_first(span)


def _score_gad7_single(df, gateway_cols, cond_cols):
    available_gw   = [c for c in gateway_cols if c in df.columns]
    available_cond = [c for c in cond_cols    if c in df.columns]
    if not available_gw:
        return pd.Series([np.nan] * len(df), index=df.index)
    scored_gw = df[available_gw].apply(pd.to_numeric, errors="coerce")
    gateway_sum = scored_gw.sum(axis=1, min_count=1)
    cond_scores = pd.DataFrame(index=df.index)
    for col in available_cond:
        raw = df[col].apply(pd.to_numeric, errors="coerce")
        raw = raw.where(~((gateway_sum == 0) & raw.isna()), other=1)
        cond_scores[col] = (raw - 1).clip(lower=0)
    return gateway_sum.add(cond_scores.sum(axis=1), fill_value=0)


def score_gad7(df):
    """English or Spanish — whichever gateway items are filled."""
    eng  = _score_gad7_single(df, GAD7_GATEWAY_ENG,  GAD7_COND_ENG)
    span = _score_gad7_single(df, GAD7_GATEWAY_SPAN, GAD7_COND_SPAN)
    return eng.combine_first(span)


def score_engelberg_qoc(df):
    def mean_available(columns):
        cols = [c for c in columns if c in df.columns]
        if not cols:
            return pd.Series([np.nan] * len(df), index=df.index)
        return df[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    return mean_available(QOC_ALL), mean_available(QOC_GENERAL), mean_available(QOC_EOL)


def score_goc_quality(df):
    if GOC_QUALITY_ITEM not in df.columns:
        return pd.Series([np.nan] * len(df), index=df.index), \
               pd.Series([np.nan] * len(df), index=df.index)
    raw = pd.to_numeric(df[GOC_QUALITY_ITEM], errors="coerce")
    dichotomized = raw.apply(lambda x: 1 if x >= 9 else (0 if pd.notna(x) else np.nan))
    return raw, dichotomized


# ---------------------------------------------------------------------------
# FIELD EXTRACTION
# ---------------------------------------------------------------------------

# The ID column name as it appears in the REDCap export header
RECORD_ID_LABEL = "Record ID"

# All the field labels the script needs (used for extraction)
ALL_NEEDED_LABELS = list(COLUMN_LABEL_MAP.keys())


def extract_fields(df_raw):
    """
    From the full export, keep only:
      - Record ID column
      - Every column whose label appears in COLUMN_LABEL_MAP
    Returns a DataFrame with only those columns (plus Record ID).
    """
    keep = []

    # Record ID column
    if RECORD_ID_LABEL in df_raw.columns:
        keep.append(RECORD_ID_LABEL)
    else:
        # Fallback: first column is usually the ID
        keep.append(df_raw.columns[0])
        print(f"  NOTE: '{RECORD_ID_LABEL}' not found; using first column '{df_raw.columns[0]}' as ID")

    # Add every column whose label is in our map (including pandas-suffixed duplicates)
    for col in df_raw.columns:
        if col in COLUMN_LABEL_MAP and col not in keep:
            keep.append(col)

    found    = [c for c in keep if c != keep[0]]
    missing  = [label for label in ALL_NEEDED_LABELS if label not in df_raw.columns]

    print(f"  Keeping {len(keep)} columns ({len(found)} scale fields + ID)")
    if missing:
        print(f"  NOTE: {len(missing)} expected field labels not found in this export:")
        for m in missing:
            print(f"    - {m}")

    return df_raw[keep].copy()


def filter_patients(df, patient_ids):
    """
    Keep only rows whose Record ID is in patient_ids.
    patient_ids should be a list of strings or ints.
    """
    id_col = RECORD_ID_LABEL if RECORD_ID_LABEL in df.columns else df.columns[0]
    before = len(df)
    # Try matching as-is first, then as string
    mask = df[id_col].isin(patient_ids) | df[id_col].astype(str).isin([str(p) for p in patient_ids])
    df = df[mask].copy()
    matched = set(df[id_col].astype(str).unique())
    unmatched = [p for p in patient_ids if str(p) not in matched]
    print(f"  Patient filter: {before} → {len(df)} rows ({len(matched)} unique patients matched of {len(patient_ids)} requested)")
    if unmatched:
        print(f"  WARNING: {len(unmatched)} ID(s) not found in CSV: {unmatched}")
    return df


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extract and score REDCap scales for GoC study.")
    parser.add_argument("--input",        required=True,
                        help="Path to REDCap CSV export (field-label headers)")
    parser.add_argument("--output",       required=True,
                        help="Path for output CSV")
    parser.add_argument("--extract-only", action="store_true",
                        help="Only extract relevant fields; skip scoring. "
                             "Good first step to inspect the data before running scores.")
    parser.add_argument("--patients",     nargs="+", default=None,
                        help="Optional: one or more Record IDs to include (space-separated). "
                             "If omitted, all rows are processed.")
    parser.add_argument("--patients-file", default=None,
                        help="Optional: path to a plain text file with one Record ID per line. "
                             "Use this instead of --patients when you have many IDs.")
    parser.add_argument("--event", default=None,
                        help="Optional: filter to a specific REDCap event name (e.g. \'baseline_arm_1\'). "
                             "Use this if PHQ-9/QOC columns are blank because the data lives in a specific event row.")
    args = parser.parse_args()

    print(f"Reading: {args.input}")
    ext = args.input.replace("'", "").replace('"', "").lower().rsplit(".", 1)[-1]
    if ext in ("xlsx", "xls"):
        df_raw = pd.read_excel(args.input, engine='openpyxl')
        print(f"  {len(df_raw)} rows, {len(df_raw.columns)} columns (Excel)")
    else:
        for encoding in ("utf-8", "latin-1", "cp1252", "utf-8-sig"):
            try:
                df_raw = pd.read_csv(args.input, low_memory=False, encoding=encoding)
                print(f"  {len(df_raw)} rows, {len(df_raw.columns)} columns (encoding: {encoding})")
                break
            except (UnicodeDecodeError, UnicodeError):
                print(f"  Encoding {encoding} failed, trying next...")
        else:
            raise ValueError("Could not read the CSV with any supported encoding "
                             "(tried utf-8, latin-1, cp1252, utf-8-sig).")

    # Step 1: extract only the fields we need
    print("\nExtracting relevant fields...")
    df = extract_fields(df_raw)

    # Step 2 (optional): filter to specific patient IDs
    patient_ids = None
    if args.patients_file:
        with open(args.patients_file, "r") as f:
            patient_ids = [line.strip() for line in f if line.strip()]
        print(f"\nLoaded {len(patient_ids)} patient IDs from {args.patients_file}")
    elif args.patients:
        patient_ids = args.patients

    if patient_ids:
        print(f"\nFiltering to {len(patient_ids)} requested patient ID(s)...")
        df = filter_patients(df, patient_ids)

    if args.event:
        event_col = "Event Name" if "Event Name" in df.columns else None
        if event_col:
            before = len(df)
            df = df[df[event_col].astype(str).str.strip() == args.event.strip()].copy()
            print(f"\nEvent filter '{args.event}': {before} → {len(df)} rows")
        else:
            print("\nWARNING: --event specified but no 'Event Name' column found in extracted data.")

    if args.extract_only:
        df.to_csv(args.output, index=False)
        print(f"\nExtraction complete. Saved {len(df)} rows, {len(df.columns)} columns → {args.output}")
        return

    # Step 3: rename columns to internal variable names
    print("\nApplying column label map...")
    df = apply_column_label_map(df)

    # Step 4: score
    out = df.copy()

    out["comm_skills_score_baseline"], out["comm_skills_n_valid_baseline"] = \
        score_comm_skills(df, COMM_SKILLS_BASELINE)

    out["comm_skills_score_post"], out["comm_skills_n_valid_post"] = \
        score_comm_skills(df, COMM_SKILLS_POST)

    out["phq9_score"]  = score_phq9(df)
    out["gad7_score"]  = score_gad7(df)

    out["engelberg_qoc_score"], out["engelberg_qoc_general"], out["engelberg_qoc_eol"] = \
        score_engelberg_qoc(df)

    out["goc_discussion_quality"], out["goc_discussion_satisfied"] = \
        score_goc_quality(df)

    scored_cols = [
        "comm_skills_score_baseline", "comm_skills_score_post",
        "phq9_score", "gad7_score",
        "engelberg_qoc_score", "engelberg_qoc_general", "engelberg_qoc_eol",
        "goc_discussion_quality", "goc_discussion_satisfied",
    ]
    print("\nScored variable summary:")
    for col in scored_cols:
        if col in out.columns:
            n    = out[col].notna().sum()
            mean = out[col].mean()
            print(f"  {col}: {n} non-missing  |  mean={mean:.2f}"
                  if pd.notna(mean) else f"  {col}: {n} non-missing")

    out.to_csv(args.output, index=False)
    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()