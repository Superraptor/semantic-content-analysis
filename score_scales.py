"""
score_scales.py

Computes scored scale variables from a REDCap CSV export for the
"Improving Advanced Cancer Patient-Centered Care by Enabling Goals of
Care Discussions" study.

Outputs scored fields per record:
  1. comm_skills_score_baseline  — Communication Skills Assessment (baseline tape)
  2. comm_skills_score_post      — Communication Skills Assessment (post-intervention tape)
  3. phq9_score                  — PHQ-9 Depression total (0–27); uses English or Spanish
                                   items, whichever is filled in for that patient
  4. gad7_score                  — GAD-7 Anxiety total (0–21)
  5. engelberg_qoc_score         — Engelberg QOC mean (0–10), general + EOL subscales
  6. goc_discussion_quality      — GoC Discussion Quality raw (0–10)
  7. goc_discussion_satisfied    — GoC Discussion Quality dichotomized (1=highly satisfied
                                   [9–10], 0=not highly satisfied [<=8])

Usage:
    python score_scales.py --input your_redcap_export.csv --output scored_output.csv

Requirements: pandas
    pip install pandas
"""

import argparse
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# SKILL ASSESSMENT ITEMS
# Binary items coded 0=No, 1=Yes, 777=NA
# Score = count of "Yes" (1) responses; 777 is treated as missing (not counted)
# ---------------------------------------------------------------------------

COMM_SKILLS_BASELINE = [
    "greeting1",       # (S) Setting: Greeting
    "ptunderstand1",   # (P) Perception: Assessed patient's/family's understanding
    "askpt1",          # (I) Invitation: Asked what patient/family wants to know
    "warning1",        # Gave a "warning shot"
    "info1",           # (K) Knowledge: Gave information about current medical condition
    "medjargon1",      # Avoided use of medical jargon
    "nurse1",          # (E) Empathic Response: Responded to emotions (NURSE)
    "wish1",           # Wish statements (for unrealistic tx goals)
    "checkin1",        # (S) Strategy: Check-in before moving on
    "understand1",     # Check for understanding
    "summary1",        # Summary
    "plan1",           # Plan
    "continuers1",     # Other Skills: Used empathic continuers (NURSE – at least one)
    "terminators1",    # Used empathic terminators
    "prognosis1",      # Prognosis (delivered as range)
    "silence1",        # Used silence appropriately
    "elicitvalues1",   # Elicited values (e.g. "What's most important to you?")
    "goalset1",        # Goal setting (in context of ongoing or future care)
    "support1",        # Explored patient identity/family support
]

COMM_SKILLS_POST = [
    "greeting2",
    "ptunderstand2",
    "askpt2",
    "warning2",
    "info2",
    "medjargon2",
    "nurse2",
    "wish2",
    "checkin2",
    "understand2",
    "summary2",
    "plan2",
    "continuers2",
    "terminators2",
    "prognosis2",
    "silence2",
    "elicitvalues2",
    "goalset2",
    "support2",
]

# ---------------------------------------------------------------------------
# PHQ-9 ITEMS
# REDCap coding: 1=Not at all, 2=Several Days, 3=More than half the days,
#                4=Nearly every day
# Standard PHQ-9 scoring: 0–3 per item → subtract 1 from each, then sum.
# phq23–phq29 are conditional (only shown if phq21 or phq22 >= 2);
# if missing, treat as 0 (patient screened negative on gateway items).
# Total range: 0–27
#
# Each patient receives either the English OR Spanish version — not both.
# The script checks which gateway items are filled and uses that language.
# ---------------------------------------------------------------------------

PHQ9_ITEMS       = ["phq21", "phq22", "phq23", "phq24", "phq25", "phq26", "phq27", "phq28", "phq29"]
PHQ9_GATEWAY     = ["phq21", "phq22"]
PHQ9_CONDITIONAL = ["phq23", "phq24", "phq25", "phq26", "phq27", "phq28", "phq29"]

PHQ9_ITEMS_SPAN       = [f"{v}span" for v in PHQ9_ITEMS]
PHQ9_GATEWAY_SPAN     = [f"{v}span" for v in PHQ9_GATEWAY]
PHQ9_CONDITIONAL_SPAN = [f"{v}span" for v in PHQ9_CONDITIONAL]

# ---------------------------------------------------------------------------
# GAD-7 ITEMS
# Gateway items f (Q13) and w (Q14) are coded 0–3 directly.
# gad71–gad75 (Q15–Q19) are conditional (shown if f+w >= 1) and coded
#   1=Not at all, 2=Several days, 3=More than half the days, 4=Nearly every day
#   → subtract 1 from each to get 0–3.
# If f+w == 0, gad71–gad75 were skipped; treat each as 0.
# Total range: 0–21
# ---------------------------------------------------------------------------

GAD7_GATEWAY = ["f", "w"]           # coded 0–3
GAD7_CONDITIONAL = ["gad71", "gad72", "gad73", "gad74", "gad75"]  # coded 1–4

# ---------------------------------------------------------------------------
# ENGELBERG QUALITY OF COMMUNICATION (QOC)
# Items qoc1–qoc16, each rated 0–10 (0=worst, 10=best).
# Overall score = mean of all 16 items.
# Subscales (based on Engelberg et al.):
#   General: qoc1–qoc9  (general communication quality)
#   End-of-Life: qoc10–qoc16  (EOL-specific communication)
# Note: confirm subscale boundaries against the published instrument if needed.
# qoc17 is the GoC Discussion Quality item — scored separately below.
# ---------------------------------------------------------------------------

QOC_GENERAL = [f"qoc{i}" for i in range(1, 10)]    # qoc1–qoc9
QOC_EOL     = [f"qoc{i}" for i in range(10, 17)]   # qoc10–qoc16
QOC_ALL     = QOC_GENERAL + QOC_EOL                 # qoc1–qoc16

# ---------------------------------------------------------------------------
# GoC DISCUSSION QUALITY
# qoc17: 0–10 scale
# Dichotomized: 1 = "highly satisfied" (9–10), 0 = "not highly satisfied" (<=8)
# ---------------------------------------------------------------------------

GOC_QUALITY_ITEM = "qoc17"


# ---------------------------------------------------------------------------
# SCORING FUNCTIONS
# ---------------------------------------------------------------------------

def score_comm_skills(df, items):
    """
    Count of 'Yes' (1) responses across binary skill items.
    777 = NA, treated as missing and excluded from count.
    Returns a Series with the count and a Series with the number of valid items.
    """
    available = [c for c in items if c in df.columns]
    if not available:
        return pd.Series([np.nan] * len(df)), pd.Series([0] * len(df))

    subset = df[available].replace(777, np.nan)
    score = subset.eq(1).sum(axis=1).where(subset.notna().any(axis=1))
    n_valid = subset.notna().sum(axis=1)
    return score, n_valid


def _score_phq9_single(df, items, gateway, conditional):
    """
    Internal: score PHQ-9 for one language version.
    Returns a Series (0–27), NaN where gateway items are missing.
    """
    available_gateway = [c for c in gateway if c in df.columns]
    available_all     = [c for c in items if c in df.columns]

    if not available_gateway:
        return pd.Series([np.nan] * len(df), index=df.index)

    scored = df[available_all].copy().apply(pd.to_numeric, errors="coerce")

    # If both gateway items = 1 (Not at all), patient screened negative
    # and skipped the rest → fill missing conditionals with 1 (→ 0 after -1)
    gateway_sum = scored[available_gateway].sum(axis=1, min_count=1)
    screened_negative = gateway_sum <= len(available_gateway)

    available_cond = [c for c in conditional if c in df.columns]
    for col in available_cond:
        scored[col] = scored[col].fillna(
            scored[col].where(~screened_negative, other=1)
        )

    return (scored - 1).sum(axis=1, min_count=len(available_gateway))


def score_phq9(df):
    """
    PHQ-9 combined: each patient received either the English or Spanish version.
    Uses whichever gateway items are non-null for that row.
    If somehow both are filled (shouldn't happen), English takes precedence.
    Returns a single Series (0–27).
    """
    eng  = _score_phq9_single(df, PHQ9_ITEMS,      PHQ9_GATEWAY,      PHQ9_CONDITIONAL)
    span = _score_phq9_single(df, PHQ9_ITEMS_SPAN,  PHQ9_GATEWAY_SPAN, PHQ9_CONDITIONAL_SPAN)

    # Use English where available, fall back to Spanish
    return eng.combine_first(span)


def score_gad7(df):
    """
    GAD-7:
      - Gateway items f and w: already coded 0–3, sum directly.
      - Conditional items gad71–gad75: coded 1–4, subtract 1 → 0–3.
        If f+w == 0, conditionals were skipped → treat as 0.
    Returns a Series (0–21).
    """
    scored = df.copy().apply(lambda col: pd.to_numeric(col, errors="coerce")
                             if col.name in GAD7_GATEWAY + GAD7_CONDITIONAL else col)

    gateway_available = [c for c in GAD7_GATEWAY if c in df.columns]
    cond_available    = [c for c in GAD7_CONDITIONAL if c in df.columns]

    if not gateway_available:
        return pd.Series([np.nan] * len(df))

    gateway_sum = scored[gateway_available].sum(axis=1, min_count=1)

    # Fill skipped conditional items with 0 (already on 0–3 scale after -1)
    cond_scores = pd.DataFrame(index=df.index)
    for col in cond_available:
        raw = scored[col].copy()
        # Where gateway=0 (screened negative) and item is missing, fill with 1 (→ 0 after -1)
        raw = raw.where(~((gateway_sum == 0) & raw.isna()), other=1)
        cond_scores[col] = (raw - 1).clip(lower=0)  # 1–4 → 0–3

    gad7 = gateway_sum.add(cond_scores.sum(axis=1), fill_value=0)
    return gad7


def score_engelberg_qoc(df):
    """
    Engelberg QOC:
      Mean of qoc1–qoc16 (each 0–10).
      Also computes general (qoc1–9) and EOL (qoc10–16) subscale means.
    Returns three Series: overall, general subscale, EOL subscale.
    """
    def mean_available(columns):
        cols = [c for c in columns if c in df.columns]
        if not cols:
            return pd.Series([np.nan] * len(df))
        subset = df[cols].apply(pd.to_numeric, errors="coerce")
        return subset.mean(axis=1)

    overall = mean_available(QOC_ALL)
    general = mean_available(QOC_GENERAL)
    eol     = mean_available(QOC_EOL)
    return overall, general, eol


def score_goc_quality(df):
    """
    GoC Discussion Quality:
      Raw: qoc17 (0–10)
      Dichotomized: 1 if >= 9, 0 if <= 8, NaN if missing
    """
    if GOC_QUALITY_ITEM not in df.columns:
        return pd.Series([np.nan] * len(df)), pd.Series([np.nan] * len(df))

    raw = pd.to_numeric(df[GOC_QUALITY_ITEM], errors="coerce")
    dichotomized = raw.apply(lambda x: 1 if x >= 9 else (0 if pd.notna(x) else np.nan))
    return raw, dichotomized


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Score REDCap scales for GoC study.")
    parser.add_argument("--input",  required=True, help="Path to REDCap CSV export")
    parser.add_argument("--output", required=True, help="Path for scored output CSV")
    args = parser.parse_args()

    print(f"Reading: {args.input}")
    df = pd.read_csv(args.input, low_memory=False)
    print(f"  {len(df)} rows, {len(df.columns)} columns")

    out = df.copy()

    # 1. Communication Skills — Baseline
    out["comm_skills_score_baseline"], out["comm_skills_n_valid_baseline"] = \
        score_comm_skills(df, COMM_SKILLS_BASELINE)

    # 2. Communication Skills — Post-intervention
    out["comm_skills_score_post"], out["comm_skills_n_valid_post"] = \
        score_comm_skills(df, COMM_SKILLS_POST)

    # 3. PHQ-9 (English or Spanish — whichever the patient received)
    out["phq9_score"] = score_phq9(df)

    # 4. GAD-7
    out["gad7_score"] = score_gad7(df)

    # 5. Engelberg QOC
    out["engelberg_qoc_score"], out["engelberg_qoc_general"], out["engelberg_qoc_eol"] = \
        score_engelberg_qoc(df)

    # 6. GoC Discussion Quality
    out["goc_discussion_quality"], out["goc_discussion_satisfied"] = \
        score_goc_quality(df)

    # Summary
    print("\nScored variable summary:")
    scored_cols = [
        "comm_skills_score_baseline", "comm_skills_score_post",
        "phq9_score", "gad7_score",
        "engelberg_qoc_score", "engelberg_qoc_general", "engelberg_qoc_eol",
        "goc_discussion_quality", "goc_discussion_satisfied",
    ]
    for col in scored_cols:
        if col in out.columns:
            n = out[col].notna().sum()
            print(f"  {col}: {n} non-missing values  |  mean={out[col].mean():.2f}")

    out.to_csv(args.output, index=False)
    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()