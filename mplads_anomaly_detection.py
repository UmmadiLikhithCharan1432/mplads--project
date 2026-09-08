"""
MPLADS Anomaly & Fraud Detection - Core Engine
================================================
This single script is the "brain" of the project. It:
  1. Loads MPLADS work-sanction data (or generates realistic sample data
     if you don't have the real file yet)
  2. Computes a few interpretable features from it
  3. Applies simple RULE-BASED flags (fast, explainable, always defensible)
  4. Applies ONE simple ML model (Isolation Forest) as the "AI" layer,
     to catch anomalies that don't match any hand-written rule
  5. Combines both into a single risk_score per work, with a plain
     English reason -- this is what your dashboard will display

Read this file top to bottom -- it's written to be learned from, not
just run. Every function does ONE thing.

Expected columns (matches the real MPLADS compiled dataset schema):
  state, nodal_district, implementing_district, house_name, member_type,
  mp_name, sanction_amount, date_of_administrative_approval, work_name,
  unique_work_number, implementing_agency_name, work_status,
  date_of_receipt_of_work_proposal_from_mp, unit
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest


# ---------------------------------------------------------------------------
# STEP 0: Sample data generator (use this until you plug in the real CSV)
# ---------------------------------------------------------------------------
def generate_sample_data(n=500, seed=42):
    """Creates a fake-but-realistic dataset matching the real MPLADS schema,
    with a handful of planted anomalies so you can verify detection works
    before you have the real data loaded."""
    rng = np.random.default_rng(seed)

    states = ["Telangana", "Maharashtra", "Bihar", "Kerala", "Punjab"]
    units = ["Road (Km)", "Handpump (Number)", "Community Hall (Number)", "Streetlight (Number)"]
    agencies = [f"Agency_{i}" for i in range(1, 21)]
    mps = [f"MP_{i}" for i in range(1, 16)]
    statuses = ["Completed", "Work in Progress", "Sanctioned", "Not Started"]

    proposal_dates = pd.to_datetime("2023-01-01") + pd.to_timedelta(
        rng.integers(0, 700, n), unit="D"
    )
    approval_delay = rng.integers(5, 90, n)  # normal delay: 5-90 days
    approval_dates = proposal_dates + pd.to_timedelta(approval_delay, unit="D")

    df = pd.DataFrame({
        "state": rng.choice(states, n),
        "implementing_district": [f"District_{d}" for d in rng.integers(1, 10, n)],
        "mp_name": rng.choice(mps, n),
        "sanction_amount": rng.normal(1_500_000, 400_000, n).clip(50_000),
        "date_of_receipt_of_work_proposal_from_mp": proposal_dates,
        "date_of_administrative_approval": approval_dates,
        "work_name": [f"Work_{i}" for i in range(n)],
        "unique_work_number": [f"UWN{i:05d}" for i in range(n)],
        "implementing_agency_name": rng.choice(agencies, n),
        "work_status": rng.choice(statuses, n, p=[0.5, 0.25, 0.15, 0.10]),
        "unit": rng.choice(units, n),
    })

    # --- Plant a few obvious anomalies so you can verify detection works ---
    # 1. Cost inflation: a few works cost 8-10x the normal amount
    inflate_idx = rng.choice(n, 6, replace=False)
    df.loc[inflate_idx, "sanction_amount"] *= rng.uniform(8, 10, 6)

    # 2. Approval delay anomaly: a few works approved 400+ days after proposal
    delay_idx = rng.choice(n, 5, replace=False)
    df.loc[delay_idx, "date_of_administrative_approval"] = (
        df.loc[delay_idx, "date_of_receipt_of_work_proposal_from_mp"]
        + pd.to_timedelta(rng.integers(400, 600, 5), unit="D")
    )

    # 3. Agency concentration: force one agency to dominate one district
    conc_idx = df[df["implementing_district"] == "District_3"].index[:15]
    df.loc[conc_idx, "implementing_agency_name"] = "Agency_1"

    return df


# ---------------------------------------------------------------------------
# STEP 1: Load real data (swap this in once you have the actual CSV)
# ---------------------------------------------------------------------------
def load_real_data(csv_path):
    df = pd.read_csv(csv_path)
    # Explicitly parse dates as DD-MM-YYYY (matches the real MPLADS export
    # format). Newer pandas versions no longer auto-detect this format via
    # parse_dates=[...] alone, so we convert it ourselves. errors="coerce"
    # turns any row with a genuinely broken date into NaT (missing) instead
    # of crashing the whole pipeline.
    for col in ["date_of_receipt_of_work_proposal_from_mp", "date_of_administrative_approval"]:
        df[col] = pd.to_datetime(df[col], format="%d-%m-%Y", errors="coerce")
    return df


# ---------------------------------------------------------------------------
# STEP 2: Feature engineering -- turn raw columns into signals
# ---------------------------------------------------------------------------
def compute_features(df):
    df = df.copy()

    # Feature 1: approval delay in days (proposal -> administrative approval)
    df["approval_delay_days"] = (
        df["date_of_administrative_approval"]
        - df["date_of_receipt_of_work_proposal_from_mp"]
    ).dt.days

    # Feature 2: cost z-score WITHIN peer group (same 'unit' = same kind of work)
    # This is what catches "this road cost way more than other roads", not
    # just "this is an expensive work" (a Km of road SHOULD cost more than
    # a handpump -- comparing within the same unit avoids that false positive)
    df["cost_zscore"] = df.groupby("unit")["sanction_amount"].transform(
        lambda x: (x - x.mean()) / x.std(ddof=0)
    )

    # Feature 3: agency's share of works within its district
    # (a single agency winning a suspicious % of a district's works)
    district_agency_counts = df.groupby(["implementing_district", "implementing_agency_name"])\
        .size().rename("agency_works_in_district").reset_index()
    district_totals = df.groupby("implementing_district").size().rename("district_total_works")
    district_agency_counts = district_agency_counts.merge(
        district_totals, on="implementing_district"
    )
    district_agency_counts["agency_concentration"] = (
        district_agency_counts["agency_works_in_district"]
        / district_agency_counts["district_total_works"]
    )
    df = df.merge(
        district_agency_counts[["implementing_district", "implementing_agency_name", "agency_concentration"]],
        on=["implementing_district", "implementing_agency_name"],
        how="left",
    )

    return df
# ---------------------------------------------------------------------------
# STEP 3: Rule-based flags -- simple, explainable, always defensible
# ---------------------------------------------------------------------------
def apply_rules(df):
    df = df.copy()
    reasons = [[] for _ in range(len(df))]
    rule_points = np.zeros(len(df))

    # Rule 1: Cost inflation -- cost is a statistical outlier vs peer works
    cost_flag = df["cost_zscore"] > 2.5
    for i in np.where(cost_flag)[0]:
        reasons[i].append(
            f"cost is {df['cost_zscore'].iloc[i]:.1f} std-dev above similar works"
        )
    rule_points += cost_flag * 40

    # Rule 2: Approval delay -- took unusually long to approve
    delay_flag = df["approval_delay_days"] > 200
    for i in np.where(delay_flag)[0]:
        reasons[i].append(
            f"took {df['approval_delay_days'].iloc[i]} days to get administrative approval"
        )
    rule_points += delay_flag * 25

    # Rule 3: Agency concentration -- one agency dominates a district
    conc_flag = df["agency_concentration"] > 0.5
    for i in np.where(conc_flag)[0]:
        reasons[i].append(
            f"agency holds {df['agency_concentration'].iloc[i]:.0%} of works in its district"
        )
    rule_points += conc_flag * 35

    df["rule_reasons"] = reasons
    df["rule_points"] = rule_points
    return df


# ---------------------------------------------------------------------------
# STEP 4: ML layer -- Isolation Forest catches what the rules didn't name
# ---------------------------------------------------------------------------
def apply_ml_layer(df):
    """Isolation Forest: works by randomly splitting the data repeatedly.
    Outliers get isolated (separated from everything else) in FEWER splits
    than normal points, because they're 'different' on some feature
    combination. It doesn't need labeled fraud examples -- it just learns
    what 'normal' looks like from the bulk of the data and scores anything
    unusual as more anomalous. This is the standard, simplest real ML
    approach for fraud/anomaly detection when you have no labels."""
    df = df.copy()

    feature_cols = ["sanction_amount", "approval_delay_days", "cost_zscore", "agency_concentration"]
    X = df[feature_cols].fillna(0)

    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(X)

    # decision_function: higher = more normal, lower/negative = more anomalous.
    # We flip and rescale it to a clean 0-100 "ml_score" for the dashboard.
    raw_scores = model.decision_function(X)
    df["ml_anomaly_score"] = (
        (raw_scores.max() - raw_scores) / (raw_scores.max() - raw_scores.min()) * 100
    )
    return df


# ---------------------------------------------------------------------------
# STEP 5: Combine rules + ML into one risk score with a readable reason
# ---------------------------------------------------------------------------
def compute_final_risk(df):
    df = df.copy()

    # Weighted blend: rules are trusted signals (60%), ML fills the gaps (40%)
    df["risk_score"] = (0.6 * df["rule_points"]) + (0.4 * df["ml_anomaly_score"])
    df["risk_score"] = df["risk_score"].clip(0, 100).round(1)

    def build_explanation(row):
        parts = list(row["rule_reasons"])
        if row["ml_anomaly_score"] > 70 and not parts:
            parts.append("flagged as statistically unusual by the anomaly model "
                          "(no single rule matched, but the overall pattern is atypical)")
        if not parts:
            return "No red flags."
        return "Flagged: " + "; ".join(parts) + "."

    df["explanation"] = df.apply(build_explanation, axis=1)
    return df.drop(columns=["rule_reasons"])


# ---------------------------------------------------------------------------
# MAIN: run the whole pipeline
# ---------------------------------------------------------------------------
def run_pipeline(csv_path=None):
    df = load_real_data(csv_path) if csv_path else generate_sample_data()

    df = compute_features(df)
    df = apply_rules(df)
    df = apply_ml_layer(df)
    df = compute_final_risk(df)

    df = df.sort_values("risk_score", ascending=False)

    # Save the full scored dataset -- this is what your API/dashboard reads
    output_cols = [
        "unique_work_number", "work_name", "mp_name", "implementing_district",
        "implementing_agency_name", "sanction_amount", "work_status",
        "risk_score", "explanation",
    ]
    df[output_cols].to_csv("scored_works.csv", index=False)

    # Print a quick summary -- these are your pitch-deck impact numbers
    flagged = df[df["risk_score"] > 40]
    print(f"Total works analyzed: {len(df)}")
    print(f"Flagged as high-risk (score > 40): {len(flagged)} ({len(flagged)/len(df):.1%})")
    print(f"Total sanction value flagged: Rs. {flagged['sanction_amount'].sum():,.0f}")
    print("\nTop 5 highest-risk works:")
    print(df[["unique_work_number", "mp_name", "risk_score", "explanation"]].head(5).to_string(index=False))

    return df


if __name__ == "__main__":
    # Change to run_pipeline("your_real_data.csv") once you have the real file
    run_pipeline()