"""
MPLADS Anomaly Detection - API
================================
Minimal FastAPI backend. On startup it runs the detection pipeline once
and keeps the scored results in memory (fine for a one-day hackathon
prototype -- no need for a database yet).

Endpoints:
  GET  /summary          -> dashboard header stats (totals, flagged count, value)
  GET  /works            -> list of works, optionally filtered/sorted
  GET  /works/{work_id}  -> single work's full detail + explanation

Run with:
  pip install fastapi uvicorn scikit-learn pandas numpy
  uvicorn main:app --reload --port 8000

Then open http://localhost:8000/docs for a free interactive test UI --
use that to check your endpoints before wiring up the frontend.
"""
import os 
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import pandas as pd

from mplads_anomaly_detection import run_pipeline

app = FastAPI(title="MPLADS Anomaly Detection API")

# Allow the frontend (running on a different port/domain) to call this API.
# Wide open ("*") is fine for a hackathon demo -- tighten this before any
# real deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Run the pipeline once at startup and cache the result in memory ---
# Swap run_pipeline() for run_pipeline("your_real_data.csv") once you have
# the real dataset.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "telangana_mplads_demo_1000.csv")

_scored_df: pd.DataFrame = run_pipeline(DATA_PATH)
def _row_to_dict(row) -> dict:
    """Converts one DataFrame row into a clean JSON-friendly dict."""
    return {
        "work_id": row["unique_work_number"],
        "work_name": row["work_name"],
        "mp_name": row["mp_name"],
        "district": row["implementing_district"],
        "agency": row["implementing_agency_name"],
        "sanction_amount": float(row["sanction_amount"]),
        "status": row["work_status"],
        "risk_score": float(row["risk_score"]),
        "explanation": row["explanation"],
    }


@app.get("/summary")
def get_summary():
    """High-level numbers for the dashboard header / pitch slide."""
    flagged = _scored_df[_scored_df["risk_score"] > 40]
    return {
        "total_works": len(_scored_df),
        "flagged_works": len(flagged),
        "flagged_percent": round(len(flagged) / len(_scored_df) * 100, 1),
        "flagged_value_inr": float(flagged["sanction_amount"].sum()),
    }


@app.get("/works")
def get_works(
    min_risk: float = Query(0, description="Only return works with risk_score >= this"),
    district: Optional[str] = Query(None, description="Filter by implementing_district"),
    limit: int = Query(100, le=1000),
):
    """List works, optionally filtered. Always sorted by risk_score descending
    so the dashboard's default view shows the most suspicious works first."""
    df = _scored_df[_scored_df["risk_score"] >= min_risk]
    if district:
        df = df[df["implementing_district"] == district]
    df = df.sort_values("risk_score", ascending=False).head(limit)
    return [_row_to_dict(row) for _, row in df.iterrows()]


@app.get("/works/{work_id}")
def get_work_detail(work_id: str):
    """Single work's full detail -- used for the dashboard's drill-down view."""
    match = _scored_df[_scored_df["unique_work_number"] == work_id]
    if match.empty:
        raise HTTPException(status_code=404, detail="Work not found")
    return _row_to_dict(match.iloc[0])
