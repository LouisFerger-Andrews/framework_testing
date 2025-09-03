# -*- coding: utf-8 -*-
"""
Appendix 1.13 — Data Freshness

Computes latency L = trecord - tevent (seconds), then:
Qx = min(1, max(0, 2 - Lx/SLO)), x in {50,95}
Adaptive weights -> AI-Data-Freshness Score (0–10)

Auto-detects timestamp pairs in the input CSV:
  Preferred pairs (first match wins):
    ('charttime','storetime'), ('event_time','record_time'),
    ('measurement_time','ingest_time'), ('chart_time','store_time'),
    ('lab_charttime','lab_storetime'), ('timestamp','recorded_at'),
    ('time_event','time_record')
Or uses a numeric 'latency_seconds' column, if present.
If none available, generates a synthetic latency demo (seeded).

Artifacts: results/freshness/
- latency_sample.csv           (subset of latencies for inspection)
- freshness_quantiles.csv      (L50, L95, and bootstrap CIs)
- ai_freshness_summary.csv     (appended each run)
- (optional) group_freshness.csv if 'sex'/'gender' present (not used in score)
"""

import os
import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# ----------------- Paths & config -----------------
FILE_DIR  = Path(__file__).parent
PROJ_ROOT = (FILE_DIR / ".." / "..").resolve()
DATA_PATH = PROJ_ROOT / "data" / "mimic_lung_day1.csv"
OUT_DIR   = PROJ_ROOT / "results" / "freshness"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# SLO (seconds). Override via env FRESHNESS_SLO.
SLO_SECONDS = float(os.environ.get("FRESHNESS_SLO", "30"))

# Bootstrap reps for CIs (kept light to run fast on laptop)
BOOT_REPS = 200
EPS = 1e-6
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ----------------- Helpers -----------------
TIME_PAIRS: List[Tuple[str, str]] = [
    ("charttime", "storetime"),
    ("event_time", "record_time"),
    ("measurement_time", "ingest_time"),
    ("chart_time", "store_time"),
    ("lab_charttime", "lab_storetime"),
    ("timestamp", "recorded_at"),
    ("time_event", "time_record"),
]

def parse_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=False)

def find_latency_seconds(df: pd.DataFrame) -> Tuple[np.ndarray, str]:
    # 1) try explicit latency column
    for col in ["latency_seconds", "latency_sec", "latency_s"]:
        if col in df.columns:
            lat = pd.to_numeric(df[col], errors="coerce").to_numpy()
            lat = lat[np.isfinite(lat)]
            lat = lat[lat >= 0]
            if lat.size > 0:
                return lat, f"explicit:{col}"

    # 2) try time pairs
    for ev_col, rec_col in TIME_PAIRS:
        if ev_col in df.columns and rec_col in df.columns:
            ev = parse_dt(df[ev_col])
            rec = parse_dt(df[rec_col])
            delta = (rec - ev).dt.total_seconds()
            lat = delta.to_numpy()
            lat = lat[np.isfinite(lat)]
            lat = lat[lat >= 0]
            if lat.size > 0:
                return lat, f"pair:{ev_col}->{rec_col}"

    # 3) none found
    return np.array([]), "none"

def bootstrap_ci(values: np.ndarray, q: float, reps: int = BOOT_REPS) -> Tuple[float, float]:
    if values.size == 0:
        return (np.nan, np.nan)
    n = values.size
    qs = []
    for _ in range(reps):
        idx = np.random.randint(0, n, size=n)
        qs.append(np.percentile(values[idx], q))
    qs = np.sort(np.array(qs, dtype=float))
    lo = float(np.percentile(qs, 2.5))
    hi = float(np.percentile(qs, 97.5))
    return lo, hi

def compute_Q(Lx: float, slo: float) -> float:
    if not np.isfinite(Lx) or slo <= 0:
        return 0.0
    return float(np.clip(2.0 - (Lx / slo), 0.0, 1.0))

def get_sex_series(df: pd.DataFrame) -> Optional[pd.Series]:
    for col in ["sex", "gender", "Sex", "Gender"]:
        if col in df.columns:
            sc = df[col].astype(str).str.strip().str.lower()
            mapped = np.where(sc.isin(["m","male","1","true","t","man"]), "Male",
                      np.where(sc.isin(["f","female","0","false","f","woman"]), "Female", "Other"))
            return pd.Series(mapped, index=df.index, name="sex_group")
    return None

# ----------------- Main -----------------
def main():
    # Load
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing data at {DATA_PATH}. Run data/make_fake_mimic.py first.")
    df = pd.read_csv(DATA_PATH)

    # Extract latencies
    lat, source = find_latency_seconds(df)

    if lat.size == 0:
        # Synthetic demo (seeded) to keep the runner functional without time columns.
        # 80% < SLO, 15% between 1x and 2x SLO, 5% beyond 2x SLO
        n = max(500, min(5000, len(df)))
        p = np.random.rand(n)
        lat = np.empty(n, dtype=float)
        fast = p < 0.80
        mid  = (p >= 0.80) & (p < 0.95)
        slow = p >= 0.95
        lat[fast] = np.random.uniform(0.1 * SLO_SECONDS, 0.8 * SLO_SECONDS, size=fast.sum())
        lat[mid]  = np.random.uniform(1.0 * SLO_SECONDS, 2.0 * SLO_SECONDS, size=mid.sum())
        lat[slow] = np.random.uniform(2.0 * SLO_SECONDS, 4.0 * SLO_SECONDS, size=slow.sum())
        source = "synthetic_demo"  # clearly labelled

    # Compute percentiles
    L50 = float(np.percentile(lat, 50))
    L95 = float(np.percentile(lat, 95))

    # Bootstrap CIs for latencies and Q-scores
    L50_lo, L50_hi = bootstrap_ci(lat, 50.0)
    L95_lo, L95_hi = bootstrap_ci(lat, 95.0)

    Q50 = compute_Q(L50, SLO_SECONDS)
    Q95 = compute_Q(L95, SLO_SECONDS)

    # Risk-proportional blend
    d50 = 1.0 - Q50
    d95 = 1.0 - Q95
    if (d50 < 0.02) and (d95 < 0.02):
        w50, w95 = 0.5, 0.5
    else:
        w50 = (d50 + EPS) / (d50 + d95 + 2*EPS)
        w95 = 1.0 - w50

    S_raw = w50 * Q50 + w95 * Q95
    Score = int(math.floor(10.0 * S_raw + 0.5))

    # Save a sample of latencies (for quick inspection)
    sample_sz = min(2000, lat.size)
    samp = pd.DataFrame({"latency_seconds": lat[:sample_sz]})
    samp["source"] = source
    samp.to_csv(OUT_DIR / "latency_sample.csv", index=False)

    # Save quantiles + CIs
    qtbl = pd.DataFrame([{
        "SLO_seconds": SLO_SECONDS,
        "source": source,
        "L50": L50, "L50_lo": L50_lo, "L50_hi": L50_hi,
        "L95": L95, "L95_lo": L95_lo, "L95_hi": L95_hi,
        "Q50": Q50, "Q95": Q95, "w50": w50, "w95": w95,
        "S_raw": S_raw, "AI_Data_Freshness_Score": Score,
    }])
    qtbl.to_csv(OUT_DIR / "freshness_quantiles.csv", index=False)

    # Optional fairness stratification (does not affect score): by sex if available
    sex_s = get_sex_series(df)
    if sex_s is not None and source.startswith("pair:"):
        # recompute latencies restricted to rows with valid times for each sex group
        ev_col, rec_col = source.split("pair:")[1].split("->")
        ev = pd.to_datetime(df[ev_col], errors="coerce")
        rec = pd.to_datetime(df[rec_col], errors="coerce")
        delta = (rec - ev).dt.total_seconds()
        rows = []
        for grp in ["Female", "Male", "Other"]:
            mask = (sex_s == grp) & delta.notna() & (delta >= 0)
            if mask.sum() == 0:
                rows.append({"group": grp, "N": 0, "L50": np.nan, "L95": np.nan, "Q50": np.nan, "Q95": np.nan})
                continue
            arr = delta[mask].to_numpy()
            l50 = float(np.percentile(arr, 50)); l95 = float(np.percentile(arr, 95))
            rows.append({"group": grp, "N": int(mask.sum()),
                         "L50": l50, "L95": l95,
                         "Q50": compute_Q(l50, SLO_SECONDS),
                         "Q95": compute_Q(l95, SLO_SECONDS)})
        pd.DataFrame(rows).to_csv(OUT_DIR / "group_freshness.csv", index=False)

    # Append summary row
    sum_path = OUT_DIR / "ai_freshness_summary.csv"
    qtbl.to_csv(sum_path, mode="a", header=not sum_path.exists(), index=False)

    # Console summary
    print(qtbl[[
        "AI_Data_Freshness_Score","SLO_seconds",
        "L50","L95","Q50","Q95","w50","w95","source"
    ]].to_string(index=False))
    print(f"\nSaved sample latencies    -> {OUT_DIR/'latency_sample.csv'}")
    print(f"Saved quantiles + CIs     -> {OUT_DIR/'freshness_quantiles.csv'}")
    print(f"Summary appended to       -> {sum_path}")

if __name__ == "__main__":
    main()
