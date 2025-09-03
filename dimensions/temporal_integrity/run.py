# -*- coding: utf-8 -*-
"""
Appendix 1.14 — Temporal Integrity

We simulate coarser sampling intervals (∆t) by attenuating "extreme" features (mins/maxes)
toward central tendency, train/evaluate your models at each ∆t, find ∆t_elbow (largest ∆t
with ≤1pp AUROC drop vs best), then compute:
  QT = min(1, ∆t_opt / ∆t_cur), where ∆t_opt = max(τ_clin, τ_phys, τ_acf, ∆t_elbow)
  QR from ∆AUROC(∆t_elbow vs ∆t_cur), with 5pp cap
  Risk-proportional blend -> AI-Temporal-Integrity Score (0–10)

This runner uses your trainers in models/ and your standard feature list.

Env overrides (minutes):
  CUR_INTERVAL_MIN   (default 60)
  TAU_CLIN_MIN       (default 30)
  TAU_PHYS_MIN       (default 15)
  TAU_ACF_MIN        (default = TAU_PHYS_MIN)

Artifacts (results/temporal_integrity/):
  temporal_sweep_<model>.csv       # AUROC across ∆t grid
  ai_temporal_integrity_summary.csv
"""

import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

# ----------------- Paths & config -----------------
FILE_DIR  = Path(__file__).parent
PROJ_ROOT = (FILE_DIR / ".." / "..").resolve()
DATA_PATH = PROJ_ROOT / "data" / "mimic_lung_day1.csv"
OUT_DIR   = PROJ_ROOT / "results" / "temporal_integrity"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Features/label (same set used elsewhere)
FEATURES: List[str] = [
    "age","vent_flag","pressors_flag",
    "hr_min","hr_max","map_min","map_max",
    "resp_rate_max","temp_max","spo2_min",
    "albumin_min","lactate_max","bun_max","creatinine_max","bilirubin_max",
    "hemoglobin_min","platelets_min","sodium_min","potassium_max","aniongap_max",
]
LABEL = "hospital_expire_flag"

# Split/seed
RANDOM_STATE = 42
TEST_SIZE = 0.30
EPS = 1e-6

# Sampling grid (minutes) — feel free to adjust/extend
DT_GRID = [1, 2, 5, 10, 15, 30, 60, 90, 120]

# Env-configured current interval and τ*
DT_CUR   = float(os.environ.get("CUR_INTERVAL_MIN", "60"))
TAU_CLIN = float(os.environ.get("TAU_CLIN_MIN", "30"))
TAU_PHYS = float(os.environ.get("TAU_PHYS_MIN", "15"))
TAU_ACF  = float(os.environ.get("TAU_ACF_MIN", str(TAU_PHYS)))

ELBOW_EPS = 0.01  # ≤1pp AUROC drop defines ∆t_elbow (vs best across grid)

# ----------------- Import your trainers -----------------
sys.path.insert(0, str(PROJ_ROOT))
from models import train_logreg_en, train_random_forest, train_gbm  # type: ignore

TRAINERS = {
    "logreg_en": train_logreg_en,
    "random_forest": train_random_forest,
    "gbm": train_gbm,
}
MODEL_NAMES = ["logreg_en", "random_forest", "gbm"]

# ----------------- Utilities -----------------
def call_trainer(trainer, df: pd.DataFrame, label_col: str, feature_cols: List[str]):
    """Small shim to support different trainer signatures."""
    try:
        return trainer(df.copy(), label_col=label_col, feature_cols=feature_cols)
    except TypeError:
        try:
            return trainer(df.copy(), label_col=label_col, features=feature_cols)
        except TypeError:
            try:
                return trainer(df.copy(), label_col, feature_cols)  # positional
            except TypeError as e:
                raise TypeError(
                    f"Trainer signature not supported. Tried keyword/positional variants. Original error: {e}"
                )

def get_proba(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        s = np.asarray(s, dtype=float)
        return (s - s.min()) / (s.max() - s.min() + 1e-9)
    p = model.predict(X)
    return np.asarray(p, dtype=float)

def metric_M(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, str]:
    y = np.asarray(y_true).astype(int)
    if len(np.unique(y)) > 1:
        return float(roc_auc_score(y, y_score)), "auroc"
    yhat = (y_score >= 0.5).astype(int)
    return float(balanced_accuracy_score(y, yhat)), "bal_acc"

def attenuation_gamma(dt_min: float, cap_min: float = 60.0) -> float:
    """
    Map ∆t to [0,1] attenuation; 0 ~ fine sampling (retain extremes),
    1 ~ coarse sampling (extremes collapse to central). Linear in minutes.
    """
    return float(np.clip(dt_min / cap_min, 0.0, 1.0))

def simulate_coarser_sampling(df: pd.DataFrame, dt_min: float, med: Dict[str, float]) -> pd.DataFrame:
    """
    Approximate effect of coarser ∆t by pulling mins/maxes toward central tendency.

    Rules:
      - For paired features (min/max), move both toward center = 0.5*(min+max).
      - For lone *_min or *_max features, move toward global median.
      - Flags (vent/pressors) and age unchanged.
    """
    g = attenuation_gamma(dt_min)
    out = df.copy()

    # define pairs present in our schema
    pairs = [
        ("hr_min", "hr_max"),
        ("map_min", "map_max"),
    ]
    for a, b in pairs:
        if a in out.columns and b in out.columns:
            va = pd.to_numeric(out[a], errors="coerce")
            vb = pd.to_numeric(out[b], errors="coerce")
            center = 0.5 * (va + vb)
            out[a] = va + g * (center - va)
            out[b] = vb + g * (center - vb)

    # Lone mins -> toward median
    lone_mins = ["spo2_min","albumin_min","hemoglobin_min","platelets_min","sodium_min"]
    for f in lone_mins:
        if f in out.columns:
            v = pd.to_numeric(out[f], errors="coerce")
            out[f] = v + g * (med[f] - v)

    # Lone maxes -> toward median
    lone_maxes = ["resp_rate_max","temp_max","lactate_max","bun_max","creatinine_max","bilirubin_max",
                  "potassium_max","aniongap_max"]
    for f in lone_maxes:
        if f in out.columns:
            v = pd.to_numeric(out[f], errors="coerce")
            out[f] = v + g * (med[f] - v)

    # leave age/flags untouched
    return out

# ----------------- Main -----------------
def main():
    # Load data
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing data at {DATA_PATH}. Run data/make_fake_mimic.py first.")
    df = pd.read_csv(DATA_PATH)

    # Basic checks
    for c in [LABEL] + FEATURES:
        if c not in df.columns:
            raise RuntimeError(f"Required column '{c}' not found in {DATA_PATH}")

    df = df.loc[df[LABEL].notna()].copy()
    df[LABEL] = df[LABEL].astype(int)

    # Single stratified split (same protocol as other dimensions)
    train_df, val_df = train_test_split(df, test_size=TEST_SIZE, stratify=df[LABEL], random_state=RANDOM_STATE)
    X_val_base = val_df[FEATURES].copy()
    y_val = val_df[LABEL].to_numpy()

    # Global medians used by the simulation transform
    med = {f: float(pd.to_numeric(train_df[f], errors="coerce").median()) for f in FEATURES if f in train_df.columns}

    # Sweep ∆t for each model
    summaries = []
    for mname in MODEL_NAMES:
        rows = []
        # evaluate across grid
        for dt in DT_GRID:
            tr_sim = simulate_coarser_sampling(train_df[FEATURES], dt, med)
            va_sim = simulate_coarser_sampling(X_val_base, dt, med)

            tr_df_sim = pd.concat([tr_sim.reset_index(drop=True), train_df[[LABEL]].reset_index(drop=True)], axis=1)
            va_df_sim = pd.concat([va_sim.reset_index(drop=True), val_df[[LABEL]].reset_index(drop=True)], axis=1)

            model = call_trainer(TRAINERS[mname], tr_df_sim, LABEL, FEATURES)
            p = get_proba(model, va_df_sim[FEATURES])
            auc, metric_type = metric_M(y_val, p)

            rows.append({"model": mname, "dt_minutes": float(dt), "metric": auc, "metric_type": metric_type})

        sweep = pd.DataFrame(rows).sort_values("dt_minutes")
        sweep.to_csv(OUT_DIR / f"temporal_sweep_{mname}.csv", index=False)

        # Find best metric and ∆t_elbow (largest ∆t within 1pp of best)
        best_metric = float(sweep["metric"].max())
        tol = best_metric - ELBOW_EPS
        elbow_rows = sweep[sweep["metric"] >= tol]
        if elbow_rows.empty:
            # fallback: use smallest dt as elbow
            dt_elbow = float(sweep["dt_minutes"].min())
            m_elbow = float(sweep.loc[sweep["dt_minutes"].idxmin(), "metric"])
        else:
            idx = elbow_rows["dt_minutes"].idxmax()
            dt_elbow = float(sweep.loc[idx, "dt_minutes"])
            m_elbow = float(sweep.loc[idx, "metric"])

        # Metric at current ∆t_cur (interpolate to nearest grid point)
        # choose nearest dt in grid
        nearest_idx = (sweep["dt_minutes"] - DT_CUR).abs().idxmin()
        m_cur = float(sweep.loc[nearest_idx, "metric"])
        dt_cur_snap = float(sweep.loc[nearest_idx, "dt_minutes"])

        # Step A — QT
        dt_opt = float(max(TAU_CLIN, TAU_PHYS, TAU_ACF, dt_elbow))
        QT = float(min(1.0, dt_opt / (DT_CUR + EPS)))

        # Step B — QR (5pp cap, note ∆AUROC defined with m_elbow - m_cur)
        delta = float(max(0.0, m_elbow - m_cur))
        QR = float(1.0 - min(1.0, delta / 0.05))

        # Step C — weights
        dT = 1.0 - QT
        dR = 1.0 - QR
        if (dT < 0.02) and (dR < 0.02):
            wT, wR = 0.5, 0.5
        else:
            wT = (dT + EPS) / (dT + dR + 2*EPS)
            wR = 1.0 - wT

        # Step D — Score
        S_raw = wT * QT + wR * QR
        Score = int(math.floor(10.0 * S_raw + 0.5))

        summaries.append({
            "model": mname,
            "DT_cur_input": DT_CUR,
            "DT_cur_used_from_grid": dt_cur_snap,
            "tau_clin": TAU_CLIN,
            "tau_phys": TAU_PHYS,
            "tau_acf": TAU_ACF,
            "DT_elbow": dt_elbow,
            "best_metric": best_metric,
            "metric_at_elbow": m_elbow,
            "metric_at_cur": m_cur,
            "QT": QT,
            "QR": QR,
            "wT": wT,
            "wR": wR,
            "S_raw": S_raw,
            "AI_Temporal_Integrity_Score": Score,
        })

    out = pd.DataFrame(summaries)
    csv_path = OUT_DIR / "ai_temporal_integrity_summary.csv"
    out.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)

    print(out[[
        "model","AI_Temporal_Integrity_Score","DT_cur_input","DT_elbow",
        "QT","QR","wT","wR","metric_at_cur","metric_at_elbow"
    ]].to_string(index=False))
    print(f"\nPer-model sweeps saved as {OUT_DIR}/temporal_sweep_<model>.csv")
    print(f"Summary appended to {csv_path}")

if __name__ == "__main__":
    main()
