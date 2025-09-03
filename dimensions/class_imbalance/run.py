# -*- coding: utf-8 -*-
"""
Appendix 1.12 — Class Imbalance

For each model in models/:
- Train baseline on original train -> M_full
- Create balanced variants (under/over), train, eval on same val -> M_bal (best of two)
- QI = IR = min(n+, n-)/max(n+, n-)
- QR = 1 - min(1, max(0, M_bal - M_full)/0.10)
- Risk-proportional blend -> AI-Class-Imbalance Score (0–10)

Artifacts (results/class_imbalance/):
- ai_class_imbalance_summary.csv
- per_class_metrics_<model>.csv
"""

import math
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, precision_score, recall_score

# ----------------- Paths & config -----------------
FILE_DIR  = Path(__file__).parent
PROJ_ROOT = (FILE_DIR / ".." / "..").resolve()
DATA_PATH = PROJ_ROOT / "data" / "mimic_lung_day1.csv"
OUT_DIR   = PROJ_ROOT / "results" / "class_imbalance"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Feature/label config (same as other dimensions)
FEATURES: List[str] = [
    "age","vent_flag","pressors_flag",
    "hr_min","hr_max","map_min","map_max",
    "resp_rate_max","temp_max","spo2_min",
    "albumin_min","lactate_max","bun_max","creatinine_max","bilirubin_max",
    "hemoglobin_min","platelets_min","sodium_min","potassium_max","aniongap_max",
]
LABEL = "hospital_expire_flag"

# Method params
RANDOM_STATE = 42
TEST_SIZE = 0.30
EPS = 1e-6

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
    """
    Minimal shim to support your model trainer signatures.
    """
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
    """Primary metric AUROC; fallback to balanced accuracy if single-class in y_true."""
    y = np.asarray(y_true).astype(int)
    if len(np.unique(y)) > 1:
        return float(roc_auc_score(y, y_score)), "auroc"
    yhat = (y_score >= 0.5).astype(int)
    return float(balanced_accuracy_score(y, yhat)), "bal_acc"

def per_class_report(y_true: np.ndarray, y_score: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    y = np.asarray(y_true).astype(int)
    yhat = (np.asarray(y_score) >= thr).astype(int)
    # Positive class metrics
    sens = float(recall_score(y, yhat, zero_division=0))
    ppv  = float(precision_score(y, yhat, zero_division=0))
    # Negative class "sensitivity" (specificity)
    tn   = int(((y==0) & (yhat==0)).sum()); fp = int(((y==0) & (yhat==1)).sum())
    spec = float(tn / (tn + fp + EPS))
    return {"sensitivity_pos": sens, "ppv_pos": ppv, "specificity_neg": spec, "threshold": thr}

def make_balanced(df: pd.DataFrame, label: str, method: str) -> Tuple[pd.DataFrame, str]:
    """
    Simple class balancing:
    - 'under': random undersample majority to minority count
    - 'over' : random oversample minority to majority count
    Returns balanced df and method actually used (or 'none' if impossible).
    """
    pos = df[df[label] == 1]
    neg = df[df[label] == 0]
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return df.copy(), "none"

    if method == "under":
        n = min(n_pos, n_neg)
        pos_s = pos.sample(n, replace=False, random_state=RANDOM_STATE)
        neg_s = neg.sample(n, replace=False, random_state=RANDOM_STATE)
    else:  # "over"
        n = max(n_pos, n_neg)
        pos_s = pos.sample(n, replace=True, random_state=RANDOM_STATE)
        neg_s = neg.sample(n, replace=True, random_state=RANDOM_STATE)

    out = pd.concat([pos_s, neg_s], axis=0).sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    return out, method

# ----------------- Main -----------------
def main():
    # Load & basic checks
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing data at {DATA_PATH}. Run data/make_fake_mimic.py first.")
    df = pd.read_csv(DATA_PATH)

    for c in [LABEL] + FEATURES:
        if c not in df.columns:
            raise RuntimeError(f"Required column '{c}' not in {DATA_PATH}")

    df = df.loc[df[LABEL].notna()].copy()
    df[LABEL] = df[LABEL].astype(int)

    # Single stratified split
    train_df, val_df = train_test_split(df, test_size=TEST_SIZE, stratify=df[LABEL], random_state=RANDOM_STATE)
    X_val = val_df[FEATURES].copy()
    y_val = val_df[LABEL].to_numpy()

    # Training class counts & imbalance ratio (QI)
    n_pos = int((train_df[LABEL] == 1).sum())
    n_neg = int((train_df[LABEL] == 0).sum())
    if max(n_pos, n_neg) == 0:
        IR = 0.0
    else:
        IR = float(min(n_pos, n_neg) / (max(n_pos, n_neg) + EPS))  # QI

    summaries = []
    for m in MODEL_NAMES:
        # ----- Baseline -----
        model_full = call_trainer(TRAINERS[m], train_df, LABEL, FEATURES)
        p_full = get_proba(model_full, X_val)
        M_full, metric_type = metric_M(y_val, p_full)

        # ----- Balanced variants -----
        best_M_bal = -np.inf
        best_method = "none"
        best_bal_df_size = len(train_df)

        for method in ["under", "over"]:
            bal_df, used = make_balanced(train_df, LABEL, method)
            if used == "none":
                continue
            try:
                model_bal = call_trainer(TRAINERS[m], bal_df, LABEL, FEATURES)
                p_bal = get_proba(model_bal, X_val)
                M_bal, _ = metric_M(y_val, p_bal)
                if M_bal > best_M_bal:
                    best_M_bal = M_bal
                    best_method = used
                    best_bal_df_size = len(bal_df)
            except Exception:
                # if a trainer fails on a variant, just skip that variant
                continue

        # If neither variant succeeded, set M_bal = M_full
        if best_method == "none":
            best_M_bal = M_full

        # ----- Robustness delta & QR -----
        delta = float(max(0.0, best_M_bal - M_full))
        QR = float(1.0 - min(1.0, delta / 0.10))  # 10 pp cap

        # ----- Weights & Score -----
        QI = IR
        dI = 1.0 - QI
        dR = 1.0 - QR
        if (dI < 0.02) and (dR < 0.02):
            wI, wR = 0.5, 0.5
        else:
            wI = (dI + EPS) / (dI + dR + 2*EPS)
            wR = 1.0 - wI

        S_raw = wI * QI + wR * QR
        Score = int(math.floor(10.0 * S_raw + 0.5))

        # ----- Per-class reporting on baseline -----
        cls_rep = per_class_report(y_val, p_full, thr=0.5)
        report_df = pd.DataFrame([{
            "model": m,
            "n_pos_train": n_pos,
            "n_neg_train": n_neg,
            "pos_rate_train": n_pos / (n_pos + n_neg + EPS),
            **cls_rep
        }])
        report_df.to_csv(OUT_DIR / f"per_class_metrics_{m}.csv", index=False)

        # ----- Collect summary -----
        summaries.append({
            "model": m,
            "metric_type": metric_type,
            "n_pos_train": n_pos,
            "n_neg_train": n_neg,
            "IR_QI": QI,
            "M_full": M_full,
            "M_bal": float(best_M_bal),
            "delta_bal_minus_full": delta,
            "QR": QR,
            "wI": wI,
            "wR": wR,
            "S_raw": S_raw,
            "AI_Class_Imbalance_Score": Score,
            "balancing_method_used": best_method,
            "n_train_balanced": best_bal_df_size,
        })

    out = pd.DataFrame(summaries)
    csv_path = OUT_DIR / "ai_class_imbalance_summary.csv"
    out.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)

    print(out[[
        "model","AI_Class_Imbalance_Score","IR_QI","QR","wI","wR",
        "M_full","M_bal","delta_bal_minus_full","balancing_method_used"
    ]].to_string(index=False))
    print(f"\nPer-model per-class metrics saved as {OUT_DIR}/per_class_metrics_<model>.csv")
    print(f"Summary appended to {csv_path}")

if __name__ == "__main__":
    main()
