# -*- coding: utf-8 -*-
"""
Appendix 1.2 Consistency / Conformance
Exact implementation per methodology:
- Step A: Empirical conformance C on validation after minimal remediation
- Step B: Robustness sweep M_psi at psi ∈ {1,3,5,10}% faults (no retraining), R from Δmax
- Step C: Adaptive weights
- Step D: AI-Conformance Score = floor(10 * (wC*C + wR*R) + 0.5)

Outputs (under results/conformance/):
- ai_conformance_summary.csv                         (appended)
- ai_conformance_<model>.json                        (per model summary)
- violations_<model>.csv                             (per feature E, N, rates on validation)
- injection_sweep_<model>.csv                        (M_psi across ψ per model)
"""

import os, sys, json, math
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn import set_config
set_config(transform_output="pandas")  # keep feature names through pipelines

import warnings
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
    category=UserWarning,
)

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Optional boosters with safe fallbacks
HAS_LGBM = False
HAS_XGB = False
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    pass
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    pass
from sklearn.ensemble import HistGradientBoostingClassifier

# ---------------- Paths & config ----------------
FILE_DIR = Path(__file__).parent
PROJ_ROOT = (FILE_DIR / ".." / "..").resolve()
DATA_PATH = PROJ_ROOT / "data" / "mimic_lung_day1.csv"
SCHEMA_PATH = PROJ_ROOT / "data" / "schema_conformance.json"

OUT_DIR = PROJ_ROOT / "results" / "conformance"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES: List[str] = [
    "age","vent_flag","pressors_flag",
    "hr_min","hr_max","map_min","map_max",
    "resp_rate_max","temp_max","spo2_min",
    "albumin_min","lactate_max","bun_max","creatinine_max","bilirubin_max",
    "hemoglobin_min","platelets_min","sodium_min","potassium_max","aniongap_max",
]
CATEGORICALS: List[str] = []   # add categorical features here if you have them
LABEL = "hospital_expire_flag"
SUBJECT_COL: Optional[str] = None  # if present, we split by subject; else row-wise split

TEST_SIZE = 0.30
RANDOM_STATE = 42
EPS = 1e-6
PSI_LEVELS = [1, 3, 5, 10]  # percent
CONFORMANCE_CAP = 0.05      # if E/N > 5%, C := 0 (Appendix cap)

# ---------------- Schema helpers ----------------
def _default_schema() -> Dict:
    """Basic plausible numeric ranges for the synthetic example; replace via data/schema_conformance.json for real data."""
    rng = {
        "age": (18, 100),
        "vent_flag": (0, 1),
        "pressors_flag": (0, 1),
        "hr_min": (30, 160), "hr_max": (50, 220),
        "map_min": (30, 120), "map_max": (50, 160),
        "resp_rate_max": (8, 60),
        "temp_max": (34.0, 42.0),
        "spo2_min": (50, 100),
        "albumin_min": (0.5, 6.0),
        "lactate_max": (0.2, 20.0),
        "bun_max": (2, 150),
        "creatinine_max": (0.2, 10.0),
        "bilirubin_max": (0.1, 30.0),
        "hemoglobin_min": (4, 20),
        "platelets_min": (5, 1000),
        "sodium_min": (110, 170),
        "potassium_max": (2.0, 8.0),
        "aniongap_max": (2, 40),
    }
    features = {}
    for k in FEATURES:
        lo, hi = rng.get(k, (-np.inf, np.inf))
        features[k] = {"dtype": "numeric", "min": lo, "max": hi, "unit": None, "unit_col": None}
    # include label conformance (0/1)
    features[LABEL] = {"dtype": "numeric", "min": 0, "max": 1, "unit": None, "unit_col": None}
    return {
        "features": features,
        "categoricals": {},   # e.g. {"admission_type": {"allowed": ["EMERGENCY","URGENT","ELECTIVE"]}}
        "units": {
            # optionally per-feature maps, e.g. "creatinine_max": {"mg/dL->µmol/L": 88.4, "µmol/L->mg/dL": 1/88.4}
        }
    }

def load_schema() -> Dict:
    if SCHEMA_PATH.exists():
        try:
            return json.loads(SCHEMA_PATH.read_text())
        except Exception:
            pass
    return _default_schema()

# ---------------- Model builders ----------------
def build_preprocessor(num_cols: List[str], cat_cols: List[str], scale_numeric: bool) -> ColumnTransformer:
    num_pipe = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_pipe.append(("scale", StandardScaler()))
    num_pipe = Pipeline(num_pipe)

    if cat_cols:
        cat_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        return ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)], remainder="drop")
    return ColumnTransformer([("num", num_pipe, num_cols)], remainder="drop")

def make_model(name: str, y_train: pd.Series, num_cols: List[str], cat_cols: List[str]) -> Pipeline:
    if name == "logreg_en":
        pre = build_preprocessor(num_cols, cat_cols, scale_numeric=True)
        clf = LogisticRegression(max_iter=4000, solver="saga", penalty="elasticnet",
                                 l1_ratio=0.2, class_weight="balanced", random_state=RANDOM_STATE)
        return Pipeline([("prep", pre), ("clf", clf)])
    if name == "random_forest":
        pre = build_preprocessor(num_cols, cat_cols, scale_numeric=False)
        clf = RandomForestClassifier(n_estimators=600, max_depth=None, min_samples_leaf=2,
                                     class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE)
        return Pipeline([("prep", pre), ("clf", clf)])
    if name == "gbm":
        pre = build_preprocessor(num_cols, cat_cols, scale_numeric=False)
        pos = int(y_train.sum()); neg = int(len(y_train) - pos)
        scale_pos = float(neg) / float(pos) if pos > 0 else 1.0
        if HAS_LGBM:
            clf = LGBMClassifier(n_estimators=800, learning_rate=0.03, num_leaves=64,
                                 subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                                 objective="binary", random_state=RANDOM_STATE, n_jobs=-1,
                                 scale_pos_weight=scale_pos, verbosity=-1)
        elif HAS_XGB:
            clf = XGBClassifier(n_estimators=900, learning_rate=0.03, max_depth=4,
                                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                                objective="binary:logistic", eval_metric="auc",
                                random_state=RANDOM_STATE, n_jobs=-1,
                                scale_pos_weight=scale_pos, tree_method="hist")
        else:
            clf = HistGradientBoostingClassifier(max_depth=None, learning_rate=0.05, max_iter=400, random_state=RANDOM_STATE)
        return Pipeline([("prep", pre), ("clf", clf)])
    raise ValueError(f"Unknown model {name}")

# ---------------- Conformance checks (Step A) ----------------
def _coerce_numeric(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    return pd.to_numeric(s, errors="coerce")

def _normalize_units(df: pd.DataFrame, feat: str, meta: Dict, units_cfg: Dict) -> pd.Series:
    """Convert to expected unit if a unit column & map are provided; else return numeric coercion."""
    s = df[feat]
    unit_col = meta.get("unit_col")
    target_unit = meta.get("unit")
    if unit_col and unit_col in df.columns and target_unit:
        conv_map = units_cfg.get(feat, {})  # per-feature conversion map
        src = df[unit_col].astype(str).str.strip().fillna("")
        vals = _coerce_numeric(s).copy()
        for u in src.dropna().unique():
            if not u or u == target_unit:
                continue
            key = f"{u}->{target_unit}"
            if key in conv_map:
                mask = src == u
                vals.loc[mask] = vals.loc[mask] * float(conv_map[key])
        return vals
    return _coerce_numeric(s)

def detect_nonconforming(val_df: pd.DataFrame, schema: Dict) -> Tuple[int, int, pd.DataFrame]:
    """
    Returns (E, N, per-feature table)
    - Minimal remediation: numeric coercion + unit normalization
    - Counts only non-missing checked cells
    - Numeric: outside [min, max] is a violation
    - Categorical: not in allowed list is a violation (if list provided), else unassessed (Nf=0)
    - Includes LABEL conformance if present in schema
    """
    feats_meta = schema.get("features", {})
    units_cfg = schema.get("units", {})
    rows = []
    E_total = 0
    N_total = 0

    for f in FEATURES + [LABEL]:
        meta = feats_meta.get(f, {"dtype": "numeric", "min": -np.inf, "max": np.inf, "unit": None, "unit_col": None})
        dtype = meta.get("dtype", "numeric")
        if dtype == "numeric":
            s = _normalize_units(val_df, f, meta, units_cfg)
            Nf = int(s.notna().sum())
            lo, hi = float(meta.get("min", -np.inf)), float(meta.get("max", np.inf))
            Ef = int(((s < lo) | (s > hi)).sum())
        else:
            allowed = meta.get("allowed", [])
            if allowed:
                s = val_df[f].astype(str)
                Nf = int(s.notna().sum())
                Ef = int((~s.isin(allowed)).sum())
            else:
                Nf, Ef = 0, 0  # unassessed without vocabulary
        rows.append({"feature": f, "checked": Nf, "violations": Ef, "violation_rate": float(Ef / (Nf + EPS)) if Nf else 0.0})
        E_total += Ef
        N_total += Nf

    tbl = pd.DataFrame(rows).sort_values(["violation_rate", "violations"], ascending=False)
    return int(E_total), int(N_total), tbl

# ---------------- Robustness sweep (Step B) ----------------
def inject_faults(df_val_X: pd.DataFrame, psi_percent: int, schema: Dict, rng: np.random.RandomState) -> pd.DataFrame:
    """
    Inject schema violations into psi% of numeric feature cells on a copy of validation X.
    Types:
      - scale10 / scale01 (unit mismatch)
      - spike (+5 std)
      - sentinel_low / sentinel_high (beyond allowed bounds)
      - flip_flag for binary 0/1 ranges
    Ensures numeric columns are float to avoid dtype warnings.
    """
    out = df_val_X.copy()
    feats_meta = schema.get("features", {})

    # make numeric features float so we can assign injected values
    for f in FEATURES:
        meta = feats_meta.get(f, {"dtype": "numeric"})
        if meta.get("dtype", "numeric") == "numeric" and f in out.columns:
            out[f] = pd.to_numeric(out[f], errors="coerce").astype(float)

    # build candidate cells
    candidates: List[Tuple[int, str]] = []
    for f in FEATURES:
        meta = feats_meta.get(f, {"dtype": "numeric"})
        if meta.get("dtype", "numeric") != "numeric" or f not in out.columns:
            continue
        mask = out[f].notna().values
        idxs = out.index.values[mask]
        for i in idxs:
            candidates.append((int(i), f))

    if not candidates:
        return out

    k = max(1, int(len(candidates) * psi_percent / 100.0))
    picks = rng.choice(len(candidates), size=k, replace=False)

    for j in picks:
        i, f = candidates[j]
        v = out.at[i, f]
        if pd.isna(v):
            continue

        lo = float(feats_meta.get(f, {}).get("min", -1000))
        hi = float(feats_meta.get(f, {}).get("max", 1000))
        is_binary = (lo == 0.0 and hi == 1.0)

        if is_binary:
            # flip 0 <-> 1 to create a categorical/range conformance fault
            out.at[i, f] = 1.0 - float(v >= 0.5)
            continue

        t = rng.choice(["scale10", "scale01", "spike", "sentinel_low", "sentinel_high"])
        if t == "scale10":
            out.at[i, f] = float(v) * 10.0
        elif t == "scale01":
            out.at[i, f] = float(v) * 0.1
        elif t == "spike":
            std = float(pd.to_numeric(out[f], errors="coerce").std(skipna=True) or 1.0)
            out.at[i, f] = float(v) + 5.0 * std
        elif t == "sentinel_low":
            out.at[i, f] = lo - abs(0.2 * (abs(lo) + 1.0))
        else:  # sentinel_high
            out.at[i, f] = hi + abs(0.2 * (abs(hi) + 1.0))

    return out

# ---------------- Metric (AUROC else balanced accuracy) ----------------
def metric_M(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, str]:
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) > 1:
        return float(roc_auc_score(y_true, y_score)), "auroc"
    yhat = (y_score >= 0.5).astype(int)
    return float(balanced_accuracy_score(y_true, yhat)), "bal_acc"

# ---------------- Per-model run ----------------
def run_one_model(model_name: str, train_df: pd.DataFrame, val_df: pd.DataFrame, schema: Dict) -> Dict:
    num_cols = [c for c in FEATURES if c not in CATEGORICALS]
    model = make_model(model_name, train_df[LABEL].astype(int), num_cols, CATEGORICALS)

    X_tr, y_tr = train_df[FEATURES], train_df[LABEL].astype(int)
    X_va, y_va = val_df[FEATURES], val_df[LABEL].astype(int)

    # Train once (fresh)
    model.fit(X_tr, y_tr)

    # Step A: empirical conformance after remediation
    E, N, violations = detect_nonconforming(val_df, schema)
    violations.to_csv(OUT_DIR / f"violations_{model_name}.csv", index=False)

    if N <= 0:
        C = 0.0
    else:
        C = 0.0 if (E / N) > CONFORMANCE_CAP else (1.0 - (E / N))

    # Baseline M0 on clean validation
    p0 = model.predict_proba(X_va)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_va)
    M0, mtype = metric_M(y_va.values, p0)

    # Step B: robustness sweep over ψ
    rng = np.random.RandomState(RANDOM_STATE)
    rows_psi = []
    deltas = []
    for psi in PSI_LEVELS:
        X_pert = inject_faults(X_va, psi, schema, rng)
        p = model.predict_proba(X_pert)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_pert)
        Mpsi, _ = metric_M(y_va.values, p)
        d = max(0.0, M0 - Mpsi)
        deltas.append(d)
        rows_psi.append({"model": model_name, "psi_percent": psi, "M0": M0, "M_psi": Mpsi, "delta": d})
    pd.DataFrame(rows_psi).to_csv(OUT_DIR / f"injection_sweep_{model_name}.csv", index=False)

    delta_max = max(deltas) if deltas else 0.0
    R = 1.0 - min(1.0, delta_max / 0.10)  # 10-point AUROC drop => R = 0 (Appendix)

    # Step C: adaptive weights
    dC = 1.0 - C
    dR = 1.0 - R
    wC = (dC + EPS) / (dC + dR + 2 * EPS)
    wR = 1.0 - wC

    # Step D: score
    S_raw = wC * C + wR * R
    Score = int(math.floor(10.0 * S_raw + 0.5))

    summary = {
        "model": model_name,
        "C_empirical": float(C),
        "E_nonconforming": int(E),
        "N_checked": int(N),
        "M0_baseline": float(M0),
        "delta_max": float(delta_max),
        "R_robustness": float(R),
        "wC": float(wC),
        "wR": float(wR),
        "S_raw": float(S_raw),
        "AI_Conformance_Score": int(Score),
        "metric_type": mtype,
        "psi_levels": PSI_LEVELS,
        "params": {"cap_E_over_N": CONFORMANCE_CAP, "test_size": TEST_SIZE, "random_state": RANDOM_STATE},
    }
    (OUT_DIR / f"ai_conformance_{model_name}.json").write_text(json.dumps(summary, indent=2))
    return summary

# ---------------- Orchestrator ----------------
def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing data at {DATA_PATH}. Run data/make_fake_mimic.py first.")
    df = pd.read_csv(DATA_PATH)

    # subject-level split if SUBJECT_COL is present; else simple stratified split
    if SUBJECT_COL and SUBJECT_COL in df.columns:
        subj = df[SUBJECT_COL].astype(str)
        y_subj = df.groupby(subj)[LABEL].max().astype(int)
        subj_ids = y_subj.index.to_numpy()
        y_s = y_subj.values
        tr_ids, va_ids = train_test_split(subj_ids, test_size=TEST_SIZE, stratify=y_s, random_state=RANDOM_STATE)
        tr_mask = subj.isin(tr_ids)
        va_mask = subj.isin(va_ids)
        train_df = df.loc[tr_mask].reset_index(drop=True)
        val_df = df.loc[va_mask].reset_index(drop=True)
    else:
        train_df, val_df = train_test_split(df, test_size=TEST_SIZE, stratify=df[LABEL].astype(int),
                                            random_state=RANDOM_STATE)
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)

    schema = load_schema()

    summaries = []
    for m in ["logreg_en", "random_forest", "gbm"]:
        summaries.append(run_one_model(m, train_df, val_df, schema))

    out = pd.DataFrame(summaries)
    csv_path = OUT_DIR / "ai_conformance_summary.csv"
    out.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)

    print(out[["model","AI_Conformance_Score","C_empirical","R_robustness","delta_max"]].to_string(index=False))
    print(f"\nSummary appended to {csv_path}")
    print(f"Violations and injection sweep files saved in {OUT_DIR}")

if __name__ == "__main__":
    main()
