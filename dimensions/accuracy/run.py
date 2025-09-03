# -*- coding: utf-8 -*-
"""
Appendix 1.7 Accuracy — empirical error burden + robustness to injected inaccuracies.

Implements:
- Step A (FER): fraction of cells (FEATURES + LABEL) violating plausibility/consistency
  (range/domain) on the dataset. Missingness is NOT counted here (that's completeness).
- Step B (Robustness): inject inaccuracies at ψ∈{1,3,5,10}% of critical items in TRAIN,
  retrain FRESH for each ψ, evaluate on CLEAN validation, compute Δmax = max(M0 − Mψ).
- Step C/D: adaptive weights and 0–10 AI Accuracy score.
- Reporting: 95% bootstrap CIs for FER, Δmax, A, B.

Artifacts (results/accuracy/):
- fer_counts.csv
- psi_curve_<model>.csv
- ai_accuracy_summary.csv
- accuracy_config.json
"""

import json, math
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

# Optional boosters (used if available)
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

# ----------------- Paths & config -----------------
FILE_DIR = Path(__file__).parent
PROJ_ROOT = (FILE_DIR / ".." / "..").resolve()
DATA_PATH = PROJ_ROOT / "data" / "mimic_lung_day1.csv"
SCHEMA_PATH = PROJ_ROOT / "data" / "schema_validity.json"  # optional ranges/domains
OUT_DIR = PROJ_ROOT / "results" / "accuracy"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Feature/label config (same list as your other runners)
FEATURES: List[str] = [
    "age","vent_flag","pressors_flag",
    "hr_min","hr_max","map_min","map_max",
    "resp_rate_max","temp_max","spo2_min",
    "albumin_min","lactate_max","bun_max","creatinine_max","bilirubin_max",
    "hemoglobin_min","platelets_min","sodium_min","potassium_max","aniongap_max",
]
CATEGORICALS: List[str] = []   # add here if you later include categoricals
LABEL = "hospital_expire_flag"
SUBJECT_COL: Optional[str] = None   # if you later have subject_id; else each row is a "subject"

# Methodology parameters
TEST_SIZE = 0.3
RANDOM_STATE = 42
EPS = 1e-6
PSI_GRID = [1, 3, 5, 10]   # percent inaccuracies injected
BOOTSTRAP_N = 400          # for CIs (bump later if you want tighter CIs)

# Models to train fresh in this run
MODEL_SPECS = {
    "logreg_en": {"type": "logreg"},
    "random_forest": {"type": "rf"},
    "gbm": {"type": "gbm"},
}

# ----------------- Utilities -----------------
def get_subject_series(df: pd.DataFrame) -> pd.Series:
    if SUBJECT_COL and SUBJECT_COL in df.columns:
        return df[SUBJECT_COL].astype(str)
    return pd.Series(df.index.astype(str), index=df.index, name="subject_id")

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

def make_model(model_name: str, y_train: pd.Series, num_cols: List[str], cat_cols: List[str]) -> Pipeline:
    if model_name == "logreg_en":
        pre = build_preprocessor(num_cols, cat_cols, scale_numeric=True)
        clf = LogisticRegression(
            max_iter=4000, solver="saga", penalty="elasticnet", l1_ratio=0.2, class_weight="balanced", random_state=RANDOM_STATE
        )
        return Pipeline([("prep", pre), ("clf", clf)])
    if model_name == "random_forest":
        pre = build_preprocessor(num_cols, cat_cols, scale_numeric=False)
        clf = RandomForestClassifier(
            n_estimators=600, max_depth=None, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE
        )
        return Pipeline([("prep", pre), ("clf", clf)])
    if model_name == "gbm":
        pre = build_preprocessor(num_cols, cat_cols, scale_numeric=False)
        pos = int(y_train.sum()); neg = int(len(y_train) - pos)
        scale_pos = float(neg) / float(pos) if pos > 0 else 1.0
        if HAS_LGBM:
            clf = LGBMClassifier(
                n_estimators=800, learning_rate=0.03, num_leaves=64,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                objective="binary", random_state=RANDOM_STATE, n_jobs=-1,
                scale_pos_weight=scale_pos, verbosity=-1,
            )
        elif HAS_XGB:
            clf = XGBClassifier(
                n_estimators=900, learning_rate=0.03, max_depth=4,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                objective="binary:logistic", eval_metric="auc",
                random_state=RANDOM_STATE, n_jobs=-1, scale_pos_weight=scale_pos, tree_method="hist",
            )
        else:
            clf = HistGradientBoostingClassifier(max_depth=None, learning_rate=0.05, max_iter=400, random_state=RANDOM_STATE)
        return Pipeline([("prep", pre), ("clf", clf)])
    raise ValueError(f"Unknown model {model_name}")

def metric_M(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, str]:
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) > 1:
        return float(roc_auc_score(y_true, y_score)), "auroc"
    yhat = (y_score >= 0.5).astype(int)
    return float(balanced_accuracy_score(y_true, yhat)), "bal_acc"

# ----------------- Schema & FER -----------------
def _default_schema() -> Dict:
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
        LABEL: (0, 1),
    }
    feats = {v: {"min": rng.get(v, (-np.inf, np.inf))[0],
                 "max": rng.get(v, (-np.inf, np.inf))[1]} for v in FEATURES + [LABEL]}
    return {"features": feats}

def load_schema() -> Dict:
    if SCHEMA_PATH.exists():
        try:
            return json.loads(SCHEMA_PATH.read_text())
        except Exception:
            pass
    return _default_schema()

def check_cell_violation(value, meta) -> bool:
    """True if value violates plausibility/consistency domain/range. Missing is NOT a violation here."""
    if pd.isna(value):
        return False
    lo = float(meta.get("min", -np.inf))
    hi = float(meta.get("max",  np.inf))
    # binary domain enforcement for 0/1
    if lo == 0.0 and hi == 1.0:
        try:
            v = float(value)
            return not (abs(v - 0.0) < 1e-9 or abs(v - 1.0) < 1e-9)
        except Exception:
            return True
    # numeric range
    try:
        v = float(value)
        return (v < lo) or (v > hi)
    except Exception:
        return True  # non-numeric in numeric field

def compute_FER(df: pd.DataFrame, schema: Dict, cols: List[str]) -> Tuple[float, int, int, pd.DataFrame]:
    """Return FER, E, N, and per-variable violation counts."""
    E = 0
    N = 0
    rows = []
    for c in cols:
        if c not in df.columns:
            continue
        meta = (schema.get("features") or {}).get(c, {})
        s = df[c]
        n = int(s.notna().sum())
        v = int(sum(check_cell_violation(x, meta) for x in s))
        rows.append({"variable": c, "checked": n, "violations": v})
        E += v
        N += n
    FER = (E / (N + EPS)) if N > 0 else 0.0
    return FER, E, N, pd.DataFrame(rows)

def bootstrap_FER_ci(df: pd.DataFrame, schema: Dict, cols: List[str], B: int, rng: np.random.RandomState) -> Tuple[float,float]:
    n = len(df)
    stats = []
    for _ in range(B):
        idx = rng.randint(0, n, size=n)
        FER_b, _, _, _ = compute_FER(df.iloc[idx], schema, cols)
        stats.append(FER_b)
    a = np.array(stats, dtype=float)
    return float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))

# ----------------- Inaccuracy injection (Step B) -----------------
def corrupt_training_data(train_df: pd.DataFrame, schema: Dict, psi_percent: float, rng: np.random.RandomState) -> pd.DataFrame:
    """
    Corrupt ψ% of critical items (FEATURES + LABEL) in TRAIN.
    - Label or binary-like (0/1 domain): flip.
    - Numeric: 50% chance push beyond bounds; else add Gaussian noise scaled to column std.
    """
    df = train_df.copy()
    critical_cols = [LABEL] + FEATURES
    cell_positions = []
    for i in range(len(df)):
        for col in critical_cols:
            if pd.notna(df.at[i, col]):
                cell_positions.append((i, col))
    total_items = len(cell_positions)
    if total_items == 0:
        return df
    k = int(round((psi_percent / 100.0) * total_items))
    if k <= 0:
        return df

    rng.shuffle(cell_positions)
    chosen = cell_positions[:k]

    for (i, col) in chosen:
        meta = (schema.get("features") or {}).get(col, {})
        lo, hi = float(meta.get("min", -np.inf)), float(meta.get("max", np.inf))
        val = df.at[i, col]
        # binary/label flip
        if col == LABEL or (lo == 0.0 and hi == 1.0):
            try:
                df.at[i, col] = 1 - int(round(float(val)))
            except Exception:
                df.at[i, col] = 1  # force to 1 if unparsable
            continue
        # numeric corruption
        try:
            x = float(val)
        except Exception:
            # if not numeric, slam out-of-range
            df.at[i, col] = hi + abs(0.2 * (abs(hi) + 1.0))
            continue
        if rng.rand() < 0.5 and np.isfinite(lo) and np.isfinite(hi):
            # push out-of-range (20% beyond closer bound)
            toward_low = rng.rand() < 0.5
            if toward_low:
                df.at[i, col] = lo - abs(0.2 * (abs(lo) + 1.0))
            else:
                df.at[i, col] = hi + abs(0.2 * (abs(hi) + 1.0))
        else:
            # additive Gaussian noise scaled to column std
            col_vals = pd.to_numeric(df[col], errors="coerce")
            sigma = float(np.nanstd(col_vals))
            if not np.isfinite(sigma) or sigma == 0.0:
                sigma = max(1.0, abs(x) * 0.1)
            df.at[i, col] = x + rng.normal(0.0, 0.75 * sigma)
    return df

# ----------------- Train/predict helpers -----------------
def fit_and_predict_proba(model_name: str, train_df: pd.DataFrame, val_df: pd.DataFrame) -> np.ndarray:
    num_cols = [c for c in FEATURES if c not in CATEGORICALS]
    model = make_model(model_name, train_df[LABEL].astype(int), num_cols, CATEGORICALS)
    X_tr, y_tr = train_df[FEATURES], train_df[LABEL].astype(int)
    X_va = val_df[FEATURES]
    model.fit(X_tr, y_tr)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_va)[:, 1]
    if hasattr(model, "decision_function"):
        s = model.decision_function(X_va)
        return (s - s.min()) / (s.max() - s.min() + 1e-9)
    return model.predict(X_va).astype(float)

# ----------------- Main procedure -----------------
def main():
    # Load data
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing data at {DATA_PATH}. Run data/make_fake_mimic.py first.")
    df = pd.read_csv(DATA_PATH)
    schema = load_schema()

    # Ensure required columns
    for c in [LABEL] + FEATURES:
        if c not in df.columns:
            raise RuntimeError(f"Required column '{c}' not in {DATA_PATH}")

    # Split once into TRAIN / VAL
    # (Row-level split is fine here since your table is one row per stay; SUBJECT_COL hook kept for parity)
    train_df, val_df = train_test_split(df, test_size=TEST_SIZE, stratify=df[LABEL].astype(int), random_state=RANDOM_STATE)
    train_df = train_df.reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)
    y_val = val_df[LABEL].astype(int).values

    # ---------- Step A: Empirical error burden (FER) ----------
    cols_checked = FEATURES + [LABEL]
    FER, E, N, fer_table = compute_FER(df, schema, cols_checked)
    A_comp = 1.0 - min(1.0, FER / 0.05)

    # Save FER table
    fer_table.sort_values("variable").to_csv(OUT_DIR / "fer_counts.csv", index=False)

    # Bootstrap CI for FER, then map to A CI conservatively
    rng = np.random.RandomState(RANDOM_STATE)
    fer_ci_low, fer_ci_high = bootstrap_FER_ci(df, schema, cols_checked, BOOTSTRAP_N, rng)
    A_ci_low  = 1.0 - min(1.0, fer_ci_high / 0.05)
    A_ci_high = 1.0 - min(1.0, fer_ci_low  / 0.05)

    # ---------- Step B: Robustness sweep ----------
    summaries = []
    for model_name in MODEL_SPECS.keys():
        # Clean baseline
        p0 = fit_and_predict_proba(model_name, train_df, val_df)
        M0, mtype0 = metric_M(y_val, p0)

        # Inject inaccuracies at ψ%, retrain fresh each time
        psi_rows = []
        worst_drop = 0.0
        worst_p = None  # predictions for the worst ψ model (for CI)
        for psi in PSI_GRID:
            tr_bad = corrupt_training_data(train_df, schema, psi, rng)
            p_bad = fit_and_predict_proba(model_name, tr_bad, val_df)
            Mpsi, _ = metric_M(y_val, p_bad)
            drop = max(0.0, M0 - Mpsi)
            psi_rows.append({"psi_percent": psi, "metric": Mpsi, "drop": drop})
            if drop > worst_drop:
                worst_drop = drop
                worst_p = p_bad

        pd.DataFrame(psi_rows).to_csv(OUT_DIR / f"psi_curve_{model_name}.csv", index=False)

        delta_max = float(worst_drop)
        # B component (cap at 10 percentage points drop)
        B_comp = 1.0 - min(1.0, delta_max / 0.10)

        # Bootstrap CI for Δmax using fixed predictions (resample validation rows)
        def bootstrap_delta_ci(y: np.ndarray, p_clean: np.ndarray, p_worst: Optional[np.ndarray], B: int, rng: np.random.RandomState) -> Tuple[float,float]:
            if p_worst is None:
                return (0.0, 0.0)
            n = len(y)
            deltas = []
            for _ in range(B):
                idx = rng.randint(0, n, size=n)
                a0, _ = metric_M(y[idx], p_clean[idx])
                a1, _ = metric_M(y[idx], p_worst[idx])
                deltas.append(max(0.0, a0 - a1))
            a = np.array(deltas, dtype=float)
            return float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))
        d_ci_low, d_ci_high = bootstrap_delta_ci(y_val, p0, worst_p, BOOTSTRAP_N, rng)
        B_ci_low  = 1.0 - min(1.0, d_ci_high / 0.10)
        B_ci_high = 1.0 - min(1.0, d_ci_low  / 0.10)

        # ---------- Step C: adaptive weights ----------
        dA = 1.0 - A_comp
        dB = 1.0 - B_comp
        if (dA < 0.02) and (dB < 0.02):
            wA, wB = 0.5, 0.5
        else:
            wA = (dA + EPS) / (dA + dB + 2 * EPS)
            wB = 1.0 - wA

        # ---------- Step D: score ----------
        S_raw = wA * A_comp + wB * B_comp
        Score = int(np.floor(10.0 * S_raw + 0.5))

        summaries.append({
            "model": model_name,
            "FER": FER, "FER_ci_low": fer_ci_low, "FER_ci_high": fer_ci_high,
            "A_component": A_comp, "A_ci_low": A_ci_low, "A_ci_high": A_ci_high,
            "delta_max": delta_max, "delta_max_ci_low": d_ci_low, "delta_max_ci_high": d_ci_high,
            "B_component": B_comp, "B_ci_low": B_ci_low, "B_ci_high": B_ci_high,
            "wA": wA, "wB": wB, "S_raw": S_raw, "AI_Accuracy_Score": int(Score),
            "M0_metric": M0, "metric_type": mtype0
        })

    # Append to cumulative CSV
    out = pd.DataFrame(summaries)
    csv_path = OUT_DIR / "ai_accuracy_summary.csv"
    out.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)

    # Config snapshot
    (OUT_DIR / "accuracy_config.json").write_text(json.dumps({
        "psi_grid": PSI_GRID,
        "features": FEATURES,
        "label": LABEL,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE
    }, indent=2))

    # Console
    print(out[["model","AI_Accuracy_Score","A_component","B_component","wA","wB","FER","delta_max","M0_metric","metric_type"]].to_string(index=False))
    print(f"\nSaved FER counts to {OUT_DIR/'fer_counts.csv'}")
    print(f"Saved ψ-sweep curves to {OUT_DIR}/psi_curve_<model>.csv")
    print(f"Summary appended to {csv_path}")

if __name__ == "__main__":
    main()
