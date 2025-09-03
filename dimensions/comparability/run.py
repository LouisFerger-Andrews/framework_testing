# -*- coding: utf-8 -*-
"""
Appendix 1.6 Comparability — MIMIC-only, model-centric score per your framework.

Implements:
1) Grouping factor (default calendar year if available; else 3 pseudo-years for smoke).
2) Variable-level screens across groups:
   - Continuous -> Kruskal–Wallis (robust to variance inequality).
   - Categorical -> Chi-square; Fisher's exact when 2x2 and any expected < 5.
   - Benjamini–Hochberg FDR (q=0.05).
3) Harmonisation on flagged variables: per-group z-score using TRAIN stats; re-train and evaluate.
4) Residual worst-group performance gap Δmax = max_g |AUROCharm,g − AUROCharm|.
5) QV, QP, adaptive weights, final 0–10 AI-Comparability score, with 95% bootstrap CIs.

Artifacts:
- results/comparability/flagged_variables.csv
- results/comparability/group_auc_<model>.csv
- results/comparability/ai_comparability_summary.csv
- results/comparability/comparability_config.json
"""

import json, math, sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

# ---------- Paths ----------
FILE_DIR = Path(__file__).parent
PROJ_ROOT = (FILE_DIR / ".." / "..").resolve()
DATA_PATH = PROJ_ROOT / "data" / "mimic_lung_day1.csv"
OUT_DIR = PROJ_ROOT / "results" / "comparability"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Config ----------
FEATURES: List[str] = [
    "age","vent_flag","pressors_flag",
    "hr_min","hr_max","map_min","map_max",
    "resp_rate_max","temp_max","spo2_min",
    "albumin_min","lactate_max","bun_max","creatinine_max","bilirubin_max",
    "hemoglobin_min","platelets_min","sodium_min","potassium_max","aniongap_max",
]
LABEL = "hospital_expire_flag"
GROUP_COL_PREFERRED = None  # set to e.g. "admission_year", "first_icu_year", "careunit", etc. If None -> auto

RANDOM_STATE = 42
TEST_SIZE = 0.30
FDR_Q = 0.05
BOOTSTRAP_N = 400  # reasonable runtime; increase later
EPS = 1e-6
# Step-4 constants:
QV_DENOM_FRACTION = 0.20  # 20% flagged -> QV=0
QP_DENOM = 0.10           # 10 AUC points drop -> QP=0
EPSILON_THRESH = 0.03     # ε for "within best" not used here; only QP uses Δmax

# Try project models if available
try:
    sys.path.insert(0, str(PROJ_ROOT))
    from models import train_logreg_en, train_random_forest, train_gbm  # type: ignore
    HAVE_MODELS_MODULE = True
except Exception:
    HAVE_MODELS_MODULE = False

# ---------- Small helpers ----------
def metric_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y = np.asarray(y_true).astype(int)
    if len(np.unique(y)) < 2:
        yhat = (y_score >= 0.5).astype(int)
        return balanced_accuracy_score(y, yhat)
    return roc_auc_score(y, y_score)

def build_local_model(name: str) -> Pipeline:
    num_prep = ColumnTransformer([("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), FEATURES)], remainder="drop")
    if name == "logreg_en":
        clf = LogisticRegression(max_iter=4000, solver="saga", penalty="elasticnet",
                                 l1_ratio=0.2, class_weight="balanced", random_state=RANDOM_STATE)
        return Pipeline([("prep", ColumnTransformer([("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), FEATURES)], remainder="drop")),
                         ("clf", clf)])
    if name == "random_forest":
        clf = RandomForestClassifier(n_estimators=600, max_depth=None, min_samples_leaf=2,
                                     class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=1)
        return Pipeline([("prep", num_prep), ("clf", clf)])
    if name == "gbm":
        clf = HistGradientBoostingClassifier(max_depth=None, learning_rate=0.05, max_iter=400, random_state=RANDOM_STATE)
        return Pipeline([("prep", num_prep), ("clf", clf)])
    raise ValueError(name)

def fit_and_predict_proba(model_name: str, train_df: pd.DataFrame, val_df: pd.DataFrame) -> np.ndarray:
    X_tr, y_tr = train_df[FEATURES], train_df[LABEL].astype(int)
    X_va = val_df[FEATURES]
    if HAVE_MODELS_MODULE:
        if model_name == "logreg_en":
            model = train_logreg_en(X_tr, y_tr)
        elif model_name == "random_forest":
            model = train_random_forest(X_tr, y_tr)
        elif model_name == "gbm":
            model = train_gbm(X_tr, y_tr)
        else:
            raise ValueError(model_name)
    else:
        model = build_local_model(model_name)
        model.fit(X_tr, y_tr)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_va)[:, 1]
    elif hasattr(model, "decision_function"):
        s = model.decision_function(X_va)
        s = (s - s.min()) / (s.max() - s.min() + 1e-9)
        return s
    else:
        return model.predict(X_va).astype(float)

def infer_group_column(df: pd.DataFrame) -> Tuple[str, pd.Series]:
    """
    Prefer an explicit year column; else derive from datetime; else 3 pseudo-years by row order.
    """
    # explicit preference
    candidates = []
    if GROUP_COL_PREFERRED and GROUP_COL_PREFERRED in df.columns:
        candidates.append(GROUP_COL_PREFERRED)
    # common time columns
    for c in ["admission_year", "first_icu_year", "chart_year"]:
        if c in df.columns:
            return c, df[c].astype(str)
    for c in ["admittime", "intime", "charttime", "admit_time"]:
        if c in df.columns:
            try:
                dt = pd.to_datetime(df[c], errors="coerce")
                years = dt.dt.year.fillna(-1).astype(int).astype(str)
                return f"{c}_year", years
            except Exception:
                pass
    # fallback: 3 bins by row position (smoke)
    n = len(df)
    k = 3
    bins = pd.qcut(np.arange(n), q=k, labels=[f"Y{i+1}" for i in range(k)])
    return "year_group", bins.astype(str)

def is_categorical_like(s: pd.Series, name: str) -> bool:
    if "flag" in name.lower():
        return True
    vals = s.dropna().unique()
    if s.dtype.kind in "biu" and len(vals) <= 8:
        return True
    if len(vals) <= 5:
        return True
    return False

def fisher_or_chi2(table: pd.DataFrame) -> float:
    # table: rows = variable levels, cols = groups
    if table.shape == (2, 2):
        # expected counts
        exp = stats.contingency.expected_freq(table.to_numpy())
        if (exp < 5).any():
            _, p = stats.fisher_exact(table.to_numpy())
            return float(p)
    chi2, p, _, _ = stats.chi2_contingency(table, correction=False)
    return float(p)

def kruskal_groups(groups: List[np.ndarray]) -> float:
    # drop groups with <2 non-nans to avoid errors
    clean = [g[~np.isnan(g)] for g in groups if np.isfinite(g).sum() >= 2]
    if len(clean) < 2:
        return 1.0
    try:
        stat, p = stats.kruskal(*clean)
        return float(p)
    except Exception:
        return 1.0

def benjamini_hochberg(pvals: List[float], q: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (p_adj, rejected_bool) arrays using BH step-up control of FDR at level q.
    """
    p = np.array(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranks = np.arange(1, m+1)
    p_sorted = p[order]
    # adjusted p-values
    p_adj = np.minimum.accumulate((p_sorted * m / ranks)[::-1])[::-1]
    p_adj = np.clip(p_adj, 0, 1)
    # rejection
    thresh = (ranks / m) * q
    reject = p_sorted <= thresh
    # ensure monotonicity (standard BH step-up)
    max_k = np.where(reject)[0].max() if reject.any() else -1
    reject_final = np.zeros(m, dtype=bool)
    if max_k >= 0:
        reject_final[:max_k+1] = True
    # unsort
    p_adj_unsorted = np.empty_like(p_adj)
    reject_unsorted = np.empty_like(reject_final)
    p_adj_unsorted[order] = p_adj
    reject_unsorted[order] = reject_final
    return p_adj_unsorted, reject_unsorted

def groupwise_zscore(train: pd.DataFrame, val: pd.DataFrame, group_col: str, vars_to_scale: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Per-group z-score on TRAIN stats only; apply to TRAIN & VAL.
    """
    train2, val2 = train.copy(), val.copy()
    if not vars_to_scale:
        return train2, val2
    gstats = train2.groupby(group_col)[vars_to_scale].agg(["mean","std"])
    # flatten columns
    gstats.columns = [f"{v}__{stat}" for v, stat in gstats.columns]
    for v in vars_to_scale:
        mu = gstats[f"{v}__mean"]; sd = gstats[f"{v}__std"].replace(0, np.nan)
        # map means/stds by group
        train2[v] = (train2[v] - train2[group_col].map(mu)) / (train2[group_col].map(sd) + 1e-9)
        val2[v]   = (val2[v]   - val2[group_col].map(mu)) / (val2[group_col].map(sd) + 1e-9)
    return train2, val2

# ---------- Orchestrator ----------
def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing data at {DATA_PATH}. Run data/make_fake_mimic.py first.")

    df = pd.read_csv(DATA_PATH)
    # ensure a grouping column
    group_name, group_series = infer_group_column(df)
    df = df.copy()
    df[group_name] = group_series.astype(str)
    # ensure stay_id for merges if present
    if "stay_id" not in df.columns:
        df["stay_id"] = np.arange(1, len(df)+1)

    # --- Step 2: variable-level screens ---
    pvals, var_types, tested_vars = [], [], []
    for v in FEATURES:
        if v not in df.columns:
            continue
        x = pd.to_numeric(df[v], errors="coerce")
        # decide type
        is_cat = is_categorical_like(x, v)
        # split by group
        groups = []
        for g, sub in df.groupby(group_name):
            xv = pd.to_numeric(sub[v], errors="coerce").values
            if np.isfinite(xv).sum() >= 2:
                groups.append(xv)
        if len(groups) < 2:
            continue  # not testable

        if is_cat:
            # build contingency: var (0/1/levels) x group
            # binarize if many levels
            s = pd.Series(x).round() if x.dtype.kind in "fc" else x
            # limit levels to <=8; else treat numeric continuous
            levs = pd.unique(s.dropna())
            if len(levs) > 8:
                # fallback to continuous test
                p = kruskal_groups(groups)
                var_types.append("continuous")
            else:
                table = pd.crosstab(df[v], df[group_name], dropna=True)
                if table.shape[0] >= 2 and table.shape[1] >= 2:
                    p = fisher_or_chi2(table)
                    var_types.append("categorical")
                else:
                    p = 1.0
                    var_types.append("categorical")
        else:
            p = kruskal_groups(groups)  # robust, Welch-friendly alternative
            var_types.append("continuous")

        tested_vars.append(v)
        pvals.append(float(p))

    if not tested_vars:
        raise RuntimeError("No variables were testable for comparability; check your grouping and features.")

    # BH FDR
    p_adj, reject = benjamini_hochberg(pvals, q=FDR_Q)
    flagged = [bool(r) for r in reject]
    V = len(tested_vars)
    F = int(sum(flagged))

    flagged_df = pd.DataFrame({
        "variable": tested_vars,
        "type": var_types,
        "p_raw": pvals,
        "p_adj_bh": p_adj,
        "flagged": flagged
    }).sort_values(["flagged","p_adj_bh","variable"], ascending=[False, True, True])
    flagged_df.to_csv(OUT_DIR / "flagged_variables.csv", index=False)

    # --- Step 3: model impact (harmonise flagged vars, retrain, compute AUROCs) ---
    # split once
    train_df, val_df = train_test_split(df, test_size=TEST_SIZE, stratify=df[LABEL].astype(int), random_state=RANDOM_STATE)
    train_df = train_df.reset_index(drop=True); val_df = val_df.reset_index(drop=True)

    # vars to harmonise (use TRAIN-only stats)
    vars_flagged = [v for v, f in zip(tested_vars, flagged) if f and v in FEATURES]
    # Apply per-group z-score on BOTH continuous and categorical flagged variables (binary as numeric 0/1)
    train_h, val_h = groupwise_zscore(train_df, val_df, group_name, vars_flagged)

    models = ["logreg_en", "random_forest", "gbm"]
    rows_summary = []

    for m in models:
        # raw model (trained/evaluated on original split)
        p_raw = fit_and_predict_proba(m, train_df, val_df)
        auc_raw = metric_auc(val_df[LABEL].values, p_raw)

        # harmonised model (train/eval on harmonised copies)
        p_h = fit_and_predict_proba(m, train_h, val_h)
        auc_h = metric_auc(val_h[LABEL].values, p_h)

        # per-group AUROCs on harmonised val
        grp_rows = []
        diffs = []
        for g, sub in val_h.groupby(group_name):
            idx = sub.index
            if len(idx) < 10:
                continue
            auc_g = metric_auc(sub[LABEL].values, p_h[sub.index - val_h.index[0]])  # align
            grp_rows.append({"model": m, "group": str(g), "n_val": len(idx), "auc_harmonised_group": auc_g})
            diffs.append(abs(auc_g - auc_h))
        if not grp_rows:
            # fallback: no groups big enough; treat gap as 0
            delta_max = 0.0
        else:
            delta_max = float(np.nanmax(diffs))

        # save per-group
        pd.DataFrame(grp_rows).to_csv(OUT_DIR / f"group_auc_{m}.csv", index=False)

        # --- Step 4: component scores ---
        QV = 1.0 - min(1.0, F / (QV_DENOM_FRACTION * V + EPS))
        QP = 1.0 - min(1.0, delta_max / QP_DENOM)

        # --- Step 5: adaptive weights ---
        dV, dP = (1.0 - QV), (1.0 - QP)
        if (dV < 0.02) and (dP < 0.02):
            wV, wP = 0.5, 0.5
        else:
            wV = (dV + EPS) / (dV + dP + 2*EPS)
            wP = 1.0 - wV

        S_raw = wV * QV + wP * QP
        Score = int(math.floor(10.0 * S_raw + 0.5))

        # --- Bootstrap CIs for group AUROCs and Δmax (fixed p_h, resample rows) ---
        bs_deltas = []
        bs_groups = {str(g): [] for g in val_h[group_name].unique()}
        rng = np.random.RandomState(RANDOM_STATE)
        val_idx = np.arange(len(val_h))
        for _ in range(BOOTSTRAP_N):
            idx = rng.randint(0, len(val_h), size=len(val_h))
            y_b = val_h[LABEL].values[idx]
            p_b = p_h[idx]
            # overall
            auc_b = metric_auc(y_b, p_b)
            # per group
            deltas = []
            for g, sub in val_h.groupby(group_name):
                gi = sub.index.values
                # indices of bootstrap sample that fall in this group
                mask = np.isin(idx, gi)
                if mask.sum() < 5:
                    continue
                auc_gb = metric_auc(y_b[mask], p_b[mask])
                bs_groups[str(g)].append(auc_gb)
                deltas.append(abs(auc_gb - auc_b))
            if deltas:
                bs_deltas.append(np.nanmax(deltas))
        def ci(a):
            a = np.array(a, dtype=float)
            if a.size == 0:
                return (np.nan, np.nan)
            return (float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5)))
        delta_ci = ci(bs_deltas)

        rows_summary.append({
            "model": m,
            "group_col": group_name,
            "V_tested": V, "F_flagged": F, "QV": QV,
            "delta_max": delta_max, "QP": QP,
            "wV": wV, "wP": wP,
            "S_raw": S_raw, "AI_Comparability_Score": Score,
            "auc_raw": auc_raw, "auc_harmonised": auc_h,
            "delta_max_ci_low": delta_ci[0], "delta_max_ci_high": delta_ci[1]
        })

    # append summary
    sum_path = OUT_DIR / "ai_comparability_summary.csv"
    pd.DataFrame(rows_summary).to_csv(sum_path, mode="a", header=not sum_path.exists(), index=False)

    # config snapshot
    (OUT_DIR / "comparability_config.json").write_text(json.dumps({
        "features_tested": tested_vars,
        "fdr_q": FDR_Q,
        "group_col_used": group_name
    }, indent=2))

    # console
    print(f"Grouping column used: {group_name}")
    print(pd.DataFrame(rows_summary)[["model","AI_Comparability_Score","QV","QP","wV","wP","delta_max"]].to_string(index=False))
    print(f"\nFlagged variables (BH q={FDR_Q}): {F} / {V} (see {OUT_DIR/'flagged_variables.csv'})")
    print(f"Summary appended to {sum_path}")

if __name__ == "__main__":
    main()
