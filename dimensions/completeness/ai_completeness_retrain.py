import os, sys, json
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Optional boosters
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

# ----------------- Paths & config -----------------
FILE_DIR = Path(__file__).parent
PROJ_ROOT = (FILE_DIR / ".." / "..").resolve()
DATA_PATH = PROJ_ROOT / "data" / "mimic_lung_day1.csv"
OUT_DIR = PROJ_ROOT / "results" / "completeness"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Feature/label config
FEATURES: List[str] = [
    "age","vent_flag","pressors_flag",
    "hr_min","hr_max","map_min","map_max",
    "resp_rate_max","temp_max","spo2_min",
    "albumin_min","lactate_max","bun_max","creatinine_max","bilirubin_max",
    "hemoglobin_min","platelets_min","sodium_min","potassium_max","aniongap_max",
]
CATEGORICALS: List[str] = []   # put names here if you later include categoricals
LABEL = "hospital_expire_flag"
SUBJECT_COL: Optional[str] = None   # if you have subject_id; None => each row is its own subject

# Methodology parameters (Appendix 1.1)
D_DEFAULT = 5
N_MIN = 30
EPS = 1e-6
WEIGHTING = "subject"   # "subject" (default) or "cell"
AGG_FUNC = "mean"       # g(.) for subject aggregation of instance preds
TEST_SIZE = 0.3         # separate validation set
RANDOM_STATE = 42

# Models to train in this run (fresh)
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

def completeness_table(val_df: pd.DataFrame, features: List[str], subj: pd.Series) -> pd.DataFrame:
    """Step 1: compute c_k, c~_k, q_k, s_k per subject on the VALIDATION set."""
    P = len(features)
    grp = val_df.groupby(subj)
    n_k = grp.size().rename("n_k")
    M_k = (n_k * P).rename("M_k")
    miss_cells = grp[features].apply(lambda x: x.isna().sum().sum()).rename("miss_cells")
    c_k = (1.0 - miss_cells / M_k.replace(0, np.nan)).fillna(0.0).rename("c_k")
    M_max = float(M_k.max()) if len(M_k) else 1.0
    q_k = (M_k / (M_max if M_max > 0 else 1.0)).rename("q_k")
    c_tilde = ((M_k * c_k) / (M_max if M_max > 0 else 1.0)).rename("c_tilde")
    beta_raw = float(q_k.median()) if len(q_k) else 0.5
    beta = max(0.1, min(0.9, beta_raw))
    s_k = (((1.0 - beta) * c_k) + (beta * c_tilde)).rename("s_k")
    out = pd.concat([n_k, M_k, miss_cells, c_k, c_tilde, q_k, s_k], axis=1).reset_index(names="subject_id")
    out["beta"] = beta
    out["P"] = P
    out["M_max"] = M_max
    return out

def stratify_by_s(scores: pd.DataFrame, D: int, n_min: int) -> List[np.ndarray]:
    """Step 3: strata by s_k, equal counts, min n_min per stratum, merging tails if needed."""
    K = len(scores)
    if K == 0:
        return []
    while D > 1 and K // D < n_min:
        D -= 1
    order = scores.sort_values("s_k", kind="mergesort").reset_index(drop=True)
    cuts = np.linspace(0, K, D + 1, dtype=int)
    strata = [order.iloc[cuts[i]:cuts[i + 1]].index.to_numpy() for i in range(D)]
    i = len(strata) - 1
    while i > 0:
        if len(strata[i]) < n_min:
            strata[i - 1] = np.concatenate([strata[i - 1], strata[i]])
            strata.pop(i)
        i -= 1
    return strata

def aggregate_subject_preds(val_df: pd.DataFrame, subj: pd.Series, proba: np.ndarray, agg: str = "mean") -> pd.DataFrame:
    df2 = val_df.copy()
    df2["_p"] = proba
    df2["_y"] = val_df[LABEL].astype(int)
    if agg == "max":
        pred_s = df2.groupby(subj)["_p"].max()
    elif agg == "median":
        pred_s = df2.groupby(subj)["_p"].median()
    else:
        pred_s = df2.groupby(subj)["_p"].mean()
    true_s = df2.groupby(subj)["_y"].max()
    out = pd.concat([pred_s.rename("yhat_subj"), true_s.rename("y_subj")], axis=1).reset_index(names="subject_id")
    return out

def confusion(y_true: np.ndarray, y_score: np.ndarray, thr: float = 0.5) -> Tuple[int,int,int,int]:
    yhat = (y_score >= thr).astype(int)
    tp = int(((y_true == 1) & (yhat == 1)).sum())
    tn = int(((y_true == 0) & (yhat == 0)).sum())
    fp = int(((y_true == 0) & (yhat == 1)).sum())
    fn = int(((y_true == 1) & (yhat == 0)).sum())
    return tp, tn, fp, fn

# ----------------- Main procedure -----------------
def run_one_model(name: str, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Dict:
    subj_val = get_subject_series(val_df)

    # Step 1: completeness on validation subjects
    comp = completeness_table(val_df, FEATURES, subj_val)
    comp_path = OUT_DIR / f"subject_completeness_{name}.csv"
    comp.to_csv(comp_path, index=False)

    # Build & train fresh model (Step 2.1)
    num_cols = [c for c in FEATURES if c not in CATEGORICALS]
    model = make_model(name, train_df[LABEL], num_cols, CATEGORICALS)
    X_tr, y_tr = train_df[FEATURES], train_df[LABEL].astype(int)
    X_va, y_va = val_df[FEATURES], val_df[LABEL].astype(int)
    model.fit(X_tr, y_tr)

    # Step 2.2: instance predictions on validation
    proba_val = model.predict_proba(X_va)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_va)

    # Step 2.3: aggregate to subject level with g(.)
    subj_pred = aggregate_subject_preds(val_df, subj_val, proba_val, agg=AGG_FUNC)

    # Join completeness + preds
    tab = comp.merge(subj_pred, on="subject_id", how="inner")
    K = len(tab)
    if K == 0:
        raise RuntimeError("No validation subjects available after merge.")

    # Step 3: completeness strata
    strata_idx = stratify_by_s(tab[["s_k"]], D_DEFAULT, N_MIN)
    if not strata_idx:
        raise RuntimeError("Could not form strata; fewer subjects than n_min.")

    # Step 4: per-stratum performance, relative rd
    md_vals: List[float] = []
    strata_rows: List[Dict] = []
    for d, idx in enumerate(strata_idx, start=1):
        part = tab.iloc[idx]
        m_d, m_type = metric_M(part["y_subj"].values, part["yhat_subj"].values)
        md_vals.append(m_d)
    mmax = max(md_vals) if md_vals else 0.0
    # Step 5: weights
    if WEIGHTING == "cell":
        weights = []
        for idx in strata_idx:
            part = tab.iloc[idx]
            weights.append(float(part["M_k"].sum() / (tab["M_k"].sum() + EPS)))
    else:
        weights = [len(idx) / K for idx in strata_idx]

    # Collect detailed stratum info, Step 6 components
    R = 0.0
    best_idx = int(np.argmax(md_vals))
    for d, idx in enumerate(strata_idx, start=1):
        part = tab.iloc[idx]
        m_d = md_vals[d-1]
        r_d = (m_d / mmax) if mmax > 0 else 0.0
        w_d = weights[d-1]
        R += w_d * r_d

        tp, tn, fp, fn = confusion(part["y_subj"].values, part["yhat_subj"].values, thr=0.5)
        P_pred_d = tp + fp
        N_pred_d = tn + fn
        p_d = 1.0 - (min(P_pred_d, N_pred_d) / (P_pred_d + N_pred_d + EPS))

        strata_rows.append({
            "model": name, "stratum": d, "size": len(part), "metric_type": "auroc_or_balacc",
            "md": m_d, "rd": r_d, "weight": w_d,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "PPredd": P_pred_d, "NPredd": N_pred_d, "p_d": p_d,
            "avg_s_k": float(part["s_k"].mean()), "avg_c_k": float(part["c_k"].mean()),
            "avg_c_tilde": float(part["c_tilde"].mean())
        })

    # Step 6: choose d* and compute C*
    best_part = tab.iloc[strata_idx[best_idx]]
    C_star = float((best_part["M_k"] * best_part["c_k"]).sum() / (best_part["M_k"].sum() + EPS))

    # Step 6: graded penalty
    P_tot = float(sum(w * row["p_d"] for w, row in zip(weights, strata_rows)))

    # Step 7: final score
    dR = 1.0 - R
    dC = 1.0 - C_star
    wR = (dR + EPS) / (dR + dC + 2 * EPS)
    wC = 1.0 - wR
    C_raw = wR * R + wC * C_star
    Score_base = int(np.floor(10.0 * C_raw + 0.5))
    AI_Completeness = max(0, Score_base - int(np.ceil(P_tot)))

    # Save stratum table
    pd.DataFrame(strata_rows).to_csv(OUT_DIR / f"ai_completeness_strata_{name}.csv", index=False)

    # Summary JSON per model
    summary = {
        "model": name,
        "params": {"weighting": WEIGHTING, "agg": AGG_FUNC, "D_default": D_DEFAULT, "n_min": N_MIN, "threshold_for_penalty": 0.5},
        "K_subjects_val": int(K),
        "beta": float(tab["beta"].iloc[0]),
        "P_features": int(tab["P"].iloc[0]),
        "M_max": float(tab["M_max"].iloc[0]),
        "metric_mmax": float(mmax),
        "R": float(R),
        "C_star": float(C_star),
        "P_tot": float(P_tot),
        "wR": float(wR),
        "wC_star": float(wC),
        "C_raw": float(C_raw),
        "Score_base": int(Score_base),
        "AI_Completeness_Score": int(AI_Completeness),
    }
    (OUT_DIR / f"ai_completeness_{name}.json").write_text(json.dumps(summary, indent=2))
    return summary

def main():
    # Load data
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing data at {DATA_PATH}. Run data/make_fake_mimic.py first.")
    df = pd.read_csv(DATA_PATH)

    # Subject ids and subject-level stratified split for a separate validation set (Step 2 precondition)
    subj = get_subject_series(df)
    # subject-level label = max label across that subject's instances
    subj_lab = df.groupby(subj)[LABEL].max().astype(int)
    subj_ids = subj_lab.index.to_numpy()
    y_subj = subj_lab.values

    # train/val split **by subject**, then map back to rows
    subj_tr, subj_va = train_test_split(subj_ids, test_size=TEST_SIZE, stratify=y_subj, random_state=RANDOM_STATE)
    tr_mask = subj.isin(subj_tr)
    va_mask = subj.isin(subj_va)
    train_df = df.loc[tr_mask].reset_index(drop=True)
    val_df = df.loc[va_mask].reset_index(drop=True)

    # Train & evaluate each model fresh (no reuse of earlier fits)
    summaries = []
    for model_name in MODEL_SPECS.keys():
        summaries.append(run_one_model(model_name, train_df, val_df))

    # Append to cumulative CSV
    out = pd.DataFrame(summaries)
    csv_path = OUT_DIR / "ai_completeness_summary_retrain.csv"
    out.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)

    print(out[["model","AI_Completeness_Score","Score_base","R","C_star","P_tot"]].to_string(index=False))
    print(f"\nSummary appended to {csv_path}")
    print(f"Subject-level completeness CSVs and per-stratum CSVs saved in {OUT_DIR}")

if __name__ == "__main__":
    main()
