# -*- coding: utf-8 -*-
"""
Appendix 1.10 — Informativeness

Low-info filter (entropy) -> redundancy filter (pairwise MI) -> task-relevance (MI with label)
-> performance elbow (fixed baseline GBT) -> QI, QP, adaptive weights -> AI-Informativeness (0–10).

Artifacts: results/informativeness/
- entropy_table.csv
- redundancy_kept.csv
- mi_with_label.csv
- elbow_curve.csv
- ai_informativeness_summary.csv
"""

import math
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif

# ----------------- Paths & config -----------------
FILE_DIR = Path(__file__).parent
PROJ_ROOT = (FILE_DIR / ".." / "..").resolve()
DATA_PATH = PROJ_ROOT / "data" / "mimic_lung_day1.csv"
OUT_DIR = PROJ_ROOT / "results" / "informativeness"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Feature/label config (same list as other dimensions)
FEATURES: List[str] = [
    "age","vent_flag","pressors_flag",
    "hr_min","hr_max","map_min","map_max",
    "resp_rate_max","temp_max","spo2_min",
    "albumin_min","lactate_max","bun_max","creatinine_max","bilirubin_max",
    "hemoglobin_min","platelets_min","sodium_min","potassium_max","aniongap_max",
]
LABEL = "hospital_expire_flag"

# Methodology params (Appendix 1.10)
RANDOM_STATE = 42
TEST_SIZE = 0.30
ALPHA_LOWINFO = 0.10        # drop bottom 10% entropy per type
TOP_FOR_REDUCING_COST = 200 # cap features before O(V^2) (no-op here; V ~ 20)
NBINS_DISC = 10             # discretization bins for entropy & feature-feature MI
MI_BETA = 0.01              # 1% of summed MI threshold (proxy for CMI)
ELBOW_EPS = 0.01            # 1 pp
EPS = 1e-6

# ----------------- Helpers -----------------
def is_categorical(s: pd.Series) -> bool:
    # treat low-cardinality as categorical
    nun = s.nunique(dropna=True)
    return (pd.api.types.is_integer_dtype(s) and nun <= 10) or nun <= 10

def discretize_series(s: pd.Series, nbins: int = NBINS_DISC) -> pd.Series:
    s_num = pd.to_numeric(s, errors="coerce")
    if s_num.notna().sum() == 0 or s_num.nunique(dropna=True) <= nbins:
        vals = s_num.dropna().unique()
        vals.sort()
        mapping = {v: i for i, v in enumerate(vals)}
        return s_num.map(mapping)
    try:
        return pd.qcut(s_num, q=nbins, duplicates="drop", labels=False)
    except Exception:
        return pd.cut(s_num, bins=nbins, labels=False, include_lowest=True)

def entropy_from_counts(counts: np.ndarray) -> float:
    p = counts.astype(float)
    p = p / (p.sum() + EPS)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))

def entropy_of_series(s: pd.Series, nbins: int = NBINS_DISC) -> float:
    if is_categorical(s):
        counts = s.value_counts(dropna=True).to_numpy()
        return entropy_from_counts(counts)
    d = discretize_series(s, nbins)
    counts = pd.Series(d).value_counts(dropna=True).to_numpy()
    return entropy_from_counts(counts)

def mi_between_features(a: pd.Series, b: pd.Series, nbins: int = NBINS_DISC) -> float:
    da = discretize_series(a, nbins)
    db = discretize_series(b, nbins)
    df = pd.DataFrame({"a": da, "b": db}).dropna()
    if df.empty:
        return 0.0
    ca = df["a"].astype(int)
    cb = df["b"].astype(int)
    n_a = int(ca.max()) + 1 if ca.notna().any() else 0
    n_b = int(cb.max()) + 1 if cb.notna().any() else 0
    if n_a <= 1 or n_b <= 1:
        return 0.0
    joint = np.zeros((n_a, n_b), dtype=float)
    for i, j in zip(ca.to_numpy(), cb.to_numpy()):
        if i >= 0 and j >= 0:
            joint[i, j] += 1.0
    total = joint.sum() + EPS
    pxy = joint / total
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = pxy / (px @ py)
        ratio[~np.isfinite(ratio)] = 1.0
        term = pxy * np.log2(ratio)
    return float(np.nansum(term))

def metric_M(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, str]:
    y = np.asarray(y_true).astype(int)
    if len(np.unique(y)) > 1:
        return float(roc_auc_score(y, y_score)), "auroc"
    yhat = (y_score >= 0.5).astype(int)
    return float(balanced_accuracy_score(y, yhat)), "bal_acc"

def build_baseline_model(feature_cols: List[str]) -> Pipeline:
    # Build the pipeline with EXACTLY the subset columns to avoid KeyError
    pre = ColumnTransformer(
        [("num", SimpleImputer(strategy="median"), feature_cols)],
        remainder="drop"
    )
    clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, random_state=RANDOM_STATE)
    return Pipeline([("prep", pre), ("clf", clf)])

def fit_predict_scores(X_train: pd.DataFrame, y_train: np.ndarray, X_val: pd.DataFrame, cols: List[str]) -> np.ndarray:
    model = build_baseline_model(cols)
    model.fit(X_train[cols], y_train)
    # HGB exposes decision_function; normalize to [0,1]
    s = model.decision_function(X_val[cols])
    s = np.asarray(s, dtype=float)
    return (s - s.min()) / (s.max() - s.min() + 1e-9)

# ----------------- Main -----------------
def main():
    # Load & basic checks
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing data at {DATA_PATH}. Run data/make_fake_mimic.py first.")
    df = pd.read_csv(DATA_PATH)

    for c in [LABEL] + FEATURES:
        if c not in df.columns:
            raise RuntimeError(f"Required column '{c}' not found in {DATA_PATH}")

    df = df.loc[df[LABEL].notna()].copy()
    df[LABEL] = df[LABEL].astype(int)

    # Split once
    train_df, val_df = train_test_split(df, test_size=TEST_SIZE, stratify=df[LABEL], random_state=RANDOM_STATE)
    X_train_full = train_df[FEATURES].copy()
    y_train = train_df[LABEL].to_numpy()
    X_val_full = val_df[FEATURES].copy()
    y_val = val_df[LABEL].to_numpy()

    # ----- 1) Low-information filter (entropy) on TRAIN -----
    ent_rows = []
    for f in FEATURES:
        ent_rows.append({"feature": f, "entropy": entropy_of_series(train_df[f])})
    ent_tbl = pd.DataFrame(ent_rows).sort_values("entropy", ascending=False).reset_index(drop=True)

    cats = [f for f in FEATURES if is_categorical(train_df[f])]
    cont = [f for f in FEATURES if f not in cats]

    def drop_bottom_alpha(tbl: pd.DataFrame, feats: List[str]) -> List[str]:
        if not feats:
            return []
        subt = tbl[tbl["feature"].isin(feats)].sort_values("entropy", ascending=False).reset_index(drop=True)
        k_drop = max(0, int(math.floor(ALPHA_LOWINFO * len(subt))))
        if k_drop == 0:
            return list(subt["feature"])
        return list(subt["feature"].iloc[:-k_drop])  # drop bottom k_drop

    keep_cats = drop_bottom_alpha(ent_tbl, cats)
    keep_cont = drop_bottom_alpha(ent_tbl, cont)
    keep_after_lowinfo = list(dict.fromkeys(keep_cats + keep_cont))
    ent_tbl["kept_after_lowinfo"] = ent_tbl["feature"].isin(keep_after_lowinfo)
    ent_tbl.to_csv(OUT_DIR / "entropy_table.csv", index=False)

    # Pre-trim before O(V^2) redundancy step
    prelist = ent_tbl[ent_tbl["kept_after_lowinfo"]].sort_values("entropy", ascending=False)["feature"].tolist()
    prelist = prelist[:min(len(prelist), TOP_FOR_REDUCING_COST)]

    # ----- 2) Redundancy filter (pairwise MI) -----
    ent_map = {r.feature: float(r.entropy) for _, r in ent_tbl.iterrows()}
    survivors: List[str] = []
    for f in prelist:  # greedy in descending entropy
        redundant = False
        for g in survivors:
            mi = mi_between_features(train_df[f], train_df[g])
            thr = 0.9 * min(ent_map[f], ent_map[g])
            if mi >= thr:
                redundant = True
                break
        if not redundant:
            survivors.append(f)
    if len(survivors) == 0:
        survivors = ent_tbl.sort_values("entropy", ascending=False)["feature"].head(min(5, len(ent_tbl))).tolist()
    pd.DataFrame({"feature": survivors}).to_csv(OUT_DIR / "redundancy_kept.csv", index=False)

    # ----- 3) Task-relevance filter (proxy CMI with MI(Y;Xj)) -----
    X_train_surv = train_df[survivors].copy()
    mi_vals = mutual_info_classif(
        X_train_surv.fillna(X_train_surv.median(numeric_only=True)),
        y_train,
        discrete_features=[is_categorical(train_df[c]) for c in survivors],
        random_state=RANDOM_STATE
    )
    mi_tbl = pd.DataFrame({"feature": survivors, "MI_y": mi_vals})
    sum_mi = float(np.sum(mi_vals)) + EPS
    thresh = MI_BETA * sum_mi
    keep_task = mi_tbl[mi_tbl["MI_y"] >= thresh]["feature"].tolist()
    if len(keep_task) == 0:
        keep_task = mi_tbl.sort_values("MI_y", ascending=False)["feature"].head(min(5, len(mi_tbl))).tolist()
    mi_tbl.to_csv(OUT_DIR / "mi_with_label.csv", index=False)

    # ----- 4) Performance elbow (fixed baseline model) -----
    rank = mi_tbl.sort_values("MI_y", ascending=False)["feature"].tolist()
    N = len(rank)
    # Reference full metric
    p_full = fit_predict_scores(X_train_full, y_train, X_val_full, rank)
    M_full, metric_type = metric_M(y_val, p_full)

    # grid of ks
    ks = sorted(set([5, 10, 20, N]))
    ks = [k for k in ks if 0 < k <= N]
    if N not in ks:
        ks.append(N)

    elbow_rows = []
    best_k = N
    prev_ok = False
    for k in ks:
        sub = rank[:k]
        p_k = fit_predict_scores(X_train_full, y_train, X_val_full, sub)
        M_k, _ = metric_M(y_val, p_k)
        delta = M_k - M_full
        ok = (abs(delta) <= ELBOW_EPS)
        if ok and prev_ok:
            best_k = k
        prev_ok = ok
        elbow_rows.append({"k": k, "metric": M_k, "delta_vs_full": delta})
    if best_k == N:
        ok_ks = [r["k"] for r in elbow_rows if abs(r["delta_vs_full"]) <= ELBOW_EPS]
        if ok_ks:
            best_k = min(ok_ks)

    elbow_df = pd.DataFrame(elbow_rows)
    elbow_df.to_csv(OUT_DIR / "elbow_curve.csv", index=False)

    S_elbow = rank[:best_k]
    S_final = S_elbow  # (Step 5 clinician/fairness reinstate would modify here)

    # ----- 6/7/8) QI, QP, weights, score -----
    mi_map = {r.feature: float(r.MI_y) for _, r in mi_tbl.iterrows()}
    info_final = float(sum(mi_map.get(f, 0.0) for f in S_final))
    QI = info_final / (sum_mi + EPS)

    p_final = fit_predict_scores(X_train_full, y_train, X_val_full, S_final)
    M_final, _ = metric_M(y_val, p_final)
    QP = (M_final / (M_full + EPS)) if M_full > 0 else 1.0

    dI = 1.0 - QI
    dP = 1.0 - QP
    if (dI < 0.02) and (dP < 0.02):
        wI, wP = 0.5, 0.5
    else:
        wI = (dI + EPS) / (dI + dP + 2*EPS)
        wP = 1.0 - wI

    S_raw = wI * QI + wP * QP
    Score = int(math.floor(10.0 * S_raw + 0.5))

    summary = pd.DataFrame([{
        "metric_type": metric_type,
        "M_full": M_full,
        "M_final": M_final,
        "QI": QI,
        "QP": QP,
        "wI": wI,
        "wP": wP,
        "S_raw": S_raw,
        "AI_Informativeness_Score": Score,
        "V_all_after_redundancy": len(survivors),
        "V_final": len(S_final),
        "k_elbow": best_k,
    }])
    sum_path = OUT_DIR / "ai_informativeness_summary.csv"
    summary.to_csv(sum_path, mode="a", header=not sum_path.exists(), index=False)

    print(summary.to_string(index=False))
    print(f"\nSaved entropy table         -> {OUT_DIR/'entropy_table.csv'}")
    print(f"Saved redundancy survivors  -> {OUT_DIR/'redundancy_kept.csv'}")
    print(f"Saved MI(Y;X) table         -> {OUT_DIR/'mi_with_label.csv'}")
    print(f"Saved elbow curve           -> {OUT_DIR/'elbow_curve.csv'}")
    print(f"Summary appended to         -> {sum_path}")

if __name__ == "__main__":
    main()
