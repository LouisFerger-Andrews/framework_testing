# -*- coding: utf-8 -*-
"""
Appendix 1.11 — Representativeness

Audits age (bins) and sex (if present). Uses your three trainers in models/.
Outputs (results/representativeness/):
- group_metrics_<model>.csv
- ai_representativeness_summary.csv
"""

import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

# ----------------- Paths & config -----------------
FILE_DIR  = Path(__file__).parent
PROJ_ROOT = (FILE_DIR / ".." / "..").resolve()
DATA_PATH = PROJ_ROOT / "data" / "mimic_lung_day1.csv"
OUT_DIR   = PROJ_ROOT / "results" / "representativeness"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Feature/label config (same list used elsewhere)
FEATURES: List[str] = [
    "age","vent_flag","pressors_flag",
    "hr_min","hr_max","map_min","map_max",
    "resp_rate_max","temp_max","spo2_min",
    "albumin_min","lactate_max","bun_max","creatinine_max","bilirubin_max",
    "hemoglobin_min","platelets_min","sodium_min","potassium_max","aniongap_max",
]
LABEL = "hospital_expire_flag"

# Method parameters
RANDOM_STATE = 42
TEST_SIZE = 0.30
EPS = 1e-6
NMIN = 100  # desired minimum per audited subgroup

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
def get_proba(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        s = np.asarray(s, dtype=float)
        return (s - s.min()) / (s.max() - s.min() + 1e-9)
    p = model.predict(X)
    return np.asarray(p, dtype=float)

def metric_M(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[Optional[float], str]:
    y = np.asarray(y_true).astype(int)
    if len(np.unique(y)) > 1:
        return float(roc_auc_score(y, y_score)), "auroc"
    return None, "auroc"  # undefined for single-class group (per spec)

def make_age_bins(age: pd.Series) -> pd.Series:
    a = pd.to_numeric(age, errors="coerce")
    bins = [-np.inf, 50, 70, 80, np.inf]
    labels = ["<50", "50-69", "70-79", "80+"]
    return pd.cut(a, bins=bins, labels=labels, right=False)

def get_sex_series(df: pd.DataFrame) -> Optional[pd.Series]:
    for col in ["sex", "gender", "Sex", "Gender"]:
        if col in df.columns:
            sc = df[col].astype(str).str.strip().str.lower()
            mapped = np.where(sc.isin(["m","male","1","true","t","man"]), "Male",
                      np.where(sc.isin(["f","female","0","false","f","woman"]), "Female", "Other"))
            return pd.Series(mapped, index=df.index, name="sex_group")
    return None

def build_groups(val_df: pd.DataFrame) -> List[Tuple[str, pd.Series]]:
    groups: List[Tuple[str, pd.Series]] = []
    if "age" in val_df.columns:
        age_g = make_age_bins(val_df["age"])
        groups += [
            ("age=<50",   age_g == "<50"),
            ("age=50-69", age_g == "50-69"),
            ("age=70-79", age_g == "70-79"),
            ("age=80+",   age_g == "80+"),
        ]
    sex_s = get_sex_series(val_df)
    if sex_s is not None:
        groups += [
            ("sex=Female", sex_s == "Female"),
            ("sex=Male",   sex_s == "Male"),
            ("sex=Other",  sex_s == "Other"),
        ]
        if "age" in val_df.columns:
            age_g = make_age_bins(val_df["age"])
            for a_lbl in ["<50","50-69","70-79","80+"]:
                for s_lbl in ["Female","Male"]:
                    groups.append((f"age={a_lbl}&sex={s_lbl}", (age_g == a_lbl) & (sex_s == s_lbl)))
    return groups

def summarize_groups(y: np.ndarray, p: np.ndarray, group_defs: List[Tuple[str, pd.Series]]) -> pd.DataFrame:
    rows = []
    for name, mask in group_defs:
        idx = mask.fillna(False).to_numpy()
        n = int(idx.sum())
        if n == 0:
            rows.append({"group": name, "N": 0, "metric": np.nan, "metric_defined": False})
            continue
        y_g = y[idx]; p_g = p[idx]
        m, _ = metric_M(y_g, p_g)
        rows.append({"group": name, "N": n, "metric": (np.nan if m is None else m), "metric_defined": (m is not None)})
    return pd.DataFrame(rows)

def call_trainer(trainer, df: pd.DataFrame, label_col: str, feature_cols: List[str]):
    """
    Minimal shim: try the common signatures your model files use.
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
                    f"Trainer signature not supported. Tried (df,label_col,feature_cols) "
                    f"and keyword variants. Original error: {e}"
                )

# ----------------- Main -----------------
def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing data at {DATA_PATH}. Run data/make_fake_mimic.py first.")
    df = pd.read_csv(DATA_PATH)

    for c in [LABEL] + FEATURES:
        if c not in df.columns:
            raise RuntimeError(f"Required column '{c}' not in {DATA_PATH}")

    df = df.loc[df[LABEL].notna()].copy()
    df[LABEL] = df[LABEL].astype(int)

    train_df, val_df = train_test_split(df, test_size=TEST_SIZE, stratify=df[LABEL], random_state=RANDOM_STATE)

    X_val = val_df[FEATURES].copy()
    y_val = val_df[LABEL].to_numpy()

    group_defs = build_groups(val_df)
    if not group_defs:
        raise RuntimeError("No audited groups could be formed (need at least 'age' or 'sex').")

    summaries = []
    for m in MODEL_NAMES:
        model = call_trainer(TRAINERS[m], train_df, LABEL, FEATURES)
        p_val = get_proba(model, X_val)

        gm = summarize_groups(y_val, p_val, group_defs)
        gm["model"] = m

        # Step A — Sample sufficiency S
        S = float(min(1.0, gm["N"].min() / float(NMIN + EPS))) if len(gm) else 0.0

        # Step B — Parity B (exclude undefined groups)
        gm_def = gm[gm["metric_defined"] & gm["metric"].notna()].copy()
        if len(gm_def) >= 1:
            Mmax = float(gm_def["metric"].max())
            B = 1.0 if Mmax <= 0 else float(gm_def["metric"].min() / (Mmax + EPS))
            B = max(0.0, min(1.0, B))
        else:
            B = 1.0  # neutral if everything undefined (unlikely)

        # Step C — Risk-proportional blend
        dS, dB = 1.0 - S, 1.0 - B
        if (dS < 0.02) and (dB < 0.02):
            wS, wB = 0.5, 0.5
        else:
            wS = (dS + EPS) / (dS + dB + 2*EPS)
            wB = 1.0 - wS

        # Step D — Score
        R_raw = wS * S + wB * B
        Score = int(math.floor(10.0 * R_raw + 0.5))

        # Save group table
        gm[["model","group","N","metric","metric_defined"]].to_csv(OUT_DIR / f"group_metrics_{m}.csv", index=False)

        summaries.append({
            "model": m,
            "S_sample_sufficiency": S,
            "B_parity": B,
            "wS": wS,
            "wB": wB,
            "R_raw": R_raw,
            "AI_Representativeness_Score": Score,
            "n_groups_total": int(len(gm)),
            "n_groups_defined": int(len(gm_def)),
            "Nmin": NMIN,
        })

    out = pd.DataFrame(summaries)
    csv_path = OUT_DIR / "ai_representativeness_summary.csv"
    out.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)

    print(out[["model","AI_Representativeness_Score","S_sample_sufficiency","B_parity","wS","wB","n_groups_defined"]].to_string(index=False))
    print(f"\nPer-model group tables saved as {OUT_DIR}/group_metrics_<model>.csv")
    print(f"Summary appended to {csv_path}")

if __name__ == "__main__":
    main()
