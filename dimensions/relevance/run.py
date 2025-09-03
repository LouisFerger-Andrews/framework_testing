# -*- coding: utf-8 -*-
r"""
Appendix 1.9 Relevance — Concept coverage (A) + Model-importance sufficiency (B).

Inputs:
- data/mimic_lung_day1.csv
- optional: data/relevance_cci_map.json
  {
    "CCI": ["Age","Ventilation", ...],
    "H":   ["Ventilation","Vasopressors", ...],
    "mapping": {"Age":["age"], "Ventilation":["vent_flag"], ...},
    "beta": 0.01
  }

Outputs (results/relevance/):
- cci_mapping_used.json
- feature_importance_<model>.csv
- concept_importance_<model>.csv
- ai_relevance_summary.csv
"""

import json, math, sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

# ---------------- Paths & config ----------------
FILE_DIR  = Path(__file__).parent
PROJ_ROOT = (FILE_DIR / ".." / "..").resolve()
DATA_PATH = PROJ_ROOT / "data" / "mimic_lung_day1.csv"
MAP_PATH  = PROJ_ROOT / "data" / "relevance_cci_map.json"
OUT_DIR   = PROJ_ROOT / "results" / "relevance"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Feature set (same as other dimensions)
FEATURES: List[str] = [
    "age","vent_flag","pressors_flag",
    "hr_min","hr_max","map_min","map_max",
    "resp_rate_max","temp_max","spo2_min",
    "albumin_min","lactate_max","bun_max","creatinine_max","bilirubin_max",
    "hemoglobin_min","platelets_min","sodium_min","potassium_max","aniongap_max",
]
LABEL = "hospital_expire_flag"

TEST_SIZE = 0.30
RANDOM_STATE = 42
EPS = 1e-6
BETA_DEFAULT = 0.01  # Ih threshold

MODEL_NAMES = ["logreg_en", "random_forest", "gbm"]

# Use your models/ trainers (keyword-only label_col)
sys.path.insert(0, str(PROJ_ROOT))
from models import train_logreg_en, train_random_forest, train_gbm  # type: ignore
_TRAINERS = {
    "logreg_en": train_logreg_en,
    "random_forest": train_random_forest,
    "gbm": train_gbm,
}

# ---------------- Mapping (CCI/H) ----------------
def _default_mapping() -> Dict:
    mapping = {
        "Age": ["age"],
        "Ventilation": ["vent_flag"],
        "Vasopressors": ["pressors_flag"],
        "Heart rate": ["hr_min","hr_max"],
        "Mean arterial pressure": ["map_min","map_max"],
        "Respiratory rate": ["resp_rate_max"],
        "Temperature": ["temp_max"],
        "Oxygen saturation": ["spo2_min"],
        "Albumin": ["albumin_min"],
        "Lactate": ["lactate_max"],
        "BUN": ["bun_max"],
        "Creatinine": ["creatinine_max"],
        "Bilirubin": ["bilirubin_max"],
        "Hemoglobin": ["hemoglobin_min"],
        "Platelets": ["platelets_min"],
        "Sodium": ["sodium_min"],
        "Potassium": ["potassium_max"],
        "Anion gap": ["aniongap_max"],
    }
    H = [
        "Age","Ventilation","Vasopressors",
        "Mean arterial pressure","Oxygen saturation","Lactate",
        "Creatinine","Bilirubin","Platelets","Hemoglobin"
    ]
    CCI = list(mapping.keys())
    return {"CCI": CCI, "H": H, "mapping": mapping, "beta": BETA_DEFAULT}

def load_mapping(features_present: List[str]) -> Tuple[List[str], List[str], Dict[str,List[str]], float]:
    if MAP_PATH.exists():
        try:
            cfg = json.loads(MAP_PATH.read_text())
        except Exception:
            cfg = _default_mapping()
    else:
        cfg = _default_mapping()

    CCI: List[str] = list(dict.fromkeys(cfg.get("CCI", [])))
    mapping_in: Dict[str, List[str]] = cfg.get("mapping", {})
    H_in: List[str] = cfg.get("H", [])
    beta: float = float(cfg.get("beta", BETA_DEFAULT))

    # Keep only mapped features that exist
    mapping: Dict[str, List[str]] = {c: [f for f in feats if f in features_present] for c, feats in mapping_in.items()}
    mapping = {c: feats for c, feats in mapping.items() if len(feats) > 0}

    covered = set(mapping.keys())
    CCI2 = [c for c in CCI if (c in covered) or (c in H_in)]
    for c in covered:
        if c not in CCI2:
            CCI2.append(c)
    H2 = [h for h in H_in if h in CCI2]
    return CCI2, H2, mapping, beta

# ---------------- Trainers & preds ----------------
def _call_trainer(name: str, train_df: pd.DataFrame):
    """Call your trainer with keyword-only label_col. No positional, no (X,y)."""
    trainer = _TRAINERS[name]
    df = train_df.copy()
    # Ensure label is non-missing and int for your trainer
    df = df.loc[df[LABEL].notna()].copy()
    df[LABEL] = df[LABEL].astype(int)
    # Keyword-only call
    return trainer(df, label_col=LABEL)

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

# Manual permutation importance (model-agnostic)
def permutation_importance_manual(model, X: pd.DataFrame, y: np.ndarray, repeats: int = 5, random_state: int = RANDOM_STATE) -> pd.Series:
    rng = np.random.RandomState(random_state)
    base_p = get_proba(model, X)
    base_score, _ = metric_M(y, base_p)
    cols = list(X.columns)
    imps = np.zeros(len(cols), dtype=float)
    for j, col in enumerate(cols):
        drops = []
        for r in range(repeats):
            x_copy = X.copy()
            shuffled = x_copy[col].sample(frac=1.0, replace=False, random_state=rng).to_numpy()
            x_copy[col] = shuffled
            p = get_proba(model, x_copy)
            s, _ = metric_M(y, p)
            drops.append(max(0.0, base_score - s))
        imps[j] = float(np.mean(drops))
    total = float(imps.sum())
    if total <= 0:
        imps = np.ones_like(imps) / len(imps)
    else:
        imps = imps / total
    return pd.Series(imps, index=cols)

# ---------------- A & B ----------------
def compute_A(CCI: List[str], H: List[str], mapping: Dict[str, List[str]], features_present: List[str]) -> Tuple[float, float, float]:
    r"""
    Returns (A, AH, AL)
    AH = fraction of H concepts with at least one mapped feature present
    AL = fraction of (CCI \\ H) concepts with at least one mapped feature present
    A  = 0.67*AH + 0.33*AL
    """
    H_set = set(H)
    C_all = set(CCI)
    L_set = C_all - H_set

    def covered(concept: str) -> bool:
        feats = mapping.get(concept, [])
        return any(f in features_present for f in feats)

    AH = (sum(1 for h in H if covered(h)) / (len(H) + EPS)) if len(H) > 0 else 1.0
    AL = (sum(1 for l in L_set if covered(l)) / (len(L_set) + EPS)) if len(L_set) > 0 else 1.0
    A = 0.67 * AH + 0.33 * AL
    return A, AH, AL

def compute_B(H: List[str], mapping: Dict[str, List[str]], feat_importance: pd.Series, beta: float) -> Tuple[float, pd.DataFrame]:
    rows = []
    for h in H:
        feats = mapping.get(h, [])
        ih = float(feat_importance.reindex(feats).fillna(0.0).sum())
        rows.append({"concept": h, "Ih": ih, "n_features": len(feats)})
    df = pd.DataFrame(rows).sort_values("Ih", ascending=False)
    B = float((df["Ih"] >= beta).sum() / len(H)) if len(H) > 0 else 1.0
    return B, df

# ---------------- Orchestrator ----------------
def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing data at {DATA_PATH}. Run data/make_fake_mimic.py first.")
    df = pd.read_csv(DATA_PATH)

    for c in [LABEL] + FEATURES:
        if c not in df.columns:
            raise RuntimeError(f"Required column '{c}' not in {DATA_PATH}")

    # Drop rows with missing label to satisfy trainer's .astype(int)
    df = df.loc[df[LABEL].notna()].copy()
    df[LABEL] = df[LABEL].astype(int)

    # Split once
    train_df, val_df = train_test_split(df, test_size=TEST_SIZE, stratify=df[LABEL], random_state=RANDOM_STATE)
    train_df = train_df.reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)

    # Mapping
    CCI, H, mapping, beta = load_mapping(features_present=FEATURES)
    (OUT_DIR / "cci_mapping_used.json").write_text(json.dumps({"CCI": CCI, "H": H, "mapping": mapping, "beta": beta}, indent=2))

    # Step A
    A, AH, AL = compute_A(CCI, H, mapping, FEATURES)

    # Per model
    rows = []
    for m in MODEL_NAMES:
        model = _call_trainer(m, train_df)

        X_val = val_df[FEATURES]
        y_val = val_df[LABEL].to_numpy()
        p_val = get_proba(model, X_val)
        M0, metric_type = metric_M(y_val, p_val)

        # Feature importances on validation (normalized)
        fi = permutation_importance_manual(model, X_val, y_val, repeats=5, random_state=RANDOM_STATE)

        # Save per-feature importances (with concept tag)
        feat_rows = []
        for f in FEATURES:
            concept = None
            for cpt, feats in mapping.items():
                if f in feats:
                    concept = cpt
                    break
            feat_rows.append({"feature": f, "importance_norm": float(fi.get(f, 0.0)), "concept": concept})
        pd.DataFrame(feat_rows).sort_values("importance_norm", ascending=False).to_csv(OUT_DIR / f"feature_importance_{m}.csv", index=False)

        # Step B
        B, concept_df = compute_B(H, mapping, fi, beta=beta)
        concept_df.to_csv(OUT_DIR / f"concept_importance_{m}.csv", index=False)

        # Step C/D
        dA = 1.0 - A
        dB = 1.0 - B
        if (dA < 0.02) and (dB < 0.02):
            wA, wB = 0.5, 0.5
        else:
            wA = (dA + EPS) / (dA + dB + 2*EPS)
            wB = 1.0 - wA

        S_raw = wA * A + wB * B
        Score = int(math.floor(10.0 * S_raw + 0.5))

        rows.append({
            "model": m,
            "A": A, "AH": AH, "AL": AL,
            "B": B, "beta": beta,
            "wA": wA, "wB": wB,
            "S_raw": S_raw, "AI_Relevance_Score": Score,
            "val_metric": M0, "metric_type": metric_type,
            "n_CCI": len(CCI), "n_H": len(H)
        })

    out = pd.DataFrame(rows)
    csv_path = OUT_DIR / "ai_relevance_summary.csv"
    out.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)

    print(out[["model","AI_Relevance_Score","A","B","wA","wB","beta","val_metric","metric_type"]].to_string(index=False))
    print(f"\nSaved mapping to {OUT_DIR/'cci_mapping_used.json'}")
    print(f"Per-feature importances: {OUT_DIR}/feature_importance_<model>.csv")
    print(f"Per-concept Ih tables:   {OUT_DIR}/concept_importance_<model>.csv")
    print(f"Summary appended to      {csv_path}")

if __name__ == "__main__":
    main()
