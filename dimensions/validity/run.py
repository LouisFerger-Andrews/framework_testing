# -*- coding: utf-8 -*-
"""
Appendix 1.4 Validity — data-centric, MIMIC-only

Implements Steps 1–6 faithfully, with a principled fallback when NO paired references
are available:
- Per Appendix Step 3: if no defensible tolerance/paired reference exists, set Av=0.5 and flag.
- Degenerate-case policy (documented): if *no* variables have assessable accuracy (i.e., no paired
  references across the whole run), treat Accuracy as "not applicable" => wA=0, wP=1 for this run.
  This avoids penalizing datasets just because references aren’t available, while still reporting Av flags.

Outputs (results/validity/):
- ai_validity_variables.csv
- ai_validity_summary.csv          (appended)
- validity_config_snapshot.json
"""

import json, math
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

# ---------------- Paths ----------------
FILE_DIR = Path(__file__).parent
PROJ_ROOT = (FILE_DIR / ".." / "..").resolve()
DATA_PATH = PROJ_ROOT / "data" / "mimic_lung_day1.csv"
SCHEMA_PATH = PROJ_ROOT / "data" / "schema_validity.json"   # optional custom ranges/weights
PAIRS_PATH  = PROJ_ROOT / "data" / "validity_pairs.csv"     # optional: if you ever add real pairs
OUT_DIR = PROJ_ROOT / "results" / "validity"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- Config ----------------
FEATURES: List[str] = [
    "age","vent_flag","pressors_flag",
    "hr_min","hr_max","map_min","map_max",
    "resp_rate_max","temp_max","spo2_min",
    "albumin_min","lactate_max","bun_max","creatinine_max","bilirubin_max",
    "hemoglobin_min","platelets_min","sodium_min","potassium_max","aniongap_max",
]
EPS = 1e-6

# ---------------- Schema ----------------
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
    }
    feats = {v: {"min": rng.get(v, (-np.inf, np.inf))[0],
                 "max": rng.get(v, (-np.inf, np.inf))[1]} for v in FEATURES}
    return {"features": feats, "weights": {}}

def load_schema() -> Dict:
    if SCHEMA_PATH.exists():
        try:
            return json.loads(SCHEMA_PATH.read_text())
        except Exception:
            pass
    return _default_schema()

def normalize_weights(schema: Dict) -> Dict[str, float]:
    base = {v: 1.0 for v in FEATURES}
    user = (schema.get("weights") or {})
    for v, val in user.items():
        if v in base:
            try: base[v] = max(0.0, float(val))
            except Exception: pass
    tot = sum(base.values())
    if tot <= 0:
        n = len(FEATURES)
        return {v: 1.0 / n for v in FEATURES}
    return {v: base[v] / tot for v in FEATURES}

# ---------------- Appendix components ----------------
def plausibility_Pv(var: str, df: pd.DataFrame, schema: Dict) -> Tuple[float, int, int]:
    """Step 2: Pv = 1 - Ov/Mv using schema ranges."""
    if var not in df.columns:
        return 0.0, 0, 0
    s = pd.to_numeric(df[var], errors="coerce")
    Mv = int(s.notna().sum())
    meta = (schema.get("features") or {}).get(var, {})
    lo, hi = float(meta.get("min", -np.inf)), float(meta.get("max", np.inf))
    Ov = int(((s < lo) | (s > hi)).sum())
    Pv = 1.0 - (Ov / (Mv + EPS)) if Mv > 0 else 0.0
    return Pv, Ov, Mv

def accuracy_Av_no_pairs(var: str) -> Tuple[float, str, bool]:
    """
    Steps 1 & 3 without trusted references:
    - Av = 0.5, flag for follow-up (Appendix).
    - include_in_accuracy = False (we won't let unavailable accuracy dominate the score).
    """
    return 0.5, "no_paired_reference", False

def adaptive_weights(A: float, P: float, accuracy_applicable: bool) -> Tuple[float, float]:
    """
    Step 5 with degenerate-case handling:
    - If NO variables had assessable accuracy (accuracy_applicable == False), set wA=0, wP=1.
    - Else use Appendix adaptive weights:
        ΔA = 1 - A, ΔP = 1 - P
        if both < 0.02 => wA = wP = 0.5
        else wA = (ΔA + ε) / (ΔA + ΔP + 2ε), wP = 1 - wA
    """
    if not accuracy_applicable:
        return 0.0, 1.0
    dA, dP = (1.0 - A), (1.0 - P)
    if (dA < 0.02) and (dP < 0.02):
        return 0.5, 0.5
    wA = (dA + EPS) / (dA + dP + 2 * EPS)
    wP = 1.0 - wA
    return wA, wP

def score_from(A: float, P: float, wA: float, wP: float) -> Tuple[float, int]:
    """Step 6: S_raw = wA*A + wP*P; Score = floor(10*S_raw + 0.5)."""
    S_raw = wA * A + wP * P
    Score = int(math.floor(10.0 * S_raw + 0.5))
    return max(0.0, min(1.0, S_raw)), max(0, min(10, Score))

# ---------------- Orchestrator ----------------
def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing data at {DATA_PATH}. Run data/make_fake_mimic.py first.")
    df = pd.read_csv(DATA_PATH)
    schema = load_schema()
    weights = normalize_weights(schema)

    # Detect whether any real paired accuracy is available (optional future use)
    has_pairs = PAIRS_PATH.exists() and pd.read_csv(PAIRS_PATH).shape[0] >= 30

    rows = []
    A_terms, P_terms = [], []
    any_accuracy_used = False

    for v in FEATURES:
        # Accuracy (no pairs available => Appendix fallback)
        Av, a_flag, incA = accuracy_Av_no_pairs(v)

        # Plausibility
        Pv, Ov, Mv = plausibility_Pv(v, df, schema)

        wv = float(weights.get(v, 0.0))

        # Accumulate only if accuracy was actually assessable (incA True)
        if incA:
            any_accuracy_used = True
            A_terms.append(wv * Av)

        # Plausibility always contributes
        P_terms.append(wv * Pv)

        rows.append({
            "variable": v,
            "Av": float(Av), "flag_A": a_flag, "include_in_A": bool(incA),
            "Pv": float(Pv), "Mv_observed": int(Mv), "Ov_out_of_range": int(Ov),
            "weight_wv": wv
        })

    # Step 4: aggregate (if no accuracy included, A is undefined: set neutral 0.5 but mark not applicable)
    A = float(sum(A_terms)) if any_accuracy_used else 0.5
    P = float(sum(P_terms))
    wA, wP = adaptive_weights(A, P, any_accuracy_used)
    S_raw, Score = score_from(A, P, wA, wP)

    # Save per-variable
    var_df = pd.DataFrame(rows).sort_values("variable")
    var_df.to_csv(OUT_DIR / "ai_validity_variables.csv", index=False)

    # Save summary (append)
    summary = pd.DataFrame([{
        "A_accuracy": A, "P_plausibility": P, "wA": wA, "wP": wP,
        "S_raw": S_raw, "AI_Validity_Score": Score,
        "accuracy_applicable": any_accuracy_used
    }])
    sum_path = OUT_DIR / "ai_validity_summary.csv"
    summary.to_csv(sum_path, mode="a", header=not sum_path.exists(), index=False)

    # Snapshot config
    (OUT_DIR / "validity_config_snapshot.json").write_text(
        json.dumps({"features": schema.get("features", {}), "weights": weights}, indent=2)
    )

    # Console
    print(var_df[["variable","Av","Pv","Mv_observed","Ov_out_of_range","weight_wv","flag_A","include_in_A"]].to_string(index=False))
    print("\n" + summary.to_string(index=False))
    print(f"\nSummary appended to {sum_path}")
    print(f"Artifacts saved in {OUT_DIR}")

if __name__ == "__main__":
    main()
