# -*- coding: utf-8 -*-
"""
Appendix 1.5 Signal Quality — MIMIC (chartevents) proxy implementation.

What this does:
1) Load first-6h minute-level signals per stay (HR, MAP invasive/NIBP, SpO2, Temp).
2) Compute per-stay proxy SNR (signal vs residual) and artefact rate (spike/step/flatline windows).
3) Quality sweep:
   - Noise-only: add noise to time-series, re-aggregate features, retrain models, AUROC vs SNR.
   - Artefact-only: inject artefacts (spikes/flatlines), re-aggregate, retrain, AUROC vs AR.
4) Threshold inference with ε=0.03: SNR_min and AR_max per model.
5) Score: QS = min(1, SNR/SNR_min), QA = min(1, AR_max/max(AR,1e-6)), adaptive weights, 0–10.

Outputs (results/signal_quality/):
- snr_ar_per_stay.csv                       (baseline SNR/AR per stay)
- noise_sweep_<model>.csv                   (cohort SNR & AUROC per noise variant)
- artifact_sweep_<model>.csv                (cohort AR  & AUROC per artifact variant)
- ai_signal_quality_summary.csv             (appended; score & CIs per model)

Requires:
- data/mimic_lung_day1.csv                  (your tabular day-1 dataset; one row per stay)
- data/timeseries_first6h.csv               (preferred real export; if missing -> smoke fallback)

Notes:
- This uses only tabular + chartevents proxies (no waveforms).
- Welch bands are replaced by a smoothing/residual SNR proxy suitable for ~1-min cadence.

"""

import os, sys, json, math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

# ---------------- Paths & constants ----------------
FILE_DIR = Path(__file__).parent
PROJ_ROOT = (FILE_DIR / ".." / "..").resolve()
DATA_DAY1 = PROJ_ROOT / "data" / "mimic_lung_day1.csv"
DATA_TS   = PROJ_ROOT / "data" / "timeseries_first6h.csv"     # preferred
OUT_DIR   = PROJ_ROOT / "results" / "signal_quality"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES: List[str] = [
    "age","vent_flag","pressors_flag",
    "hr_min","hr_max","map_min","map_max",
    "resp_rate_max","temp_max","spo2_min",
    "albumin_min","lactate_max","bun_max","creatinine_max","bilirubin_max",
    "hemoglobin_min","platelets_min","sodium_min","potassium_max","aniongap_max",
]
LABEL = "hospital_expire_flag"

# First-6h time-series columns expected
SIG_COLS = ["hr", "map_invasive", "map_nibp", "spo2", "temp"]

# quality windows & detectors
ROLL_WIN = 15      # minutes for smoothing (signal estimate)
WIN_AR   = 5       # minutes per artefact window
SPIKE_K  = 6.0     # MAD z threshold for spikes
STEP_K   = 6.0     # MAD z threshold for steps/derivative
FLAT_EPS = 1e-3    # near-flat std threshold (relative)
EPS      = 1e-6
RANDOM_STATE = 42
TEST_SIZE = 0.30
BOOTSTRAP_N = 500
EPSILON_THRESH = 0.03  # ε for SNR_min/AR_max

# Noise & artefact sweep levels
NOISE_LEVELS = [0.0, 0.5, 1.0, 1.5, 2.0]  # scales residual std
ARTIF_RATES  = [0.0, 0.01, 0.03, 0.05, 0.10]  # fraction of windows to corrupt

# Try to use your models module; fall back to local definitions if not found
try:
    sys.path.insert(0, str(PROJ_ROOT))
    from models import train_logreg_en, train_random_forest, train_gbm  # type: ignore
    HAVE_MODELS_MODULE = True
except Exception:
    HAVE_MODELS_MODULE = False

# ---------------- Utilities ----------------
def ensure_stay_id(df: pd.DataFrame) -> pd.DataFrame:
    if "stay_id" not in df.columns:
        df = df.copy()
        df["stay_id"] = np.arange(1, len(df) + 1)
    return df

def split_train_val(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tr, va = train_test_split(df, test_size=TEST_SIZE, stratify=df[LABEL].astype(int), random_state=RANDOM_STATE)
    return tr.reset_index(drop=True), va.reset_index(drop=True)

def metric_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y = np.asarray(y_true).astype(int)
    if len(np.unique(y)) < 2:
        # fallback per the appendix guidance when a slice has single class
        yhat = (y_score >= 0.5).astype(int)
        return balanced_accuracy_score(y, yhat)
    return roc_auc_score(y, y_score)

def build_local_model(name: str) -> Pipeline:
    """
    Local model fallback matching your earlier baselines:
    - logreg_en: impute+scale+elasticnet LR
    - random_forest: impute+RF
    - gbm: impute+HGB (as a GBM substitute)
    """
    num = ColumnTransformer([("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), FEATURES)], remainder="drop")
    if name == "logreg_en":
        clf = LogisticRegression(max_iter=4000, solver="saga", penalty="elasticnet",
                                 l1_ratio=0.2, class_weight="balanced", random_state=RANDOM_STATE)
        return Pipeline([("prep", ColumnTransformer([("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), FEATURES)], remainder="drop")),
                         ("clf", clf)])
    if name == "random_forest":
        clf = RandomForestClassifier(n_estimators=600, max_depth=None, min_samples_leaf=2,
                                     class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=1)
        return Pipeline([("prep", num), ("clf", clf)])
    if name == "gbm":
        clf = HistGradientBoostingClassifier(max_depth=None, learning_rate=0.05, max_iter=400, random_state=RANDOM_STATE)
        return Pipeline([("prep", num), ("clf", clf)])
    raise ValueError(f"Unknown model {name}")

def fit_and_auc(model_name: str, train_df: pd.DataFrame, val_df: pd.DataFrame) -> float:
    X_tr, y_tr = train_df[FEATURES], train_df[LABEL].astype(int)
    X_va, y_va = val_df[FEATURES],   val_df[LABEL].astype(int)
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
        p = model.predict_proba(X_va)[:, 1]
    elif hasattr(model, "decision_function"):
        s = model.decision_function(X_va)
        # rank-scale to 0..1 for AUC; not perfect but fine for fallback
        s = (s - s.min()) / (s.max() - s.min() + 1e-9)
        p = s
    else:
        p = model.predict(X_va).astype(float)
    return metric_auc(y_va, p)

# ---------------- Time-series handling ----------------
def load_or_make_timeseries(day1: pd.DataFrame) -> pd.DataFrame:
    """
    Preferred: read data/timeseries_first6h.csv.
    Fallback: synthesize 6h@1min per stay with plausible noise around day-1 aggregates.
    """
    if DATA_TS.exists():
        ts = pd.read_csv(DATA_TS)
        # normalize columns
        need = ["stay_id","timestamp"]
        for c in need:
            if c not in ts.columns:
                raise FileNotFoundError(f"{DATA_TS} missing required column '{c}'")
        # ensure expected signal columns exist (absent ones will be created as NaN)
        for c in SIG_COLS:
            if c not in ts.columns:
                ts[c] = np.nan
        ts["timestamp"] = pd.to_datetime(ts["timestamp"], errors="coerce")
        ts = ts.sort_values(["stay_id","timestamp"]).reset_index(drop=True)
        return ts

    # --- Smoke fallback ---
    rng = np.random.RandomState(7)
    day1 = ensure_stay_id(day1)
    rows = []
    for _, r in day1.iterrows():
        sid = int(r["stay_id"])
        # 6h @ 1/min
        t0 = pd.Timestamp("2020-01-01 00:00:00") + pd.Timedelta(minutes=rng.randint(0, 60))
        for m in range(6*60):
            ts = t0 + pd.Timedelta(minutes=m)
            # build around aggregates if present, else defaults
            hr_lo = float(r.get("hr_min", 70)); hr_hi = float(r.get("hr_max", 110))
            map_lo = float(r.get("map_min", 60)); map_hi = float(r.get("map_max", 95))
            spo2_lo = float(r.get("spo2_min", 93)); spo2_hi = 99.0
            temp_lo = float(r.get("temp_max", 37.5)) - 0.5; temp_hi = float(r.get("temp_max", 37.5)) + 0.3

            hr  = np.clip(rng.normal((hr_lo+hr_hi)/2, 0.10*(hr_hi-hr_lo)+3.0), 30, 220)
            map_i = np.clip(rng.normal((map_lo+map_hi)/2, 0.10*(map_hi-map_lo)+2.5), 30, 160)
            map_n = np.clip(map_i + rng.normal(0, 3.0), 30, 160)
            spo2 = np.clip(rng.normal(max(92.0, spo2_lo+1), 1.2), 50, 100)
            temp = np.clip(rng.normal((temp_lo+temp_hi)/2, 0.1), 34.0, 42.0)

            rows.append({"stay_id": sid, "timestamp": ts, "hr": hr, "map_invasive": map_i, "map_nibp": map_n, "spo2": spo2, "temp": temp})
    ts = pd.DataFrame(rows)
    return ts

def rolling_mean(series: pd.Series, win: int) -> pd.Series:
    return series.rolling(win, min_periods=max(3, win//3), center=True).mean()

def series_snr_db(x: pd.Series, win: int = ROLL_WIN) -> float:
    x = pd.to_numeric(x, errors="coerce")
    if x.dropna().shape[0] < max(10, win):
        return np.nan
    s = rolling_mean(x, win)
    r = x - s
    psig = float(np.nanvar(s))
    pnoi = float(np.nanvar(r))
    return 10.0 * np.log10((psig + EPS) / (pnoi + EPS))

def artefact_windows(x: pd.Series, win: int = WIN_AR) -> Tuple[int, int]:
    """
    Count artefact windows via 3 detectors: spikes, steps, flatlines.
    """
    x = pd.to_numeric(x, errors="coerce").copy()
    n = len(x)
    if n < win + 2:
        return 0, 0
    # global MAD
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) + EPS
    # steps (first differences)
    dx = x.diff()
    mad_dx = np.nanmedian(np.abs(dx - np.nanmedian(dx))) + EPS

    flags = []
    for start in range(0, n, win):
        seg = x.iloc[start:start+win]
        if seg.isna().all():
            flags.append(False)
            continue
        seg_med = np.nanmedian(seg)
        seg_mad = np.nanmedian(np.abs(seg - seg_med)) + EPS
        # spike: any point far from local median
        spike = bool(np.nanmax(np.abs(seg - seg_med)) > SPIKE_K * seg_mad)
        # step: large change in this window
        dx_seg = dx.iloc[max(start,1):start+win]
        step = bool(np.nanmax(np.abs(dx_seg - np.nanmedian(dx_seg))) > STEP_K * mad_dx)
        # flatline: very low variation
        flat = bool(np.nanstd(seg) < FLAT_EPS * (np.nanstd(x) + EPS))
        flags.append(spike or step or flat)
    k = len(flags)
    a = int(np.sum(flags))
    return a, k

def combine_snr_ar_per_stay(ts: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-stay median SNR across available channels and union artefact rate across channels.
    """
    out = []
    for sid, g in ts.groupby("stay_id"):
        snrs = []
        total_w = 0
        art_w = 0
        for c in SIG_COLS:
            if c not in g.columns:
                continue
            s = g[c]
            if s.notna().sum() < 10:
                continue
            sdb = series_snr_db(s, ROLL_WIN)
            if not np.isnan(sdb):
                snrs.append(sdb)
            a, k = artefact_windows(s, WIN_AR)
            art_w += a
            total_w += k
        if total_w == 0:
            ar = np.nan
        else:
            ar = art_w / total_w
        out.append({"stay_id": sid, "snr_db": (np.median(snrs) if len(snrs) else np.nan), "artefact_rate": ar})
    return pd.DataFrame(out)

def aggregate_timeseries(ts: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate time-series into day-1 feature slices aligning with your FEATURES.
    """
    rows = []
    for sid, g in ts.groupby("stay_id"):
        hr_min = np.nanmin(g["hr"]) if "hr" in g else np.nan
        hr_max = np.nanmax(g["hr"]) if "hr" in g else np.nan
        maps = []
        if "map_invasive" in g: maps.append(g["map_invasive"].values)
        if "map_nibp" in g:     maps.append(g["map_nibp"].values)
        maps = np.concatenate([m for m in maps if m is not None], axis=0) if maps else np.array([np.nan])
        map_min = np.nanmin(maps); map_max = np.nanmax(maps)
        spo2_min = np.nanmin(g["spo2"]) if "spo2" in g else np.nan
        temp_max = np.nanmax(g["temp"]) if "temp" in g else np.nan
        rows.append({"stay_id": sid, "hr_min": hr_min, "hr_max": hr_max, "map_min": map_min, "map_max": map_max, "spo2_min": spo2_min, "temp_max": temp_max})
    return pd.DataFrame(rows)

# ---------------- Quality sweep transforms ----------------
def add_noise(ts: pd.DataFrame, level: float, rng: np.random.RandomState) -> pd.DataFrame:
    """
    Add noise proportional to residual std per channel: x' = x + level * std(residual)
    """
    out = ts.copy()
    for c in SIG_COLS:
        if c not in out.columns:
            continue
        x = pd.to_numeric(out[c], errors="coerce")
        s = rolling_mean(x, ROLL_WIN)
        resid = x - s
        sigma = float(np.nanstd(resid))
        if sigma <= 0 or np.isnan(sigma):
            continue
        noise = rng.normal(0.0, sigma * level, size=len(x))
        out[c] = (x + noise)
    return out

def inject_artefacts(ts: pd.DataFrame, target_rate: float, rng: np.random.RandomState) -> pd.DataFrame:
    """
    Corrupt a fraction of 5-min windows with spikes or flatlines (chosen randomly).
    """
    out = ts.copy()
    # gather windows indexes per stay
    for sid, g in out.groupby("stay_id"):
        idx = g.index
        n = len(idx)
        if n < 2*WIN_AR:
            continue
        num_windows = n // WIN_AR
        k_corrupt = int(max(0, round(num_windows * target_rate)))
        if k_corrupt == 0:
            continue
        win_starts = list(range(0, num_windows * WIN_AR, WIN_AR))
        rng.shuffle(win_starts)
        corr_starts = win_starts[:k_corrupt]
        for w0 in corr_starts:
            w_idx = idx[w0:w0+WIN_AR]
            mode = rng.choice(["spike","flatline"], p=[0.6, 0.4])
            for c in SIG_COLS:
                if c not in out.columns:
                    continue
                seg = pd.to_numeric(out.loc[w_idx, c], errors="coerce")
                if seg.isna().all():
                    continue
                if mode == "spike":
                    med = float(np.nanmedian(seg))
                    amp = float(np.nanstd(seg) + 1.0) * rng.uniform(4.0, 8.0)
                    spike_pos = rng.choice(w_idx)
                    out.at[spike_pos, c] = med + amp * rng.choice([-1, 1])
                else:
                    # flatline to local median
                    med = float(np.nanmedian(seg))
                    out.loc[w_idx, c] = med
    return out

# ---------------- Main orchestration ----------------
def main():
    # Load day-1 base table
    if not DATA_DAY1.exists():
        raise FileNotFoundError(f"Missing {DATA_DAY1}. Run your data/make_fake_mimic.py first.")
    base = pd.read_csv(DATA_DAY1)
    base = ensure_stay_id(base)

    # Load or synthesize first-6h time-series
    ts0 = load_or_make_timeseries(base)

    # Compute baseline SNR/AR per stay
    snr_ar = combine_snr_ar_per_stay(ts0)
    snr_ar.to_csv(OUT_DIR / "snr_ar_per_stay.csv", index=False)

    # Cohort baseline (medians)
    SNR_med = float(np.nanmedian(snr_ar["snr_db"]))
    AR_med  = float(np.nanmedian(snr_ar["artefact_rate"]))

    # Prepare a baseline aggregate from ts0 to override the time-series-derived features
    agg0 = aggregate_timeseries(ts0)
    # Merge with base (override hr/map/spo2/temp)
    merged0 = base.drop(columns=["hr_min","hr_max","map_min","map_max","spo2_min","temp_max"], errors="ignore").merge(
        agg0, on="stay_id", how="inner"
    )

    # Split once for all variants
    train_df, val_df = split_train_val(merged0)

    models = ["logreg_en", "random_forest", "gbm"]
    noise_curves = {m: [] for m in models}
    art_curves   = {m: [] for m in models}

    rng = np.random.RandomState(RANDOM_STATE)

    # --- Noise-only sweep ---
    for level in NOISE_LEVELS:
        ts_n = add_noise(ts0, level, rng)
        snr_ar_n = combine_snr_ar_per_stay(ts_n)
        SNR_med_n = float(np.nanmedian(snr_ar_n["snr_db"]))
        # aggregate and merge
        agg_n = aggregate_timeseries(ts_n)
        merged_n = base.drop(columns=["hr_min","hr_max","map_min","map_max","spo2_min","temp_max"], errors="ignore").merge(
            agg_n, on="stay_id", how="inner"
        )
        # keep same rows as split
        tr_n = merged_n.loc[merged_n["stay_id"].isin(train_df["stay_id"])].reset_index(drop=True)
        va_n = merged_n.loc[merged_n["stay_id"].isin(val_df["stay_id"])].reset_index(drop=True)

        for m in models:
            auc = fit_and_auc(m, tr_n, va_n)
            noise_curves[m].append({"noise_level": level, "cohort_snr_db": SNR_med_n, "auc": auc})

    # --- Artefact-only sweep ---
    for rate in ARTIF_RATES:
        ts_a = inject_artefacts(ts0, rate, rng)
        snr_ar_a = combine_snr_ar_per_stay(ts_a)
        AR_med_a = float(np.nanmedian(snr_ar_a["artefact_rate"]))
        agg_a = aggregate_timeseries(ts_a)
        merged_a = base.drop(columns=["hr_min","hr_max","map_min","map_max","spo2_min","temp_max"], errors="ignore").merge(
            agg_a, on="stay_id", how="inner"
        )
        tr_a = merged_a.loc[merged_a["stay_id"].isin(train_df["stay_id"])].reset_index(drop=True)
        va_a = merged_a.loc[merged_a["stay_id"].isin(val_df["stay_id"])].reset_index(drop=True)

        for m in models:
            auc = fit_and_auc(m, tr_a, va_a)
            art_curves[m].append({"artifact_target": rate, "cohort_ar": AR_med_a, "auc": auc})

    # --- Threshold inference & scoring per model ---
    rows_summary = []
    for m in models:
        # Save curves
        pd.DataFrame(noise_curves[m]).sort_values("cohort_snr_db", ascending=True).to_csv(OUT_DIR / f"noise_sweep_{m}.csv", index=False)
        pd.DataFrame(art_curves[m]).sort_values("cohort_ar", ascending=True).to_csv(OUT_DIR / f"artifact_sweep_{m}.csv", index=False)

        # Best AUC on each curve
        best_auc_noise = max(x["auc"] for x in noise_curves[m]) if noise_curves[m] else float("nan")
        best_auc_art   = max(x["auc"] for x in art_curves[m])   if art_curves[m]   else float("nan")

        # SNR_min: lowest SNR whose AUC within ε of best
        SNR_min = np.nan
        if noise_curves[m]:
            nc = sorted(noise_curves[m], key=lambda x: x["cohort_snr_db"])
            eligible = [x["cohort_snr_db"] for x in nc if x["auc"] >= (best_auc_noise - EPSILON_THRESH)]
            if len(eligible):
                SNR_min = float(min(eligible))

        # AR_max: highest AR whose AUC within ε of best
        AR_max = np.nan
        if art_curves[m]:
            ac = sorted(art_curves[m], key=lambda x: x["cohort_ar"])
            eligible = [x["cohort_ar"] for x in ac if x["auc"] >= (best_auc_art - EPSILON_THRESH)]
            if len(eligible):
                AR_max = float(max(eligible))

        # Component scores
        QS = float(min(1.0, (SNR_med / (SNR_min + EPS)))) if np.isfinite(SNR_min) else 0.0
        QA = float(min(1.0, (AR_max / max(AR_med, EPS)))) if np.isfinite(AR_max) else 0.0

        # Adaptive weights
        dS, dA = (1.0 - QS), (1.0 - QA)
        if (dS < 0.02) and (dA < 0.02):
            wS, wA = 0.5, 0.5
        else:
            wS = (dS + EPS) / (dS + dA + 2*EPS)
            wA = 1.0 - wS

        S_raw = wS * QS + wA * QA
        Score = int(math.floor(10.0 * S_raw + 0.5))

        # --- Bootstrap CIs for SNR, AR, QS, QA and S_raw ---
        # Resample stays to compute cohort medians, then apply fixed thresholds
        stays = snr_ar["stay_id"].values
        snrs  = snr_ar["snr_db"].values
        ars   = snr_ar["artefact_rate"].values
        bs_SNR, bs_AR, bs_QS, bs_QA, bs_Sraw = [], [], [], [], []
        if len(stays) >= 10:
            rng = np.random.RandomState(RANDOM_STATE)
            n = len(stays)
            for _ in range(BOOTSTRAP_N):
                idx = rng.randint(0, n, size=n)
                SNR_b = float(np.nanmedian(snrs[idx]))
                AR_b  = float(np.nanmedian(ars[idx]))
                QS_b  = float(min(1.0, (SNR_b / (SNR_min + EPS)))) if np.isfinite(SNR_min) else 0.0
                QA_b  = float(min(1.0, (AR_max / max(AR_b, EPS)))) if np.isfinite(AR_max) else 0.0
                dSb, dAb = (1.0 - QS_b), (1.0 - QA_b)
                if (dSb < 0.02) and (dAb < 0.02):
                    wSb, wAb = 0.5, 0.5
                else:
                    wSb = (dSb + EPS) / (dSb + dAb + 2*EPS)
                    wAb = 1.0 - wSb
                Sraw_b = wSb * QS_b + wAb * QA_b
                bs_SNR.append(SNR_b); bs_AR.append(AR_b); bs_QS.append(QS_b); bs_QA.append(QA_b); bs_Sraw.append(Sraw_b)

            def ci(a):
                a = np.array(a)
                return float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))
            SNR_ci = ci(bs_SNR); AR_ci = ci(bs_AR); QS_ci = ci(bs_QS); QA_ci = ci(bs_QA); Sraw_ci = ci(bs_Sraw)
        else:
            SNR_ci = (np.nan, np.nan); AR_ci = (np.nan, np.nan); QS_ci = (np.nan, np.nan); QA_ci = (np.nan, np.nan); Sraw_ci = (np.nan, np.nan)

        rows_summary.append({
            "model": m,
            "SNR_median_db": SNR_med, "AR_median": AR_med,
            "SNR_min_db": SNR_min, "AR_max": AR_max,
            "QS": QS, "QA": QA, "wS": wS, "wA": wA,
            "S_raw": S_raw, "AI_SignalQuality_Score": Score,
            "best_auc_noise": best_auc_noise, "best_auc_artifact": best_auc_art,
            "SNR_ci_low": SNR_ci[0], "SNR_ci_high": SNR_ci[1],
            "AR_ci_low": AR_ci[0],   "AR_ci_high": AR_ci[1],
            "QS_ci_low": QS_ci[0],   "QS_ci_high": QS_ci[1],
            "QA_ci_low": QA_ci[0],   "QA_ci_high": QA_ci[1],
            "Sraw_ci_low": Sraw_ci[0], "Sraw_ci_high": Sraw_ci[1]
        })

    # Append summary
    sum_path = OUT_DIR / "ai_signal_quality_summary.csv"
    pd.DataFrame(rows_summary).to_csv(sum_path, mode="a", header=not sum_path.exists(), index=False)

    # Console
    pretty = pd.DataFrame(rows_summary)[["model","AI_SignalQuality_Score","QS","QA","wS","wA","SNR_median_db","AR_median","SNR_min_db","AR_max"]]
    print(pretty.to_string(index=False))
    print(f"\nSummary appended to {sum_path}")
    print(f"Curves and per-stay metrics saved in {OUT_DIR}")

if __name__ == "__main__":
    main()
