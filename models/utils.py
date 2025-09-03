import os, json, warnings
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, recall_score,
    precision_score, brier_score_loss
)
from sklearn.exceptions import ConvergenceWarning
from joblib import dump

def ensure_dirs(results_dir: str, models_dir: str) -> None:
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

def build_preprocessor(num_features: List[str], cat_features: Optional[List[str]], scale_numeric: bool) -> ColumnTransformer:
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scale", StandardScaler()))
    num_pipe = Pipeline(num_steps)
    if cat_features:
        cat_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        return ColumnTransformer(
            [("num", num_pipe, num_features), ("cat", cat_pipe, cat_features)],
            remainder="drop",
        )
    return ColumnTransformer([("num", num_pipe, num_features)], remainder="drop")

def crossval_oof_proba(model: Pipeline, X: pd.DataFrame, y: pd.Series, n_splits: int = 5, seed: int = 42) -> np.ndarray:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    proba = np.zeros(len(y), dtype=float)
    for tr, te in skf.split(X, y):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr = y.iloc[tr]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            model.fit(X_tr, y_tr)
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(X_te)[:, 1]
        elif hasattr(model, "decision_function"):
            s = model.decision_function(X_te)
            p = (s - s.min()) / (s.max() - s.min() + 1e-9)
        else:
            p = model.predict(X_te).astype(float)
        proba[te] = p
    return proba

def compute_metrics(y_true: np.ndarray, proba: np.ndarray, t: float) -> Dict[str, float]:
    yhat = (proba >= t).astype(int)
    tn = ((y_true == 0) & (yhat == 0)).sum()
    fp = ((y_true == 0) & (yhat == 1)).sum()
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    return {
        "auc": roc_auc_score(y_true, proba),
        "auprc": average_precision_score(y_true, proba),
        "f1": f1_score(y_true, yhat, zero_division=0),
        "recall": recall_score(y_true, yhat, zero_division=0),
        "precision": precision_score(y_true, yhat, zero_division=0),
        "specificity": spec,
        "brier": brier_score_loss(y_true, proba),
        "threshold": float(t),
    }

def best_threshold(y_true: np.ndarray, proba: np.ndarray, target: str = "f1") -> float:
    grid = np.linspace(0.05, 0.95, 19)
    best_t, best_val = 0.5, -1.0
    for t in grid:
        m = compute_metrics(y_true, proba, t)
        val = m.get(target, 0.0)
        if val > best_val:
            best_val, best_t = val, float(t)
    return best_t

def save_artifacts(name: str, model: Pipeline, features: List[str], cats: List[str],
                   label_col: str, threshold: float, class_ratio: Dict[str,int],
                   results_dir: str, models_dir: str, oof_proba: np.ndarray, metrics_row: dict) -> None:
    dump(model, os.path.join(models_dir, f"{name}.joblib"))
    with open(os.path.join(models_dir, f"{name}.json"), "w") as f:
        json.dump({
            "model_name": name,
            "features": features,
            "categoricals": cats,
            "label": label_col,
            "threshold": float(threshold),
            "class_ratio": class_ratio,
        }, f, indent=2)
    np.save(os.path.join(results_dir, f"oof_{name}_proba.npy"), oof_proba)
    with open(os.path.join(results_dir, f"{name}_metrics.json"), "w") as f:
        json.dump(metrics_row, f, indent=2)
