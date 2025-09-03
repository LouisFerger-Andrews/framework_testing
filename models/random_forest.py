from typing import List, Optional, Dict
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from .utils import (
    ensure_dirs, build_preprocessor, crossval_oof_proba,
    best_threshold, compute_metrics, save_artifacts
)

MODEL_NAME = "random_forest"

def train(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
    cat_features: Optional[List[str]] = None,
    results_dir: str = "framework_testing/results",
    models_dir: str = "framework_testing/models",
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict:
    ensure_dirs(results_dir, models_dir)
    X = df[feature_cols].copy()
    y = df[label_col].astype(int).copy()
    num = [c for c in feature_cols if c not in (cat_features or [])]

    pre = build_preprocessor(num, cat_features, scale_numeric=False)
    clf = RandomForestClassifier(
        n_estimators=600,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=random_state,
    )
    pipe = Pipeline([("prep", pre), ("clf", clf)])

    oof = crossval_oof_proba(pipe, X, y, n_splits=n_splits, seed=random_state)
    t = best_threshold(y.values, oof, target="f1")
    metrics = compute_metrics(y.values, oof, t)
    row = {"model": MODEL_NAME, **metrics}

    pipe.fit(X, y)
    class_ratio = {"neg": int((y == 0).sum()), "pos": int((y == 1).sum())}
    save_artifacts(MODEL_NAME, pipe, feature_cols, cat_features or [], label_col, t, class_ratio,
                   results_dir, models_dir, oof, row)
    return row
