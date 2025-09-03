from typing import List, Optional, Dict
import pandas as pd
from sklearn.pipeline import Pipeline

from .utils import (
    ensure_dirs, build_preprocessor, crossval_oof_proba,
    best_threshold, compute_metrics, save_artifacts
)

MODEL_NAME = "gbm"

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

def _make_gbm(y, random_state: int = 42):
    pos = int(y.sum())
    neg = int(len(y) - pos)
    scale_pos = float(neg) / float(pos) if pos > 0 else 1.0
    if HAS_LGBM:
        return LGBMClassifier(
            n_estimators=800, learning_rate=0.03,
            num_leaves=64, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, objective="binary",
            random_state=random_state, n_jobs=-1,
            scale_pos_weight=scale_pos,
        )
    if HAS_XGB:
        return XGBClassifier(
            n_estimators=900, learning_rate=0.03, max_depth=4,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            objective="binary:logistic", eval_metric="auc",
            random_state=random_state, n_jobs=-1,
            scale_pos_weight=scale_pos, tree_method="hist",
        )
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        max_depth=None, learning_rate=0.05, max_iter=400, random_state=random_state
    )

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
    clf = _make_gbm(y, random_state)
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
