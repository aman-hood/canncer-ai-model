from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (brier_score_loss, log_loss, precision_recall_curve, roc_auc_score,
                             accuracy_score, auc, confusion_matrix)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from .preprocessing import FeatureEngineer


def build_pipeline(features: list[str], numeric: list[str], categorical: list[str]) -> Pipeline:
    prep = ColumnTransformer([
        ("num", Pipeline([("engineer", FeatureEngineer()), ("scale", StandardScaler())]), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    base = [
        ("xgb", XGBClassifier(n_estimators=180, max_depth=3, learning_rate=.05, subsample=.85,
                              colsample_bytree=.85, eval_metric="mlogloss", random_state=42)),
        ("gbm", GradientBoostingClassifier(random_state=42)),
        ("rf", RandomForestClassifier(n_estimators=250, class_weight="balanced", random_state=42, n_jobs=-1)),
        ("lr", LogisticRegression(max_iter=1500, multi_class="auto")),
    ]
    return Pipeline([("prep", prep), ("model", StackingClassifier(
        estimators=base, final_estimator=LogisticRegression(max_iter=1500),
        stack_method="predict_proba", passthrough=False, cv=5,
    ))])


def train_and_evaluate(frame: pd.DataFrame, artifacts_dir: str = "artifacts") -> tuple[Pipeline, dict]:
    target = frame["level"]
    X = frame.drop(columns=["level", "patient_id"])
    numeric = [c for c in X.columns if c != "gender"]
    pipe = build_pipeline(list(X.columns), numeric, ["gender"])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    probabilities = cross_val_predict(pipe, X, target, cv=cv, method="predict_proba", n_jobs=None)
    labels = sorted(target.unique(), key=["Low", "Medium", "High"].index)
    high = (target == "High").astype(int).to_numpy()
    high_idx = labels.index("High")
    predicted_labels = np.asarray(labels)[probabilities.argmax(axis=1)]
    metrics = {
        "accuracy": float(accuracy_score(target, predicted_labels)),
        "roc_auc_ovr": float(roc_auc_score(target, probabilities, multi_class="ovr", labels=labels)),
        "high_risk_roc_auc": float(roc_auc_score(high, probabilities[:, high_idx])),
        "pr_auc": float(auc(*precision_recall_curve(high, probabilities[:, high_idx])[:2][::-1])),
        "brier_score": float(brier_score_loss(high, probabilities[:, high_idx])),
        "log_loss": float(log_loss(target, probabilities, labels=labels)),
        "confusion_matrix": confusion_matrix(target, predicted_labels, labels=labels).tolist(),
        "labels": labels,
    }
    Path(artifacts_dir).mkdir(exist_ok=True)
    (Path(artifacts_dir) / "cv_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pipe.fit(X, target)
    calibrated = CalibratedClassifierCV(pipe, cv=3, method="sigmoid")
    calibrated.fit(X, target)
    return calibrated, metrics
