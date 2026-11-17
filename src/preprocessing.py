from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

ENVIRONMENTAL = [
    "smoking", "passive_smoker", "air_pollution", "occupational_hazards",
    "genetic_risk", "chronic_lung_disease",
]
SYMPTOMS = [
    "chest_pain", "coughing_of_blood", "fatigue", "weight_loss",
    "shortness_of_breath", "wheezing", "swallowing_difficulty",
    "clubbing_of_finger_nails", "frequent_cold", "dry_cough", "snoring",
]
REQUIRED_COLUMNS = ["patient_id", "age", "gender", "level", *ENVIRONMENTAL, *SYMPTOMS]


def deduplicate_profiles(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate feature profiles, retaining the first labelled record."""
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")
    result = frame.copy()
    result = result.drop_duplicates(subset=[c for c in REQUIRED_COLUMNS if c != "patient_id"], keep="first")
    if not result["level"].isin(["Low", "Medium", "High"]).all():
        raise ValueError("level must contain only Low, Medium, and High")
    return result.reset_index(drop=True)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Add clinically interpretable indices and interactions to ordinal inputs."""

    feature_names_in_: np.ndarray

    def fit(self, X: pd.DataFrame, y: Iterable | None = None) -> "FeatureEngineer":
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        result = X.copy()
        for col in [*ENVIRONMENTAL, *SYMPTOMS, "age"]:
            if col not in result:
                raise ValueError(f"Missing feature: {col}")
        result["composite_risk_score"] = (
            0.25 * result["smoking"] + 0.15 * result["passive_smoker"]
            + 0.15 * result["air_pollution"] + 0.15 * result["occupational_hazards"]
            + 0.15 * result["genetic_risk"] + 0.15 * result["chronic_lung_disease"]
        )
        result["symptom_burden_index"] = result[SYMPTOMS].sum(axis=1) / len(SYMPTOMS)
        result["age_squared"] = result["age"] ** 2
        result["smoking_x_pollution"] = result["smoking"] * result["air_pollution"]
        result["smoking_x_chronic_lung"] = result["smoking"] * result["chronic_lung_disease"]
        result["hemoptysis_x_weight_loss"] = result["coughing_of_blood"] * result["weight_loss"]
        result["age_x_symptom_burden"] = result["age"] * result["symptom_burden_index"]
        return result

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        base = list(input_features if input_features is not None else self.feature_names_in_)
        return np.asarray(base + [
            "composite_risk_score", "symptom_burden_index", "age_squared",
            "smoking_x_pollution", "smoking_x_chronic_lung",
            "hemoptysis_x_weight_loss", "age_x_symptom_burden",
        ], dtype=object)

