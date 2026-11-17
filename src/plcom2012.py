from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import pandas as pd

# Published PLCOm2012 variable names and coefficients. Keep these centralized so
# clinical deployments can audit or replace them against their licensed reference.
COEFFICIENTS = {
    "intercept": -4.532506, "age": 0.077886, "age_squared": -0.000000,
    "race_black": 0.482616, "race_other": 0.0, "education": -0.040371,
    "bmi": -0.017613, "copd": 0.355306, "personal_cancer": 0.285386,
    "family_history": 0.267826, "smoking_former": 0.0, "smoking_current": 0.0,
    "intensity": 0.025719, "inverse_intensity": -0.001822,
    "duration": 0.019102, "quit_time": -0.030857,
}


@dataclass
class ClinicalInputs:
    age: float = 62
    race: str = "White"
    education: int = 4
    bmi: float = 27
    copd: int = 0
    personal_history_of_cancer: int = 0
    family_history_lung_cancer: int = 0
    smoking_status: str = "never"
    smoking_intensity: float = 0
    smoking_duration: float = 0
    smoking_quit_time: float = 0


def adapt_dataset_row(row: pd.Series) -> ClinicalInputs:
    smoking = float(row.get("smoking", 1))
    status = "current" if smoking >= 7 else ("former" if smoking >= 3 else "never")
    return ClinicalInputs(
        age=float(row["age"]), bmi=float(max(15, 32 - 1.5 * float(row.get("obesity", 1)))),
        copd=int(float(row.get("chronic_lung_disease", 1)) >= 5),
        family_history_lung_cancer=int(float(row.get("genetic_risk", 1)) >= 5),
        smoking_status=status, smoking_intensity=max(0, smoking * 3),
        smoking_duration=max(0, (float(row["age"]) - 18) * smoking / 8),
        smoking_quit_time=0 if status == "current" else 10,
    )


def logit_contributions(inputs: ClinicalInputs) -> dict[str, float]:
    c = COEFFICIENTS
    age = inputs.age - 62
    intensity = inputs.smoking_intensity / 10
    contributions = {
        "Intercept": c["intercept"],
        "Age": c["age"] * age + c["age_squared"] * age * age,
        "Race": c["race_black"] if inputs.race.lower() == "black" else c["race_other"],
        "Education": c["education"] * (inputs.education - 4),
        "BMI": c["bmi"] * (inputs.bmi - 27),
        "COPD": c["copd"] * inputs.copd,
        "Personal cancer": c["personal_cancer"] * inputs.personal_history_of_cancer,
        "Family history": c["family_history"] * inputs.family_history_lung_cancer,
        "Smoking status": c.get(f"smoking_{inputs.smoking_status.lower()}", 0),
        "Smoking intensity": c["intensity"] * intensity + c["inverse_intensity"] * (1 / intensity if intensity > 0 else 0),
        "Smoking duration": c["duration"] * (inputs.smoking_duration - 27),
        "Smoking quit time": c["quit_time"] * (inputs.smoking_quit_time - 10),
    }
    return contributions


def plcom2012_risk(inputs: ClinicalInputs) -> float:
    logit = sum(logit_contributions(inputs).values())
    return 1 / (1 + math.exp(-logit))


def score_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    values = [plcom2012_risk(adapt_dataset_row(row)) for _, row in frame.iterrows()]
    return pd.DataFrame({"plco_risk": values}, index=frame.index)

