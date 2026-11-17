from __future__ import annotations

import pandas as pd

from .plcom2012 import ClinicalInputs, logit_contributions


def triage_logic(ml_probability: float, plco_probability: float) -> tuple[str, str]:
    ml_high = ml_probability >= 0.5
    plco_high = plco_probability >= 0.0151
    if ml_high and plco_high:
        return "Concordant High Priority", "Both engines indicate elevated risk; expedite clinical review and LDCT eligibility assessment."
    if ml_high:
        return "Symptom-Driven Alert", "The symptom model is elevated despite lower epidemiological risk; assess active symptoms and diagnostic pathways."
    if plco_high:
        return "Latent Epidemiological Risk", "Epidemiological risk is elevated despite lower symptom-model risk; verify smoking history and consider LDCT."
    return "Concordant Low", "Both engines are below alert thresholds; continue shared decision-making and routine prevention."


def plco_attribution(inputs: ClinicalInputs) -> pd.DataFrame:
    values = logit_contributions(inputs)
    return pd.DataFrame({"factor": list(values), "delta_logit": list(values.values())}).sort_values("delta_logit")

