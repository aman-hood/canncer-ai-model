# Dual-Path Lung Cancer Risk Intelligence

This Streamlit command center combines a deduplicated machine-learning cohort
with a PLCOm2012 epidemiological adapter. It is clinical decision support, not
a diagnosis or a substitute for clinician review.

## Run

1. Put the supplied `cancer_patient_data_sets.csv` at the repository root, or
   upload a CSV directly using the **Upload patient cohort CSV** control in the
   app.
2. Install dependencies: `python -m pip install -r requirements.txt`.
3. Train: `python train.py`.
4. Launch: `streamlit run app.py`.

The training command refuses malformed or missing data, deduplicates identical
feature profiles before cross-validation, and writes model and metrics artifacts.
