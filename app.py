from io import BytesIO
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.explainability import plco_attribution, triage_logic
from src.plcom2012 import ClinicalInputs, adapt_dataset_row, plcom2012_risk
from src.preprocessing import REQUIRED_COLUMNS, deduplicate_profiles

st.set_page_config(page_title="Lung Risk Intelligence", page_icon="🫁", layout="wide")


@st.cache_data
def load_data(uploaded_csv: bytes | None = None) -> pd.DataFrame:
    if uploaded_csv is not None:
        return deduplicate_profiles(pd.read_csv(BytesIO(uploaded_csv)))

    path = Path("cancer_patient_data_sets.csv")
    if not path.exists():
        raise FileNotFoundError(
            "Upload a CSV dataset or place cancer_patient_data_sets.csv at the project root."
        )
    return deduplicate_profiles(pd.read_csv(path))


@st.cache_resource
def load_model():
    import joblib
    path = Path("artifacts/stacking_pipeline.joblib")
    if not path.exists():
        return None
    return joblib.load(path)


def gauge(title: str, value: float, maximum: float = 1.0, threshold: float | None = None):
    fig = go.Figure(go.Indicator(mode="gauge+number", value=value * 100, number={"suffix": "%"},
        title={"text": title}, gauge={"axis": {"range": [0, maximum * 100]},
        "bar": {"color": "#10B981"}, "steps": [{"range": [0, maximum * 35], "color": "#1F2937"},
                                               {"range": [maximum * 35, maximum * 70], "color": "#374151"},
                                               {"range": [maximum * 70, maximum * 100], "color": "#7F1D1D"}]}))
    if threshold is not None:
        fig.add_vline(x=threshold * 100, line_dash="dash", line_color="#F59E0B")
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=55, b=10), template="plotly_dark")
    return fig


st.title("🫁 Dual-Path Lung Cancer Risk Intelligence")
st.caption("Clinical decision support only — not a diagnosis. Validate formula governance before clinical deployment.")
uploaded_file = st.file_uploader(
    "Upload patient cohort CSV",
    type=["csv"],
    help="The uploaded CSV must include the required patient, risk-factor, and symptom columns.",
)
try:
    data = load_data(uploaded_file.getvalue() if uploaded_file is not None else None)
except (
    FileNotFoundError,
    pd.errors.EmptyDataError,
    pd.errors.ParserError,
    UnicodeDecodeError,
    ValueError,
) as exc:
    st.error(str(exc))
    st.stop()
if uploaded_file is not None:
    st.caption(f"Using uploaded dataset: {uploaded_file.name}")

model = load_model()
metrics_path = Path("artifacts/cv_metrics.json")
metrics = __import__("json").loads(metrics_path.read_text()) if metrics_path.exists() else {}
high_pct = (data["level"].eq("High").mean())
stages = st.segmented_control("Architecture Flow Explorer", ["Data Sources", "Preprocessing", "Models", "Outputs"], default="Outputs")
with st.expander(f"{stages} details", expanded=True):
    st.write({"Data Sources": "Validated cohort + strict feature-profile deduplication",
              "Preprocessing": "FeatureEngineer branch + PLCOm2012 clinical adapter",
              "Models": "XGB, GBM, RF, LR stacked with 5-fold OOF calibration",
              "Outputs": "Risk probabilities, LDCT flags, SHAP and clinical concordance"}[stages])

kpi = st.columns(5)
kpi[0].metric("Deduplicated cohort", len(data))
kpi[1].metric("5-fold accuracy", f"{metrics.get('accuracy', 0):.1%}")
kpi[2].metric("Brier score", f"{metrics.get('brier_score', 0):.3f}")
kpi[3].metric("High-risk cohort", f"{high_pct:.1%}")
kpi[4].metric("LDCT eligible", "Train cohort pending")

tab1, tab2, tab3 = st.tabs(["Patient Simulator", "Model Diagnostics", "Cohort Explorer"])
with tab1:
    st.subheader("Dual-engine patient simulator")
    row = data.iloc[0]
    with st.sidebar:
        st.header("Clinical control deck")
        selected = st.selectbox("Load cohort patient", data["patient_id"].astype(str).tolist())
        row = data[data["patient_id"].astype(str) == selected].iloc[0]
        age = st.slider("Age", 18, 100, int(row["age"]))
        bmi = st.slider("BMI", 15.0, 50.0, float(adapt_dataset_row(row).bmi))
        cigs = st.slider("Cigarettes/day", 0.0, 80.0, float(adapt_dataset_row(row).smoking_intensity))
        years = st.slider("Years smoked", 0.0, 70.0, float(adapt_dataset_row(row).smoking_duration))
        family = st.checkbox("Family history", bool(adapt_dataset_row(row).family_history_lung_cancer))
    clinical = adapt_dataset_row(row)
    clinical.age, clinical.bmi, clinical.smoking_intensity, clinical.smoking_duration = age, bmi, cigs, years
    clinical.family_history_lung_cancer = int(family)
    plco = plcom2012_risk(clinical)
    ml = 0.0
    if model is not None:
        probs = model.predict_proba(pd.DataFrame([row.drop(labels=["level", "patient_id"])]))[0]
        labels = getattr(model, "classes_", ["Low", "Medium", "High"])
        ml = float(probs[list(labels).index("High")])
    else:
        ml = min(0.99, max(0.01, float(row["symptom_burden_index"]) / 9 if "symptom_burden_index" in row else 0.1))
    label, note = triage_logic(ml, plco)
    a, b = st.columns(2)
    a.plotly_chart(gauge("ML high-risk probability", ml), use_container_width=True)
    b.plotly_chart(gauge("PLCOm2012 6-year risk", plco, maximum=max(.05, plco * 1.5), threshold=.0151), use_container_width=True)
    st.info(f"**{label}** — {note}")
    st.plotly_chart(go.Figure(go.Bar(x=plco_attribution(clinical)["delta_logit"], y=plco_attribution(clinical)["factor"], orientation="h")).update_layout(template="plotly_dark", height=420), use_container_width=True)

with tab2:
    st.subheader("5-fold validation diagnostics")
    st.json(metrics)
    if metrics.get("confusion_matrix"):
        st.plotly_chart(go.Figure(go.Heatmap(z=metrics["confusion_matrix"], x=metrics["labels"], y=metrics["labels"], colorscale="Viridis")).update_layout(template="plotly_dark"), use_container_width=True)

with tab3:
    st.subheader("Cohort explorer")
    view = data.copy()
    view["plco_6yr_risk"] = [plcom2012_risk(adapt_dataset_row(r)) for _, r in data.iterrows()]
    st.dataframe(view, use_container_width=True, hide_index=True)
