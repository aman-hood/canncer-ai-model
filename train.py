from pathlib import Path
import joblib
import pandas as pd

from src.ensemble import train_and_evaluate
from src.preprocessing import deduplicate_profiles


def main() -> None:
    path = Path("cancer_patient_data_sets.csv")
    if not path.exists():
        raise FileNotFoundError("cancer_patient_data_sets.csv was not found. Place the supplied cohort at the project root.")
    frame = deduplicate_profiles(pd.read_csv(path))
    model, metrics = train_and_evaluate(frame)
    Path("artifacts").mkdir(exist_ok=True)
    joblib.dump(model, "artifacts/stacking_pipeline.joblib")
    print(metrics)


if __name__ == "__main__":
    main()

