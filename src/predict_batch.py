"""Phase 3 (batch): score a CSV of complaints.
Run: python -m src.predict_batch --input data/batch/sample_batch.csv
"""
import argparse
import joblib
import pandas as pd
from src import config
from src.model import FEATURES
from src.pipeline import validate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(config.BATCH_DIR / "sample_batch.csv"))
    ap.add_argument("--output", default=str(config.REPORTS / "batch_predictions.csv"))
    a = ap.parse_args()
    df = pd.read_csv(a.input)
    check = validate(df, require_target=False)
    if not check["passed"]:
        bad = [c["check"] for c in check["checks"] if not c["passed"]]
        raise SystemExit(f"Input failed validation: {bad}")
    df[config.TEXT] = df[config.TEXT].str.lower().str.strip()
    model = joblib.load(config.MODEL_PATH)
    prob = model.predict_proba(df[FEATURES])[:, 1]
    out = df.assign(escalation_probability=prob.round(4), escalate=(prob >= 0.5).astype(int))
    config.REPORTS.mkdir(exist_ok=True)
    out.to_csv(a.output, index=False)
    print(f"Scored {len(out)} complaints -> {a.output}")
    print(f"Flagged for escalation: {int(out['escalate'].sum())} ({out['escalate'].mean():.1%})")
    print(out[["channel", "product_category", "escalation_probability", "escalate"]].head(8).to_string(index=False))


if __name__ == "__main__":
    main()
