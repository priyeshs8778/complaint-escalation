"""Phase 2: train the Random Forest, log metrics, save the model artifact.
Run: python -m src.train
"""
import json
import time
import joblib
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from src import config
from src.model import FEATURES, build_pipeline

PARAMS = {"n_estimators": 300, "max_depth": None, "test_size": 0.2, "seed": 42}


def log_to_mlflow(params, metrics, model_path):
    """MLflow is a bonus: training must still succeed if it is unavailable."""
    try:
        import mlflow
        mlflow.set_tracking_uri(f"sqlite:///{config.ROOT / 'mlflow.db'}")
        mlflow.set_experiment("complaint-escalation")
        with mlflow.start_run(run_name="random-forest"):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(model_path))
        print("MLflow run logged (mlflow.db)")
    except Exception as e:  # noqa: BLE001
        print(f"MLflow logging skipped: {e}")


def main():
    df = pd.read_csv(config.PROCESSED)
    X, y = df[FEATURES], df[config.TARGET]
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=PARAMS["test_size"], stratify=y, random_state=PARAMS["seed"])
    print(f"Train rows: {len(Xtr)}  Test rows: {len(Xte)}")

    model = build_pipeline(PARAMS["n_estimators"], PARAMS["max_depth"], PARAMS["seed"])
    t0 = time.time()
    model.fit(Xtr, ytr)
    print(f"Trained in {time.time() - t0:.1f}s")

    proba = model.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "accuracy": accuracy_score(yte, pred),
        "precision": precision_score(yte, pred),
        "recall": recall_score(yte, pred),
        "f1": f1_score(yte, pred),
        "roc_auc": roc_auc_score(yte, proba),
    }
    metrics = {k: round(float(v), 4) for k, v in metrics.items()}
    print(json.dumps(metrics, indent=2))
    print(classification_report(yte, pred, target_names=["no_escalation", "escalation"]))
    print("Confusion matrix:\n", confusion_matrix(yte, pred))

    config.MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, config.MODEL_PATH)
    print(f"Saved model -> {config.MODEL_PATH}")

    config.REPORTS.mkdir(exist_ok=True)
    (config.REPORTS / "metrics.json").write_text(json.dumps(
        {"params": PARAMS, "metrics": metrics, "n_train": len(Xtr), "n_test": len(Xte)}, indent=2))
    # Held-out test split is reused by the Responsible AI analysis.
    Xte.assign(**{config.TARGET: yte.values}).to_csv(config.REPORTS / "test_split.csv", index=False)
    log_to_mlflow(PARAMS, metrics, config.MODEL_PATH)


if __name__ == "__main__":
    main()
