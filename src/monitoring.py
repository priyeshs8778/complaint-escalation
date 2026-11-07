"""Phase 5 (runtime half): Prometheus metrics and a JSONL prediction log."""
import json
import time
from prometheus_client import Counter, Histogram, Gauge
from src import config

REQUESTS = Counter("predict_requests_total", "Prediction requests", ["status"])
LATENCY = Histogram("predict_latency_seconds", "Prediction latency in seconds")
ESCALATIONS = Counter("predicted_escalations_total", "Predictions flagged as escalation")
LAST_PROB = Gauge("last_escalation_probability", "Most recent predicted probability")


def log_prediction(features: dict, probability: float, label: int) -> None:
    config.REPORTS.mkdir(exist_ok=True)
    rec = {"ts": time.time(), **features, "probability": round(probability, 4), "label": label}
    with open(config.PRED_LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")
