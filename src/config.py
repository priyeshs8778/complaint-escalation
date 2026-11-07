"""Central paths and constants so every phase reads and writes the same places."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "complaints_raw.csv"
PROCESSED = ROOT / "data" / "processed" / "complaints_processed.csv"
REFERENCE = ROOT / "data" / "processed" / "reference_features.csv"
BATCH_DIR = ROOT / "data" / "batch"
MODEL_PATH = ROOT / "models" / "escalation_model.joblib"
REPORTS = ROOT / "reports"
PRED_LOG = REPORTS / "prediction_log.jsonl"

NUMERIC = ["order_value", "days_since_order", "prior_complaints",
           "customer_tenure_months", "response_wait_hours"]
CATEGORICAL = ["product_category", "channel"]
TEXT = "complaint_text"
TARGET = "escalated"
SENSITIVE = "product_category"  # group column used for the fairness check
