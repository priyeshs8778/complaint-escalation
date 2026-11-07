"""Phase 3 (real-time): FastAPI service. Run: uvicorn src.api:app --port 8000"""
import time
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field
from typing import Literal
from src import config, monitoring
from src.explainer import explain
from src.model import FEATURES

app = FastAPI(title="Complaint Escalation Risk API", version="1.0")
_model = None


def get_model():
    global _model
    if _model is None:
        if not config.MODEL_PATH.exists():
            raise HTTPException(503, "Model not found. Run `make train` first.")
        _model = joblib.load(config.MODEL_PATH)
    return _model


class Complaint(BaseModel):
    order_value: float = Field(ge=0, le=20000)
    days_since_order: int = Field(ge=0, le=365)
    prior_complaints: int = Field(ge=0, le=100)
    customer_tenure_months: int = Field(ge=0, le=600)
    response_wait_hours: float = Field(ge=0, le=2000)
    product_category: Literal["electronics", "clothing", "home", "grocery", "beauty"]
    channel: Literal["email", "chat", "phone", "social"]
    complaint_text: str = Field(min_length=1)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": config.MODEL_PATH.exists()}


@app.post("/predict")
def predict(c: Complaint):
    t0 = time.time()
    try:
        row = c.model_dump()
        row["complaint_text"] = row["complaint_text"].lower().strip()
        prob = float(get_model().predict_proba(pd.DataFrame([row])[FEATURES])[0, 1])
        label = int(prob >= 0.5)
        monitoring.REQUESTS.labels("ok").inc()
        monitoring.LAST_PROB.set(prob)
        if label:
            monitoring.ESCALATIONS.inc()
        monitoring.log_prediction(row, prob, label)
        return {"escalation_probability": round(prob, 4), "escalate": bool(label),
                "explanation": explain(row, prob)}
    except HTTPException:
        monitoring.REQUESTS.labels("error").inc()
        raise
    finally:
        monitoring.LATENCY.observe(time.time() - t0)


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
