import pytest
from fastapi.testclient import TestClient
from src import config
from src.api import app

pytestmark = pytest.mark.skipif(not config.MODEL_PATH.exists(), reason="model not trained")
client = TestClient(app)
GOOD = {"order_value": 320, "days_since_order": 12, "prior_complaints": 3,
        "customer_tenure_months": 8, "response_wait_hours": 40, "product_category": "electronics",
        "channel": "social", "complaint_text": "I will contact a lawyer"}


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_predict_returns_probability_and_explanation():
    r = client.post("/predict", json=GOOD)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["escalation_probability"] <= 1
    assert body["explanation"]["risk_level"] in {"LOW", "MEDIUM", "HIGH"}


def test_predict_rejects_invalid_input():
    assert client.post("/predict", json={**GOOD, "channel": "pigeon"}).status_code == 422
    assert client.post("/predict", json={**GOOD, "order_value": -1}).status_code == 422


def test_metrics_endpoint():
    client.post("/predict", json=GOOD)
    assert "predict_requests_total" in client.get("/metrics").text
