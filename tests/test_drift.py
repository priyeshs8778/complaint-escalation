from src.drift import categorical_drift, numeric_drift
from src.generate_data import generate


def test_no_drift_on_same_distribution():
    a, b = generate(2000, seed=1), generate(500, seed=2)
    assert not numeric_drift(a["order_value"], b["order_value"])["drift"]
    assert not categorical_drift(a["channel"], b["channel"])["drift"]


def test_drift_detected_on_shift():
    a, b = generate(2000, seed=1), generate(500, seed=2, drifted=True)
    assert numeric_drift(a["order_value"], b["order_value"])["drift"]
    assert categorical_drift(a["channel"], b["channel"])["drift"]
