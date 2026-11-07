from src import config
from src.generate_data import generate
from src.pipeline import validate


def test_generated_data_passes_validation():
    assert validate(generate(300))["passed"]


def test_validation_catches_bad_values():
    df = generate(100)
    df.loc[0, "channel"] = "pigeon"
    df.loc[1, "order_value"] = -10
    rep = validate(df)
    assert not rep["passed"]
    failed = {c["check"] for c in rep["checks"] if not c["passed"]}
    assert "channel values valid" in failed


def test_validation_catches_missing_column():
    df = generate(50).drop(columns=[config.TEXT])
    assert not validate(df)["passed"]
