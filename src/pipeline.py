"""Phase 1: ingest -> validate -> process. One command: python -m src.pipeline"""
import json
import sys
import pandas as pd
from src import config

EXPECTED = config.NUMERIC + config.CATEGORICAL + [config.TEXT, config.TARGET]
RANGES = {"order_value": (0, 20000), "days_since_order": (0, 365),
          "prior_complaints": (0, 100), "customer_tenure_months": (0, 600),
          "response_wait_hours": (0, 2000)}
ALLOWED = {"product_category": {"electronics", "clothing", "home", "grocery", "beauty"},
           "channel": {"email", "chat", "phone", "social"}}


def validate(df: pd.DataFrame, require_target: bool = True) -> dict:
    cols = EXPECTED if require_target else [c for c in EXPECTED if c != config.TARGET]
    checks = []

    def add(name, ok, detail=""):
        checks.append({"check": name, "passed": bool(ok), "detail": str(detail)})

    missing = [c for c in cols if c not in df.columns]
    add("required columns present", not missing, missing or "all present")
    if missing:
        return {"passed": False, "checks": checks, "rows": len(df)}
    add("no null values", df[cols].isna().sum().sum() == 0, int(df[cols].isna().sum().sum()))
    add("no duplicate rows", not df.duplicated().any(), int(df.duplicated().sum()))
    for c, (lo, hi) in RANGES.items():
        add(f"{c} in [{lo}, {hi}]", df[c].between(lo, hi).all())
    for c, allowed in ALLOWED.items():
        add(f"{c} values valid", df[c].isin(allowed).all(), sorted(set(df[c]) - allowed) or "ok")
    add("complaint_text not empty", (df[config.TEXT].str.strip().str.len() > 0).all())
    if require_target:
        add("target is binary", set(df[config.TARGET].unique()) <= {0, 1})
    return {"passed": all(c["passed"] for c in checks), "checks": checks, "rows": len(df)}


def run() -> int:
    print(f"[1/4] Source: {config.RAW}")
    df = pd.read_csv(config.RAW)
    print(f"[2/4] Loaded {len(df)} rows x {df.shape[1]} columns")
    report = validate(df)
    for c in report["checks"]:
        print(f"      {'PASS' if c['passed'] else 'FAIL'}  {c['check']}  {c['detail']}")
    config.REPORTS.mkdir(exist_ok=True)
    (config.REPORTS / "validation_report.json").write_text(json.dumps(report, indent=2))
    if not report["passed"]:
        print("Validation failed. Stopping.")
        return 1
    print("[3/4] Processing: dedupe, normalise text, clip outliers")
    df = df.drop_duplicates().copy()
    df[config.TEXT] = df[config.TEXT].str.lower().str.strip()
    df["order_value"] = df["order_value"].clip(upper=df["order_value"].quantile(0.995))
    config.PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.PROCESSED, index=False)
    # Reference sample (no target) is what drift detection compares new data against.
    df.drop(columns=[config.TARGET]).to_csv(config.REFERENCE, index=False)
    print(f"[4/4] Wrote {config.PROCESSED} ({len(df)} rows)")
    print(f"      Escalation rate: {df[config.TARGET].mean():.1%}")
    print(df.head(5).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(run())
