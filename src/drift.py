"""Phase 5 (drift): compare a new batch against the training reference data.

Numeric features: Population Stability Index (PSI) and a two-sample KS test.
Categorical features: PSI on category shares and a chi-square test.
Alert rule: PSI >= 0.2 on a feature (a common industry threshold) or p < 0.01.

Run: python -m src.drift --current data/batch/drifted_batch.csv
Exit code 0 = no drift, 2 = drift alert (so CI or cron can act on it).
"""
import argparse
import json
import sys
import numpy as np
import pandas as pd
from scipy import stats
from src import config

PSI_ALERT = 0.2
P_ALERT = 0.01
EPS = 1e-4


def psi_from_shares(ref, cur):
    ref = np.clip(np.asarray(ref, float), EPS, None)
    cur = np.clip(np.asarray(cur, float), EPS, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def numeric_drift(ref: pd.Series, cur: pd.Series) -> dict:
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, 11)))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    ks = stats.ks_2samp(ref, cur)
    psi = psi_from_shares(r, c)
    return {"type": "numeric", "psi": round(psi, 4), "p_value": float(ks.pvalue),
            "ref_mean": round(float(ref.mean()), 3), "cur_mean": round(float(cur.mean()), 3),
            "drift": bool(psi >= PSI_ALERT or ks.pvalue < P_ALERT)}


def categorical_drift(ref: pd.Series, cur: pd.Series) -> dict:
    cats = sorted(set(ref) | set(cur))
    r = ref.value_counts(normalize=True).reindex(cats, fill_value=0)
    c = cur.value_counts(normalize=True).reindex(cats, fill_value=0)
    table = np.array([ref.value_counts().reindex(cats, fill_value=0),
                      cur.value_counts().reindex(cats, fill_value=0)])
    p = float(stats.chi2_contingency(table + 1e-9)[1])
    psi = psi_from_shares(r, c)
    return {"type": "categorical", "psi": round(psi, 4), "p_value": p,
            "ref_share": {k: round(float(v), 3) for k, v in r.items()},
            "cur_share": {k: round(float(v), 3) for k, v in c.items()},
            "drift": bool(psi >= PSI_ALERT or p < P_ALERT)}


def run(current_path: str) -> dict:
    ref = pd.read_csv(config.REFERENCE)
    cur = pd.read_csv(current_path)
    feats = {}
    for col in config.NUMERIC:
        feats[col] = numeric_drift(ref[col], cur[col])
    for col in config.CATEGORICAL:
        feats[col] = categorical_drift(ref[col], cur[col])
    drifted = [k for k, v in feats.items() if v["drift"]]
    return {"reference_rows": len(ref), "current_rows": len(cur), "current_file": str(current_path),
            "thresholds": {"psi": PSI_ALERT, "p_value": P_ALERT},
            "features": feats, "drifted_features": drifted,
            "share_drifted": round(len(drifted) / len(feats), 3),
            "alert": len(drifted) > 0}


def to_html(rep: dict) -> str:
    rows = ""
    for name, v in rep["features"].items():
        color = "#f8d7da" if v["drift"] else "#d4edda"
        rows += (f"<tr style='background:{color}'><td>{name}</td><td>{v['type']}</td>"
                 f"<td>{v['psi']}</td><td>{v['p_value']:.2e}</td>"
                 f"<td>{'DRIFT' if v['drift'] else 'ok'}</td></tr>")
    banner = ("<h2 style='color:#b00020'>ALERT: data drift detected in "
              f"{len(rep['drifted_features'])} of {len(rep['features'])} features</h2>"
              if rep["alert"] else "<h2 style='color:#1b7f3b'>No drift detected</h2>")
    return (f"<html><body style='font-family:Arial;margin:24px'><h1>Drift Report</h1>{banner}"
            f"<p>Reference rows: {rep['reference_rows']} | Current rows: {rep['current_rows']} | "
            f"File: {rep['current_file']}</p>"
            "<table border='1' cellpadding='6' style='border-collapse:collapse'>"
            "<tr><th>Feature</th><th>Type</th><th>PSI</th><th>p-value</th><th>Status</th></tr>"
            f"{rows}</table><p>Alert rule: PSI &ge; {PSI_ALERT} or p &lt; {P_ALERT}.</p></body></html>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--current", default=str(config.BATCH_DIR / "drifted_batch.csv"))
    a = ap.parse_args()
    rep = run(a.current)
    config.REPORTS.mkdir(exist_ok=True)
    (config.REPORTS / "drift_report.json").write_text(json.dumps(rep, indent=2))
    (config.REPORTS / "drift_report.html").write_text(to_html(rep))
    print(f"Compared {rep['current_rows']} current rows against {rep['reference_rows']} reference rows")
    for k, v in rep["features"].items():
        print(f"  {'DRIFT' if v['drift'] else 'ok   '}  {k:<24} PSI={v['psi']:<8} p={v['p_value']:.2e}")
    if rep["alert"]:
        print(f"ALERT: drift in {rep['drifted_features']}")
        sys.exit(2)
    print("No drift detected")


if __name__ == "__main__":
    main()
