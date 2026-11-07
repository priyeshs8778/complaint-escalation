"""Phase 6: Responsible AI analysis on the held-out test split.

1. Explainability: SHAP values for the Random Forest, aggregated back to the original
   input columns, saved as a bar chart and a JSON ranking.
2. Fairness: per-group selection rate, recall and false positive rate across
   product_category, with demographic parity and equalised odds gaps.

Run: python -m src.responsible_ai
"""
import json
import warnings
import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score
from src import config
from src.model import FEATURES

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import shap  # noqa: E402

warnings.filterwarnings("ignore")
MAX_GAP = 0.20  # flag a group gap above 20 percentage points


def feature_names(prep):
    names, owners = [], []
    for name, trans, cols in prep.transformers_:
        if name == "remainder":
            continue
        out = trans.get_feature_names_out()
        for n in out:
            names.append(n)
            if name == "num":
                owners.append(n)
            elif name == "cat":
                owners.append(next(c for c in cols if n.startswith(c)))
            else:
                owners.append(config.TEXT)
    return names, owners


def explainability(model, test):
    prep, rf = model.named_steps["prep"], model.named_steps["rf"]
    X = prep.transform(test[FEATURES])
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    sample = X[:300]
    sv = shap.TreeExplainer(rf).shap_values(sample)
    sv = np.asarray(sv)
    # Normalise the many shapes shap has used across versions to (rows, features) for class 1
    if sv.ndim == 3:
        sv = sv[:, :, 1] if sv.shape[2] == 2 else sv[1]
    _, owners = feature_names(prep)
    imp = pd.Series(np.abs(sv).mean(axis=0), index=owners).groupby(level=0).sum()
    imp = (imp / imp.sum()).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    imp.iloc[::-1].plot.barh(ax=ax, color="#1F3864")
    ax.set_xlabel("Share of total mean |SHAP| (importance)")
    ax.set_title("What drives escalation predictions (SHAP, by input column)")
    fig.tight_layout()
    fig.savefig(config.REPORTS / "shap_importance.png", dpi=140)
    plt.close(fig)
    return {k: round(float(v), 4) for k, v in imp.items()}


def fairness(model, test):
    prob = model.predict_proba(test[FEATURES])[:, 1]
    pred = (prob >= 0.5).astype(int)
    y = test[config.TARGET].to_numpy()
    rows = []
    for g, idx in test.groupby(config.SENSITIVE).groups.items():
        pos = test.index.get_indexer(idx)
        yt, yp = y[pos], pred[pos]
        neg = (yt == 0).sum()
        rows.append({"group": g, "n": int(len(pos)), "n_positive": int(yt.sum()),
                     "actual_rate": round(float(yt.mean()), 3),
                     "selection_rate": round(float(yp.mean()), 3),
                     "recall_tpr": round(float(recall_score(yt, yp, zero_division=0)), 3),
                     "false_positive_rate": round(float(((yp == 1) & (yt == 0)).sum() / neg), 3) if neg else None})
    df = pd.DataFrame(rows)
    gaps = {
        "demographic_parity_gap": round(float(df["selection_rate"].max() - df["selection_rate"].min()), 3),
        "tpr_gap": round(float(df["recall_tpr"].max() - df["recall_tpr"].min()), 3),
        "fpr_gap": round(float(df["false_positive_rate"].max() - df["false_positive_rate"].min()), 3),
    }
    gaps["within_threshold"] = bool(all(v <= MAX_GAP for v in gaps.values()))
    return df, gaps


def main():
    model = joblib.load(config.MODEL_PATH)
    test = pd.read_csv(config.REPORTS / "test_split.csv")
    imp = explainability(model, test)
    print("Top drivers (share of SHAP importance):")
    for k, v in list(imp.items())[:6]:
        print(f"  {k:<24} {v:.1%}")
    table, gaps = fairness(model, test)
    print(f"\nFairness across '{config.SENSITIVE}':")
    print(table.to_string(index=False))
    print("\nGaps (max - min across groups):", gaps)
    print("RESULT:", "within the 20pt threshold" if gaps["within_threshold"] else "REVIEW: gap above threshold")
    small = table[table["n_positive"] < 30]["group"].tolist()
    if small:
        print(f"CAVEAT: groups with fewer than 30 true escalations ({small}) give noisy recall estimates;"
              " collect more data before acting on these gaps.")
    (config.REPORTS / "responsible_ai.json").write_text(json.dumps(
        {"shap_importance": imp, "fairness_group": config.SENSITIVE,
         "fairness_table": table.to_dict("records"), "gaps": gaps, "threshold": MAX_GAP}, indent=2))
    print("Wrote reports/responsible_ai.json and reports/shap_importance.png")


if __name__ == "__main__":
    main()
