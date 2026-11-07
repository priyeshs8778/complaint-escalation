"""Phase 7: Streamlit dashboard over real project outputs. Run: streamlit run dashboard.py"""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from src import config  # noqa: E402
from src.explainer import explain  # noqa: E402

st.set_page_config(page_title="Complaint Escalation Dashboard", layout="wide")
st.title("Complaint Escalation Risk Dashboard")


def load_json(name):
    p = config.REPORTS / name
    return json.loads(p.read_text()) if p.exists() else None


tab_pred, tab_model, tab_drift, tab_rai = st.tabs(
    ["Batch predictions", "Model metrics", "Drift", "Responsible AI"])

with tab_pred:
    p = config.REPORTS / "batch_predictions.csv"
    if not p.exists():
        st.warning("No batch predictions yet. Run `make batch`.")
    else:
        df = pd.read_csv(p)
        c1, c2, c3 = st.columns(3)
        threshold = st.slider("Escalation probability threshold", 0.0, 1.0, 0.5, 0.05)
        flagged = df[df["escalation_probability"] >= threshold]
        c1.metric("Complaints scored", len(df))
        c2.metric("Flagged at threshold", len(flagged))
        c3.metric("Flag rate", f"{len(flagged) / len(df):.1%}")
        channel = st.multiselect("Channel", sorted(df["channel"].unique()), default=sorted(df["channel"].unique()))
        view = flagged[flagged["channel"].isin(channel)].sort_values("escalation_probability", ascending=False)
        st.dataframe(view[["channel", "product_category", "order_value", "prior_complaints",
                           "response_wait_hours", "complaint_text", "escalation_probability"]],
                     hide_index=True)
        st.bar_chart(view.groupby("product_category")["escalation_probability"].count())

        st.subheader("Explain one complaint")
        if len(view):
            i = st.selectbox("Row", view.index.tolist())
            row = df.loc[i].to_dict()
            st.json(explain(row, float(row["escalation_probability"])))

with tab_model:
    m = load_json("metrics.json")
    if m:
        cols = st.columns(len(m["metrics"]))
        for col, (k, v) in zip(cols, m["metrics"].items()):
            col.metric(k, v)
        st.caption(f"Trained on {m['n_train']} rows, tested on {m['n_test']} rows. Params: {m['params']}")
    else:
        st.warning("No metrics found. Run `make train`.")

with tab_drift:
    d = load_json("drift_report.json")
    if d:
        (st.error if d["alert"] else st.success)(
            f"{'ALERT' if d['alert'] else 'OK'}: {len(d['drifted_features'])} of {len(d['features'])} "
            f"features drifted ({d['current_file']})")
        st.dataframe(pd.DataFrame(d["features"]).T[["type", "psi", "p_value", "drift"]])
    else:
        st.warning("No drift report. Run `make drift`.")

with tab_rai:
    r = load_json("responsible_ai.json")
    if r:
        img = config.REPORTS / "shap_importance.png"
        if img.exists():
            st.image(str(img))
        st.subheader(f"Fairness across {r['fairness_group']}")
        st.dataframe(pd.DataFrame(r["fairness_table"]), hide_index=True)
        st.write("Gaps:", r["gaps"])
    else:
        st.warning("No analysis found. Run `make rai`.")
