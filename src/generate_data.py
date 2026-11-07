"""Create a reproducible synthetic e-commerce complaints dataset.

Usage:
  python -m src.generate_data                 # raw training data
  python -m src.generate_data --drifted       # shifted batch used for the drift demo
"""
import argparse
import numpy as np
import pandas as pd
from src import config

CATEGORIES = ["electronics", "clothing", "home", "grocery", "beauty"]
CHANNELS = ["email", "chat", "phone", "social"]

HIGH_RISK = ["I will contact a lawyer", "reporting this to consumer court",
             "this is fraud and I want a chargeback", "I will post this on social media",
             "worst service ever and I demand a full refund", "escalate to your manager now"]
MID_RISK = ["the item arrived damaged", "package never arrived", "wrong item delivered",
            "refund still not received", "support has not replied"]
LOW_RISK = ["a small question about my order", "please update my delivery address",
            "the colour is slightly different", "how do I return an item",
            "just checking the delivery date"]


def _text(rng, risk):
    parts = []
    pool = HIGH_RISK if risk > 0.7 else MID_RISK if risk > 0.35 else LOW_RISK
    parts.append(rng.choice(pool))
    if rng.random() < 0.5:
        parts.append(rng.choice(MID_RISK + LOW_RISK))
    return ". ".join(parts) + "."


def generate(n=3000, seed=42, drifted=False):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "order_value": rng.gamma(2.5, 40, n).round(2),
        "days_since_order": rng.integers(1, 45, n),
        "prior_complaints": rng.poisson(0.8, n),
        "customer_tenure_months": rng.integers(1, 96, n),
        "response_wait_hours": rng.gamma(2.0, 8, n).round(1),
        "product_category": rng.choice(CATEGORIES, n, p=[.3, .25, .2, .15, .1]),
        "channel": rng.choice(CHANNELS, n, p=[.35, .3, .2, .15]),
    })
    if drifted:
        # Simulate a bad week: pricier orders, longer waits, more repeat complaints,
        # and a shift in the product and channel mix.
        df["order_value"] = (df["order_value"] * 1.9).round(2)
        df["response_wait_hours"] = (df["response_wait_hours"] * 2.2).round(1)
        df["prior_complaints"] = df["prior_complaints"] + rng.poisson(1.2, n)
        df["product_category"] = rng.choice(CATEGORIES, n, p=[.6, .1, .1, .1, .1])
        df["channel"] = rng.choice(CHANNELS, n, p=[.1, .1, .2, .6])

    z = (-3.0
         + 0.012 * df["order_value"]
         + 0.55 * df["prior_complaints"]
         + 0.05 * df["response_wait_hours"]
         + 0.02 * df["days_since_order"]
         - 0.01 * df["customer_tenure_months"]
         + df["channel"].map({"email": 0, "chat": 0, "phone": 0.3, "social": 0.9}))
    risk = 1 / (1 + np.exp(-z))
    df[config.TEXT] = [_text(rng, r + rng.normal(0, 0.15)) for r in risk]
    text_boost = df[config.TEXT].str.contains("lawyer|court|fraud|social media|manager", regex=True) * 1.2
    p = 1 / (1 + np.exp(-2.2 * (z + text_boost - 0.3)))
    df[config.TARGET] = (rng.random(n) < p).astype(int)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--drifted", action="store_true")
    ap.add_argument("--n", type=int, default=3000)
    a = ap.parse_args()
    if a.drifted:
        out = config.BATCH_DIR / "drifted_batch.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        generate(a.n // 3, seed=7, drifted=True).drop(columns=[config.TARGET]).to_csv(out, index=False)
    else:
        out = config.RAW
        out.parent.mkdir(parents=True, exist_ok=True)
        generate(a.n).to_csv(out, index=False)
        # A clean, unlabelled batch for the normal batch-inference demo
        generate(200, seed=11).drop(columns=[config.TARGET]).to_csv(
            config.BATCH_DIR / "sample_batch.csv", index=False)
    print(f"wrote {out}")
