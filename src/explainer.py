"""Human-readable reasons for a prediction (the 'explainer' part of the hybrid system).

Default is a deterministic rule-based explainer, so the project needs no API key.
If ANTHROPIC_API_KEY is set and USE_LLM_EXPLAINER=1, an LLM rewrites the reasons as a
short business note. That path is optional and not exercised by the tests or CI.
"""
import json
import os


def rule_based(row: dict, probability: float) -> dict:
    reasons = []
    text = str(row.get("complaint_text", "")).lower()
    if any(k in text for k in ["lawyer", "court", "fraud", "social media", "manager", "chargeback"]):
        reasons.append("Complaint text contains a high-risk threat or escalation phrase")
    if row.get("prior_complaints", 0) >= 2:
        reasons.append(f"Customer has {row['prior_complaints']} prior complaints")
    if row.get("response_wait_hours", 0) >= 24:
        reasons.append(f"Customer has waited {row['response_wait_hours']} hours for a response")
    if row.get("order_value", 0) >= 250:
        reasons.append(f"High order value ({row['order_value']})")
    if row.get("channel") == "social":
        reasons.append("Complaint was raised on a public social channel")
    level = "HIGH" if probability >= 0.7 else "MEDIUM" if probability >= 0.4 else "LOW"
    return {"risk_level": level,
            "reasons": reasons or ["No single dominant risk factor"],
            "recommended_action": {"HIGH": "Route to a senior agent immediately",
                                   "MEDIUM": "Prioritise in the next response cycle",
                                   "LOW": "Handle through the standard queue"}[level]}


def explain(row: dict, probability: float) -> dict:
    base = rule_based(row, probability)
    if os.getenv("USE_LLM_EXPLAINER") == "1" and os.getenv("ANTHROPIC_API_KEY"):
        try:
            import anthropic
            client = anthropic.Anthropic()
            msg = client.messages.create(
                model="claude-sonnet-5-5", max_tokens=300,
                messages=[{"role": "user", "content":
                           "Rewrite this complaint escalation assessment as a two sentence "
                           "business note. Return plain text only.\n" + json.dumps(base)}])
            base["llm_note"] = msg.content[0].text
        except Exception as e:  # noqa: BLE001
            base["llm_note_error"] = str(e)
    return base
