# Complaint Escalation Risk: End-to-End MLOps

A production-style MLOps project that predicts whether an e-commerce customer complaint will escalate
(legal threats, chargebacks, social media blow-ups). It covers all eight capstone phases: data pipeline,
training, real-time and batch serving, CI/CD, monitoring and drift, Responsible AI, dashboard, and a
one-command golden path.

**Author:** <YOUR FULL NAME> | **BITS ID:** <YOUR BITS ID>

## Scope

This repo is the deployable core of a larger hybrid system (Random Forest + CNN + BiLSTM + LLM explainer).
For this capstone it ships the Random Forest on tabular and TF-IDF text features, plus a rule-based explainer.
The CNN and BiLSTM are intentionally excluded to keep the image small and CI fast. The optional LLM explainer is
off by default and needs `ANTHROPIC_API_KEY` plus `USE_LLM_EXPLAINER=1`. The data is synthetic and generated
reproducibly (seed 42).

## Architecture

```
generate_data -> pipeline (validate, process) -> train (RF + MLflow) -> models/escalation_model.joblib
                                                                    |-> FastAPI /predict  (+ /metrics -> Prometheus)
                                                                    |-> predict_batch (CSV in, CSV out)
reference data + new batch -> drift (PSI, KS, chi-square) -> drift_report.{json,html}
held-out test split -> responsible_ai (SHAP + fairness) -> responsible_ai.json, shap_importance.png
reports/* -> Streamlit dashboard
```

## Golden path (clean clone to full evidence)

```bash
git clone <REPO_URL> && cd complaint-escalation-mlops
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
make setup
make all
```

`make all` wipes generated state, then runs data, train, batch, drift, rai and tests. Evidence lands in `reports/`.
Then:

```bash
make dashboard        # Streamlit UI at http://localhost:8501
make api              # FastAPI at http://localhost:8000/docs
make up               # Docker: API + Prometheus (http://localhost:9090)
```

No `make`? Use the commands in the Makefile directly (for example `python -m src.pipeline`).

## Phases

| Phase | Command | Output |
|---|---|---|
| 1. Data pipeline | `make data` | `data/processed/complaints_processed.csv`, `reports/validation_report.json` (12 checks) |
| 2. Training | `make train` | `models/escalation_model.joblib`, `reports/metrics.json`, MLflow run in `mlflow.db` |
| 3. Serving | `make api` / `make batch` | `POST /predict`, `reports/batch_predictions.csv` |
| 4. CI/CD and infra | `.github/workflows/ci.yml`, `Dockerfile`, `docker-compose.yml` | lint, pipeline, train, tests, image build, container smoke test |
| 5. Monitoring and drift | `/metrics`, `make drift` | Prometheus metrics, `reports/drift_report.html` |
| 6. Responsible AI | `make rai` | SHAP importance chart, fairness table across `product_category` |
| 7. Dashboard | `make dashboard` | four tabs reading real `reports/` outputs |
| 8. Golden path | `make all` | everything above from a clean state |

## Try the API

```bash
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' -d '{
  "order_value": 320, "days_since_order": 12, "prior_complaints": 3,
  "customer_tenure_months": 8, "response_wait_hours": 40,
  "product_category": "electronics", "channel": "social",
  "complaint_text": "I will contact a lawyer, refund still not received"}'
```

## Monitoring and drift

- Runtime: request count, latency histogram and predicted-escalation counter at `/metrics`; every prediction is
  appended to `reports/prediction_log.jsonl`.
- Drift: `python -m src.drift --current <csv>` compares a batch with the training reference using PSI, a KS test
  (numeric) and chi-square (categorical). Alert rule: PSI >= 0.2 or p < 0.01. It exits with code 2 on alert.
  `data/batch/drifted_batch.csv` simulates a bad week and triggers the alert; `sample_batch.csv` does not.

## Responsible AI

- **Explainability:** SHAP on the Random Forest, aggregated to input columns. Complaint text is the top driver.
- **Fairness:** selection rate, recall and false-positive rate per product category, with gap checks against a
  20-point threshold. The recall gap currently exceeds it, driven by small groups (beauty has 16 true
  escalations in the test split). The report flags this caveat rather than hiding it.

## Limitations

- Synthetic data: metrics show the pipeline works, not real-world accuracy.
- The Docker image and CI workflow are defined here and run on GitHub Actions; they are not part of `make all`.
- Secrets: nothing is hard-coded. Use `.env` (see `.env.example`) and GitHub Secrets.

## Layout

```
src/            pipeline, model, train, api, batch, drift, responsible_ai, explainer, monitoring
tests/          pipeline, drift and API tests (9)
dashboard.py    Streamlit app
monitoring/     Prometheus config
docs/           VIDEO_SCRIPT.md
```
