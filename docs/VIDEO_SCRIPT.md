# Recording script (target 11:30, hard limit 12:00)

Before recording: run `make all` once so `reports/` is full. Open and name these in advance: VS Code (repo),
two terminals (T1 commands, T2 API server already running via `make api`), browser tabs for GitHub repo,
GitHub Actions green run, `localhost:8000/docs`, Prometheus `localhost:9090`, drift_report.html, Streamlit.
Hide `.env`. Never show API keys. Record the whole screen with webcam on.

| Time | Say | Show |
|---|---|---|
| 0:00-0:45 | "I am <NAME>, BITS ID <ID>. This is my complaint escalation MLOps project." | Webcam, intro slide, GitHub repo URL, `git log -1` commit hash |
| 0:45-2:15 | "Synthetic complaints data is validated with 12 checks, then processed." | `head data/raw/complaints_raw.csv`; T1: `python -m src.pipeline`; point at PASS lines and processed output |
| 2:15-3:45 | "Random Forest on tabular plus TF-IDF text; metrics logged to MLflow." | `src/train.py` for 20 seconds; `reports/metrics.json` (AUC about 0.90); `ls -lh models/`; MLflow run |
| 3:45-5:15 | "One live API call, one batch run." | `localhost:8000/docs` POST /predict, then T1: `make batch` and the predictions table |
| 5:15-7:00 | "CI runs lint, pipeline, training, tests and Docker build; the container is deployed." | Green GitHub Actions run; T1: `docker ps` (api + prometheus up) |
| 7:00-8:30 | "Prometheus scrapes API metrics; drift is checked against training data." | Prometheus query `predict_requests_total`; T1: `make drift` showing ALERT; open `drift_report.html` |
| 8:30-9:45 | "SHAP shows text is the main driver; fairness is checked across product category." | `shap_importance.png`; fairness table; mention the flagged recall gap and small samples honestly |
| 9:45-10:45 | "The dashboard reads these real outputs." | Streamlit: move the threshold slider, open the Drift tab |
| 10:45-12:00 | "From a clean state, one command reproduces everything." | README golden-path section; T1: show `make all` from a prior clean run log or run it (about 20 s) and scroll the evidence list |

Tips: use the pre-generated logs for CI and Docker build if slow; keep each explanation to what it does, what ran, what
the output proves.
