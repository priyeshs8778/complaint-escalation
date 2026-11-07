PY ?= python
export MLFLOW_DISABLE_AGENT_HINT=1

.PHONY: setup data train batch drift rai test lint api dashboard up down all clean

setup:
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) -m src.generate_data
	$(PY) -m src.generate_data --drifted
	$(PY) -m src.pipeline

train:
	$(PY) -m src.train

batch:
	$(PY) -m src.predict_batch --input data/batch/sample_batch.csv

drift:
	-$(PY) -m src.drift --current data/batch/drifted_batch.csv

rai:
	$(PY) -m src.responsible_ai

lint:
	flake8 src tests dashboard.py --max-line-length 120 --extend-ignore=E203,W503

test:
	$(PY) -m pytest -q

api:
	uvicorn src.api:app --port 8000

dashboard:
	streamlit run dashboard.py

up:
	docker compose up -d --build

down:
	docker compose down

# Golden path: from a clean clone to every evidence artifact, one command.
all: clean data train batch drift rai test
	@echo ""
	@echo "GOLDEN PATH COMPLETE. Evidence in reports/:"
	@ls reports
	@echo "Next: 'make dashboard' (UI) and 'make up' (Docker + Prometheus)."

clean:
	rm -rf data/raw data/processed data/batch reports models mlflow.db mlruns
	mkdir -p data/raw data/processed data/batch reports models
