.PHONY: setup setup-tm synth collect ingest panel features bakeoff bakeoff-tm report test lint all clean

PY ?= python

setup:
	$(PY) -m pip install -e ".[dev]"

# Tsetlin needs numpy<2 -> keep it in its own venv (.venv-tm).
setup-tm:
	py -3.12 -m venv .venv-tm
	.venv-tm/Scripts/python -m pip install -U pip
	.venv-tm/Scripts/python -m pip install "numpy<2" "scikit-learn==1.5.2" pandas pyarrow pyyaml python-dotenv xgboost lightgbm matplotlib tmu==0.8.3

bakeoff-tm:
	.venv-tm/Scripts/python -m src.models.bakeoff --config config/bakeoff.yaml
	.venv-tm/Scripts/python -m src.analysis.clauses --config config/bakeoff.yaml

synth:
	$(PY) -m src.ingest.make_synthetic --n-matches 60

collect:
	$(PY) -m src.ingest.collect --config config/collect.yaml

ingest:
	$(PY) -m src.ingest.odds_api --config config/ingest.yaml

panel:
	$(PY) -m src.panel.build_panel --config config/features.yaml

features:
	$(PY) -m src.features.booleanize --config config/features.yaml

bakeoff:
	$(PY) -m src.models.bakeoff --config config/bakeoff.yaml

report:
	$(PY) -m src.analysis.clauses --config config/bakeoff.yaml

# full dry run on synthetic data
all: synth panel features bakeoff report

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests

clean:
	rm -rf data/raw/* data/panel/* data/features/* results/bakeoff_*.json
