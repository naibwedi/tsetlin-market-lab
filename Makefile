.PHONY: setup synth collect ingest panel features bakeoff report test lint all clean

PY ?= python

setup:
	$(PY) -m pip install -e ".[dev]"
	@echo "For Tsetlin models (Linux/WSL/Docker): pip install -e '.[tm]'"

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
