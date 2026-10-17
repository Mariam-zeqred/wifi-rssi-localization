PY ?= python3
export PYTHONPATH := src

.PHONY: all check lint test smoke data figures clean

# Everything the README reports: download (hash-checked) + models + figures + results.json
all: data figures

check: lint test

lint:
	ruff check src tests
	ruff format --check src tests

test:
	$(PY) -m pytest -q

# The full pipeline on a small synthetic building (no download, ~10 s). Proves the code runs.
smoke:
	$(PY) -m rssi_loc run --synthetic --out results/smoke --figures figures/smoke

# Download trainingData.csv and validationData.csv into data/ and verify their SHA-256.
data:
	$(PY) -m rssi_loc download --data data

# kNN + MLP on the real data: results/results.json and figures/*.png (a few minutes on a laptop).
figures:
	$(PY) -m rssi_loc run --data data --out results --figures figures

clean:
	rm -rf .pytest_cache .ruff_cache results/smoke figures/smoke
