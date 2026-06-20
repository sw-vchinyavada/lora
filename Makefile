# LoRA Credit Scoring — Simple commands
# Uses project venv when present (no need to activate first)

PY := $(shell test -f venv/bin/python && echo $(CURDIR)/venv/bin/python || echo python3)
.PHONY: install train run mfi-api mfi-front clean clean-start docker-train docker-up help

install:
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install sqlalchemy python-jose[cryptography] bcrypt

clean:
	@echo "Cleaning generated files..."
	@rm -f models/baseline/*.pkl models/lora/*.pt models/lora/*.bin models/preprocessor.pkl models/dataset_name.txt 2>/dev/null || true
	@rm -f results/metrics/*.json results/metrics/*.csv results/figures/*.png results/reports/*.html 2>/dev/null || true
	@rm -f data/raw/*.csv data/raw/*.zip data/synthetic/*.csv mfi_portal/data/*.db 2>/dev/null || true
	@find . -type d -name __pycache__ -not -path "./venv/*" 2>/dev/null | xargs rm -rf 2>/dev/null || true
	@echo "Done."

train:
	$(PY) scripts/train.py --dataset zimbabwe_synthetic --n-samples 50000 --epochs 15 --export-figures

train-quick:
	$(PY) scripts/train.py --dataset zimbabwe_synthetic --n-samples 5000 --epochs 10 --export-figures

export-figures:
	$(PY) scripts/export_results.py

run:
	$(PY) -m app.main

mfi-api:
	cd mfi_portal/backend && $(PY) -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

mfi-front:
	cd mfi_portal/frontend && npm run dev

docker-train:
	docker compose --profile train run --rm train

docker-up:
	docker compose up -d app

clean-start:
	@chmod +x scripts/clean_start.sh && ./scripts/clean_start.sh

help:
	@echo "clean       - remove generated files (models, results, data)"
	@echo "clean-start - full clean + instructions (./scripts/clean_start.sh)"
	@echo "install     - pip install deps"
	@echo "train       - train models"
	@echo "run         - launch demo UI (:7860)"
	@echo "mfi-api     - MFI portal API (:8000)"
	@echo "mfi-front   - MFI portal frontend (:5174)"
	@echo "docker-train - train (Docker)"
	@echo "docker-up   - start UI (Docker)"
