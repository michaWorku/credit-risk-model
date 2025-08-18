.PHONY: setup lint test unit integration train api dashboard monitor
setup: ## install dev deps
	pip install -r requirements.txt -r requirements-dev.txt
	pre-commit install

lint:	pre-commit run --all-files

unit:	pytest -q tests/unit --cov=src --cov-report=xml
integration:	pytest -q tests/integration

test: lint unit integration

train:	python -m src.models.train --config configs/training.yaml
api:	uvicorn src.api.main:app --reload --port 8000
monitor:	python -m src.monitoring.drift