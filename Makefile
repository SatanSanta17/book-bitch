.PHONY: run-backend run-frontend format lint test ingest-demo

ENV_FILE ?= .env

run-backend:
	uvicorn app.main:app --reload --env-file $(ENV_FILE)

run-frontend:
	streamlit run frontend/ui.py

format:
	pip install ruff black
	ruff check app frontend tests --fix
	black app frontend tests

lint:
	pip install ruff mypy
	ruff check app frontend tests
	mypy app

test:
	pytest

ingest-demo:
	python app/scripts/demo_ingest.py data/sample.pdf demo-book