PYTHON ?= python3.11
VENV ?= .venv
VENV_PYTHON := $(VENV)/bin/python
VENV_BIN := $(VENV)/bin

.PHONY: setup install-dev lint format typecheck test verify api ui

setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV_PYTHON) -m pip install --upgrade pip

install-dev:
	$(VENV_PYTHON) -m pip install -e ".[core,api,ui,dev]"

lint:
	$(VENV_BIN)/ruff check .

format:
	$(VENV_BIN)/ruff format .

typecheck:
	$(VENV_BIN)/mypy src scripts

test:
	$(VENV_BIN)/pytest

verify:
	$(VENV_PYTHON) scripts/verify_environment.py
	$(MAKE) lint
	$(MAKE) typecheck
	$(MAKE) test

api:
	$(VENV_BIN)/uvicorn ivoirevoice.api.app:app --reload

ui:
	$(VENV_PYTHON) -m ivoirevoice.ui.app

