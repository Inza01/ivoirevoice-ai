PYTHON ?= python3.11
VENV ?= .venv
VENV_PYTHON := $(VENV)/bin/python
VENV_BIN := $(VENV)/bin
DIOULA_DATA_DIR ?= $(IVOIREVOICE_DIOULA_DATA_DIR)
ARTIFACTS_DIR ?= $(if $(IVOIREVOICE_ARTIFACTS_DIR),$(IVOIREVOICE_ARTIFACTS_DIR),../artifacts)

.PHONY: setup install-dev lint format typecheck test verify api ui audit-dioula
.PHONY: manifest-dioula curate-dioula compare-dioula-splits freeze-dioula-v01
.PHONY: validate-dioula-v01

setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV_PYTHON) -m pip install --upgrade pip

install-dev:
	$(VENV_PYTHON) -m pip install -e ".[core,data,api,ui,dev]"

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

audit-dioula:
	IVOIREVOICE_DIOULA_DATA_DIR="$(DIOULA_DATA_DIR)" \
	IVOIREVOICE_ARTIFACTS_DIR="$(ARTIFACTS_DIR)" \
	$(VENV_PYTHON) -m ivoirevoice.data.audit --config configs/data/dioula.yaml

manifest-dioula:
	IVOIREVOICE_DIOULA_DATA_DIR="$(DIOULA_DATA_DIR)" \
	IVOIREVOICE_ARTIFACTS_DIR="$(ARTIFACTS_DIR)" \
	$(VENV_PYTHON) -m ivoirevoice.data.manifest --config configs/data/dioula.yaml

curate-dioula:
	IVOIREVOICE_DIOULA_DATA_DIR="$(DIOULA_DATA_DIR)" \
	IVOIREVOICE_ARTIFACTS_DIR="$(ARTIFACTS_DIR)" \
	$(VENV_PYTHON) -m ivoirevoice.data.curation --config configs/data/dioula.yaml

compare-dioula-splits:
	IVOIREVOICE_DIOULA_DATA_DIR="$(DIOULA_DATA_DIR)" \
	IVOIREVOICE_ARTIFACTS_DIR="$(ARTIFACTS_DIR)" \
	$(VENV_PYTHON) -m ivoirevoice.data.split_comparison --config configs/data/dioula.yaml

freeze-dioula-v01:
	IVOIREVOICE_DIOULA_DATA_DIR="$(DIOULA_DATA_DIR)" \
	IVOIREVOICE_ARTIFACTS_DIR="$(ARTIFACTS_DIR)" \
	$(VENV_PYTHON) -m ivoirevoice.data.freeze --config configs/data/dioula.yaml

validate-dioula-v01:
	IVOIREVOICE_DIOULA_DATA_DIR="$(DIOULA_DATA_DIR)" \
	IVOIREVOICE_ARTIFACTS_DIR="$(ARTIFACTS_DIR)" \
	$(VENV_PYTHON) -m ivoirevoice.data.freeze --config configs/data/dioula.yaml --validate-only
