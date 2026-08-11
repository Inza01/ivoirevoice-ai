PYTHON ?= python3.11
VENV ?= .venv
VENV_PYTHON := $(VENV)/bin/python
VENV_BIN := $(VENV)/bin
DIOULA_DATA_DIR ?= $(IVOIREVOICE_DIOULA_DATA_DIR)
ARTIFACTS_DIR ?= $(if $(IVOIREVOICE_ARTIFACTS_DIR),$(IVOIREVOICE_ARTIFACTS_DIR),../artifacts)
MODEL_CACHE_DIR ?= $(if $(IVOIREVOICE_MODEL_CACHE_DIR),$(IVOIREVOICE_MODEL_CACHE_DIR),../cache/models)
CHECKPOINT_DIR ?= $(if $(IVOIREVOICE_CHECKPOINT_DIR),$(IVOIREVOICE_CHECKPOINT_DIR),../checkpoints)
DIOULA_PILOT_MODEL_PATH ?= $(IVOIREVOICE_DIOULA_PILOT_MODEL_PATH)
MODEL ?= whisper_tiny
EXPERIMENT_CONFIG := configs/experiments/baseline_dy_$(MODEL).yaml
UI_HOST ?= 127.0.0.1
UI_PORT ?= 7860
REVIEW_HOST ?= 127.0.0.1
REVIEW_PORT ?= 7861
TRAINING_REPORTS_DIR ?= reports/data
SMOKE_TRAINING_CONFIG := configs/experiments/smoke_overfit_whisper_tiny_dy.yaml
PILOT_TRAINING_CONFIG := configs/experiments/pilot_finetune_whisper_tiny_dy.yaml
FULL_TRAINING_CONFIG := configs/experiments/full_finetune_whisper_tiny_dy.yaml
CONFIRM_FINAL_HOLDOUT ?= $(IVOIREVOICE_CONFIRM_FINAL_HOLDOUT)

.PHONY: setup install-dev lint format typecheck test compile audit-repository
.PHONY: harness-check verify-fast verify
.PHONY: api ui audit-dioula
.PHONY: manifest-dioula curate-dioula compare-dioula-splits freeze-dioula-v01
.PHONY: validate-dioula-v01 check-ml-environment inspect-baseline-models
.PHONY: baseline-dy-smoke baseline-dy-pilot baseline-dy-full compare-dy-baselines
.PHONY: audit-dioula-training review-dioula-training smoke-overfit-dy
.PHONY: pilot-finetune-dy
.PHONY: full-finetune-preflight full-finetune-fp16-diagnostic
.PHONY: full-finetune-dev full-finetune-refit
.PHONY: evaluate-final-holdout-dy

setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV_PYTHON) -m pip install --upgrade pip

install-dev:
	$(VENV_PYTHON) -m pip install -e ".[core,data,ml,api,ui,dev]"

lint:
	$(VENV_BIN)/ruff check .

format:
	$(VENV_BIN)/ruff format .

typecheck:
	$(VENV_BIN)/mypy src scripts

test:
	$(VENV_BIN)/pytest

compile:
	$(VENV_PYTHON) -m compileall -q src scripts tests

audit-repository:
	$(VENV_PYTHON) scripts/audit_repository.py

harness-check:
	$(VENV_PYTHON) scripts/check_harness.py

verify-fast:
	$(VENV_PYTHON) scripts/verify_environment.py
	$(MAKE) compile
	$(MAKE) audit-repository
	$(MAKE) harness-check
	$(MAKE) lint
	$(MAKE) typecheck

verify: verify-fast
	$(MAKE) test

api:
	$(VENV_BIN)/uvicorn ivoirevoice.api.app:app --reload

ui:
	IVOIREVOICE_DIOULA_DATA_DIR="$(DIOULA_DATA_DIR)" \
	IVOIREVOICE_ARTIFACTS_DIR="$(ARTIFACTS_DIR)" \
	IVOIREVOICE_MODEL_CACHE_DIR="$(MODEL_CACHE_DIR)" \
	IVOIREVOICE_DIOULA_PILOT_MODEL_PATH="$(DIOULA_PILOT_MODEL_PATH)" \
	IVOIREVOICE_UI_HOST="$(UI_HOST)" \
	IVOIREVOICE_UI_PORT="$(UI_PORT)" \
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

check-ml-environment:
	IVOIREVOICE_ARTIFACTS_DIR="$(ARTIFACTS_DIR)" \
	$(VENV_PYTHON) -m ivoirevoice.evaluation.environment

inspect-baseline-models:
	IVOIREVOICE_ARTIFACTS_DIR="$(ARTIFACTS_DIR)" \
	$(VENV_PYTHON) -m ivoirevoice.evaluation.compatibility

baseline-dy-smoke:
	IVOIREVOICE_DIOULA_DATA_DIR="$(DIOULA_DATA_DIR)" \
	IVOIREVOICE_ARTIFACTS_DIR="$(ARTIFACTS_DIR)" \
	IVOIREVOICE_MODEL_CACHE_DIR="$(MODEL_CACHE_DIR)" \
	$(VENV_PYTHON) -m ivoirevoice.evaluation.baseline \
		--experiment $(EXPERIMENT_CONFIG) --level smoke

baseline-dy-pilot:
	IVOIREVOICE_DIOULA_DATA_DIR="$(DIOULA_DATA_DIR)" \
	IVOIREVOICE_ARTIFACTS_DIR="$(ARTIFACTS_DIR)" \
	IVOIREVOICE_MODEL_CACHE_DIR="$(MODEL_CACHE_DIR)" \
	$(VENV_PYTHON) -m ivoirevoice.evaluation.baseline \
		--experiment $(EXPERIMENT_CONFIG) --level pilot

baseline-dy-full:
	IVOIREVOICE_DIOULA_DATA_DIR="$(DIOULA_DATA_DIR)" \
	IVOIREVOICE_ARTIFACTS_DIR="$(ARTIFACTS_DIR)" \
	IVOIREVOICE_MODEL_CACHE_DIR="$(MODEL_CACHE_DIR)" \
	IVOIREVOICE_CONFIRM_FULL="$(CONFIRM_FULL)" \
	$(VENV_PYTHON) -m ivoirevoice.evaluation.baseline \
		--experiment $(EXPERIMENT_CONFIG) --level full

compare-dy-baselines:
	IVOIREVOICE_ARTIFACTS_DIR="$(ARTIFACTS_DIR)" \
	$(VENV_PYTHON) -m ivoirevoice.evaluation.comparison

audit-dioula-training:
	IVOIREVOICE_DIOULA_DATA_DIR="$(DIOULA_DATA_DIR)" \
	IVOIREVOICE_ARTIFACTS_DIR="$(ARTIFACTS_DIR)" \
	IVOIREVOICE_MODEL_CACHE_DIR="$(MODEL_CACHE_DIR)" \
	IVOIREVOICE_TRAINING_REPORTS_DIR="$(TRAINING_REPORTS_DIR)" \
	$(VENV_PYTHON) -m ivoirevoice.training.audit \
		--experiment $(SMOKE_TRAINING_CONFIG)

review-dioula-training:
	IVOIREVOICE_DIOULA_DATA_DIR="$(DIOULA_DATA_DIR)" \
	IVOIREVOICE_ARTIFACTS_DIR="$(ARTIFACTS_DIR)" \
	IVOIREVOICE_MODEL_CACHE_DIR="$(MODEL_CACHE_DIR)" \
	IVOIREVOICE_TRAINING_REPORTS_DIR="$(TRAINING_REPORTS_DIR)" \
	IVOIREVOICE_REVIEW_HOST="$(REVIEW_HOST)" \
	IVOIREVOICE_REVIEW_PORT="$(REVIEW_PORT)" \
	$(VENV_PYTHON) -m ivoirevoice.training.manual_review \
		--experiment $(SMOKE_TRAINING_CONFIG)

smoke-overfit-dy:
	IVOIREVOICE_DIOULA_DATA_DIR="$(DIOULA_DATA_DIR)" \
	IVOIREVOICE_ARTIFACTS_DIR="$(ARTIFACTS_DIR)" \
	IVOIREVOICE_MODEL_CACHE_DIR="$(MODEL_CACHE_DIR)" \
	IVOIREVOICE_TRAINING_REPORTS_DIR="$(TRAINING_REPORTS_DIR)" \
	$(VENV_PYTHON) -m ivoirevoice.training.smoke_overfit \
		--experiment $(SMOKE_TRAINING_CONFIG)

pilot-finetune-dy:
	IVOIREVOICE_DIOULA_DATA_DIR="$(DIOULA_DATA_DIR)" \
	IVOIREVOICE_ARTIFACTS_DIR="$(ARTIFACTS_DIR)" \
	IVOIREVOICE_MODEL_CACHE_DIR="$(MODEL_CACHE_DIR)" \
	IVOIREVOICE_CHECKPOINT_DIR="$(CHECKPOINT_DIR)" \
	$(VENV_PYTHON) -m ivoirevoice.training.pilot_finetune \
		--experiment $(PILOT_TRAINING_CONFIG)

full-finetune-preflight:
	IVOIREVOICE_DIOULA_DATA_DIR="$(DIOULA_DATA_DIR)" \
	IVOIREVOICE_ARTIFACTS_DIR="$(ARTIFACTS_DIR)" \
	IVOIREVOICE_MODEL_CACHE_DIR="$(MODEL_CACHE_DIR)" \
	IVOIREVOICE_CHECKPOINT_DIR="$(CHECKPOINT_DIR)" \
	IVOIREVOICE_DIOULA_PILOT_MODEL_PATH="$(DIOULA_PILOT_MODEL_PATH)" \
	$(VENV_PYTHON) -m ivoirevoice.training.full_finetune \
		--config $(FULL_TRAINING_CONFIG) --stage preflight

full-finetune-fp16-diagnostic:
	IVOIREVOICE_DIOULA_DATA_DIR="$(DIOULA_DATA_DIR)" \
	IVOIREVOICE_ARTIFACTS_DIR="$(ARTIFACTS_DIR)" \
	IVOIREVOICE_MODEL_CACHE_DIR="$(MODEL_CACHE_DIR)" \
	IVOIREVOICE_CHECKPOINT_DIR="$(CHECKPOINT_DIR)" \
	IVOIREVOICE_DIOULA_PILOT_MODEL_PATH="$(DIOULA_PILOT_MODEL_PATH)" \
	$(VENV_PYTHON) -m ivoirevoice.training.full_finetune \
		--config $(FULL_TRAINING_CONFIG) --stage fp16-diagnostic

full-finetune-dev:
	IVOIREVOICE_DIOULA_DATA_DIR="$(DIOULA_DATA_DIR)" \
	IVOIREVOICE_ARTIFACTS_DIR="$(ARTIFACTS_DIR)" \
	IVOIREVOICE_MODEL_CACHE_DIR="$(MODEL_CACHE_DIR)" \
	IVOIREVOICE_CHECKPOINT_DIR="$(CHECKPOINT_DIR)" \
	IVOIREVOICE_DIOULA_PILOT_MODEL_PATH="$(DIOULA_PILOT_MODEL_PATH)" \
	$(VENV_PYTHON) -m ivoirevoice.training.full_finetune \
		--config $(FULL_TRAINING_CONFIG) --stage development

full-finetune-refit:
	IVOIREVOICE_DIOULA_DATA_DIR="$(DIOULA_DATA_DIR)" \
	IVOIREVOICE_ARTIFACTS_DIR="$(ARTIFACTS_DIR)" \
	IVOIREVOICE_MODEL_CACHE_DIR="$(MODEL_CACHE_DIR)" \
	IVOIREVOICE_CHECKPOINT_DIR="$(CHECKPOINT_DIR)" \
	IVOIREVOICE_DIOULA_PILOT_MODEL_PATH="$(DIOULA_PILOT_MODEL_PATH)" \
	$(VENV_PYTHON) -m ivoirevoice.training.full_finetune \
		--config $(FULL_TRAINING_CONFIG) --stage refit

evaluate-final-holdout-dy:
	IVOIREVOICE_DIOULA_DATA_DIR="$(DIOULA_DATA_DIR)" \
	IVOIREVOICE_ARTIFACTS_DIR="$(ARTIFACTS_DIR)" \
	IVOIREVOICE_MODEL_CACHE_DIR="$(MODEL_CACHE_DIR)" \
	IVOIREVOICE_CHECKPOINT_DIR="$(CHECKPOINT_DIR)" \
	IVOIREVOICE_DIOULA_PILOT_MODEL_PATH="$(DIOULA_PILOT_MODEL_PATH)" \
	IVOIREVOICE_CONFIRM_FINAL_HOLDOUT="$(CONFIRM_FINAL_HOLDOUT)" \
	$(VENV_PYTHON) -m ivoirevoice.training.full_finetune \
		--config $(FULL_TRAINING_CONFIG) --stage final-evaluation
