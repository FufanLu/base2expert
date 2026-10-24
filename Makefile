PY ?= .venv/bin/python

.PHONY: dev test lint typecheck check push

dev:            ## local CPU env for tests
	python3 -m venv .venv && .venv/bin/pip install -q uv && .venv/bin/uv pip install --python .venv/bin/python -e ".[dev]"

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests $(wildcard tools pretrain)

typecheck:
	$(PY) -m mypy

check: lint typecheck test

push:           ## copy the code to the cluster (needs an open ssh master: ssh explorer true)
	bash cluster/push.sh
