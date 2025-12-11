#!/usr/bin/env bash

uv run mypy src/torchxai/explainers --follow-imports=skip     # type check
uv run ruff check src/torchxai/explainers     # linter
uv run ruff format src/torchxai/explainers --check $@ # formatter $@