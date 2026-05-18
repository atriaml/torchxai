#!/usr/bin/env bash

dirs=(
  "data_types"
  "explainers"
  "metrics"
)

for dir in "${dirs[@]}"; do
  uv run mypy src/torchxai/$dir --follow-imports=skip     # type check
  uv run ruff check src/torchxai/$dir     # linter
  uv run ruff format src/torchxai/$dir --check $@ # formatter $@
done