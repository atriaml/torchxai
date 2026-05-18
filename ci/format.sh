#!/usr/bin/env bash

dirs=(
  "data_types"
  "explainers"
  "metrics"
)

for dir in "${dirs[@]}"; do
  uv run ruff check src/torchxai/$dir --fix --unsafe-fixes $@   # linter with unsafe fixes for B905
  uv run ruff format src/torchxai/$dir # formatter
done
