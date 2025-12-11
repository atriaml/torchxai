#!/usr/bin/env bash

uv run ruff check src --fix --unsafe-fixes $@   # linter with unsafe fixes for B905
uv run ruff format src # formatter
