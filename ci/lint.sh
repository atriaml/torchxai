#!/usr/bin/env bash

uv run mypy src --follow-imports=skip     # type check
uv run ruff check src     # linter
uv run ruff format src --check $@ # formatter $@