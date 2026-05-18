# Developer Instructions

This file documents setup, coding conventions, CI workflows, and contribution guidelines for TorchXAI contributors and maintainers.

---

## Environment Setup

### Prerequisites

- Python 3.11 (enforced in `pyproject.toml`)
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Install with uv (recommended)

```bash
uv sync          # install all deps including dev group
```

### Install with pip (editable)

```bash
pip install -e ".[dev]"
# or simply:
pip install -e .
```

---

## Running Tests

```bash
# All tests
pytest tests/

# Only metrics
pytest tests/metrics/ -m metrics

# Only explainers
pytest tests/explainers/ -m explainers

# Specific file with verbose output
pytest tests/metrics/faithfulness/test_infidelity.py -v

# With coverage
pytest tests/ --cov=src/torchxai --cov-report=term-missing
```

Test markers are defined in `pyproject.toml`:
- `metrics` — tests for XAI metrics
- `explainers` — tests for attribution methods

---

## Linting and Formatting

The project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting.

```bash
# Check lint
ruff check src/ tests/

# Auto-fix fixable issues
ruff check --fix src/ tests/

# Format code
ruff format src/ tests/

# CI scripts (used in GitHub Actions)
bash ci/explainers/lint.sh
bash ci/explainers/format.sh
```

---

## Type Checking

```bash
mypy src/torchxai
```

mypy configuration is in `pyproject.toml`. Import errors for untyped third-party packages are suppressed by default.

---

## Code Conventions

- Python 3.11+ syntax (use `|` union types, `match`, etc. where appropriate).
- All public functions and classes must have NumPy-style or Google-style docstrings.
- Module-level docstrings required for all public modules.
- Use `TensorOrTupleOfTensorsGeneric` and other type aliases from `torchxai.data_types._common` for type annotations.
- All `ExplanationTarget`-derived classes use Pydantic v2 validators.
- Attribution methods must handle both single-input and tuple-of-inputs consistently.
- Use `_format_tensor_into_tuples` from Captum internally when normalizing inputs.

---

## Project Layout Conventions

- Internal implementation modules are prefixed with `_` (e.g., `_explainer.py`, `_saliency.py`).
- Public re-exports are centralized in `__init__.py` files for each subpackage.
- Multi-target variants of metrics live in `metrics/<category>/multi_target/`.
- Tests mirror the `src/` structure under `tests/`.

---

## Adding a New Explainer

1. Create `src/torchxai/explainers/_grad/_my_method.py` (or `_perturbation/`).
2. Implement a class that inherits from `Explainer` and overrides `_init_single_target_explanation_fn()`.
3. Optionally override `_init_multi_target_explanation_fn()` for an efficient multi-target implementation.
4. Implement `explain()` with the correct signature (see existing explainers as reference).
5. Register the explainer in `_factory.py`:`AVAILABLE_EXPLAINERS["my_method"] = MyMethodExplainer`.
6. Add it to `explainers/__init__.py` exports.
7. Write tests in `tests/explainers/`.

---

## Adding a New Metric

1. Create `src/torchxai/metrics/<category>/my_metric.py`.
2. Implement `_my_metric(...)` (internal helper, accepts raw tensors).
3. Implement `my_metric(...)` (public API, handles `multi_target` and `return_dict` flags).
4. If multi-target is needed, create `src/torchxai/metrics/<category>/multi_target/my_metric.py` with `_multi_target_my_metric(...)`.
5. Add the function to `metrics/__init__.py` imports and `__all__`.
6. Write tests in `tests/metrics/<category>/test_my_metric.py`.

---

## Pending Tasks

The following are known issues / TODOs to address in future releases:

| Item | Location | Action |
|------|----------|--------|
| Export or remove ABPC metric | `metrics/faithfulness/abpc.py` | Add to `__init__.py` or delete |
| Export or remove Selectivity metric | `metrics/faithfulness/selectivity.py` | Add full implementation with batch support or delete |
| Expose NoiseTunnel explainer | `explainers/_grad/_noise_tunnel.py` | Register in factory and export, or document as internal only |
| Fill or remove `data_types/common.py` | `data_types/common.py` | Use or delete empty placeholder |
| Add top-level convenience exports | `torchxai/__init__.py` | Re-export main symbols for `import torchxai; torchxai.SaliencyExplainer` UX |
| Fill `instructures.md` | project root | ✅ Done |
