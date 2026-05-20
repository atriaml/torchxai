# Metrics Overview

TorchXAI provides evaluation metrics that quantify how good an attribution is across five axes: **axiomatic correctness**, **faithfulness**, **complexity**, **robustness**, and **localization**.

---

## Metric Summary

Column key:

- **Perturbation Type** — *Ordered*: features removed in attribution-ranked order. *Unordered*: random subset removal. *—*: no perturbation needed.
- **Requires Model** — whether the model's forward function is called during evaluation.
- **Requires Baseline** — whether a reference input is needed.
- **FM** — feature mask support (group features into segments before evaluation).
- **MT** — efficient multi-target computation (✓) vs. must be run once per target (✗).
- **Chunking** — whether computation can be split across feature chunks for memory efficiency.
- **↑ / ↓** — direction in which a better attribution scores.

| Type | Metric | API | Perturbation | Requires Model | Requires Baseline | FM | MT | Chunking |
|------|--------|-----|:------------:|:--------------:|:-----------------:|:--:|:--:|:--------:|
| Axiomatic | [Completeness](metrics/completeness.md) ↓ | `completeness` | — | ✓ | ✓ | — | ✓ | ✗ |
| Axiomatic | [Non-Sensitivity](metrics/non_sensitivity.md) ↓ | `non_sensitivity` | Unordered | ✓ | ✓ | ✓ | ✓ | ✓ |
| Faithfulness | [Area Over Perturbation Curve](metrics/aopc.md) ↑ desc / ↓ asc | `aopc` | Ordered | ✓ | ✓ | ✓ | ✗ | ✓ |
| Faithfulness | [Area Between Perturbation Curves](metrics/abpc.md) ↑ | `abpc` | Ordered | ✓ | ✓ | ✓ | ✗ | ✓ |
| Faithfulness | [Faithfulness Correlation](metrics/faithfulness_corr.md) ↑ | `faithfulness_corr` | Unordered | ✓ | ✓ | ✓ | ✓ | ✓ |
| Faithfulness | [Faithfulness Estimation](metrics/faithfulness_estimate.md) ↑ | `faithfulness_estimate` | Ordered | ✓ | ✓ | ✓ | ✗ | ✓ |
| Faithfulness | [Infidelity](metrics/infidelity.md) ↓ | `infidelity` | Unordered | ✓ | ✗ | — | ✓ | — |
| Faithfulness | [Monotonicity](metrics/monotonicity.md) ↑ | `monotonicity` | Ordered | ✓ | ✓ | ✓ | ✗ | ✓ |
| Faithfulness | [Monotonicity Correlation](metrics/monotonicity_corr.md) ↑ | `monotonicity_corr` | Unordered | ✓ | ✓ | ✓ | ✓ | ✓ |
| Faithfulness | [Sensitivity-N](metrics/sensitivity_n.md) ↓ | `sensitivity_n` | Unordered | ✓ | ✓ | ✓ | ✓ | ✓ |
| Complexity | [Entropy-based Complexity](metrics/complexity_entropy.md) ↓ | `complexity_entropy` | — | ✗ | ✗ | ✓ | — | — |
| Complexity | [Sundararajan Complexity](metrics/complexity_sundararajan.md) ↓ | `complexity_sundararajan` | — | ✗ | ✗ | ✓ | — | — |
| Complexity | [Effective Complexity](metrics/effective_complexity.md) ↓ | `effective_complexity` | — | ✗ | ✗ | ✓ | — | — |
| Complexity | [Sparseness](metrics/sparseness.md) ↑ | `sparseness` | — | ✗ | ✗ | ✓ | — | — |
| Robustness | [Max Sensitivity](metrics/sensitivity_max.md) ↓ | `sensitivity_max` | Unordered | ✓ | ✗ | — | ✓ | — |
| Robustness | [Avg Sensitivity](metrics/sensitivity_avg.md) ↓ | `sensitivity_avg` | Unordered | ✓ | ✗ | — | ✓ | — |
| Localization | [Attribution Localization](metrics/attribution_localization.md) ↑ | `attribution_localization` | — | ✗ | ✗ | ✓ | — | — |

---

## Categories

### Axiomatic

Axiomatic metrics verify formal properties an attribution *should* satisfy by construction.

- **[Completeness](metrics/completeness.md)** (`completeness`) — attributions must sum to the output difference between input and baseline (summation-to-delta / conservation property).
- **[Non-Sensitivity](metrics/non_sensitivity.md)** (`non_sensitivity`) — fraction of features with zero attribution that nonetheless cause a model output change when perturbed; lower is better.

### Faithfulness

Faithfulness metrics measure whether high-attribution features actually drive model predictions.

- **[Area Over Perturbation Curve](metrics/aopc.md)** (`aopc`) — average model output drop as features are removed in descending, ascending, and random attribution order.
- **[Area Between Perturbation Curves](metrics/abpc.md)** (`abpc`) — gap between the descending and ascending AOPC curves; a larger gap indicates a more faithful ranking of features.
- **[Faithfulness Correlation](metrics/faithfulness_corr.md)** (`faithfulness_corr`) — Pearson correlation between attribution magnitudes and the change in model output when each feature is masked out.
- **[Faithfulness Estimation](metrics/faithfulness_estimate.md)** (`faithfulness_estimate`) — output change when progressively removing features in ascending attribution order; measures necessity of attributed features.
- **[Infidelity](metrics/infidelity.md)** (`infidelity`) — mean-squared error between attribution-weighted perturbation magnitudes and actual model output changes.
- **[Monotonicity](metrics/monotonicity.md)** (`monotonicity`) — fraction of features for which model output decreases monotonically as features are added back from a baseline.
- **[Monotonicity Correlation](metrics/monotonicity_corr.md)** (`monotonicity_corr`) — Spearman correlation between attribution magnitudes and output variance under unordered random perturbations.
- **[Sensitivity-N](metrics/sensitivity_n.md)** (`sensitivity_n`) — Pearson correlation between attributions and output change across random n-feature perturbations.

### Complexity

Complexity metrics quantify how many features an explanation relies on; simpler explanations are easier to interpret.

- **[Entropy-based Complexity](metrics/complexity_entropy.md)** (`complexity_entropy`, `complexity_entropy_feature_grouped`) — Shannon entropy of fractional attribution contributions; lower = more concentrated / simpler explanation. The feature-grouped variant pools attributions within groups defined by a feature mask.
- **[Sundararajan Complexity](metrics/complexity_sundararajan.md)** (`complexity_sundararajan`, `complexity_sundararajan_feature_grouped`) — count of features (or feature groups) with non-zero attribution magnitude.
- **[Effective Complexity](metrics/effective_complexity.md)** (`effective_complexity`) — number of features whose attribution magnitude exceeds a threshold, i.e., the effective support of the explanation.
- **[Sparseness](metrics/sparseness.md)** (`sparseness`, `sparseness_feature_grouped`) — Gini index of attribution magnitudes; higher = more sparse / concentrated on fewer features. Feature-grouped variant available.

### Robustness

Robustness metrics test how stable explanations are under small input perturbations.

- **[Max Sensitivity](metrics/sensitivity_max.md)** (`sensitivity_max`) — worst-case attribution distance when the input is randomly perturbed within a small radius.
- **[Avg Sensitivity](metrics/sensitivity_avg.md)** (`sensitivity_avg`) — mean attribution distance over random input perturbations.

### Localization

Localization metrics measure whether attributions concentrate in the correct spatial region.

- **[Attribution Localization](metrics/attribution_localization.md)** (`attribution_localization`) — ratio of positive attribution mass inside a ground-truth region to total attribution magnitude.

---

## Quick start

```python
import torch
from torchxai.metrics import (
    completeness,
    non_sensitivity,
    aopc,
    abpc,
    monotonicity_corr,
    sensitivity_max,
    sensitivity_avg,
    sparseness,
)
from torchxai.explainers import SaliencyExplainer, IntegratedGradientsExplainer
from torchxai.data_types import SingleTargetAcrossBatch

model = torch.nn.Sequential(torch.nn.Linear(10, 5), torch.nn.ReLU(), torch.nn.Linear(5, 3))
model.eval()

inputs   = torch.randn(1, 10)
baseline = torch.zeros(1, 10)
target   = SingleTargetAcrossBatch(index=0)

# Attributions from Integrated Gradients
attrs = IntegratedGradientsExplainer(model).explain(
    inputs=inputs, baselines=baseline, target=target
)

# Axiomatic: completeness (should be near zero for IG)
score = completeness(model, inputs, attrs, baseline, target=target)
print("Completeness:", score.item())

# Faithfulness: monotonicity correlation
mc = monotonicity_corr(model, inputs, attrs, baseline, target=target)
print("Monotonicity corr:", mc.item())

# Complexity: sparseness (Gini index, ↑ better)
sparse = sparseness(attrs)
print("Sparseness:", sparse.item())

# Robustness: max and avg sensitivity (computed together)
max_sens = sensitivity_max(SaliencyExplainer(model), inputs, target=target)
avg_sens = sensitivity_avg(SaliencyExplainer(model), inputs, target=target)
print("Max sensitivity:", max_sens.item())
print("Avg sensitivity:", avg_sens.item())
```
