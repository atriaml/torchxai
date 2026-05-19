# TorchXAI

TorchXAI is a lightweight PyTorch toolkit for evaluating machine learning models using explainability techniques. It offers efficient implementations of explainability metrics that integrate seamlessly with the Captum ecosystem, with a focus on batch computation and task/data-agnostic evaluation to make scalable XAI evaluation easy.

## Installation

```bash
pip install torchxai-tools
```

The PyPI distribution is named `torchxai-tools`; the import name is `torchxai`.

```python
from torchxai.explainers import SaliencyExplainer   # import name is torchxai
```

## Quick start

### Generating explanations

```python
import torch
import torch.nn as nn
from torchxai.explainers import SaliencyExplainer, IntegratedGradientsExplainer
from torchxai.data_types import SingleTargetAcrossBatch

model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 3))
model.eval()
inputs = torch.randn(1, 10)

# Single target
explainer = SaliencyExplainer(model)
attrs = explainer.explain(inputs=inputs, target=SingleTargetAcrossBatch(index=0))
print(attrs.shape)   # (1, 10)

# All three classes in one call
explainer_mt = SaliencyExplainer(model, multi_target=True)
targets = [SingleTargetAcrossBatch(index=i) for i in range(3)]
attrs_list = explainer_mt.explain(inputs=inputs, target=targets)
print(len(attrs_list), attrs_list[0].shape)   # 3, (1, 10)
```

### With a baseline (IntegratedGradients, DeepLift, …)

```python
from torchxai.explainers import IntegratedGradientsExplainer

baseline = torch.zeros_like(inputs)
explainer = IntegratedGradientsExplainer(model)
attrs = explainer.explain(
    inputs=inputs,
    baselines=baseline,
    target=SingleTargetAcrossBatch(index=0),
)
```

### Evaluating explanation quality

```python
from torchxai.metrics.axiomatic import completeness
from captum.attr import Saliency

net = ...   # your model
saliency = Saliency(net)
input = torch.randn(2, 3, 32, 32, requires_grad=True)
baselines = torch.zeros(2, 3, 32, 32)

attribution = saliency.attribute(input, target=3)
score = completeness(net, input, attribution, baselines)
print("Completeness:", score)
```

## Supported explainers

| Explainer | Type | `baselines` | Baseline distribution | `feature_mask` | `sliding_window_shapes` |
|-----------|------|:-----------:|:---------------------:|:--------------:|:-----------------------:|
| `SaliencyExplainer` | Gradient | ✗ | ✗ | ✗ | ✗ |
| `InputXGradientExplainer` | Gradient | ✗ | ✗ | ✗ | ✗ |
| `GuidedBackpropExplainer` | Gradient | ✗ | ✗ | ✗ | ✗ |
| `RandomExplainer` | Baseline | ✗ | ✗ | ✗ | ✗ |
| `IntegratedGradientsExplainer` | Gradient | ✓ | ✗ | ✗ | ✗ |
| `DeepLiftExplainer` | Gradient | ✓ | ✗ | ✗ | ✗ |
| `InputXBaselineGradientExplainer` | Gradient | ✓ | ✗ | ✗ | ✗ |
| `DeepLiftShapExplainer` | Gradient | ✓ | ✓ | ✗ | ✗ |
| `GradientShapExplainer` | Gradient | ✓ | ✓ | ✗ | ✗ |
| `FeatureAblationExplainer` | Perturbation | ✓ | ✗ | optional | ✗ |
| `LimeExplainer` | Perturbation | ✓ | ✗ | optional | ✗ |
| `KernelShapExplainer` | Perturbation | ✓ | ✗ | optional | ✗ |
| `OcclusionExplainer` | Perturbation | ✓ | ✗ | ✗ | ✓ |

## Supported metrics
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

## Documentation

Full documentation including per-explainer API reference and end-to-end examples (image classification, BERT sequence classification, NER):

**[saifullah3396.github.io/torchxai](https://saifullah3396.github.io/torchxai/)**

## License

MIT — see [LICENSE.txt](LICENSE.txt).
