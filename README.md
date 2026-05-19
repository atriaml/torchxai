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

| Explainer | Requires baseline | Notes |
|---|:---:|---|
| `SaliencyExplainer` | ✗ | |
| `InputXGradientExplainer` | ✗ | |
| `GuidedBackpropExplainer` | ✗ | Not compatible with transformers |
| `RandomExplainer` | ✗ | Baseline for sanity-checking |
| `IntegratedGradientsExplainer` | ✓ | |
| `DeepLiftExplainer` | ✓ | Not compatible with transformers |
| `InputXBaselineGradientExplainer` | ✓ | |
| `DeepLiftShapExplainer` | ✓ distribution | Not compatible with transformers |
| `GradientShapExplainer` | ✓ distribution | |
| `FeatureAblationExplainer` | ✗ | Optional `feature_mask` |
| `LimeExplainer` | ✗ | Optional `feature_mask` |
| `KernelShapExplainer` | ✗ | Optional `feature_mask` |
| `OcclusionExplainer` | ✗ | Requires `sliding_window_shapes` |

## Documentation

Full documentation including per-explainer API reference and end-to-end examples (image classification, BERT sequence classification, NER):

**[saifullah3396.github.io/torchxai](https://saifullah3396.github.io/torchxai/)**

## License

MIT — see [LICENSE.txt](LICENSE.txt).
