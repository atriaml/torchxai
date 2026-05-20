# Usage

## Generating explanations

All explainers share the same `explain()` interface. Choose an explainer, construct it with `multi_target=False` (default) or `multi_target=True`, then call `explain()`.

### Single target

```python
import torch
import torch.nn as nn
from torchxai.explainers import SaliencyExplainer
from torchxai.data_types import SingleTargetAcrossBatch

model = nn.Sequential(nn.Linear(10, 3), nn.ReLU())
model.eval()
inputs = torch.randn(1, 10)

explainer = SaliencyExplainer(model)
attrs = explainer.explain(inputs=inputs, target=SingleTargetAcrossBatch(index=0))
print(attrs.shape)   # (1, 10)
```

### Multiple targets in one call

```python
from torchxai.explainers import SaliencyExplainer
from torchxai.data_types import SingleTargetAcrossBatch

targets = [SingleTargetAcrossBatch(index=i) for i in range(3)]

explainer = SaliencyExplainer(model, multi_target=True)
attrs_list = explainer.explain(inputs=inputs, target=targets)
# attrs_list is a list[Tensor], one per target
for i, attr in enumerate(attrs_list):
    print(f"class {i}: {attr.shape}")   # (1, 10)
```

### With a baseline

Methods such as `IntegratedGradientsExplainer` and `DeepLiftExplainer` require a reference baseline:

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

### With a feature mask

Perturbation methods (`FeatureAblationExplainer`, `LimeExplainer`, `KernelShapExplainer`) accept an optional `feature_mask` to group elements into segments:

```python
from torchxai.explainers import FeatureAblationExplainer

# group every 2 features into one segment
feature_mask = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3, 4, 4]])

explainer = FeatureAblationExplainer(model)
attrs = explainer.explain(
    inputs=inputs,
    feature_mask=feature_mask,
    target=SingleTargetAcrossBatch(index=0),
)
```

---

## Evaluating explanation quality

TorchXAI provides axiomatic metrics to quantify how good an attribution is.

```python
from torchxai.metrics.axiomatic import completeness
from captum.attr import Saliency
import torch

net = ...   # your model
saliency = Saliency(net)
input = torch.randn(2, 3, 32, 32, requires_grad=True)
baselines = torch.zeros(2, 3, 32, 32)

attribution = saliency.attribute(input, target=3)
score = completeness(net, input, attribution, baselines)
print("Completeness:", score)
```

---

## API Reference

- [Explainers overview](explainers.md) — all explainer classes, input patterns, comparison table
- [Explainer examples](explainers/examples/image_classification.md) — end-to-end worked examples
