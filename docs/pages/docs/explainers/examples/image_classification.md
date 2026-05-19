---
title: Image Classification
summary: Multi-target attribution examples on a simple CNN image classifier
---

# Image Classification Examples

These examples show how to use torchxai explainers on a minimal image classification model. Each section covers one **input pattern** — the minimal set of arguments that pattern of explainer requires. Within each pattern the examples show single-target attribution, multi-target attribution, and a verification that both give identical results for the same target.

## Setup

```python
import torch
import torch.nn as nn

from torchxai.data_types import SingleTargetAcrossBatch


class TinyCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Linear(32 * 4 * 4, num_classes)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


torch.manual_seed(0)
model    = TinyCNN().eval()
inputs   = torch.randn(1, 3, 32, 32)   # single 3-channel 32x32 image
baseline = torch.zeros_like(inputs)    # zero baseline (used where required)
```

## Explainer Input Requirements

| Explainer | `baselines` | `feature_mask` | `sliding_window_shapes` |
|---|:---:|:---:|:---:|
| `SaliencyExplainer` | ✗ | ✗ | ✗ |
| `InputXGradientExplainer` | ✗ | ✗ | ✗ |
| `GuidedBackpropExplainer` | ✗ | ✗ | ✗ |
| `RandomExplainer` | ✗ | ✗ | ✗ |
| `FeatureAblationExplainer` | ✗ | optional | ✗ |
| `LimeExplainer` | ✗ | optional | ✗ |
| `KernelShapExplainer` | ✗ | optional | ✗ |
| `IntegratedGradientsExplainer` | ✓ | ✗ | ✗ |
| `DeepLiftExplainer` | ✓ | ✗ | ✗ |
| `DeepLiftShapExplainer` | ✓ distribution | ✗ | ✗ |
| `GradientShapExplainer` | ✓ distribution | ✗ | ✗ |
| `OcclusionExplainer` | ✗ | ✗ | ✓ |

---

## Pattern A — inputs + target

No baseline or mask required. Applies to: `SaliencyExplainer`, `InputXGradientExplainer`, `GuidedBackpropExplainer`, `RandomExplainer`, `FeatureAblationExplainer`, `LimeExplainer`, `KernelShapExplainer`.

```python
from torchxai.explainers import SaliencyExplainer

# Replace SaliencyExplainer with any Pattern-A explainer above

# Single target: attributions for all 10 classes in 10 sequential calls
explainer = SaliencyExplainer(model, multi_target=False)
attrs = [
    explainer.explain(inputs=inputs, target=SingleTargetAcrossBatch(index=i))
    for i in range(10)
]
attrs_tensor = torch.stack(attrs)   # (10, 1, 3, 32, 32)
print("Single target collected attributions shape:", attrs_tensor.shape)

# Multi-target: same attributions in one call
explainer_mt = SaliencyExplainer(model, multi_target=True)
attrs_mt = explainer_mt.explain(
    inputs=inputs,
    target=[SingleTargetAcrossBatch(index=i) for i in range(10)],
)
attrs_mt_tensor = torch.stack(attrs_mt)   # (10, 1, 3, 32, 32)
assert torch.allclose(attrs_tensor, attrs_mt_tensor), f"Multi-target attributions do not match single-target attributions:\n{attrs_tensor}\n{attrs_mt_tensor}"
print("Multi-target attributions shape:", attrs_mt_tensor.shape)
```

---

## Pattern B — inputs + baseline + target

A single reference tensor (same shape as `inputs`) is required. Applies to: `IntegratedGradientsExplainer`, `DeepLiftExplainer`, `InputXBaselineGradientExplainer`.

```python
from torchxai.explainers import IntegratedGradientsExplainer

# Replace with DeepLiftExplainer or InputXBaselineGradientExplainer as needed

explainer = IntegratedGradientsExplainer(model, multi_target=False)
attrs = explainer.explain(
    inputs=inputs, baselines=baseline, target=SingleTargetAcrossBatch(index=0)
)
print("Single target attributions shape:", attrs.shape)

explainer_mt = IntegratedGradientsExplainer(model, multi_target=True)
attrs_mt = explainer_mt.explain(
    inputs=inputs,
    baselines=baseline,
    target=[SingleTargetAcrossBatch(index=0), SingleTargetAcrossBatch(index=1)],
)
assert torch.allclose(attrs, attrs_mt[0]), (
    "Multi-target attributions do not match single-target attributions for class 0"
)
```

---

## Pattern C — inputs + baseline distribution + target

`baselines` is a **stacked set of reference samples** rather than a single tensor. Applies to: `DeepLiftShapExplainer`, `GradientShapExplainer`.

```python
from torchxai.explainers import GradientShapExplainer
# Replace with DeepLiftShapExplainer as needed

baselines_dist = baseline.expand(5, -1, -1, -1)   # (5, 3, 32, 32) reference distribution

explainer = GradientShapExplainer(model, multi_target=False, n_samples=100)
attrs = explainer.explain(
    inputs=inputs,
    baselines=baselines_dist,
    target=SingleTargetAcrossBatch(index=0),
)
print(attrs.shape)   # (1, 3, 32, 32)

explainer_mt = GradientShapExplainer(model, multi_target=True, n_samples=100)
attrs_mt = explainer_mt.explain(
    inputs=inputs,
    baselines=baselines_dist,
    target=[SingleTargetAcrossBatch(index=idx) for idx in range(10)],
)

# for GradientShap, we use a higher tolerance due to the randomness in sampling from the baseline distribution
assert torch.allclose(attrs, attrs_mt[0], atol=1e-3), (
    "Multi-target attributions do not match single-target attributions for class 0",
    (attrs - attrs_mt[0]).abs().max().item()
)
```

---

## Pattern D — inputs + feature_mask + target

A `feature_mask` groups input elements into segments so the explainer scores whole regions rather than individual pixels. Applies to: `FeatureAblationExplainer`, `LimeExplainer`, `KernelShapExplainer`.

The same explainer class works with or without a mask; the mask is optional. Pixel-level attribution uses no mask; segment-level attribution passes one.

```python
from torchxai.explainers import FeatureAblationExplainer

# Replace with LimeExplainer or KernelShapExplainer as needed

# Build an 8x8 super-pixel grid — 64 segments over a 32x32 image
grid = 8
cell_size = 32 // grid
feature_mask = torch.zeros(1, 1, 32, 32, dtype=torch.long)
for i in range(grid):
    for j in range(grid):
        feature_mask[0, 0, i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size] = i * grid + j

print("Feature mask shape:", feature_mask.shape)  # (1, 1, 32, 32)
print("Total unique segments in feature mask:", feature_mask.unique().numel())  # Should be 64

explainer = FeatureAblationExplainer(model, multi_target=False)

# Without mask — pixel-level attribution
attrs_pixel = explainer.explain(
    inputs=inputs,
    target=SingleTargetAcrossBatch(index=0),
)
print(attrs_pixel.shape)   # (1, 3, 32, 32)

# With mask — segment-level attribution
attrs_seg = explainer.explain(
    inputs=inputs,
    feature_mask=feature_mask,
    target=SingleTargetAcrossBatch(index=0),
)
print(attrs_seg.shape)   # (1, 3, 32, 32) — one score per segment, broadcast back

# Multi-target with mask
explainer_mt = FeatureAblationExplainer(model, multi_target=True)
attrs_seg_mt = explainer_mt.explain(
    inputs=inputs,
    feature_mask=feature_mask,
    target=[SingleTargetAcrossBatch(index=idx) for idx in range(10)],
)
assert torch.allclose(attrs_seg, attrs_seg_mt[0]), "Multi-target attributions do not match single-target attributions for class 0"
```

---

## Pattern E — inputs + sliding_window_shapes + target

`OcclusionExplainer` patches out rectangular windows instead of using a feature mask. The `sliding_window_shapes` tuple specifies the window size per input channel.

```python
from torchxai.explainers import OcclusionExplainer

explainer = OcclusionExplainer(model, multi_target=False)
attrs = explainer.explain(
    inputs=inputs,
    sliding_window_shapes=(1, 8, 8),   # occlude one 8x8 patch per channel
    target=SingleTargetAcrossBatch(index=0),
)
print(attrs.shape)   # (1, 3, 32, 32)

explainer_mt = OcclusionExplainer(model, multi_target=True)
attrs_mt = explainer_mt.explain(
    inputs=inputs,
    sliding_window_shapes=(1, 8, 8),
    target=[SingleTargetAcrossBatch(index=idx) for idx in range(10)],
)
assert torch.allclose(attrs, attrs_mt[0])
```
