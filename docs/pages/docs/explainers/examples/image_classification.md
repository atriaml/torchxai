---
title: Image Classification
summary: Multi-target attribution examples on a simple CNN image classifier
---

# Image Classification Examples

These examples show how to use torchxai explainers on a minimal image classification model.
Each section covers one **input pattern** — the minimal set of arguments that pattern of explainer requires.
Each pattern uses the shared `compare()` helper to run single-target sequentially, then multi-target in one call, verify they match, and report timing.

## Setup

```python
import torch
import torch.nn as nn
import time
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
inputs   = torch.randn(1, 3, 32, 32)   # single 3-channel 32×32 image
baseline = torch.zeros_like(inputs)    # zero baseline (used where required)

# Targets for all 10 output classes
targets = [SingleTargetAcrossBatch(index=i) for i in range(10)]


def compare(explainer_cls, model, explain_kwargs, targets, atol=1e-5, **init_kwargs):
    """Compare sequential single-target calls vs one multi-target call.

    Verifies results match and reports timing and attribution shape.
    Pass atol>1e-5 for stochastic methods (e.g. GradientShap).
    """
    explainer = explainer_cls(model, multi_target=False, **init_kwargs)
    t0 = time.perf_counter()
    attrs = [explainer.explain(**explain_kwargs, target=t) for t in targets]
    elapsed_single = time.perf_counter() - t0
    attrs_tensor = torch.stack(attrs)

    explainer_mt = explainer_cls(model, multi_target=True, **init_kwargs)
    t0 = time.perf_counter()
    attrs_mt = explainer_mt.explain(**explain_kwargs, target=targets)
    elapsed_mt = time.perf_counter() - t0
    attrs_mt_tensor = torch.stack(attrs_mt)

    assert torch.allclose(attrs_tensor, attrs_mt_tensor, atol=atol), \
        "Results differ between single-target and multi-target"
    speedup = elapsed_single / elapsed_mt if elapsed_mt > 0 else float("inf")
    print(f"shape  : {attrs_mt_tensor.shape}")
    print(f"single : {elapsed_single:.3f}s  |  multi : {elapsed_mt:.3f}s  |  speedup : {speedup:.1f}x")
    return attrs_mt_tensor
```

## Explainer Input Requirements

| Explainer | `baselines` | `feature_mask` | `sliding_window_shapes` |
|---|:---:|:---:|:---:|
| `SaliencyExplainer` | ✗ | optional | ✗ |
| `InputXGradientExplainer` | ✗ | optional | ✗ |
| `GuidedBackpropExplainer` | ✗ | optional | ✗ |
| `RandomExplainer` | ✗ | optional | ✗ |
| `FeatureAblationExplainer` | ✗ | optional | ✗ |
| `LimeExplainer` | ✗ | optional | ✗ |
| `KernelShapExplainer` | ✗ | optional | ✗ |
| `IntegratedGradientsExplainer` | ✓ | optional | ✗ |
| `DeepLiftExplainer` | ✓ | optional | ✗ |
| `InputXBaselineGradientExplainer` | ✓ | optional | ✗ |
| `DeepLiftShapExplainer` | ✓ distribution | optional | ✗ |
| `GradientShapExplainer` | ✓ distribution | optional | ✗ |
| `OcclusionExplainer` | ✗ | ✗ | ✓ |

---

## Pattern A — inputs + target

No baseline or mask required. Applies to: `SaliencyExplainer`, `InputXGradientExplainer`, `GuidedBackpropExplainer`, `RandomExplainer`, `FeatureAblationExplainer`, `LimeExplainer`, `KernelShapExplainer`.

```python
from torchxai.explainers import SaliencyExplainer
# Replace SaliencyExplainer with any Pattern-A explainer

compare(SaliencyExplainer, model, dict(inputs=inputs), targets)
```

---

## Pattern B — inputs + baseline + target

A single reference tensor (same shape as `inputs`) is required. Applies to: `IntegratedGradientsExplainer`, `DeepLiftExplainer`, `InputXBaselineGradientExplainer`.

```python
from torchxai.explainers import IntegratedGradientsExplainer
# Replace with DeepLiftExplainer or InputXBaselineGradientExplainer as needed

compare(IntegratedGradientsExplainer, model, dict(inputs=inputs, baselines=baseline), targets)
```

---

## Pattern C — inputs + baseline distribution + target

`baselines` is a **stacked set of reference samples** rather than a single tensor. Applies to: `DeepLiftShapExplainer`, `GradientShapExplainer`.

!!! note
    GradientShap randomly samples a baseline from the distribution on each call, so results may vary slightly between single-target runs and the multi-target call. Use `atol=1e-3`.

```python
from torchxai.explainers import GradientShapExplainer
# Replace with DeepLiftShapExplainer as needed

baselines_dist = baseline.expand(5, -1, -1, -1)   # (5, 3, 32, 32) reference distribution

compare(GradientShapExplainer, model,
        dict(inputs=inputs, baselines=baselines_dist), targets, atol=1e-3)
```

---

## Pattern D — inputs + feature_mask + target

A `feature_mask` groups input elements into segments so the explainer scores whole regions rather than individual pixels. Applies to: `FeatureAblationExplainer`, `LimeExplainer`, `KernelShapExplainer`.

The same explainer class works with or without a mask; the mask is optional. Pixel-level attribution uses no mask; segment-level attribution passes one.

```python
from torchxai.explainers import FeatureAblationExplainer
# Replace with LimeExplainer or KernelShapExplainer as needed

# 8×8 super-pixel grid — 64 segments over a 32×32 image
grid = 8
cell_size = 32 // grid
feature_mask = torch.zeros(1, 1, 32, 32, dtype=torch.long)
for i in range(grid):
    for j in range(grid):
        feature_mask[0, 0, i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size] = i * grid + j

print("Without feature mask (pixel-level):")
compare(FeatureAblationExplainer, model, dict(inputs=inputs), targets)

print("\nWith feature mask (segment-level):")
compare(FeatureAblationExplainer, model, dict(inputs=inputs, feature_mask=feature_mask), targets)
```

---

## Pattern E — inputs + sliding_window_shapes + target

`OcclusionExplainer` patches out rectangular windows instead of using a feature mask. The `sliding_window_shapes` tuple specifies the window size per input channel.

```python
from torchxai.explainers import OcclusionExplainer

compare(OcclusionExplainer, model, dict(inputs=inputs, sliding_window_shapes=(1, 8, 8)), targets)
```
