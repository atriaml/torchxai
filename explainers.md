# Explainers Overview

TorchXAI wraps [Captum](https://captum.ai/) attribution methods and adds first-class **multi-target** support: explain multiple output targets in a single forward pass.

## Available Explainers

### Base Classes

- **[Explainer / FeatureAttributionExplainer](explainers/base_explainer.md)** — abstract bases; inherit `FeatureAttributionExplainer` to build a custom explainer

### Gradient-Based Methods

- **[Saliency](explainers/saliency.md)** — gradient magnitude per input dimension
- **[Input × Gradient](explainers/input_x_gradient.md)** — input scaled by its gradient
- **[Input × Baseline Gradient](explainers/input_x_baseline_gradient.md)** — input-minus-baseline scaled by gradient
- **[Guided Backpropagation](explainers/guided_backprop.md)** — gradient with ReLU clamping on backward pass
- **[DeepLIFT](explainers/deeplift.md)** — reference-based attribution comparing activations to a baseline
- **[DeepLIFT SHAP](explainers/deeplift_shap.md)** — DeepLIFT averaged over a baseline distribution
- **[Integrated Gradients](explainers/integrated_gradients.md)** — path-integral from baseline to input
- **[GradientShap](explainers/gradient_shap.md)** — gradient-based Shapley approximation with random baselines

### Perturbation-Based Methods

- **[Feature Ablation](explainers/feature_ablation.md)** — systematically zeros out features or groups
- **[Occlusion](explainers/occlusion.md)** — sliding-window patch replacement
- **[LIME](explainers/lime.md)** — locally-linear surrogate model
- **[Kernel SHAP](explainers/kernel_shap.md)** — Shapley values via LIME kernel weighting

### Baseline Methods

- **[Random](explainers/random.md)** — random attributions for sanity-checking and comparison

---

## Input Patterns

Each explainer belongs to one of five input patterns. Choose the pattern for your explainer and your use case:

| Pattern | Required arguments | Explainers |
|---------|-------------------|------------|
| **A** | `inputs`, `target` | Saliency, InputXGradient, GuidedBackprop, Random |
| **B** | `inputs`, `baselines`, `target` | IntegratedGradients, DeepLift, InputXBaselineGradient |
| **C** | `inputs`, `baselines` (distribution), `target` | GradientShap, DeepLiftShap |
| **D** | `inputs`, `feature_mask` (optional), `target` | FeatureAblation, LIME, KernelShap |
| **E** | `inputs`, `sliding_window_shapes`, `target` | Occlusion |

### Full comparison table

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

---

## Quick Start

```python
import torch
import torch.nn as nn
from torchxai.explainers import SaliencyExplainer, IntegratedGradientsExplainer
from torchxai.data_types import SingleTargetAcrossBatch

model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 3))
model.eval()

inputs   = torch.randn(1, 10)
baseline = torch.zeros(1, 10)
target   = SingleTargetAcrossBatch(index=0)   # explain class 0

# Pattern A — no baseline needed
explainer = SaliencyExplainer(model)
attrs = explainer.explain(inputs=inputs, target=target)
print(attrs.shape)   # (1, 10)

# Pattern B — baseline required
explainer_ig = IntegratedGradientsExplainer(model)
attrs_ig = explainer_ig.explain(inputs=inputs, baselines=baseline, target=target)
print(attrs_ig.shape)   # (1, 10)
```

---

## Multi-Target Mode

Pass `multi_target=True` at construction, then supply a **list** of targets. The explainer returns a `list[Tensor]`, one per target, in a single forward-backward pass.

```python
from torchxai.explainers import SaliencyExplainer
from torchxai.data_types import SingleTargetAcrossBatch

targets = [SingleTargetAcrossBatch(index=i) for i in range(3)]   # classes 0, 1, 2

explainer = SaliencyExplainer(model, multi_target=True)
attrs_list = explainer.explain(inputs=inputs, target=targets)

for cls_idx, attr in enumerate(attrs_list):
    print(f"class {cls_idx}: {attr.shape}")   # each (1, 10)
```

This is equivalent to calling `explain()` once per target but can be significantly faster because shared computation (forward pass, intermediate activations) is reused.

---

## End-to-End Examples

Worked examples covering all five input patterns:

- **[Image Classification](explainers/examples/image_classification.md)** — TinyCNN with 10 output classes; all explainer patterns on image tensors
- **[Sequence Classification](explainers/examples/sequence_classification.md)** — BERT with embedding-level inputs; all patterns for sentence-level targets
- **[Token Classification](explainers/examples/token_classification.md)** — BERT-NER; multi-target across all token positions in one call

---

## Sanity-Checking with Random Attributions

`RandomExplainer` provides random-noise attributions. Use it to verify that your real explainer is producing signal above chance:

```python
from torchxai.explainers import RandomExplainer, SaliencyExplainer
from torchxai.data_types import SingleTargetAcrossBatch

target = SingleTargetAcrossBatch(index=0)

random_attrs   = RandomExplainer(model, random_seed=42).explain(inputs=inputs, target=target)
saliency_attrs = SaliencyExplainer(model).explain(inputs=inputs, target=target)

print("Random  :", random_attrs.abs().mean().item())
print("Saliency:", saliency_attrs.abs().mean().item())
```

If saliency attribution magnitudes are comparable to random, the model is likely not using that feature meaningfully — or the explainer is misconfigured.
