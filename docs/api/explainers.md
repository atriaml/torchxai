# Explainers Overview

TorchXAI provides various explainable AI methods for PyTorch models.

## Available Explainers

### Gradient-Based Methods

- **[Base Explainer](base_explainer.md)** - Abstract base class for all explainers
- **[Saliency](saliency.md)** - Gradient-based saliency methods
- **[Input × Gradient](input_x_gradient.md)** - Input-scaled gradient attribution methods
- **[Input × Baseline Gradient](input_x_baseline_gradient.md)** - Baseline-scaled gradient attribution methods
- **[Guided Backpropagation](guided_backprop.md)** - Modified gradient methods with ReLU handling
- **[DeepLIFT](deeplift.md)** - Reference-based attribution with baseline comparison
- **[DeepLIFT SHAP](deeplift_shap.md)** - Shapley value computation using DeepLIFT with training baselines
- **[Integrated Gradients](integrated_gradients.md)** - Path-integrated attribution methods
- **[GradientShap](gradient_shap.md)** - Noise-based Shapley value approximations

### Perturbation-Based Methods

- **[Feature Ablation](feature_ablation.md)** - Systematic feature removal attribution methods
- **[Occlusion](occlusion.md)** - Sliding-window perturbation attribution methods
- **[LIME](lime.md)** - Local interpretable model-agnostic explanations
- **[Kernel SHAP](kernel_shap.md)** - Shapley value computation using LIME framework

## Method Comparison

| Method | Type | Requires Baseline | Multiple Baselines | Feature Groups | Theory | Best For |
|--------|------|------------------|-------------------|---------------|--------|----------|
| Saliency | Gradient | ❌ | ❌ | ❌ | Gradient | Quick analysis |
| Input × Gradient | Gradient | ❌ | ❌ | ❌ | Gradient | Input-scaled importance |
| Input × Baseline Gradient | Gradient | ✅ | ❌ | ❌ | Gradient | Baseline-relative |
| Guided Backpropagation | Gradient | ❌ | ❌ | ❌ | Modified gradient | Positive contributions |
| DeepLIFT | Gradient | ✅ | ❌ | ❌ | Axiom-based | Non-linear models |
| DeepLIFT SHAP | Gradient | ✅ | ✅ | ❌ | Shapley + DeepLIFT | Training baselines |
| Integrated Gradients | Gradient | ✅ | ❌ | ❌ | Path integration | Path independence |
| GradientShap | Gradient | ✅ | ✅ | ❌ | Shapley + Gradients | Robust Shapley |
| Feature Ablation | Perturbation | ✅ | ❌ | ✅ | Direct measurement | Feature groups |
| Occlusion | Perturbation | ✅ | ❌ | ❌ | Direct measurement | Spatial/visual data |
| LIME | Perturbation | ✅ | ❌ | ✅ | Local linear | Local explanations |
| Kernel SHAP | Perturbation | ✅ | ❌ | ✅ | Shapley theory | Shapley values |

## Quick Example

```python
from torchxai.explainers._grad import (
    SaliencyExplainer, InputXGradientExplainer, 
    InputXBaselineGradientExplainer, GuidedBackpropExplainer,
    DeepLiftExplainer, DeepLiftShapExplainer,
    IntegratedGradientsExplainer, GradientShapExplainer
)
from torchxai.explainers._perturbation import (
    FeatureAblationExplainer, OcclusionExplainer, 
    LimeExplainer, KernelShapExplainer
)
import torch
from collections import OrderedDict
from torchxai.data_types import ExplanationInputs

# Different models for different methods
tabular_model = torch.nn.Sequential(torch.nn.Linear(10, 5), torch.nn.ReLU(), torch.nn.Linear(5, 2))
image_model = torch.nn.Sequential(
    torch.nn.Conv2d(3, 16, 3), torch.nn.ReLU(),
    torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(),
    torch.nn.Linear(16, 10)
)

# Training baseline samples (for SHAP methods)
train_baselines_tabular = torch.randn(100, 10)

# Different explainers
gradient_explainers = {
    "saliency": SaliencyExplainer(tabular_model),
    "input_grad": InputXGradientExplainer(tabular_model),
    "input_baseline_grad": InputXBaselineGradientExplainer(tabular_model),
    "guided_backprop": GuidedBackpropExplainer(tabular_model),
    "deeplift": DeepLiftExplainer(tabular_model),
    "deeplift_shap": DeepLiftShapExplainer(tabular_model),
    "integrated_gradients": IntegratedGradientsExplainer(tabular_model, n_steps=100),
    "gradient_shap": GradientShapExplainer(tabular_model, n_samples=50)
}

# Perturbation explainers
perturbation_explainers = {
    "feature_ablation": FeatureAblationExplainer(
        tabular_model,
        internal_batch_size=32,
        weight_attributions=True
    ),
    "occlusion": OcclusionExplainer(
        image_model, 
        sliding_window_shapes=(8, 8), 
        strides=(4, 4),
        internal_batch_size=10
    ),
    "lime": LimeExplainer(
        tabular_model,
        n_samples=200,
        alpha=0.01,
        internal_batch_size=50
    ),
    "kernel_shap": KernelShapExplainer(
        tabular_model,
        n_samples=500,
        internal_batch_size=50
    )
}

# Example inputs
tabular_inputs = OrderedDict({"features": torch.randn(1, 10)})
image_inputs = OrderedDict({"image": torch.randn(1, 3, 32, 32)})
target = torch.tensor([1])

# Feature mask for grouping (perturbation methods)
feature_mask = torch.tensor([[0, 0, 1, 1, 2, 2, 2, 3, 3, 4]])

# Different input configurations
simple_inputs = ExplanationInputs(inputs=tabular_inputs, target=target)
baseline_inputs = ExplanationInputs(
    inputs=tabular_inputs, target=target,
    baselines=OrderedDict({"features": torch.zeros(1, 10)})
)
shap_inputs = ExplanationInputs(
    inputs=tabular_inputs, target=target,
    baselines=OrderedDict({"features": train_baselines_tabular})
)
image_occlusion_inputs = ExplanationInputs(
    inputs=image_inputs, target=torch.tensor([5]),
    baselines=OrderedDict({"image": torch.zeros(1, 3, 32, 32)})
)
grouped_inputs = ExplanationInputs(
    inputs=tabular_inputs, target=target,
    baselines=OrderedDict({"features": torch.zeros(1, 10)}),
    feature_mask=feature_mask
)

# Compute attributions with different methods
results = {}

# Gradient methods
for name, explainer in gradient_explainers.items():
    if name in ["deeplift_shap", "gradient_shap"]:
        results[name] = explainer.explain(shap_inputs)
    elif name in ["input_baseline_grad", "deeplift", "integrated_gradients"]:
        results[name] = explainer.explain(baseline_inputs)
    else:
        results[name] = explainer.explain(simple_inputs)

# Perturbation methods
results["feature_ablation"] = perturbation_explainers["feature_ablation"].explain(grouped_inputs)
results["occlusion"] = perturbation_explainers["occlusion"].explain(image_occlusion_inputs)
results["lime"] = perturbation_explainers["lime"].explain(grouped_inputs)
results["kernel_shap"] = perturbation_explainers["kernel_shap"].explain(grouped_inputs)

# Compare direct methods vs approximation methods
print("Feature Ablation (direct):", results["feature_ablation"]["features"].sum())
print("Kernel SHAP (Shapley approximation):", results["kernel_shap"]["features"].sum())
print("LIME (local linear):", results["lime"]["features"].sum())
```
