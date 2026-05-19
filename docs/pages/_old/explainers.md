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

### Baseline Methods

- **[Random](random.md)** - Random attribution baseline for comparison and sanity checking

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
| Random | Baseline | ❌ | ❌ | ❌ | Random noise | Baseline comparison |

## Quick Example with Baseline Comparison

```python
from torchxai.explainers._grad import SaliencyExplainer, IntegratedGradientsExplainer
from torchxai.explainers._perturbation import FeatureAblationExplainer, KernelShapExplainer
from torchxai.explainers.random import RandomExplainer
import torch
from collections import OrderedDict
from torchxai.data_types import ExplanationInputs

# Model for comparison
model = torch.nn.Sequential(torch.nn.Linear(10, 5), torch.nn.ReLU(), torch.nn.Linear(5, 2))

# Different explainers including random baseline
explainers = {
    "random": RandomExplainer(model, random_seed=42),
    "saliency": SaliencyExplainer(model),
    "integrated_gradients": IntegratedGradientsExplainer(model, n_steps=50),
    "feature_ablation": FeatureAblationExplainer(model, internal_batch_size=32),
    "kernel_shap": KernelShapExplainer(model, n_samples=100),
}

# Example input
inputs = OrderedDict({"features": torch.randn(1, 10)})
target = torch.tensor([1])

# Input configurations
simple_inputs = ExplanationInputs(inputs=inputs, target=target)
baseline_inputs = ExplanationInputs(
    inputs=inputs, target=target,
    baselines=OrderedDict({"features": torch.zeros(1, 10)})
)

# Compute attributions with different methods
results = {}
results["random"] = explainers["random"].explain(simple_inputs)
results["saliency"] = explainers["saliency"].explain(simple_inputs)
results["integrated_gradients"] = explainers["integrated_gradients"].explain(baseline_inputs)
results["feature_ablation"] = explainers["feature_ablation"].explain(baseline_inputs)
results["kernel_shap"] = explainers["kernel_shap"].explain(baseline_inputs)

# Compare attribution magnitudes (sanity check)
print("Attribution Analysis:")
print("=" * 50)
for method, attribution in results.items():
    attr_magnitude = torch.abs(attribution["features"]).mean().item()
    attr_sum = attribution["features"].sum().item()
    print(f"{method:20}: magnitude={attr_magnitude:.4f}, sum={attr_sum:.4f}")

# Statistical significance test (example)
import scipy.stats as stats

random_attrs = results["random"]["features"].flatten().numpy()
saliency_attrs = results["saliency"]["features"].flatten().numpy()

# Test if saliency attributions are significantly different from random
t_stat, p_value = stats.ttest_ind(random_attrs, saliency_attrs)
print(f"\nSaliency vs Random: t-stat={t_stat:.4f}, p-value={p_value:.4f}")
if p_value < 0.05:
    print("Saliency attributions are significantly different from random!")
else:
    print("Saliency attributions are not significantly different from random.")
```

## Sanity Checking with Random Baseline

The Random explainer serves several important purposes:

1. **Baseline Comparison**: Establishes whether other methods provide signal above noise
2. **Statistical Testing**: Enables significance tests for attribution quality
3. **Method Validation**: Helps identify when attribution methods fail
4. **Reproducibility**: With fixed seeds, provides consistent baseline results

Always compare your attributions against random baselines to ensure meaningful explanations!
