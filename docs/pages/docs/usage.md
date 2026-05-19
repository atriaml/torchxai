
# Minimal usage examples

### 1. Generating explanations for multiple targets

This example shows computing saliency explanation heatmaps on a simple pytorch model.

```python
from torchxai.explainers._grad._saliency import SaliencyExplainer
import torch

# initialize explainer
model = torch.nn.Linear(10, 2)
explainer = SaliencyExplainer(model)

# generate explanations
inputs = torch.randn(1, 10)
attributions = explainer.explain(inputs, target=0)

# generate explanations for multiple targets together
inputs = torch.randn(1, 10)
explainer = SaliencyExplainer(model, multi_target=True)
attributions = explainer.explain(inputs, target=[0, 1])
```

### 2. Evaluating the explanations with one of the supported metrics

This example shows computing a completeness score for a Captum saliency map:

```python
from torchxai.metrics.axiomatic import completeness
from captum.attr import Saliency
import torch

net = ImageClassifier()                 # your model
saliency = Saliency(net)
input = torch.randn(2, 3, 32, 32, requires_grad=True)
baselines = torch.zeros(2, 3, 32, 32)

# computes saliency maps for class 3
attribution = saliency.attribute(input, target=3)

# computes completeness score for saliency maps
completeness_score = completeness(net, input, attribution, baselines)
print("Completeness:", completeness_score)
```

## API Reference

- [Explainers](explainers.md) - Core explainer classes
