# TorchXAI Documentation

Welcome to TorchXAI, an explainable AI library for PyTorch.

## Quick Start

```python
from torchxai.explainers._grad._saliency import SaliencyExplainer
import torch

# Initialize explainer
model = torch.nn.Linear(10, 2)
explainer = SaliencyExplainer(model)

# Generate explanations
inputs = torch.randn(1, 10)
attributions = explainer.explain(inputs, target=0)
```

## API Reference

- [Explainers](api/explainers.md) - Core explainer classes
