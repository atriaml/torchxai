---
title: Base Explainer
---

# Base Explainer

TorchXAI provides two abstract base classes that all concrete explainers inherit from.

---

## `Explainer`

The root abstract base class. Defines the `explain()` interface and a default `__repr__`.

::: torchxai.explainers.Explainer

---

## `FeatureAttributionExplainer`

Extends `Explainer` with `multi_target` support, configurable batch sizes, and the routing logic that dispatches `explain()` calls to single-target or multi-target Captum attribution functions. **All concrete explainers inherit from this class.**

::: torchxai.explainers.FeatureAttributionExplainer

---

## Implementing a custom explainer

Subclass `FeatureAttributionExplainer` and implement `_init_single_target_explanation_fn()` plus `explain()`:

```python
from collections.abc import Callable
import torch
from captum.attr import Saliency
from torchxai.explainers import FeatureAttributionExplainer
from torchxai.data_types import SingleTargetAcrossBatch


class MySaliencyExplainer(FeatureAttributionExplainer):
    def _init_single_target_explanation_fn(self) -> Callable:
        return Saliency(self._model).attribute

    def explain(self, inputs: torch.Tensor, target, **kwargs) -> torch.Tensor:
        return self._default_explain(inputs=inputs, target=target, **kwargs)


model = torch.nn.Sequential(torch.nn.Linear(10, 3), torch.nn.ReLU())
explainer = MySaliencyExplainer(model, multi_target=False)

attrs = explainer.explain(
    inputs=torch.randn(1, 10),
    target=SingleTargetAcrossBatch(index=0),
)
print(attrs.shape)   # (1, 10)
```

To override the multi-target implementation (e.g. for a more efficient batched approach), also implement `_init_multi_target_explanation_fn()`. Otherwise the default falls back to iterating `_init_single_target_explanation_fn()` once per target.
