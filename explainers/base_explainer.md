# Base Explainer

The abstract base class that all TorchXAI explainers inherit from.

::: torchxai.explainers.Explainer

## Key Features

- **Multi-target Support**: Automatically handles single-target and multi-target explanations
- **Flexible Input Handling**: Works with `ExplanationInputs` for structured input management  
- **Automatic Function Inspection**: Dynamically inspects explainer function signatures
- **Batch Processing**: Built-in support for batch explanations with configurable batch sizes
<!-- 
## Implementation Example

Here's how to implement a custom explainer by inheriting from the base `Explainer` class:

```python
from collections.abc import Callable
from torchxai.explainers.explainer import Explainer
from torchxai.data_types import ExplanationInputs, ExplanationTupleInputs
from captum.attr import Saliency
import torch

class MyCustomExplainer(Explainer):
    """Custom explainer implementation."""
    
    def _init_single_target_explanation_fn(self) -> Callable:
        """Initialize single-target explanation function."""
        return Saliency(self._model).attribute
    
    def _init_multi_target_explanation_fn(self) -> Callable:
        """Initialize multi-target explanation function."""
        # Return your multi-target implementation
        return MyMultiTargetMethod(self._model).attribute
    
    def _explain(self, explanation_tuple_inputs: ExplanationTupleInputs):
        """Internal method to compute attributions."""
        return self._explanation_fn(
            inputs=explanation_tuple_inputs.inputs,
            target=explanation_tuple_inputs.target,
            additional_forward_args=explanation_tuple_inputs.additional_forward_args,
        )

# Usage with ExplanationInputs
import torch
from collections import OrderedDict

model = torch.nn.Linear(10, 2)
explainer = MyCustomExplainer(model, multi_target=False)

# Create structured inputs
inputs = torch.randn(2, 10)  # batch size 2
target = torch.tensor([0, 1])  # targets for each sample

explanation_inputs = ExplanationInputs(
    inputs=OrderedDict({"input_features": inputs}),
    target=target,
)

# Get explanations as OrderedDict
attributions = explainer.explain(explanation_inputs)
# Returns: OrderedDict[str, torch.Tensor]

# Multi-target example
explainer_mt = MyCustomExplainer(model, multi_target=True)
mt_target = [torch.tensor([0, 1]), torch.tensor([1, 0])]  # 2 targets
explanation_inputs_mt = ExplanationInputs(
    inputs=OrderedDict({"input_features": inputs}),
    target=mt_target,
)

# Returns: list[OrderedDict[str, torch.Tensor]] - one dict per target
mt_attributions = explainer_mt.explain(explanation_inputs_mt)
```

## New vs Old API

The updated `Explainer` class now uses structured inputs via `ExplanationInputs` instead of raw tensors:

- **Old**: `explainer.explain(inputs, target, additional_forward_args=None)`
- **New**: `explainer.explain(ExplanationInputs(...))`

This provides better type safety, clearer parameter organization, and support for multiple input features with named keys. -->
