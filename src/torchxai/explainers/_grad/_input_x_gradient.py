from collections.abc import Callable
from typing import Any

import torch
from captum._utils.common import _format_output, _format_tensor_into_tuples, _is_tuple
from captum._utils.gradient import (
    apply_gradient_requirements,
    undo_gradient_requirements,
)
from captum.attr import GradientAttribution, InputXGradient

from torchxai.data_types import (
    ExplanationTargetType,
    TargetType,
    TensorOrTupleOfTensorsGeneric,
)
from torchxai.explainers._explainer import Explainer
from torchxai.explainers._utils import (
    _compute_gradients_sequential_autograd,
    _compute_gradients_vmap_autograd,
    _verify_target_for_multi_target_impl,
)


class MultiTargetInputXGradient(GradientAttribution):
    """Multi-target Input × Gradient attribution.

    This class extends Captum's GradientAttribution to support computing
    Input × Gradient attributions for multiple targets simultaneously.

    The method multiplies input features by their gradients to provide
    attributions that consider both the magnitude of the input and its sensitivity.

    Args:
        forward_func: The forward function of the model to be explained.
        gradient_func: Function for computing gradients. Automatically selects
            between vmap and sequential methods based on PyTorch version.
        grad_batch_size: Batch size for gradient computations. Defaults to 10.
    """

    def __init__(
        self,
        forward_func: Callable,
        gradient_func=(
            _compute_gradients_vmap_autograd
            if torch.__version__ >= "2.3.0"
            else _compute_gradients_sequential_autograd
        ),
        grad_batch_size: int = 10,
    ) -> None:
        super().__init__(forward_func)
        self._gradient_func = gradient_func
        self._grad_batch_size = grad_batch_size

    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: list[TargetType],
        additional_forward_args: Any = None,
    ) -> list[TensorOrTupleOfTensorsGeneric]:
        """Compute multi-target Input × Gradient attributions.

        Args:
            inputs: Input tensors for which to compute attributions.
            target: List of target indices for multi-target attribution.
            additional_forward_args: Additional arguments for the forward function.

        Returns:
            List of attribution tensors, one for each target in the target list.
            Each element has the same structure as the input tensors.
        """
        # Keeps track whether original input is a tuple or not before
        # converting it into a tuple.
        is_inputs_tuple = _is_tuple(inputs)

        inputs = _format_tensor_into_tuples(inputs)
        gradient_mask = apply_gradient_requirements(inputs)

        # verify that the target is valid
        _verify_target_for_multi_target_impl(inputs, target)

        multi_target_gradients = self._gradient_func(
            self.forward_func,
            inputs,
            target,
            additional_forward_args,
            grad_batch_size=self._grad_batch_size,
        )

        def gradients_to_attributions(gradients):
            attributions = tuple(
                input * gradient
                for input, gradient in zip(inputs, gradients, strict=False)
            )
            return attributions

        multi_target_attributions = [
            gradients_to_attributions(grad) for grad in multi_target_gradients
        ]

        undo_gradient_requirements(inputs, gradient_mask)
        return [
            _format_output(is_inputs_tuple, per_target_attributions)
            for per_target_attributions in multi_target_attributions
        ]


class InputXGradientExplainer(Explainer):
    """Input × Gradient explainer for computing input-scaled gradient attributions.

    This explainer computes attributions by multiplying input features with their
    gradients, providing a measure that considers both the magnitude of the input
    and its sensitivity to the output. Supports both single-target and multi-target
    modes with structured input/output via ExplanationInputs.

    The Input × Gradient method combines the input magnitude with gradient information,
    making it useful for understanding feature importance in the context of actual input values.

    Args:
        model: The PyTorch model whose output is to be explained.
        multi_target: Whether to use multi-target mode. When True, can compute
            attributions for multiple targets simultaneously. Defaults to False.
        internal_batch_size: Batch size for internal computations. Defaults to 64.
        grad_batch_size: Batch size for gradient computations. Defaults to 64.

    Examples:
        Single-target usage:
        >>> import torch
        >>> from collections import OrderedDict
        >>> from torchxai.data_types import ExplanationInputs
        >>>
        >>> model = torch.nn.Linear(10, 2)
        >>> explainer = InputXGradientExplainer(model)
        >>>
        >>> explanation_inputs = ExplanationInputs(
        ...     inputs=OrderedDict({"input": torch.randn(2, 10)}),
        ...     target=torch.tensor([0, 1]),
        ... )
        >>> attributions = explainer.explain(explanation_inputs)
        >>> # Returns: OrderedDict({"input": torch.Tensor})

        Multi-target usage:
        >>> explainer_mt = InputXGradientExplainer(model, multi_target=True)
        >>> explanation_inputs_mt = ExplanationInputs(
        ...     inputs=OrderedDict({"input": torch.randn(2, 10)}),
        ...     target=[torch.tensor([0]), torch.tensor([1])],
        ... )
        >>> mt_attributions = explainer_mt.explain(explanation_inputs_mt)
        >>> # Returns: [OrderedDict({"input": torch.Tensor}), OrderedDict({"input": torch.Tensor})]
    """

    def _init_single_target_explanation_fn(self) -> Callable:
        """Initialize single-target Input × Gradient attribution function.

        Returns:
            Captum InputXGradient attribution function for single targets.
        """
        return InputXGradient(self._model).attribute

    def _init_multi_target_explanation_fn(self) -> Callable:
        """Initialize multi-target Input × Gradient attribution function.

        Returns:
            MultiTargetInputXGradient attribution function for multiple targets.
        """
        return MultiTargetInputXGradient(
            self._model, grad_batch_size=self._grad_batch_size
        ).attribute

    def explain(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: ExplanationTargetType,
        additional_forward_args: tuple[Any, ...] | None = None,
    ) -> TensorOrTupleOfTensorsGeneric | list[TensorOrTupleOfTensorsGeneric]:
        """Compute Input × Gradient attributions for the given inputs.

        This method provides a backward-compatible interface that accepts individual
        parameters and constructs ExplanationInputs internally before calling the
        parent class explain method.

        Args:
            inputs: Input tensors for attribution computation. Should be an OrderedDict
                mapping feature names to tensors when used with this explainer.
            target: Target indices for attribution computation. Can be a tensor
                (single-target) or list of tensors (multi-target).
            additional_forward_args: Additional arguments for model forward pass.

        Returns:
            For single-target mode: OrderedDict mapping feature names to attribution tensors.
            For multi-target mode: List of OrderedDicts, one per target.

        Examples:
            >>> # Single tensor input (wrapped automatically)
            >>> attributions = explainer.explain(
            ...     inputs=torch.randn(2, 10), target=torch.tensor([0, 1])
            ... )
            >>>
            >>> # Multiple features (use OrderedDict)
            >>> attributions = explainer.explain(
            ...     inputs=OrderedDict(
            ...         {"feat1": torch.randn(2, 5), "feat2": torch.randn(2, 5)}
            ...     ),
            ...     target=torch.tensor([0, 1]),
            ... )
        """
        return self._default_explain(
            inputs=inputs,
            target=target,
            additional_forward_args=additional_forward_args,
        )
