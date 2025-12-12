from collections import OrderedDict
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from captum._utils.common import _format_output, _is_tuple
from captum.attr._core.gradient_shap import (
    GradientAttribution,
    InputBaselineXGradient,
    _scale_input,
)
from captum.attr._utils.common import _format_input_baseline

from torchxai.data_types import ExplanationInputs, ExplanationTargetType
from torchxai.data_types.common import (
    BaselineType,
    TargetType,
    TensorOrTupleOfTensorsGeneric,
)
from torchxai.explainers._utils import (
    _compute_gradients_sequential_autograd,
    _compute_gradients_vmap_autograd,
    _verify_target_for_multi_target_impl,
)
from torchxai.explainers.explainer import Explainer


class MultiTargetInputBaselineXGradient(GradientAttribution):
    """Multi-target Input × Baseline × Gradient attribution.

    This class extends Captum's InputBaselineXGradient to support computing
    attributions for multiple targets simultaneously in the GradientShap context.

    The method computes (input - baseline) × gradient for each target, which is
    used as a component in the GradientShap algorithm.
    """

    def __init__(
        self,
        forward_func: Callable,
        multiply_by_inputs: bool = True,
        gradient_func=(
            _compute_gradients_vmap_autograd
            if torch.__version__ >= "2.3.0"
            else _compute_gradients_sequential_autograd
        ),
        grad_batch_size: int = 10,
    ) -> None:
        super().__init__(forward_func)
        self.gradient_func = gradient_func
        self.grad_batch_size = grad_batch_size
        self._multiply_by_inputs = multiply_by_inputs

    def attribute(  # type: ignore
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines: BaselineType = None,
        target: list[TargetType] | None = None,
        additional_forward_args: Any = None,
        return_convergence_delta: bool = False,
    ) -> (
        tuple[list[TensorOrTupleOfTensorsGeneric], list[torch.Tensor]]
        | list[TensorOrTupleOfTensorsGeneric]
    ):
        # Keeps track whether original input is a tuple or not before
        # converting it into a tuple.
        is_inputs_tuple = _is_tuple(inputs)
        inputs, baselines = _format_input_baseline(inputs, baselines)

        # verify that the target is valid
        assert isinstance(target, list), (
            "Target must be a list for multi-target attribution."
        )
        _verify_target_for_multi_target_impl(inputs, target)

        rand_coefficient = torch.tensor(
            np.random.uniform(0.0, 1.0, inputs[0].shape[0]),
            device=inputs[0].device,
            dtype=inputs[0].dtype,
        )
        input_baseline_scaled = tuple(
            _scale_input(input, baseline, rand_coefficient)
            for input, baseline in zip(inputs, baselines, strict=False)
        )
        multi_target_gradients = self.gradient_func(
            self.forward_func,
            input_baseline_scaled,
            target,
            additional_forward_args,
            grad_batch_size=self.grad_batch_size,
        )

        def gradients_to_attributions(per_target_gradients):
            if self.multiplies_by_inputs:
                input_baseline_diffs = tuple(
                    input - baseline
                    for input, baseline in zip(inputs, baselines, strict=False)
                )
                return tuple(
                    input_baseline_diff * grad
                    for input_baseline_diff, grad in zip(
                        input_baseline_diffs, per_target_gradients, strict=False
                    )
                )
            else:
                return per_target_gradients

        multi_target_attributions = [
            gradients_to_attributions(per_target_gradients)
            for per_target_gradients in multi_target_gradients
        ]

        assert isinstance(multi_target_attributions, list), (
            "Expected multi_target_attributions to be a list."
        )
        assert len(multi_target_attributions) == len(target), (
            "Length of multi_target_attributions does not match length of target."
        )
        if return_convergence_delta:
            start_point, end_point = baselines, inputs
            # computes approximation error based on the completeness axiom
            delta = [
                self.compute_convergence_delta(
                    per_target_attribution,
                    start_point,
                    end_point,
                    additional_forward_args=additional_forward_args,
                    target=single_target,
                )
                for single_target, per_target_attribution in zip(
                    target, multi_target_attributions, strict=False
                )
            ]
            return [
                _format_output(is_inputs_tuple, attributions)
                for attributions in multi_target_attributions
            ], delta
        return [
            _format_output(is_inputs_tuple, attributions)
            for attributions in multi_target_attributions
        ]

    def has_convergence_delta(self) -> bool:
        return True

    @property
    def multiplies_by_inputs(self):
        return self._multiply_by_inputs


class InputXBaselineGradientExplainer(Explainer):
    """Input × Baseline Gradient explainer for computing scaled baseline-gradient attributions.

    This explainer computes attributions by multiplying (input - baseline) with their
    gradients, providing a measure that considers both the deviation from baseline
    and gradient sensitivity. This method is particularly useful when you have
    meaningful baseline references. Supports both single-target and multi-target
    modes with structured input/output.

    The Input × Baseline Gradient method provides attributions that are grounded
    in both the input magnitude relative to a baseline and gradient information.

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
        >>> explainer = InputXBaselineGradientExplainer(model)
        >>>
        >>> explanation_inputs = ExplanationInputs(
        ...     inputs=OrderedDict({"input": torch.randn(2, 10)}),
        ...     target=torch.tensor([0, 1]),
        ...     baselines=OrderedDict({"input": torch.zeros(2, 10)}),
        ... )
        >>> attributions = explainer.explain(explanation_inputs)
        >>> # Returns: OrderedDict({"input": torch.Tensor})

        Multi-target usage:
        >>> explainer_mt = InputXBaselineGradientExplainer(model, multi_target=True)
        >>> explanation_inputs_mt = ExplanationInputs(
        ...     inputs=OrderedDict({"input": torch.randn(2, 10)}),
        ...     target=[torch.tensor([0]), torch.tensor([1])],
        ...     baselines=OrderedDict({"input": torch.zeros(2, 10)}),
        ... )
        >>> mt_attributions = explainer_mt.explain(explanation_inputs_mt)
        >>> # Returns: [OrderedDict({"input": torch.Tensor}), OrderedDict({"input": torch.Tensor})]
    """

    def _init_single_target_explanation_fn(self) -> Callable:
        """Initialize single-target Input × Gradient attribution function.

        Returns:
            Captum InputXGradient attribution function for single targets.
        """
        return InputBaselineXGradient(self._model).attribute

    def _init_multi_target_explanation_fn(self) -> Callable:
        """Initialize multi-target Input × Gradient attribution function.

        Returns:
            MultiTargetInputXGradient attribution function for multiple targets.
        """
        return MultiTargetInputBaselineXGradient(
            self._model, grad_batch_size=self._grad_batch_size
        ).attribute

    def _build_inputs(
        self,
        inputs: OrderedDict[str, torch.Tensor] | torch.Tensor,
        target: ExplanationTargetType,
        baselines: OrderedDict[str, torch.Tensor] | torch.Tensor | None = None,
        additional_forward_args: tuple[Any, ...] | None = None,
    ):
        """Build ExplanationInputs from individual parameters."""
        return ExplanationInputs(
            inputs=inputs,
            target=target,
            baselines=baselines,
            additional_forward_args=additional_forward_args,
        )

    def explain(
        self,
        inputs: OrderedDict[str, torch.Tensor] | torch.Tensor,
        target: ExplanationTargetType,
        baselines: OrderedDict[str, torch.Tensor] | torch.Tensor | None = None,
        additional_forward_args: tuple[Any, ...] | None = None,
    ) -> OrderedDict[str, torch.Tensor] | list[OrderedDict[str, torch.Tensor]]:
        """Compute Input × Baseline Gradient attributions for the given inputs.

        This method provides a backward-compatible interface that accepts individual
        parameters and constructs ExplanationInputs internally before calling the
        parent class explain method.

        Args:
            inputs: Input tensors for attribution computation. Should be an OrderedDict
                mapping feature names to tensors when used with this explainer.
            target: Target indices for attribution computation. Can be a tensor
                (single-target) or list of tensors (multi-target).
            baselines: Baseline tensors representing reference values. Required for
                this method as it computes (input - baseline) × gradient.
            additional_forward_args: Additional arguments for model forward pass.

        Returns:
            For single-target mode: OrderedDict mapping feature names to attribution tensors.
            For multi-target mode: List of OrderedDicts, one per target.

        Examples:
            >>> # With explicit baselines
            >>> attributions = explainer.explain(
            ...     inputs=OrderedDict({"input": torch.randn(2, 10)}),
            ...     target=torch.tensor([0, 1]),
            ...     baselines=OrderedDict({"input": torch.zeros(2, 10)}),
            ... )
            >>>
            >>> # Multiple features with baselines
            >>> attributions = explainer.explain(
            ...     inputs=OrderedDict(
            ...         {"feat1": torch.randn(2, 5), "feat2": torch.randn(2, 5)}
            ...     ),
            ...     target=torch.tensor([0, 1]),
            ...     baselines=OrderedDict(
            ...         {"feat1": torch.zeros(2, 5), "feat2": torch.zeros(2, 5)}
            ...     ),
            ... )
        """
        return super().explain(
            inputs=inputs,
            target=target,
            baselines=baselines,
            additional_forward_args=additional_forward_args,
        )
