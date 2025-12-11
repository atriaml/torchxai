from collections import OrderedDict
from collections.abc import Callable
from functools import partial
from typing import Any

import torch
from captum._utils.common import (
    _expand_additional_forward_args,
    _expand_target,
    _format_additional_forward_args,
    _format_output,
    _is_tuple,
)
from captum.attr import IntegratedGradients
from captum.attr._utils.approximation_methods import approximation_parameters
from captum.attr._utils.common import (
    _format_input_baseline,
    _reshape_and_sum,
    _validate_input,
)
from torch import Tensor
from torch.nn import Module

from torchxai.data_types.common import (
    BaselineType,
    TargetType,
    TensorOrTupleOfTensorsGeneric,
)
from torchxai.explainers._utils import (
    _batch_attribution_multi_target,
    _compute_gradients_sequential_autograd,
    _compute_gradients_vmap_autograd,
    _verify_target_for_multi_target_impl,
)
from torchxai.explainers.explainer import Explainer


class MultiTargetIntegratedGradients(IntegratedGradients):
    """Multi-target Integrated Gradients attribution.

    This class extends Captum's IntegratedGradients to support computing
    Integrated Gradients attributions for multiple targets simultaneously.

    Integrated Gradients computes attributions by integrating gradients along
    a straight path from a baseline to the input, providing path-independent
    and axiomatically sound attributions.

    Args:
        forward_func: The forward function of the model to be explained.
        multiply_by_inputs: Whether to multiply gradients by (input - baseline).
            Defaults to True.
        gradient_func: Function for computing gradients. Automatically selects
            between vmap and sequential methods based on PyTorch version.
        grad_batch_size: Batch size for gradient computations. Defaults to 10.
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
        super().__init__(forward_func, multiply_by_inputs)
        self.gradient_func = gradient_func
        self.grad_batch_size = grad_batch_size

    def attribute(  # type: ignore
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: list[TargetType],
        baselines: BaselineType = None,
        additional_forward_args: Any = None,
        n_steps: int = 50,
        method: str = "gausslegendre",
        internal_batch_size: None | int = None,
        return_convergence_delta: bool = False,
    ) -> (
        tuple[list[TensorOrTupleOfTensorsGeneric], list[Tensor]]
        | list[TensorOrTupleOfTensorsGeneric]
    ):
        """Compute multi-target Integrated Gradients attributions.

        Args:
            inputs: Input tensors for which to compute attributions.
            baselines: Baseline tensors representing reference values. If None,
                uses zero baselines.
            target: List of target indices for multi-target attribution.
            additional_forward_args: Additional arguments for the forward function.
            n_steps: Number of steps for the integral approximation. Defaults to 50.
            method: Approximation method for the integral. Defaults to 'gausslegendre'.
            internal_batch_size: Batch size for internal computations.
            return_convergence_delta: Whether to return convergence delta for
                completeness check. Defaults to False.

        Returns:
            List of attribution tensors, one for each target in the target list.
            If return_convergence_delta is True, returns tuples of (attributions, delta).
        """
        # Keeps track whether original input is a tuple or not before
        # converting it into a tuple.
        is_inputs_tuple = _is_tuple(inputs)

        inputs, baselines = _format_input_baseline(inputs, baselines)  # t

        _validate_input(inputs, baselines, n_steps, method)

        # verify that the target is valid
        _verify_target_for_multi_target_impl(inputs, target)

        if internal_batch_size is not None:
            num_examples = inputs[0].shape[0]
            multi_target_attributions = _batch_attribution_multi_target(
                self,
                num_examples,
                internal_batch_size,
                n_steps,
                inputs=inputs,
                baselines=baselines,
                target=target,
                additional_forward_args=additional_forward_args,
                method=method,
            )
        else:
            multi_target_attributions = self._attribute(
                inputs=inputs,
                baselines=baselines,
                target=target,
                additional_forward_args=additional_forward_args,
                n_steps=n_steps,
                method=method,
            )

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

    def _attribute(  # type: ignore
        self,
        inputs: tuple[Tensor, ...],
        baselines: tuple[Tensor | int | float, ...],
        target: list[TargetType],
        additional_forward_args: Any = None,
        n_steps: int = 50,
        method: str = "gausslegendre",
        step_sizes_and_alphas: None | tuple[list[float], list[float]] = None,
    ) -> list[tuple[Tensor, ...]]:  # Fix: Correct return type
        if step_sizes_and_alphas is None:
            # retrieve step size and scaling factor for specified
            # approximation method
            step_sizes_func, alphas_func = approximation_parameters(method)
            step_sizes, alphas = step_sizes_func(n_steps), alphas_func(n_steps)
        else:
            step_sizes, alphas = step_sizes_and_alphas

        # scale features and compute gradients. (batch size is abbreviated as bsz)
        # scaled_features' dim -> (bsz * #steps x inputs[0].shape[1:], ...)
        scaled_features_tpl = tuple(
            torch.cat(
                [baseline + alpha * (input - baseline) for alpha in alphas], dim=0
            ).requires_grad_()
            for input, baseline in zip(inputs, baselines, strict=False)
        )

        additional_forward_args = _format_additional_forward_args(
            additional_forward_args
        )
        # apply number of steps to additional forward args
        # currently, number of steps is applied only to additional forward arguments
        # that are nd-tensors. It is assumed that the first dimension is
        # the number of batches.
        # dim -> (bsz * #steps x additional_forward_args[0].shape[1:], ...)
        input_additional_args = (
            _expand_additional_forward_args(additional_forward_args, n_steps)
            if additional_forward_args is not None
            else None
        )

        expanded_target = [_expand_target(t, n_steps) for t in target]

        # grads: dim -> (bsz * #steps x inputs[0].shape[1:], ...)
        multi_target_gradients = self.gradient_func(
            forward_fn=self.forward_func,
            inputs=scaled_features_tpl,
            target=expanded_target,
            additional_forward_args=input_additional_args,
            grad_batch_size=self.grad_batch_size,
        )

        # flattening grads so that we can multilpy it with step-size
        # calling contiguous to avoid `memory whole` problems
        def gradients_to_attributions(grads):
            scaled_grads = [
                grad.contiguous().view(n_steps, -1)
                * torch.tensor(step_sizes).view(n_steps, 1).to(grad.device)
                for grad in grads
            ]

            # aggregates across all steps for each tensor in the input tuple
            # total_grads has the same dimensionality as inputs
            total_grads = tuple(
                _reshape_and_sum(
                    scaled_grad, n_steps, grad.shape[0] // n_steps, grad.shape[1:]
                )
                for (scaled_grad, grad) in zip(scaled_grads, grads, strict=False)
            )

            # computes attribution for each tensor in input tuple
            # attributions has the same dimensionality as inputs
            if not self.multiplies_by_inputs:
                attributions = total_grads
            else:
                attributions = tuple(
                    total_grad * (input - baseline)
                    for total_grad, input, baseline in zip(
                        total_grads, inputs, baselines, strict=False
                    )
                )
            return attributions

        multi_target_gradients = [
            gradients_to_attributions(grad) for grad in multi_target_gradients
        ]

        return multi_target_gradients


class IntegratedGradientsExplainer(Explainer):
    """Integrated Gradients explainer for computing path-integrated attributions.

    This explainer computes attributions using Integrated Gradients, which integrates
    gradients along a straight path from a baseline input to the actual input.
    This method satisfies important axioms including sensitivity and implementation
    invariance, making it a robust attribution method. Supports both single-target
    and multi-target modes with structured input/output.

    The Integrated Gradients method provides theoretically grounded attributions
    by computing the integral of gradients along the path from baseline to input.

    Args:
        model: The PyTorch model whose output is to be explained.
        multi_target: Whether to use multi-target mode. When True, can compute
            attributions for multiple targets simultaneously. Defaults to False.
        internal_batch_size: Batch size for internal computations. Defaults to 64.
        grad_batch_size: Batch size for gradient computations. Defaults to 64.
        n_steps: Number of steps for the integral approximation. Defaults to 50.

    Example:
        Single-target usage:
        >>> import torch
        >>> from collections import OrderedDict
        >>> from torchxai.data_types import ExplanationInputs
        >>>
        >>> model = torch.nn.Sequential(
        ...     torch.nn.Linear(10, 5), torch.nn.ReLU(), torch.nn.Linear(5, 2)
        ... )
        >>> explainer = IntegratedGradientsExplainer(model, n_steps=100)
        >>>
        >>> explanation_inputs = ExplanationInputs(
        ...     inputs=OrderedDict({"input": torch.randn(2, 10)}),
        ...     target=torch.tensor([0, 1]),
        ...     baselines=OrderedDict({"input": torch.zeros(2, 10)}),
        ... )
        >>> attributions = explainer.explain(explanation_inputs)
        >>> # Returns: OrderedDict({"input": torch.Tensor})

        Multi-target usage:
        >>> explainer_mt = IntegratedGradientsExplainer(model, multi_target=True)
        >>> explanation_inputs_mt = ExplanationInputs(
        ...     inputs=OrderedDict({"input": torch.randn(2, 10)}),
        ...     target=[torch.tensor([0]), torch.tensor([1])],
        ...     baselines=OrderedDict({"input": torch.zeros(2, 10)}),
        ... )
        >>> mt_attributions = explainer_mt.explain(explanation_inputs_mt)
        >>> # Returns: [OrderedDict({"input": torch.Tensor}), OrderedDict({"input": torch.Tensor})]
    """

    def __init__(
        self,
        model: Module,
        multi_target: bool = False,
        internal_batch_size: int = 50,
        grad_batch_size: int = 64,
        n_steps: int = 50,
        return_convergence_delta: bool = False,
    ) -> None:
        """Initialize the IntegratedGradientsExplainer.

        Args:
            model: The model whose output is to be explained.
            multi_target: Whether to use multi-target mode. Defaults to False.
            internal_batch_size: Batch size for internal computations. Defaults to 50.
            grad_batch_size: Batch size for gradient computations. Defaults to 64.
            n_steps: Number of steps for the integral approximation. Defaults to 50.
            return_convergence_delta: Whether to return convergence delta for
                completeness check. Defaults to False.
        """
        self.n_steps = n_steps
        self.return_convergence_delta = return_convergence_delta

        super().__init__(
            model, multi_target, internal_batch_size, grad_batch_size=grad_batch_size
        )

    def _init_single_target_explanation_fn(self) -> Callable:
        """Initialize single-target Integrated Gradients attribution function.

        Returns:
            Captum IntegratedGradients attribution function for single targets.
        """
        return partial(
            IntegratedGradients(self._model).attribute,
            n_steps=self.n_steps,
            return_convergence_delta=self.return_convergence_delta,
        )  # type: ignore

    def _init_multi_target_explanation_fn(self) -> Callable:
        """Initialize multi-target Integrated Gradients attribution function.

        Returns:
            MultiTargetIntegratedGradients attribution function for multiple targets.
        """
        return partial(
            MultiTargetIntegratedGradients(
                self._model, grad_batch_size=self._grad_batch_size
            ).attribute,
            n_steps=self.n_steps,
            return_convergence_delta=self.return_convergence_delta,
        )

    def explain(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: TargetType,
        baselines: BaselineType | None = None,
        additional_forward_args: Any = None,
    ) -> OrderedDict[str, torch.Tensor] | list[OrderedDict[str, torch.Tensor]]:
        """Compute Integrated Gradients attributions for the given inputs.

        This method provides a backward-compatible interface that accepts individual
        parameters and constructs ExplanationInputs internally before calling the
        parent class explain method.

        Args:
            inputs: Input tensors for attribution computation. Should be an OrderedDict
                mapping feature names to tensors when used with this explainer.
            target: Target indices for attribution computation. Can be a tensor
                (single-target) or list of tensors (multi-target).
            baselines: Baseline tensors representing reference values. If None,
                uses zero baselines. Should match the structure of inputs.
            additional_forward_args: Additional arguments for model forward pass.

        Returns:
            For single-target mode: OrderedDict mapping feature names to attribution tensors.
            For multi-target mode: List of OrderedDicts, one per target.

            If return_convergence_delta was set to True during initialization, the return
            format may include convergence delta information depending on the underlying
            implementation.

        Note:
            The number of integration steps and convergence delta behavior are controlled
            by parameters set during initialization (n_steps and return_convergence_delta).
            More steps generally provide more accurate approximations but increase computation time.

        Example:
            >>> # With convergence delta enabled at initialization
            >>> explainer = IntegratedGradientsExplainer(
            ...     model, n_steps=100, return_convergence_delta=True
            ... )
            >>> attributions = explainer.explain(
            ...     inputs=OrderedDict({"input": torch.randn(2, 10)}),
            ...     target=torch.tensor([0, 1]),
            ...     baselines=OrderedDict({"input": torch.zeros(2, 10)}),
            ... )
            >>>
            >>> # With automatic zero baselines
            >>> attributions = explainer.explain(
            ...     inputs=torch.randn(2, 10), target=torch.tensor([0, 1])
            ... )
        """
        return super().explain(
            inputs=inputs,
            target=target,
            baselines=baselines,
            additional_forward_args=additional_forward_args,
        )
