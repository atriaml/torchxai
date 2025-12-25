import warnings
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
from captum._utils.common import (
    _format_output,
    _format_tensor_into_tuples,
    _is_tuple,
    _register_backward_hook,
)
from captum._utils.gradient import (
    apply_gradient_requirements,
    undo_gradient_requirements,
)
from captum.attr import GradientAttribution, GuidedBackprop
from torch import Tensor
from torch.nn import Module
from torch.utils.hooks import RemovableHandle

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


class MultiTargetGuidedBackprop(GradientAttribution):
    """Multi-target Guided Backpropagation attribution.

    This class extends Captum's GradientAttribution to support computing
    Guided Backpropagation attributions for multiple targets simultaneously.

    Guided Backpropagation modifies ReLU backward passes to only backpropagate
    positive gradients, highlighting features that positively contribute to the prediction.

    Args:
        model: The PyTorch model instance.
        use_relu_grad_output: If True, performs Deconvolution instead of
            Guided Backpropagation. Defaults to False.
        gradient_func: Function for computing gradients. Automatically selects
            between vmap and sequential methods based on PyTorch version.
        grad_batch_size: Batch size for gradient computations. Defaults to 10.
    """

    def __init__(
        self,
        model: Module,
        use_relu_grad_output: bool = False,
        gradient_func=(
            _compute_gradients_vmap_autograd
            if torch.__version__ >= "2.3.0"
            else _compute_gradients_sequential_autograd
        ),
        grad_batch_size: int = 10,
    ) -> None:
        r"""
        Args:

            model (nn.Module): The reference to PyTorch model instance.
        """
        GradientAttribution.__init__(self, model)
        self._model = model
        self._backward_hooks: list[RemovableHandle] = []
        self._use_relu_grad_output = use_relu_grad_output
        self._gradient_func = gradient_func
        self._grad_batch_size = grad_batch_size
        assert isinstance(self._model, torch.nn.Module), (
            "Given model must be an instance of torch.nn.Module to properly hook"
            " ReLU layers."
        )

    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: list[TargetType],
        additional_forward_args: Any = None,
    ) -> list[TensorOrTupleOfTensorsGeneric]:
        """Compute multi-target Guided Backpropagation attributions.

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

        # set hooks for overriding ReLU gradients
        warnings.warn(
            "Setting backward hooks on ReLU activations."
            "The hooks will be removed after the attribution is finished",
            stacklevel=2,
        )
        try:
            self._model.apply(self._register_hooks)

            multi_target_gradients = self._gradient_func(
                self.forward_func,
                inputs,
                target,
                additional_forward_args,
                grad_batch_size=self._grad_batch_size,
            )
        finally:
            self._remove_hooks()

        undo_gradient_requirements(inputs, gradient_mask)
        return [
            _format_output(is_inputs_tuple, per_target_gradients)
            for per_target_gradients in multi_target_gradients
        ]

    def _register_hooks(self, module: Module):
        if isinstance(module, torch.nn.ReLU):
            hooks = _register_backward_hook(module, self._backward_hook, self)
            self._backward_hooks.extend(hooks)

    def _backward_hook(
        self,
        module: Module,
        grad_input: Tensor | tuple[Tensor, ...],
        grad_output: Tensor | tuple[Tensor, ...],
    ):
        to_override_grads = grad_output if self._use_relu_grad_output else grad_input
        if isinstance(to_override_grads, tuple):
            return tuple(
                F.relu(to_override_grad) for to_override_grad in to_override_grads
            )
        else:
            return F.relu(to_override_grads)

    def _remove_hooks(self):
        for hook in self._backward_hooks:
            hook.remove()


class GuidedBackpropExplainer(Explainer):
    """Guided Backpropagation explainer for computing modified gradient attributions.

    This explainer computes attributions using Guided Backpropagation, which modifies
    the backward pass through ReLU activations to only propagate positive gradients.
    This technique highlights features that have a positive influence on the prediction.
    Supports both single-target and multi-target modes with structured input/output.

    The Guided Backpropagation method helps visualize which input features contribute
    positively to the model's decision by suppressing negative gradients at ReLU layers.

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
        >>> model = torch.nn.Sequential(
        ...     torch.nn.Linear(10, 5), torch.nn.ReLU(), torch.nn.Linear(5, 2)
        ... )
        >>> explainer = GuidedBackpropExplainer(model)
        >>>
        >>> explanation_inputs = ExplanationInputs(
        ...     inputs=OrderedDict({"input": torch.randn(2, 10)}),
        ...     target=torch.tensor([0, 1]),
        ... )
        >>> attributions = explainer.explain(explanation_inputs)
        >>> # Returns: OrderedDict({"input": torch.Tensor})

        Multi-target usage:
        >>> explainer_mt = GuidedBackpropExplainer(model, multi_target=True)
        >>> explanation_inputs_mt = ExplanationInputs(
        ...     inputs=OrderedDict({"input": torch.randn(2, 10)}),
        ...     target=[torch.tensor([0]), torch.tensor([1])],
        ... )
        >>> mt_attributions = explainer_mt.explain(explanation_inputs_mt)
        >>> # Returns: [OrderedDict({"input": torch.Tensor}), OrderedDict({"input": torch.Tensor})]
    """

    def _init_single_target_explanation_fn(self) -> Callable:
        """Initialize single-target Guided Backpropagation attribution function.

        Returns:
            Captum GuidedBackprop attribution function for single targets.
        """
        return GuidedBackprop(self._model).attribute

    def _init_multi_target_explanation_fn(self) -> Callable:
        """Initialize multi-target Guided Backpropagation attribution function.

        Returns:
            MultiTargetGuidedBackprop attribution function for multiple targets.
        """
        return MultiTargetGuidedBackprop(
            self._model, grad_batch_size=self._grad_batch_size
        ).attribute

    def explain(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: ExplanationTargetType,
        additional_forward_args: tuple[Any, ...] | None = None,
    ) -> TensorOrTupleOfTensorsGeneric | list[TensorOrTupleOfTensorsGeneric]:
        """Compute Guided Backpropagation attributions for the given inputs.

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

        Note:
            This method temporarily modifies ReLU backward hooks during computation.
            Hooks are automatically removed after attribution computation completes.

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
