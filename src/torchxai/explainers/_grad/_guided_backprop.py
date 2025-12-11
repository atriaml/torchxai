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

from torchxai.data_types.common import TargetType, TensorOrTupleOfTensorsGeneric
from torchxai.explainers._utils import (
    _compute_gradients_sequential_autograd,
    _compute_gradients_vmap_autograd,
    _verify_target_for_multi_target_impl,
)
from torchxai.explainers.explainer import Explainer


class MultiTargetGuidedBackprop(GradientAttribution):
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
        self.model = model
        self.backward_hooks: list[RemovableHandle] = []
        self.use_relu_grad_output = use_relu_grad_output
        self.gradient_func = gradient_func
        self.grad_batch_size = grad_batch_size
        assert isinstance(self.model, torch.nn.Module), (
            "Given model must be an instance of torch.nn.Module to properly hook"
            " ReLU layers."
        )

    def attribute(  # type: ignore[override]
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: TargetType = None,
        additional_forward_args: Any = None,
    ) -> list[TensorOrTupleOfTensorsGeneric]:
        r"""
        Computes attribution by overriding relu gradients. Based on constructor
        flag use_relu_grad_output, performs either GuidedBackpropagation if False
        and Deconvolution if True. This class is the parent class of both these
        methods, more information on usage can be found in the docstrings for each
        implementing class.
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
            self.model.apply(self._register_hooks)

            multi_target_gradients = self.gradient_func(
                self.forward_func,
                inputs,
                target,
                additional_forward_args,
                grad_batch_size=self.grad_batch_size,
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
            self.backward_hooks.extend(hooks)

    def _backward_hook(
        self,
        module: Module,
        grad_input: Tensor | tuple[Tensor, ...],
        grad_output: Tensor | tuple[Tensor, ...],
    ):
        to_override_grads = grad_output if self.use_relu_grad_output else grad_input
        if isinstance(to_override_grads, tuple):
            return tuple(
                F.relu(to_override_grad) for to_override_grad in to_override_grads
            )
        else:
            return F.relu(to_override_grads)

    def _remove_hooks(self):
        for hook in self.backward_hooks:
            hook.remove()


class GuidedBackpropExplainer(Explainer):
    """
    A Explainer class for handling Guided Backpropagation attribution using the Captum library.

    Args:
        model (torch.nn.Module): The model whose output is to be explained.
    """

    def _init_explanation_fn(self) -> Callable:
        """
        Initializes the explanation function.

        Returns:
            Attribution: The initialized explanation function.
        """
        if self._is_multi_target:
            return MultiTargetGuidedBackprop(
                self._model, grad_batch_size=self._grad_batch_size
            ).attribute
        return GuidedBackprop(self._model).attribute

    def explain(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: TargetType,
        additional_forward_args: Any = None,
    ) -> TensorOrTupleOfTensorsGeneric:
        """
        Compute the Guided Backpropagation attributions for the given inputs.

        Args:
            inputs (TensorOrTupleOfTensorsGeneric): The input tensor(s) for which attributions are computed.
            target (TargetType): The target(s) for computing attributions.
            additional_forward_args (Any): Additional arguments to forward function.

        Returns:
            TensorOrTupleOfTensorsGeneric: The computed attributions.
        """
        return self._explanation_fn(
            inputs=inputs,
            target=target,
            additional_forward_args=additional_forward_args,
        )
