import typing
import warnings
from collections import OrderedDict
from collections.abc import Callable
from functools import partial
from typing import Any

import torch
from captum._utils.common import (
    _expand_target,
    _format_additional_forward_args,
    _format_baseline,
    _format_output,
    _format_tensor_into_tuples,
    _is_tuple,
    _run_forward,
    _select_targets,
)
from captum._utils.gradient import (
    apply_gradient_requirements,
    undo_gradient_requirements,
)
from captum.attr import DeepLift
from captum.attr._core.deep_lift import SUPPORTED_NON_LINEAR, nonlinear
from captum.attr._utils.common import (
    _call_custom_attribution_func,
    _tensorize_baseline,
    _validate_input,
)
from torch import Tensor, nn
from torch.nn import Module

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

# replace the softmax with nonlinear as the normalization in the softmax function is not invariant to the batch size!
# the softmax implementation results in differnt deltas for higher or lower batch sizes. Seems incorrect!
# also see https://github.com/pytorch/captum/issues/519
SUPPORTED_NON_LINEAR[nn.Softmax] = nonlinear


class MultiTargetDeepLift(DeepLift):
    """Multi-target DeepLIFT attribution.

    This class extends Captum's DeepLift to support computing
    DeepLIFT attributions for multiple targets simultaneously.

    DeepLIFT assigns contribution scores based on the difference from a reference
    baseline, propagating relevance scores through the network while handling
    non-linear activations appropriately.

    Args:
        model: The PyTorch model instance.
        multiply_by_inputs: Whether to multiply gradients by inputs. Defaults to True.
        eps: Small value to avoid division by zero. Defaults to 1e-10.
        gradient_func: Function for computing gradients. Automatically selects
            between vmap and sequential methods based on PyTorch version.
        grad_batch_size: Batch size for gradient computations. Defaults to 10.
    """

    def __init__(
        self,
        model: Module,
        multiply_by_inputs: bool = True,
        eps: float = 1e-10,
        gradient_func=(
            _compute_gradients_vmap_autograd
            if torch.__version__ >= "2.3.0"
            else _compute_gradients_sequential_autograd
        ),
        grad_batch_size: int = 10,
    ) -> None:
        super().__init__(model, multiply_by_inputs, eps)

        self.gradient_func = gradient_func
        self.grad_batch_size = grad_batch_size

    def attribute(  # type: ignore[override]
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: list[TargetType],
        baselines: BaselineType = None,
        additional_forward_args: Any = None,
        return_convergence_delta: bool = False,
        custom_attribution_func: None | Callable[..., tuple[Tensor, ...]] = None,
    ) -> (
        tuple[list[TensorOrTupleOfTensorsGeneric], list[Tensor]]
        | list[TensorOrTupleOfTensorsGeneric]
    ):
        """Compute multi-target DeepLIFT attributions.

        Args:
            inputs: Input tensors for which to compute attributions.
            baselines: Baseline tensors representing reference values. If None,
                uses zero baselines.
            target: List of target indices for multi-target attribution.
            additional_forward_args: Additional arguments for the forward function.
            return_convergence_delta: Whether to return convergence delta for
                completeness check. Defaults to False.
            custom_attribution_func: Custom function for computing final attributions
                from gradients. Defaults to None.

        Returns:
            List of attribution tensors, one for each target in the target list.
            If return_convergence_delta is True, returns tuples of (attributions, delta).
        """
        # Keeps track whether original input is a tuple or not before
        # converting it into a tuple.
        is_inputs_tuple = _is_tuple(inputs)

        inputs = _format_tensor_into_tuples(inputs)
        baselines = typing.cast(
            tuple[torch.Tensor, ...], _format_baseline(baselines, inputs)
        )

        gradient_mask = apply_gradient_requirements(inputs)

        _validate_input(inputs, baselines)

        # verify that the target is valid
        _verify_target_for_multi_target_impl(inputs, target)

        # set hooks for baselines
        warnings.warn(
            """Setting forward, backward hooks and attributes on non-linear
            activations. The hooks and attributes will be removed
            after the attribution is finished""",
            stacklevel=2,
        )
        baselines = _tensorize_baseline(inputs, baselines)
        main_model_hooks = []
        try:
            main_model_hooks = self._hook_main_model()

            self.model.apply(self._register_hooks)

            additional_forward_args = _format_additional_forward_args(
                additional_forward_args
            )

            expanded_target = [_expand_target(t, 2) for t in target]
            wrapped_forward_func = self._construct_forward_func(
                self.model,
                (inputs, baselines),
                expanded_target,
                additional_forward_args,
            )
            multi_target_gradients = self.gradient_func(
                wrapped_forward_func, inputs, grad_batch_size=self.grad_batch_size
            )

            def gradients_to_attributions(gradients):
                if custom_attribution_func is None:
                    if self.multiplies_by_inputs:
                        attributions = tuple(
                            (input - baseline) * gradient
                            for input, baseline, gradient in zip(
                                inputs, baselines, gradients, strict=False
                            )
                        )
                    else:
                        attributions = gradients
                else:
                    attributions = _call_custom_attribution_func(
                        custom_attribution_func, gradients, inputs, baselines
                    )
                return attributions

            multi_target_attributions = [
                gradients_to_attributions(per_target_grad)
                for per_target_grad in multi_target_gradients
            ]
        finally:
            # Even if any error is raised, remove all hooks before raising
            self._remove_hooks(main_model_hooks)

        undo_gradient_requirements(inputs, gradient_mask)

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

    def _construct_forward_func(
        self,
        forward_func: Callable,
        inputs: tuple,
        target: TargetType | list[TargetType] | None = None,
        additional_forward_args: Any = None,
    ) -> Callable:
        def forward_fn():
            model_out = _run_forward(
                forward_func, inputs, None, additional_forward_args
            )
            if isinstance(target, (tuple, list)):
                return torch.stack(
                    [
                        _select_targets(
                            torch.cat((model_out[:, 0], model_out[:, 1])), single_target
                        )
                        for single_target in target
                    ],
                    dim=1,
                )
            else:
                return _select_targets(
                    torch.cat((model_out[:, 0], model_out[:, 1])), target
                )

        if hasattr(forward_func, "device_ids"):
            forward_fn.device_ids = forward_func.device_ids  # type: ignore
        return forward_fn

    def _backward_hook(
        self, module: Module, grad_input: Tensor, grad_output: Tensor
    ) -> Tensor:
        r"""
        `grad_input` is the gradient of the neuron with respect to its input
        `grad_output` is the gradient of the neuron with respect to its output
        we can override `grad_input` according to chain rule with.
        `grad_output` * delta_out / delta_in.

        """
        # before accessing the attributes from the module we want
        # to ensure that the properties exist, if not, then it is
        # likely that the module is being reused.
        attr_criteria = self.satisfies_attribute_criteria(module)
        if not attr_criteria:
            raise RuntimeError(
                f"A Module {module} was detected that does not contain some of "
                "the input/output attributes that are required for DeepLift "
                "computations. This can occur, for example, if "
                "your module is being used more than once in the network."
                "Please, ensure that module is being used only once in the "
                "network."
            )

        multipliers = SUPPORTED_NON_LINEAR[type(module)](
            module, module.input, module.output, grad_input, grad_output, eps=self.eps
        )

        # in deeplift we delete the input/output attributes but in multi-target case, we keep them as
        # we need them for computing attributions for multiple targets during autograd pass
        # del module.input
        # del module.output

        return multipliers


class DeepLiftExplainer(Explainer):
    """DeepLIFT explainer for computing reference-based attributions.

    This explainer computes attributions using DeepLIFT (Deep Learning Important FeaTures),
    which assigns contribution scores based on the difference from a reference baseline.
    DeepLIFT handles non-linear activations by decomposing them into linear components
    and properly attributing relevance through the network. Supports both single-target
    and multi-target modes with structured input/output.

    The DeepLIFT method provides more stable attributions than simple gradients by
    using reference baselines and handling activation functions appropriately.

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
        >>> explainer = DeepLiftExplainer(model)
        >>>
        >>> explanation_inputs = ExplanationInputs(
        ...     inputs=OrderedDict({"input": torch.randn(2, 10)}),
        ...     target=torch.tensor([0, 1]),
        ...     baselines=OrderedDict({"input": torch.zeros(2, 10)}),
        ... )
        >>> attributions = explainer.explain(explanation_inputs)
        >>> # Returns: OrderedDict({"input": torch.Tensor})

        Multi-target usage:
        >>> explainer_mt = DeepLiftExplainer(model, multi_target=True)
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
        internal_batch_size: int = 64,
        grad_batch_size: int = 64,
    ) -> None:
        super().__init__(model, multi_target, internal_batch_size, grad_batch_size)

    def _init_single_target_explanation_fn(self) -> Callable:
        """Initialize single-target DeepLIFT attribution function.

        Returns:
            Captum DeepLift attribution function for single targets.
        """
        return partial(DeepLift(self._model).attribute)  # type: ignore[return-value]

    def _init_multi_target_explanation_fn(self) -> Callable:
        """Initialize multi-target DeepLIFT attribution function.

        Returns:
            MultiTargetDeepLift attribution function for multiple targets.
        """
        return partial(
            MultiTargetDeepLift(
                self._model, grad_batch_size=self._grad_batch_size
            ).attribute
        )

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
        """Compute DeepLIFT attributions for the given inputs.

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

        Note:
            This method temporarily modifies activation functions during computation.
            Hooks and attributes are automatically removed after attribution computation.

        Examples:
            >>> # With explicit baselines
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
