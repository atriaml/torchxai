from collections.abc import Callable
from functools import partial
from typing import Any

import torch
from captum._utils.common import _format_output, _format_tensor_into_tuples, _is_tuple
from captum._utils.gradient import (
    apply_gradient_requirements,
    undo_gradient_requirements,
)
from captum.attr import GradientAttribution, Saliency

from torchxai.data_types import (
    ExplanationTargetType,
    TargetType,
    TensorOrTupleOfTensorsGeneric,
)
from torchxai.explainers._explainer import FeatureAttributionExplainer
from torchxai.explainers._utils import (
    _compute_gradients_sequential_autograd,
    _compute_gradients_vmap_autograd,
    _verify_target_for_multi_target_impl,
)


class MultiTargetSaliency(GradientAttribution):
    """Multi-target saliency attribution using gradients.

    This class extends Captum's GradientAttribution to support computing
    saliency attributions for multiple targets simultaneously.

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
            if torch.__version__ >= "2.1.0"
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
        abs: bool = True,
        additional_forward_args: Any = None,
    ) -> list[TensorOrTupleOfTensorsGeneric]:
        """Compute multi-target saliency attributions.

        Args:
            inputs: Input tensors for which to compute attributions.
            target: List of target indices for multi-target attribution.
            abs: Whether to return absolute values of gradients. Defaults to True.
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

        # No need to format additional_forward_args here.
        # They are being formated in the `_run_forward` function in `common.py`
        multi_target_gradients = self._gradient_func(
            self.forward_func,
            inputs,
            target,
            additional_forward_args,
            grad_batch_size=self._grad_batch_size,
        )

        def gradients_to_attributions(gradients):
            if abs:
                attributions = tuple(torch.abs(gradient) for gradient in gradients)
            else:
                attributions = gradients
            return attributions

        multi_target_attributions = [
            gradients_to_attributions(per_target_grad)
            for per_target_grad in multi_target_gradients
        ]

        undo_gradient_requirements(inputs, gradient_mask)
        return [
            _format_output(is_inputs_tuple, attributions)
            for attributions in multi_target_attributions
        ]


class SaliencyExplainer(FeatureAttributionExplainer):
    """Saliency explainer for computing gradient-based attributions.

    This explainer computes saliency maps using gradients of the model output
    with respect to inputs. Supports both single-target and multi-target modes
    for both single-target and multi-target scenarios.

    The saliency method computes the gradient of the output with respect to the input,
    providing a measure of how much each input feature contributes to the prediction.
    Raw gradients are returned (abs=False) to preserve sign information.

    Args:
        model: The PyTorch model whose output is to be explained.
        multi_target: Whether to use multi-target mode. When True, can compute
            attributions for multiple targets simultaneously. Defaults to False.
        internal_batch_size: Batch size for internal computations. Defaults to 64.
        grad_batch_size: Batch size for gradient computations. Defaults to 64.

    Examples:
        Single-target usage:
        >>> import torch
        >>> from torchxai.data_types import SingleTargetAcrossBatch
        >>>
        >>> model = torch.nn.Linear(10, 2)
        >>> explainer = SaliencyExplainer(model)
        >>> attributions = explainer.explain(
        ...     inputs=torch.randn(1, 10),
        ...     target=SingleTargetAcrossBatch(index=0),
        ... )
        >>> attributions.shape   # (1, 10)

        Multi-target usage:
        >>> explainer_mt = SaliencyExplainer(model, multi_target=True)
        >>> mt_attributions = explainer_mt.explain(
        ...     inputs=torch.randn(1, 10),
        ...     target=[SingleTargetAcrossBatch(index=0), SingleTargetAcrossBatch(index=1)],
        ... )
        >>> len(mt_attributions), mt_attributions[0].shape   # 2, (1, 10)
    """

    def _init_single_target_explanation_fn(self) -> Callable:
        """Initialize single-target saliency attribution function.

        Returns:
            Captum Saliency attribution function for single targets.
        """
        return partial(Saliency(self._model).attribute, abs=False)

    def _init_multi_target_explanation_fn(self) -> Callable:
        """Initialize multi-target saliency attribution function.

        Returns:
            MultiTargetSaliency attribution function for multiple targets.
        """
        return partial(
            MultiTargetSaliency(
                self._model, grad_batch_size=self._grad_batch_size
            ).attribute,
            abs=False,
        )

    def explain(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: ExplanationTargetType | list[ExplanationTargetType],
        additional_forward_args: tuple[Any, ...] | None = None,
    ) -> TensorOrTupleOfTensorsGeneric | list[TensorOrTupleOfTensorsGeneric]:
        """Compute saliency attributions for the given inputs.

        Args:
            inputs: Input tensor(s) for attribution computation.
            target: An `ExplanationTargetType` (e.g. `SingleTargetAcrossBatch`) for single-target
                mode, or a list of them for multi-target mode.
            additional_forward_args: Additional arguments for model forward pass.

        Returns:
            Tensor in single-target mode. List of Tensors, one per target, in multi-target mode.

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
