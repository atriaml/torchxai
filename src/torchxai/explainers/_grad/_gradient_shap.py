from collections.abc import Callable
from functools import partial
from typing import Any

import torch
from captum.attr import GradientShap
from captum.attr._core.gradient_shap import InputBaselineXGradient
from captum.attr._core.noise_tunnel import NoiseTunnel
from captum.attr._utils.common import _format_callable_baseline
from torch.nn.modules import Module

from torchxai.data_types import (
    BaselineType,
    ExplanationTargetType,
    TargetType,
    TensorOrTupleOfTensorsGeneric,
)
from torchxai.explainers._explainer import Explainer
from torchxai.explainers._grad._input_x_baseline_gradient import (
    MultiTargetInputBaselineXGradient,
)
from torchxai.explainers._grad._noise_tunnel import MultiTargetNoiseTunnel
from torchxai.explainers._utils import (
    _compute_gradients_sequential_autograd,
    _compute_gradients_vmap_autograd,
)


class GradientShap_(GradientShap):
    """Custom implementation of GradientShap with support for batch size in noise tunnel.

    This class extends Captum's GradientShap to provide better control over
    batch processing during noise tunnel operations, improving memory efficiency
    for large-scale attribution computations.

    Args:
        forward_func: The forward function of the model to be explained.
        multiply_by_inputs: Whether to multiply gradients by (input - baseline).
            Defaults to True.
    """

    def attribute(  # type: ignore
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines: BaselineType,
        n_samples: int = 5,
        n_samples_batch_size: int | None = None,
        stdevs: float | tuple[float, ...] = 0.0,
        target: TargetType = None,
        additional_forward_args: Any = None,
        return_convergence_delta: bool = False,
    ) -> (
        TensorOrTupleOfTensorsGeneric
        | tuple[TensorOrTupleOfTensorsGeneric, torch.Tensor]
    ):
        baselines = _format_callable_baseline(baselines, inputs)
        assert isinstance(baselines[0], torch.Tensor), (
            f"Baselines distribution has to be provided in a form of a torch.Tensor {baselines[0]}."
        )

        input_min_baseline_x_grad = InputBaselineXGradient(
            self.forward_func, self.multiplies_by_inputs
        )
        input_min_baseline_x_grad.gradient_func = self.gradient_func

        nt = NoiseTunnel(input_min_baseline_x_grad)

        attributions = nt.attribute.__wrapped__(  # type: ignore
            nt,  # self
            inputs,
            nt_type="smoothgrad",
            nt_samples=n_samples,
            nt_samples_batch_size=n_samples_batch_size,
            stdevs=stdevs,
            draw_baseline_from_distrib=True,
            baselines=baselines,
            target=target,
            additional_forward_args=additional_forward_args,
            return_convergence_delta=return_convergence_delta,
        )

        return attributions


class MultiTargetGradientShap(GradientShap):
    """Multi-target GradientShap attribution.

    This class extends Captum's GradientShap to support computing
    GradientShap attributions for multiple targets simultaneously.

    GradientShap combines the ideas of Integrated Gradients and SHAP by
    using a distribution of baselines and adding noise to compute more
    robust attribution estimates.

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
        self._gradient_func = gradient_func
        self._grad_batch_size = grad_batch_size

    def attribute(  # type: ignore
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines: TensorOrTupleOfTensorsGeneric
        | Callable[..., TensorOrTupleOfTensorsGeneric],
        n_samples: int = 5,
        n_samples_batch_size: int | None = None,
        stdevs: float | tuple[float, ...] = 0.0,
        target: TargetType = None,
        additional_forward_args: Any = None,
        return_convergence_delta: bool = False,
    ) -> (
        list[TensorOrTupleOfTensorsGeneric]
        | tuple[list[TensorOrTupleOfTensorsGeneric], list[torch.Tensor]]
    ):
        # since `baselines` is a distribution, we can generate it using a function
        # rather than passing it as an input argument
        baselines = _format_callable_baseline(baselines, inputs)
        assert isinstance(baselines[0], torch.Tensor), (
            "Baselines distribution has to be provided in a form "
            f"of a torch.Tensor {baselines[0]}."
        )

        input_min_baseline_x_grad = MultiTargetInputBaselineXGradient(
            self.forward_func,
            self.multiplies_by_inputs,
            gradient_func=self._gradient_func,
            grad_batch_size=self._grad_batch_size,
        )
        nt = MultiTargetNoiseTunnel(input_min_baseline_x_grad)

        attributions = nt.attribute(
            inputs,
            nt_type="smoothgrad",
            nt_samples=n_samples,
            nt_samples_batch_size=n_samples_batch_size,
            stdevs=stdevs,
            target=target,
            draw_baseline_from_distrib=True,
            baselines=baselines,
            additional_forward_args=additional_forward_args,
            return_convergence_delta=return_convergence_delta,
        )

        return attributions


class GradientShapExplainer(Explainer):
    """GradientShap explainer for computing noise-based Shapley value approximations.

    This explainer computes attributions using GradientShap, which combines ideas from
    Integrated Gradients and SHAP. It uses a distribution of baselines rather than a
    single baseline and adds noise to create more robust attribution estimates. The method
    approximates Shapley values through gradient-based computations. Supports both
    single-target and multi-target modes with structured input/output.

    GradientShap provides more robust attributions by using baseline distributions
    and noise, making it less sensitive to specific baseline choices.

    Args:
        model: The PyTorch model whose output is to be explained.
        multi_target: Whether to use multi-target mode. When True, can compute
            attributions for multiple targets simultaneously. Defaults to False.
        internal_batch_size: Batch size for internal computations. Defaults to 64.
        grad_batch_size: Batch size for gradient computations. Defaults to 64.
        n_samples: Number of random samples used to approximate Shapley values.
            Defaults to 25.
        return_convergence_delta: Whether to return convergence delta for
            completeness check. Defaults to False.

    Examples:
        Single-target usage:
        >>> import torch
        >>> from collections import OrderedDict
        >>> from torchxai.data_types import ExplanationInputs
        >>>
        >>> model = torch.nn.Sequential(
        ...     torch.nn.Linear(10, 5), torch.nn.ReLU(), torch.nn.Linear(5, 2)
        ... )
        >>> explainer = GradientShapExplainer(model, n_samples=50)
        >>>
        >>> # Create baseline distribution (required for GradientShap)
        >>> baseline_dist = torch.randn(100, 10)  # 100 baseline samples
        >>> explanation_inputs = ExplanationInputs(
        ...     inputs=OrderedDict({"input": torch.randn(2, 10)}),
        ...     target=torch.tensor([0, 1]),
        ...     baselines=OrderedDict({"input": baseline_dist}),
        ... )
        >>> attributions = explainer.explain(explanation_inputs)
        >>> # Returns: OrderedDict({"input": torch.Tensor})

        Multi-target usage:
        >>> explainer_mt = GradientShapExplainer(model, multi_target=True)
        >>> explanation_inputs_mt = ExplanationInputs(
        ...     inputs=OrderedDict({"input": torch.randn(2, 10)}),
        ...     target=[torch.tensor([0]), torch.tensor([1])],
        ...     baselines=OrderedDict({"input": baseline_dist}),
        ... )
        >>> mt_attributions = explainer_mt.explain(explanation_inputs_mt)
        >>> # Returns: [OrderedDict({"input": torch.Tensor}), OrderedDict({"input": torch.Tensor})]
    """

    def __init__(
        self,
        model: Module,
        multi_target: bool = False,
        internal_batch_size: int = 1,
        grad_batch_size: int = 64,
        n_samples: int = 25,
        stddevs: float = 0.0,
        return_convergence_delta: bool = False,
    ) -> None:
        """Initialize the GradientShapExplainer.

        Args:
            model: The model whose output is to be explained.
            multi_target: Whether to use multi-target mode. Defaults to False.
            internal_batch_size: Batch size for internal computations. Defaults to 1.
            grad_batch_size: Batch size for gradient computations. Defaults to 64.
            n_samples: Number of random samples for Shapley value approximation.
                Defaults to 25.
            return_convergence_delta: Whether to return convergence delta for
                completeness check. Defaults to False.
        """
        self._n_samples = n_samples
        self._return_convergence_delta = return_convergence_delta
        self._stdevs = stddevs

        super().__init__(
            model, multi_target, internal_batch_size, grad_batch_size=grad_batch_size
        )

    def _init_single_target_explanation_fn(self) -> Callable:
        """Initialize single-target GradientShap attribution function.

        Returns:
            Custom GradientShap attribution function for single targets.
        """
        return partial(
            GradientShap_(self._model).attribute,
            n_samples=self._n_samples,
            n_samples_batch_size=self._internal_batch_size,
            return_convergence_delta=self._return_convergence_delta,
            stdevs=self._stdevs,
        )

    def _init_multi_target_explanation_fn(self) -> Callable:
        """Initialize multi-target GradientShap attribution function.

        Returns:
            MultiTargetGradientShap attribution function for multiple targets.
        """
        return partial(
            MultiTargetGradientShap(
                self._model, grad_batch_size=self._grad_batch_size
            ).attribute,
            n_samples=self._n_samples,
            n_samples_batch_size=self._internal_batch_size,
            return_convergence_delta=self._return_convergence_delta,
            stdevs=self._stdevs,
        )

    def explain(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: ExplanationTargetType,
        baselines: TensorOrTupleOfTensorsGeneric | None = None,
        additional_forward_args: tuple[Any, ...] | None = None,
    ) -> TensorOrTupleOfTensorsGeneric | list[TensorOrTupleOfTensorsGeneric]:
        """Compute GradientShap attributions for the given inputs.

        This method provides a backward-compatible interface that accepts individual
        parameters and constructs ExplanationInputs internally before calling the
        parent class explain method.

        Args:
            inputs: Input tensors for attribution computation. Should be an OrderedDict
                mapping feature names to tensors when used with this explainer.
            target: Target indices for attribution computation. Can be a tensor
                (single-target) or list of tensors (multi-target).
            baselines: Baseline distribution for GradientShap. Must be provided as
                a tensor distribution or callable that generates baseline samples.
            additional_forward_args: Additional arguments for model forward pass.

        Returns:
            For single-target mode: OrderedDict mapping feature names to attribution tensors.
            For multi-target mode: List of OrderedDicts, one per target.

        Note:
            GradientShap requires a distribution of baselines rather than a single baseline.
            The number of samples and noise parameters are controlled by initialization settings.

        Examples:
            >>> # With baseline distribution
            >>> baseline_dist = torch.randn(50, 10)  # 50 baseline samples
            >>> attributions = explainer.explain(
            ...     inputs=OrderedDict({"input": torch.randn(2, 10)}),
            ...     target=torch.tensor([0, 1]),
            ...     baselines=OrderedDict({"input": baseline_dist}),
            ... )
        """
        return self._default_explain(
            inputs=inputs,
            target=target,
            baselines=baselines,
            additional_forward_args=additional_forward_args,
        )
