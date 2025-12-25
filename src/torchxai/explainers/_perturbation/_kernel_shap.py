from collections.abc import Callable, Generator
from functools import partial
from typing import Any

import torch
from captum._utils.common import _format_additional_forward_args
from captum._utils.models.linear_model import SkLearnLinearRegression
from captum.attr._core.lime import construct_feature_mask
from captum.attr._utils.common import _format_input_baseline
from torch import Tensor
from torch.distributions.categorical import Categorical
from torch.nn import Module

from torchxai.data_types import (
    BaselineType,
    ExplanationTargetType,
    TargetType,
    TensorOrTupleOfTensorsGeneric,
)
from torchxai.explainers._perturbation._lime import Lime, MultiTargetLime
from torchxai.explainers._utils import (
    _expand_feature_mask_to_target,
    _weight_attributions,
)
from torchxai.explainers.explainer import Explainer


def kernel_shap_frozen_features_perturb_generator(
    original_inp: Tensor | tuple[Tensor, ...], **kwargs
) -> Generator[Tensor, None, None]:
    assert "num_select_distribution" in kwargs and "num_interp_features" in kwargs, (
        "num_select_distribution and num_interp_features are necessary"
        " to use kernel_shap_perturb_func"
    )
    if isinstance(original_inp, Tensor):
        device = original_inp.device
    else:
        device = original_inp[0].device
    num_features = kwargs["num_interp_features"]
    yield torch.ones(1, num_features, device=device, dtype=torch.long)
    perturbed = torch.zeros(1, num_features, device=device, dtype=torch.long)
    if "frozen_features" in kwargs and kwargs["frozen_features"] is not None:
        perturbed[0, kwargs["frozen_features"]] = (
            1  # freeze the features, useful for padding/cls/sep tokens in sequences
        )
    yield perturbed
    while True:
        num_selected_features = kwargs["num_select_distribution"].sample()
        rand_vals = torch.randn(1, num_features)
        threshold = torch.kthvalue(
            rand_vals, num_features - num_selected_features
        ).values.item()
        perturbed = (rand_vals > threshold).to(device=device).long()
        if "frozen_features" in kwargs and kwargs["frozen_features"] is not None:
            perturbed[0, kwargs["frozen_features"]] = 1
        yield perturbed


class KernelShap(Lime):
    """Kernel SHAP attribution method using LIME framework.

    Kernel SHAP is a method that uses the LIME framework to compute Shapley Values
    by setting appropriate loss functions, weighting kernels and regularization terms
    to theoretically obtain Shapley Values more efficiently than direct computation.
    """

    def __init__(self, forward_func: Callable) -> None:
        r"""
        Args:

            forward_func (Callable): The forward function of the model or
                        any modification of it.
        """
        Lime.__init__(
            self,
            forward_func,
            interpretable_model=SkLearnLinearRegression(),
            similarity_func=self.kernel_shap_similarity_kernel,
            perturb_func=kernel_shap_frozen_features_perturb_generator,
        )
        self.inf_weight = 1000000.0

    def attribute(  # type: ignore
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Any = None,
        feature_mask: None | Tensor | tuple[Tensor, ...] = None,
        n_samples: int = 25,
        perturbations_per_eval: int = 1,
        frozen_features: list[torch.Tensor] | None = None,
        return_input_shape: bool = True,
        show_progress: bool = False,
    ) -> TensorOrTupleOfTensorsGeneric:
        formatted_inputs, baselines = _format_input_baseline(inputs, baselines)
        feature_mask, num_interp_features = construct_feature_mask(
            feature_mask, formatted_inputs
        )
        num_features_list = torch.arange(num_interp_features, dtype=torch.float)
        denom = num_features_list * (num_interp_features - num_features_list)
        probs = (num_interp_features - 1) / denom
        probs[0] = 0.0
        return self._attribute_kwargs(
            inputs=inputs,
            baselines=baselines,
            target=target,
            additional_forward_args=additional_forward_args,
            feature_mask=feature_mask,
            n_samples=n_samples,
            perturbations_per_eval=perturbations_per_eval,
            frozen_features=frozen_features,
            return_input_shape=return_input_shape,
            num_select_distribution=Categorical(probs),
            show_progress=show_progress,
        )

    def kernel_shap_similarity_kernel(
        self, _, __, interpretable_sample: Tensor, **kwargs
    ) -> Tensor:
        assert "num_interp_features" in kwargs, (
            "Must provide num_interp_features to use default similarity kernel"
        )
        num_selected_features = int(interpretable_sample.sum(dim=1).item())
        num_features = kwargs["num_interp_features"]
        if num_selected_features == 0 or num_selected_features == num_features:
            # weight should be theoretically infinite when
            # num_selected_features = 0 or num_features
            # enforcing that trained linear model must satisfy
            # end-point criteria. In practice, it is sufficient to
            # make this weight substantially larger so setting this
            # weight to 1000000 (all other weights are 1).
            similarities = self.inf_weight
        else:
            similarities = 1.0
        return torch.tensor([similarities])


class MultiTargetKernelShap(MultiTargetLime):
    """Multi-target Kernel SHAP attribution method.

    This class extends MultiTargetLime to support computing Kernel SHAP attributions
    for multiple targets simultaneously using the LIME framework with SHAP weighting.
    """

    def __init__(self, forward_func: Callable) -> None:
        r"""
        Args:

            forward_func (Callable): The forward function of the model or
                        any modification of it.
        """
        super().__init__(
            forward_func,
            interpretable_model=SkLearnLinearRegression(),
            similarity_func=self.kernel_shap_similarity_kernel,
            perturb_func=kernel_shap_frozen_features_perturb_generator,
        )
        self.inf_weight = 1000000.0

    def attribute(  # type: ignore
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: list[TargetType],
        baselines: BaselineType = None,
        additional_forward_args: Any = None,
        feature_mask: None | Tensor | tuple[Tensor, ...] = None,
        n_samples: int = 25,
        perturbations_per_eval: int = 1,
        frozen_features: list[torch.Tensor] | None = None,
        return_input_shape: bool = True,
        show_progress: bool = False,
    ) -> list[TensorOrTupleOfTensorsGeneric] | TensorOrTupleOfTensorsGeneric:
        formatted_inputs, baselines = _format_input_baseline(inputs, baselines)
        feature_mask, num_interp_features = construct_feature_mask(
            feature_mask, formatted_inputs
        )
        num_features_list = torch.arange(num_interp_features, dtype=torch.float)
        denom = num_features_list * (num_interp_features - num_features_list)
        probs = (num_interp_features - 1) / denom
        probs[0] = 0.0
        return self._attribute_kwargs(
            inputs=inputs,
            baselines=baselines,
            target=target,
            additional_forward_args=additional_forward_args,
            feature_mask=feature_mask,
            n_samples=n_samples,
            perturbations_per_eval=perturbations_per_eval,
            frozen_features=frozen_features,
            return_input_shape=return_input_shape,
            num_select_distribution=Categorical(probs),
            show_progress=show_progress,
        )

    def kernel_shap_similarity_kernel(
        self, _, __, interpretable_sample: Tensor, **kwargs
    ) -> Tensor:
        assert "num_interp_features" in kwargs, (
            "Must provide num_interp_features to use default similarity kernel"
        )
        num_selected_features = int(interpretable_sample.sum(dim=1).item())
        num_features = kwargs["num_interp_features"]
        if num_selected_features == 0 or num_selected_features == num_features:
            # weight should be theoretically infinite when
            # num_selected_features = 0 or num_features
            # enforcing that trained linear model must satisfy
            # end-point criteria. In practice, it is sufficient to
            # make this weight substantially larger so setting this
            # weight to 1000000 (all other weights are 1).
            similarities = self.inf_weight
        else:
            similarities = 1.0
        return torch.tensor([similarities])


class KernelShapExplainer(Explainer):
    """Kernel SHAP explainer for computing Shapley values using LIME framework.

    This explainer computes attributions using Kernel SHAP, which uses the LIME framework
    with specific weighting and sampling strategies to efficiently compute Shapley values.
    Kernel SHAP provides theoretically grounded explanations that satisfy Shapley value
    axioms (efficiency, symmetry, dummy, additivity) while being more computationally
    efficient than direct Shapley value computation. Supports both single-target and
    multi-target modes with structured input/output.

    Kernel SHAP is particularly effective for tabular data and provides globally
    consistent explanations across different inputs.

    Args:
        model: The PyTorch model whose output is to be explained.
        multi_target: Whether to use multi-target mode. When True, can compute
            attributions for multiple targets simultaneously. Defaults to False.
        internal_batch_size: Batch size for internal computations (perturbations
            per evaluation). Defaults to 1.
        n_samples: Number of coalition samples for Shapley value estimation.
            Defaults to 100.
        weight_attributions: Whether to weight attributions by feature group sizes
            when using feature masks. Defaults to True.

    Examples:
        Single-target usage for tabular data:
        >>> import torch
        >>> from collections import OrderedDict
        >>> from torchxai.data_types import ExplanationInputs
        >>>
        >>> model = torch.nn.Sequential(
        ...     torch.nn.Linear(10, 5), torch.nn.ReLU(), torch.nn.Linear(5, 2)
        ... )
        >>> explainer = KernelShapExplainer(model, n_samples=500)
        >>>
        >>> explanation_inputs = ExplanationInputs(
        ...     inputs=OrderedDict({"features": torch.randn(1, 10)}),
        ...     target=torch.tensor([1]),
        ...     baselines=OrderedDict({"features": torch.zeros(1, 10)}),
        ... )
        >>> attributions = explainer.explain(explanation_inputs)
        >>> # Returns: OrderedDict({"features": torch.Tensor})

        Multi-target usage:
        >>> explainer_mt = KernelShapExplainer(model, multi_target=True, n_samples=500)
        >>> explanation_inputs_mt = ExplanationInputs(
        ...     inputs=OrderedDict({"features": torch.randn(1, 10)}),
        ...     target=[torch.tensor([0]), torch.tensor([1])],
        ...     baselines=OrderedDict({"features": torch.zeros(1, 10)}),
        ... )
        >>> mt_attributions = explainer_mt.explain(explanation_inputs_mt)
        >>> # Returns: [OrderedDict({"features": torch.Tensor}), OrderedDict({"features": torch.Tensor})]
    """

    def __init__(
        self,
        model: Module,
        multi_target: bool = False,
        internal_batch_size: int = 1,
        n_samples: int = 100,
        weight_attributions: bool = True,
    ) -> None:
        """Initialize the KernelShapExplainer.

        Args:
            model: The model whose output is to be explained.
            multi_target: Whether to use multi-target mode. Defaults to False.
            internal_batch_size: Batch size for internal computations. Defaults to 1.
            n_samples: Number of coalition samples for Shapley estimation. Defaults to 100.
            weight_attributions: Whether to weight attributions by feature groups. Defaults to True.
        """
        self._n_samples = n_samples
        self._weight_attributions = weight_attributions

        super().__init__(model, multi_target, internal_batch_size)

    def _init_single_target_explanation_fn(self) -> Callable:
        """Initialize single-target Kernel SHAP attribution function.

        Returns:
            KernelShap attribution function for single targets.
        """
        expl_func = partial(
            KernelShap(self._model).attribute,
            n_samples=self._n_samples,
            perturbations_per_eval=self._internal_batch_size,
        )
        return self._expl_fn_with_post_process(expl_func)

    def _init_multi_target_explanation_fn(self) -> Callable:
        """Initialize multi-target Kernel SHAP attribution function.

        Returns:
            MultiTargetKernelShap attribution function for multiple targets.
        """
        expl_func = partial(
            MultiTargetKernelShap(self._model).attribute,
            n_samples=self._n_samples,
            perturbations_per_eval=self._internal_batch_size,
        )
        return self._expl_fn_with_post_process(expl_func)

    def _expl_fn_with_post_process(self, expl_func: Callable) -> Callable:
        def wrapped(
            inputs: TensorOrTupleOfTensorsGeneric,
            baselines: BaselineType = None,
            target: TargetType = None,
            additional_forward_args: Any = None,
            feature_mask: None | Tensor | tuple[Tensor, ...] = None,
            **kwargs,
        ) -> TensorOrTupleOfTensorsGeneric | list[TensorOrTupleOfTensorsGeneric]:
            feature_mask = (
                _expand_feature_mask_to_target(feature_mask, inputs)
                if feature_mask is not None
                else None
            )
            additional_forward_args = _format_additional_forward_args(
                additional_forward_args
            )

            attributions = expl_func(
                inputs=inputs,
                baselines=baselines,
                target=target,
                additional_forward_args=additional_forward_args,
                feature_mask=feature_mask,
                **kwargs,
            )

            # Apply feature weighting if requested
            if self._weight_attributions and feature_mask is not None:
                if self._multi_target:
                    attributions = [
                        _weight_attributions(attribution, feature_mask)
                        for attribution in attributions
                    ]
                else:
                    attributions = _weight_attributions(attributions, feature_mask)
            return attributions

        return wrapped

    def explain(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: ExplanationTargetType,
        baselines: TensorOrTupleOfTensorsGeneric | None = None,
        feature_mask: TensorOrTupleOfTensorsGeneric | None = None,
        additional_forward_args: tuple[Any, ...] | None = None,
        frozen_features: list[torch.Tensor] | None = None,
    ) -> TensorOrTupleOfTensorsGeneric | list[TensorOrTupleOfTensorsGeneric]:
        """Compute Kernel SHAP attributions for the given inputs.

        This method provides a backward-compatible interface that accepts individual
        parameters and constructs ExplanationInputs internally before calling the
        parent class explain method.

        Args:
            inputs: Input tensors for attribution computation. Should be an OrderedDict
                mapping feature names to tensors when used with this explainer.
            target: Target indices for attribution computation. Can be a tensor
                (single-target) or list of tensors (multi-target).
            baselines: Baseline tensors for coalition sampling (typically zeros).
                If None, uses zero baselines matching input shape.
            feature_mask: Masks representing feature groups for aggregation. Features
                with the same mask value are treated as a single coalition member.
            additional_forward_args: Additional arguments for model forward pass.
            frozen_features: List of feature indices to keep unchanged during perturbation.
                Useful for special tokens like CLS, SEP in NLP models.

        Returns:
            For single-target mode: OrderedDict mapping feature names to attribution tensors.
            For multi-target mode: List of OrderedDicts, one per target.

        Note:
            Kernel SHAP uses coalition sampling with SHAP-specific weighting to estimate
            Shapley values. The number of samples significantly affects accuracy and
            computation time. More samples provide better Shapley value approximations.

        Examples:
            >>> # For tabular data with feature grouping
            >>> feature_mask = torch.tensor([[0, 0, 1, 1, 2, 2, 2, 3, 3, 4]])
            >>> attributions = explainer.explain(
            ...     inputs=OrderedDict({"features": torch.randn(1, 10)}),
            ...     target=torch.tensor([1]),
            ...     baselines=OrderedDict({"features": torch.zeros(1, 10)}),
            ...     feature_mask=feature_mask,
            ... )
        """
        return self._default_explain(
            inputs=inputs,
            target=target,
            baselines=baselines,
            feature_mask=feature_mask,
            additional_forward_args=additional_forward_args,
            frozen_features=frozen_features,
        )
