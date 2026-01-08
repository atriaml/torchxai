import math
import typing
import warnings
from collections.abc import Callable
from functools import partial
from typing import Any, Literal

import torch
from captum._utils.common import (
    _flatten_tensor_or_tuple,
    _format_additional_forward_args,
    _format_output,
    _is_tuple,
    _reduce_list,
    _run_forward,
)
from captum._utils.models.linear_model import SkLearnLasso
from captum._utils.models.model import Model
from captum.attr import LimeBase
from captum.attr._core.lime import (
    construct_feature_mask,
    default_from_interp_rep_transform,
    default_perturb_func,
)
from captum.attr._utils.batching import _batch_example_iterator
from captum.attr._utils.common import _format_input_baseline
from torch import Tensor
from torch.nn import CosineSimilarity, Module

from torchxai.data_types import (
    BaselineType,
    ExplanationTargetType,
    TargetType,
    TensorOrTupleOfTensorsGeneric,
)
from torchxai.explainers._explainer import Explainer
from torchxai.explainers._perturbation._lime_base import MultiTargetLimeBase
from torchxai.explainers._utils import (
    _expand_feature_mask_to_target,
    _run_forward_multi_target,
    _weight_attributions,
)


def get_exp_kernel_similarity_function(
    distance_mode: str = "cosine", kernel_width: float = 1.0
) -> Callable:
    def default_exp_kernel(original_inp, perturbed_inp, __, **kwargs):
        flattened_original_inp = _flatten_tensor_or_tuple(original_inp).float()
        flattened_perturbed_inp = _flatten_tensor_or_tuple(perturbed_inp).float()
        if distance_mode == "cosine":
            cos_sim = CosineSimilarity(dim=0)
            distance = 1 - cos_sim(flattened_original_inp, flattened_perturbed_inp)
        elif distance_mode == "euclidean":
            distance = torch.norm(flattened_original_inp - flattened_perturbed_inp)
        else:
            raise ValueError("distance_mode must be either cosine or euclidean.")
        return math.exp(-1 * (distance**2) / (2 * (kernel_width**2)))

    return default_exp_kernel


def get_exp_kernel_similarity_function_with_interpretable_inps(
    distance_mode: str = "cosine", kernel_width: float = 0.25
) -> Callable:
    def default_exp_kernel(_, __, interpretable_inps, **kwargs):
        flattened_original_inp = torch.ones_like(interpretable_inps).squeeze().float()
        flattened_perturbed_inp = interpretable_inps.squeeze().float()
        if distance_mode == "cosine":
            cos_sim = CosineSimilarity(dim=0)
            distance = 1 - cos_sim(flattened_original_inp, flattened_perturbed_inp)
        elif distance_mode == "euclidean":
            distance = torch.norm(flattened_original_inp - flattened_perturbed_inp)
        else:
            raise ValueError("distance_mode must be either cosine or euclidean.")
        return math.exp(-1 * (distance**2) / (2 * (kernel_width**2)))

    return default_exp_kernel


def frozen_features_perturb_func(original_inp, **kwargs):
    assert "num_interp_features" in kwargs, (
        "Must provide num_interp_features to use default interpretable sampling function"
    )
    if isinstance(original_inp, Tensor):
        device = original_inp.device
    else:
        device = original_inp[0].device

    probs = torch.ones(1, kwargs["num_interp_features"]) * 0.5
    perturbation = torch.bernoulli(probs).to(device=device).long()
    if "frozen_features" in kwargs and kwargs["frozen_features"] is not None:
        frozen_features = kwargs["frozen_features"]
        assert all(
            feature_idx < kwargs["num_interp_features"]
            for feature_idx in frozen_features[
                0
            ]  # this will always have a batch size of 1
        ), "Frozen features must be less than num_interp_features"
        perturbation[0, frozen_features[0]] = (
            1  # freeze the features, useful for padding/cls/sep tokens in sequences
        )
    return perturbation


class Lime(LimeBase):
    """LIME (Local Interpretable Model-agnostic Explanations) attribution method.

    This class extends Captum's LimeBase to provide LIME functionality with
    local linear approximations using perturbed samples around the input.
    """

    def __init__(
        self,
        forward_func: Callable,
        interpretable_model: Model | None = None,
        similarity_func: Callable | None = None,
        perturb_func: Callable | None = None,
    ) -> None:
        if interpretable_model is None:
            interpretable_model = SkLearnLasso(alpha=0.01)

        if similarity_func is None:
            similarity_func = get_exp_kernel_similarity_function()

        if perturb_func is None:
            perturb_func = default_perturb_func

        LimeBase.__init__(
            self,
            forward_func,
            interpretable_model,
            similarity_func,
            perturb_func,
            True,
            default_from_interp_rep_transform,
            None,
        )

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
            show_progress=show_progress,
        )

    def _attribute_kwargs(  # type: ignore
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
        **kwargs,
    ) -> TensorOrTupleOfTensorsGeneric:
        is_inputs_tuple = _is_tuple(inputs)
        formatted_inputs, baselines = _format_input_baseline(inputs, baselines)
        bsz = formatted_inputs[0].shape[0]

        feature_mask, num_interp_features = construct_feature_mask(
            feature_mask, formatted_inputs
        )

        if num_interp_features > 10000:
            warnings.warn(
                "Attempting to construct interpretable model with > 10000 features."
                "This can be very slow or lead to OOM issues. Please provide a feature"
                "mask which groups input features to reduce the number of interpretable"
                "features. ",
                stacklevel=2,
            )

        coefs: Tensor
        if bsz > 1:
            test_output = _run_forward(
                self.forward_func, inputs, target, additional_forward_args
            )
            if isinstance(test_output, Tensor) and torch.numel(test_output) > 1:
                if torch.numel(test_output) == bsz:
                    warnings.warn(
                        "You are providing multiple inputs for Lime / Kernel SHAP "
                        "attributions. This trains a separate interpretable model "
                        "for each example, which can be time consuming. It is "
                        "recommended to compute attributions for one example at a time.",
                        stacklevel=2,
                    )
                    output_list = []
                    for (
                        curr_inps,
                        curr_target,
                        curr_additional_args,
                        curr_baselines,
                        curr_feature_mask,
                        curr_frozen_features,
                    ) in _batch_example_iterator(
                        bsz,
                        formatted_inputs,
                        target,
                        additional_forward_args,
                        baselines,
                        feature_mask,
                        frozen_features,
                    ):
                        kwargs["frozen_features"] = curr_frozen_features
                        coefs = super().attribute(
                            inputs=curr_inps if is_inputs_tuple else curr_inps[0],
                            target=curr_target,
                            additional_forward_args=curr_additional_args,
                            n_samples=n_samples,
                            perturbations_per_eval=perturbations_per_eval,
                            baselines=(
                                curr_baselines if is_inputs_tuple else curr_baselines[0]
                            ),
                            feature_mask=(
                                curr_feature_mask
                                if is_inputs_tuple
                                else curr_feature_mask[0]
                            ),
                            num_interp_features=num_interp_features,
                            show_progress=show_progress,
                            **kwargs,
                        )
                        if return_input_shape:
                            output_list.append(
                                self._convert_output_shape(
                                    curr_inps,
                                    curr_feature_mask,
                                    coefs,
                                    num_interp_features,
                                    is_inputs_tuple,
                                )
                            )
                        else:
                            output_list.append(coefs.reshape(1, -1))  # type: ignore

                    return _reduce_list(output_list)
                else:
                    raise AssertionError(
                        "Invalid number of outputs, forward function should return a"
                        "scalar per example or a scalar per input batch."
                    )
            else:
                assert perturbations_per_eval == 1, (
                    "Perturbations per eval must be 1 when forward function"
                    "returns single value per batch!"
                )

        coefs = super().attribute(
            inputs=inputs,  # type: ignore
            target=target,
            additional_forward_args=additional_forward_args,
            n_samples=n_samples,
            perturbations_per_eval=perturbations_per_eval,
            baselines=baselines if is_inputs_tuple else baselines[0],
            feature_mask=feature_mask if is_inputs_tuple else feature_mask[0],
            num_interp_features=num_interp_features,
            show_progress=show_progress,
            **kwargs,
        )
        if return_input_shape:
            return self._convert_output_shape(
                formatted_inputs,
                feature_mask,
                coefs,
                num_interp_features,
                is_inputs_tuple,
            )
        else:
            return coefs

    @typing.overload
    def _convert_output_shape(
        self,
        formatted_inp: tuple[Tensor, ...],
        feature_mask: tuple[Tensor, ...],
        coefs: Tensor,
        num_interp_features: int,
        is_inputs_tuple: Literal[True],
    ) -> tuple[Tensor, ...]: ...

    @typing.overload
    def _convert_output_shape(
        self,
        formatted_inp: tuple[Tensor, ...],
        feature_mask: tuple[Tensor, ...],
        coefs: Tensor,
        num_interp_features: int,
        is_inputs_tuple: Literal[False],
    ) -> Tensor: ...

    def _convert_output_shape(
        self,
        formatted_inp: tuple[Tensor, ...],
        feature_mask: tuple[Tensor, ...],
        coefs: Tensor,
        num_interp_features: int,
        is_inputs_tuple: bool,
    ) -> Tensor | tuple[Tensor, ...]:
        coefs = coefs.flatten()
        attr = [
            torch.zeros_like(single_inp, dtype=torch.float)
            for single_inp in formatted_inp
        ]
        for tensor_ind in range(len(formatted_inp)):
            for single_feature in range(num_interp_features):
                attr[tensor_ind] += (
                    coefs[single_feature].item()
                    * (feature_mask[tensor_ind] == single_feature).float()
                )
        return _format_output(is_inputs_tuple, tuple(attr))


class MultiTargetLime(MultiTargetLimeBase):
    """Multi-target LIME attribution method.

    This class extends MultiTargetLimeBase to support computing LIME attributions
    for multiple targets simultaneously using local linear models.
    """

    def __init__(
        self,
        forward_func: Callable,
        interpretable_model: Model | None = None,
        similarity_func: Callable | None = None,
        perturb_func: Callable | None = None,
    ) -> None:
        if interpretable_model is None:
            interpretable_model = SkLearnLasso(alpha=0.01)

        if similarity_func is None:
            similarity_func = get_exp_kernel_similarity_function()

        if perturb_func is None:
            perturb_func = default_perturb_func

        LimeBase.__init__(
            self,
            forward_func,
            interpretable_model,
            similarity_func,
            perturb_func,
            True,
            default_from_interp_rep_transform,
            None,
        )

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
        show_progress: bool = False,
    ) -> list[TensorOrTupleOfTensorsGeneric]:
        return self._attribute_kwargs(
            inputs=inputs,
            baselines=baselines,
            target=target,
            additional_forward_args=additional_forward_args,
            feature_mask=feature_mask,
            n_samples=n_samples,
            perturbations_per_eval=perturbations_per_eval,
            frozen_features=frozen_features,
            show_progress=show_progress,
        )

    def _attribute_kwargs(  # type: ignore
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: list[TargetType],
        baselines: BaselineType = None,
        additional_forward_args: Any = None,
        feature_mask: None | Tensor | tuple[Tensor, ...] = None,
        n_samples: int = 25,
        perturbations_per_eval: int = 1,
        frozen_features: list[torch.Tensor] | None = None,
        show_progress: bool = False,
        **kwargs,
    ) -> list[TensorOrTupleOfTensorsGeneric]:
        is_inputs_tuple = _is_tuple(inputs)
        formatted_inputs, baselines = _format_input_baseline(inputs, baselines)
        bsz = formatted_inputs[0].shape[0]

        feature_mask, num_interp_features = construct_feature_mask(
            feature_mask, formatted_inputs
        )

        if num_interp_features > 10000:
            warnings.warn(
                "Attempting to construct interpretable model with > 10000 features."
                "This can be very slow or lead to OOM issues. Please provide a feature"
                "mask which groups input features to reduce the number of interpretable"
                "features. ",
                stacklevel=2,
            )

        multi_target_coefs: list[Tensor]
        if bsz > 1:
            test_output = _run_forward_multi_target(
                self.forward_func, inputs, target, additional_forward_args
            )

            n_targets = len(target) if isinstance(target, list) else 1

            # if target is provided as list of torch tensors then we just convert them into list of lists
            if isinstance(target, list) and isinstance(target[0], Tensor):
                if target[0].shape[0] > 1:
                    target = [t.tolist() for t in target]  # type: ignore
                else:
                    target = [t.item() for t in target]  # type: ignore

            if (
                isinstance(target, list)
                and isinstance(target[0], list)
                and isinstance(target[0][0], int)
            ):
                assert len(target[0]) == bsz

                # convert the list of tensors to multiple ids for each example
                target = list(zip(*target, strict=True))
            elif (
                isinstance(target, list)
                and isinstance(target[0], list)
                and isinstance(target[0][0], tuple)
            ):
                assert len(target[0]) == bsz

                # convert the list of tensors to multiple ids for each example
                target = list(zip(*target, strict=True))

            if isinstance(test_output, Tensor) and torch.numel(test_output) > n_targets:
                if test_output.shape[0] == bsz:
                    warnings.warn(
                        "You are providing multiple inputs for Lime / Kernel SHAP "
                        "attributions. This trains a separate interpretable model "
                        "for each example, which can be time consuming. It is "
                        "recommended to compute attributions for one example at a time.",
                        stacklevel=2,
                    )
                    output_list = []
                    for (
                        curr_inps,
                        curr_target,
                        curr_additional_args,
                        curr_baselines,
                        curr_feature_mask,
                        curr_frozen_features,
                    ) in _batch_example_iterator(
                        bsz,
                        formatted_inputs,
                        target,
                        additional_forward_args,
                        baselines,
                        feature_mask,
                        frozen_features,
                    ):
                        kwargs["frozen_features"] = curr_frozen_features
                        if isinstance(curr_target, list) and isinstance(
                            curr_target[0], tuple
                        ):
                            curr_target = [[item] for item in curr_target[0]]

                        multi_target_coefs = super().attribute(
                            inputs=curr_inps if is_inputs_tuple else curr_inps[0],
                            target=curr_target,  # type: ignore
                            additional_forward_args=curr_additional_args,
                            n_samples=n_samples,
                            perturbations_per_eval=perturbations_per_eval,
                            baselines=(
                                curr_baselines if is_inputs_tuple else curr_baselines[0]
                            ),
                            feature_mask=(
                                curr_feature_mask
                                if is_inputs_tuple
                                else curr_feature_mask[0]
                            ),
                            num_interp_features=num_interp_features,
                            show_progress=show_progress,
                            **kwargs,
                        )
                        output_list.append(
                            [
                                self._convert_output_shape(
                                    curr_inps,
                                    curr_feature_mask,
                                    coefs,
                                    num_interp_features,
                                    is_inputs_tuple,
                                )
                                for coefs in multi_target_coefs
                            ]
                        )

                    # switch from per sample target output to per target output
                    # each element of this output now contains the batch attributions for a single target
                    output_list = list(zip(*output_list, strict=False))

                    return [_reduce_list(output) for output in output_list]  # type: ignore
                else:
                    raise AssertionError(
                        "Invalid number of outputs, forward function should return a"
                        "scalar per example or a scalar per input batch."
                    )
            else:
                assert perturbations_per_eval == 1, (
                    "Perturbations per eval must be 1 when forward function"
                    "returns single value per batch!"
                )

        multi_target_coefs = super().attribute(
            inputs=inputs,
            target=target,
            additional_forward_args=additional_forward_args,
            n_samples=n_samples,
            perturbations_per_eval=perturbations_per_eval,
            baselines=baselines if is_inputs_tuple else baselines[0],
            feature_mask=feature_mask if is_inputs_tuple else feature_mask[0],
            num_interp_features=num_interp_features,
            show_progress=show_progress,
            **kwargs,
        )

        return [
            self._convert_output_shape(
                formatted_inputs,
                feature_mask,
                coefs,
                num_interp_features,
                is_inputs_tuple,
            )
            for coefs in multi_target_coefs
        ]

    def _convert_output_shape(
        self,
        formatted_inp: tuple[Tensor, ...],
        feature_mask: tuple[Tensor, ...],
        coefs: Tensor,
        num_interp_features: int,
        is_inputs_tuple: bool,
    ) -> Tensor | tuple[Tensor, ...]:
        coefs = coefs.flatten()
        attr = [
            torch.zeros_like(single_inp, dtype=torch.float)
            for single_inp in formatted_inp
        ]
        for tensor_ind in range(len(formatted_inp)):
            for single_feature in range(num_interp_features):
                attr[tensor_ind] += (
                    coefs[single_feature].item()
                    * (feature_mask[tensor_ind] == single_feature).float()
                )
        return _format_output(is_inputs_tuple, tuple(attr))


class LimeExplainer(Explainer):
    """LIME explainer for local interpretable model-agnostic explanations.

    This explainer computes attributions using LIME (Local Interpretable Model-agnostic
    Explanations), which explains individual predictions by learning locally faithful
    linear models around the input. LIME perturbs the input and trains a simple
    interpretable model on the perturbed samples to approximate the model's behavior
    locally. Supports both single-target and multi-target modes with structured input/output.

    LIME is particularly useful for understanding complex models by providing locally
    accurate explanations using interpretable linear models.

    Args:
        model: The PyTorch model whose output is to be explained.
        multi_target: Whether to use multi-target mode. When True, can compute
            attributions for multiple targets simultaneously. Defaults to False.
        internal_batch_size: Batch size for internal computations (perturbations
            per evaluation). Defaults to 1.
        n_samples: Number of perturbed samples to generate for LIME. Defaults to 100.
        alpha: Regularization parameter for the LASSO interpretable model. Defaults to 0.01.
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
        >>> explainer = LimeExplainer(model, n_samples=200, alpha=0.01)
        >>>
        >>> explanation_inputs = ExplanationInputs(
        ...     inputs=OrderedDict({"features": torch.randn(1, 10)}),
        ...     target=torch.tensor([1]),
        ...     baselines=OrderedDict({"features": torch.zeros(1, 10)}),
        ... )
        >>> attributions = explainer.explain(explanation_inputs)
        >>> # Returns: OrderedDict({"features": torch.Tensor})

        Multi-target usage:
        >>> explainer_mt = LimeExplainer(model, multi_target=True, n_samples=200)
        >>> explanation_inputs_mt = ExplanationInputs(
        ...     inputs=OrderedDict({"features": torch.randn(1, 10)}),
        ...     target=[torch.tensor([0]), torch.tensor([1])],
        ...     baselines=OrderedDict({"features": torch.zeros(1, 10)}),
        ... )
        >>> mt_attributions = explainer_mt.explain(explanation_inputs_mt)
        >>> # Returns: [OrderedDict({"features": torch.Tensor}), OrderedDict({"features": torch.Tensor})]
    """

    __repr_attrs__ = [
        "_multi_target",
        "_internal_batch_size",
        "_n_samples",
        "_alpha",
        "_weight_attributions",
        "_show_progress",
    ]

    def __init__(
        self,
        model: Module,
        multi_target: bool = False,
        internal_batch_size: int = 1,
        n_samples: int = 100,
        alpha: float = 0.01,
        weight_attributions: bool = True,
        show_progress: bool = False,
    ) -> None:
        """Initialize the LimeExplainer.

        Args:
            model: The model whose output is to be explained.
            multi_target: Whether to use multi-target mode. Defaults to False.
            internal_batch_size: Batch size for internal computations. Defaults to 1.
            n_samples: Number of perturbed samples for LIME. Defaults to 100.
            alpha: Regularization parameter for LASSO. Defaults to 0.01.
            weight_attributions: Whether to weight attributions by feature groups. Defaults to True.
        """
        self._n_samples = n_samples
        self._alpha = alpha
        self._weight_attributions = weight_attributions
        self._show_progress = show_progress

        super().__init__(model, multi_target, internal_batch_size)

    def _init_single_target_explanation_fn(self) -> Callable:
        """Initialize single-target LIME attribution function.

        Returns:
            LIME attribution function for single targets.
        """
        expl_func = partial(
            Lime(
                self._model,
                interpretable_model=SkLearnLasso(alpha=self._alpha),
                perturb_func=frozen_features_perturb_func,
            ).attribute,
            n_samples=self._n_samples,
            perturbations_per_eval=self._internal_batch_size,
            show_progress=self._show_progress,
        )
        return self._expl_fn_with_post_process(expl_func)

    def _init_multi_target_explanation_fn(self) -> Callable:
        """Initialize multi-target LIME attribution function.

        Returns:
            MultiTargetLime attribution function for multiple targets.
        """

        expl_func = partial(
            MultiTargetLime(
                self._model,
                interpretable_model=SkLearnLasso(alpha=self._alpha),
                perturb_func=frozen_features_perturb_func,
            ).attribute,
            n_samples=self._n_samples,
            perturbations_per_eval=self._internal_batch_size,
            show_progress=self._show_progress,
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
        """Compute LIME attributions for the given inputs.

        This method provides a backward-compatible interface that accepts individual
        parameters and constructs ExplanationInputs internally before calling the
        parent class explain method.

        Args:
            inputs: Input tensors for attribution computation. Should be an OrderedDict
                mapping feature names to tensors when used with this explainer.
            target: Target indices for attribution computation. Can be a tensor
                (single-target) or list of tensors (multi-target).
            baselines: Baseline tensors for perturbation (typically zeros). If None,
                uses zero baselines matching input shape.
            feature_mask: Masks representing feature groups for aggregation. If provided,
                features with the same mask value are grouped together.
            additional_forward_args: Additional arguments for model forward pass.
            frozen_features: List of feature indices to keep unchanged during perturbation.
                Useful for special tokens like CLS, SEP in NLP models.

        Returns:
            For single-target mode: OrderedDict mapping feature names to attribution tensors.
            For multi-target mode: List of OrderedDicts, one per target.

        Note:
            LIME trains a local linear model on perturbed samples. The number of samples
            and regularization strength are controlled by initialization parameters.

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

        # Get base attributions
        return self._default_explain(
            inputs=inputs,
            target=target,
            baselines=baselines,
            feature_mask=feature_mask,
            additional_forward_args=additional_forward_args,
            frozen_features=frozen_features,
        )
