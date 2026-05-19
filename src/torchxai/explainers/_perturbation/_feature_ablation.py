import math
from collections.abc import Callable
from functools import partial
from typing import Any, cast

import torch
from captum._utils.common import (
    _expand_additional_forward_args,
    _expand_target,
    _format_additional_forward_args,
    _format_feature_mask,
    _format_output,
    _is_tuple,
)
from captum._utils.progress import progress
from captum.attr import (
    FeatureAblation as CaptumFeatureAblation,
    PerturbationAttribution,
)
from captum.attr._utils.common import _format_input_baseline
from torch import Tensor, dtype

from torchxai.data_types import (
    BaselineType,
    ExplanationTargetType,
    TargetType,
    TensorOrTupleOfTensorsGeneric,
)
from torchxai.explainers._explainer import FeatureAttributionExplainer
from torchxai.explainers._utils import (
    _expand_feature_mask_to_target,
    _run_forward_multi_target,
    _weight_attributions,
)


class FeatureAblation(CaptumFeatureAblation):
    """Feature Ablation attribution method.

    This implementation extends Captum's FeatureAblation with improved weighting
    that considers the total number of elements in each feature group instead of
    just the number of overlaps, providing more accurate attribution scaling.
    """

    def __init__(self, forward_func: Callable) -> None:
        r"""
        Args:

            forward_func (Callable): The forward function of the model or
                        any modification of it.
        """
        PerturbationAttribution.__init__(self, forward_func)
        self.use_weights = False

        # only used when perturbations_per_eval > 1, where the 1st dim of forward_func's
        # output must grow as the input batch size. If forward's output is aggregated,
        # we cannot expand the input to include more perturbations in one call.
        # If it's False, we will force the validation by comparing the outpus of
        # the original input and the modified input whose batch size expanded based on
        # perturbations_per_eval. Set the flag to True if the output of the modified
        # input grow as expected. Once it turns to True, we will assume the model's
        # behavior stays consistent and no longer check again
        self._is_output_shape_valid = False

    def attribute(  # type: ignore[override]
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Any = None,
        feature_mask: None | Tensor | tuple[Tensor, ...] = None,
        perturbations_per_eval: int = 1,
        show_progress: bool = False,
        **kwargs: Any,
    ) -> TensorOrTupleOfTensorsGeneric:
        # Keeps track whether original input is a tuple or not before
        # converting it into a tuple.
        is_inputs_tuple = _is_tuple(inputs)
        inputs, baselines = _format_input_baseline(inputs, baselines)
        additional_forward_args = _format_additional_forward_args(
            additional_forward_args
        )
        num_examples = inputs[0].shape[0]
        feature_mask = _format_feature_mask(feature_mask, inputs)

        assert (
            isinstance(perturbations_per_eval, int) and perturbations_per_eval >= 1
        ), "Perturbations per evaluation must be an integer and at least 1."
        with torch.no_grad():
            attr_progress = None
            if show_progress:
                feature_counts = self._get_feature_counts(
                    inputs, feature_mask, **kwargs
                )
                total_forwards = (
                    sum(
                        math.ceil(count / perturbations_per_eval)
                        for count in feature_counts
                    )
                    + 1
                )  # add 1 for the initial eval
                attr_progress = progress(
                    desc=f"{self.get_name()} attribution", total=total_forwards
                )
                attr_progress.update(0)

            # Computes initial evaluation with all features, which is compared
            # to each ablated result.
            initial_eval = self._strict_run_forward(
                self.forward_func, inputs, target, additional_forward_args
            )

            if attr_progress is not None:
                attr_progress.update()

            # number of elements in the output of forward_func
            n_outputs = initial_eval.numel() if isinstance(initial_eval, Tensor) else 1

            # flatten eval outputs into 1D (n_outputs)
            # add the leading dim for n_feature_perturbed
            flattened_initial_eval = initial_eval.reshape(1, -1)

            # Initialize attribution totals and counts
            attrib_type = cast(dtype, flattened_initial_eval.dtype)

            total_attrib = [
                # attribute w.r.t each output element
                torch.zeros(
                    (n_outputs,) + input.shape[1:],
                    dtype=attrib_type,
                    device=input.device,
                )
                for input in inputs
            ]

            # Weights are used in cases where ablations may be overlapping.
            weights = None
            if self.use_weights:
                weights = [
                    torch.zeros(
                        (n_outputs,) + input.shape[1:], device=input.device
                    ).float()
                    for input in inputs
                ]

            # Iterate through each feature tensor for ablation
            for i in range(len(inputs)):
                # Skip any empty input tensors
                if torch.numel(inputs[i]) == 0:
                    continue

                for (
                    current_inputs,
                    current_add_args,
                    current_target,
                    current_mask,
                ) in self._ith_input_ablation_generator(
                    i,
                    inputs,
                    additional_forward_args,
                    target,
                    baselines,
                    feature_mask,
                    perturbations_per_eval,
                    **kwargs,
                ):
                    # modified_eval has (n_feature_perturbed * n_outputs) elements
                    # shape:
                    #   agg mode: (*initial_eval.shape)
                    #   non-agg mode:
                    #     (feature_perturbed * batch_size, *initial_eval.shape[1:])
                    modified_eval = self._strict_run_forward(
                        self.forward_func,
                        current_inputs,
                        current_target,
                        current_add_args,
                    )

                    if attr_progress is not None:
                        attr_progress.update()

                    # if perturbations_per_eval > 1, the output shape must grow with
                    # input and not be aggregated
                    if perturbations_per_eval > 1 and not self._is_output_shape_valid:
                        current_batch_size = current_inputs[0].shape[0]

                        # number of perturbation, which is not the same as
                        # perturbations_per_eval when not enough features to perturb
                        n_perturb = current_batch_size / num_examples

                        current_output_shape = modified_eval.shape

                        # use initial_eval as the forward of perturbations_per_eval = 1
                        initial_output_shape = initial_eval.shape

                        assert (
                            # check if the output is not a scalar
                            current_output_shape
                            and initial_output_shape
                            # check if the output grow in same ratio, i.e., not agg
                            and current_output_shape[0]
                            == n_perturb * initial_output_shape[0]
                        ), (
                            "When perturbations_per_eval > 1, forward_func's output "
                            "should be a tensor whose 1st dim grow with the input "
                            f"batch size: when input batch size is {num_examples}, "
                            f"the output shape is {initial_output_shape}; "
                            f"when input batch size is {current_batch_size}, "
                            f"the output shape is {current_output_shape}"
                        )

                        self._is_output_shape_valid = True

                    # reshape the leading dim for n_feature_perturbed
                    # flatten each feature's eval outputs into 1D of (n_outputs)
                    modified_eval = modified_eval.reshape(-1, n_outputs)
                    # eval_diff in shape (n_feature_perturbed, n_outputs)
                    eval_diff = flattened_initial_eval - modified_eval

                    # append the shape of one input example
                    # to make it broadcastable to mask
                    eval_diff = eval_diff.reshape(
                        eval_diff.shape + (inputs[i].dim() - 1) * (1,)
                    )
                    eval_diff = eval_diff.to(total_attrib[i].device)

                    if self.use_weights:
                        assert weights is not None, (
                            "weights should not be None when use_weights is True"
                        )
                        # this line is the only change from the original captum code where we multiply the weights
                        # by the total number of elements in the feature group. Note that the sum over the mask
                        # for a single sample here is the total number of elements in the feature group.
                        weights[i] += (
                            current_mask.float().sum(dim=0)
                            * current_mask[0][0].float().sum()
                        )

                    total_attrib[i] += (eval_diff * current_mask.to(attrib_type)).sum(
                        dim=0
                    )

            if attr_progress is not None:
                attr_progress.close()

            # Divide total attributions by counts and return formatted attributions
            if self.use_weights:
                assert weights is not None, (
                    "weights should not be None when use_weights is True"
                )
                attrib = tuple(
                    single_attrib.float() / weight
                    for single_attrib, weight in zip(
                        total_attrib, weights, strict=False
                    )
                )
            else:
                attrib = tuple(total_attrib)
            return _format_output(is_inputs_tuple, attrib)


class MultiTargetFeatureAblation(FeatureAblation):
    """Multi-target Feature Ablation attribution method.

    This class extends FeatureAblation to support computing feature ablation
    attributions for multiple targets simultaneously by systematically removing
    features and measuring the impact on each target.
    """

    def attribute(  # type: ignore[override]
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: list[TargetType],
        baselines: BaselineType = None,
        additional_forward_args: Any = None,
        feature_mask: None | Tensor | tuple[Tensor, ...] = None,
        perturbations_per_eval: int = 1,
        show_progress: bool = False,
        **kwargs: Any,
    ) -> list[TensorOrTupleOfTensorsGeneric]:
        is_inputs_tuple = _is_tuple(inputs)
        inputs, baselines = _format_input_baseline(inputs, baselines)
        additional_forward_args = _format_additional_forward_args(
            additional_forward_args
        )
        num_examples = inputs[0].shape[0]
        feature_mask = _format_feature_mask(feature_mask, inputs)

        assert (
            isinstance(perturbations_per_eval, int) and perturbations_per_eval >= 1
        ), "Perturbations per evaluation must be an integer and at least 1."
        with torch.no_grad():
            attr_progress = None
            if show_progress:
                feature_counts = self._get_feature_counts(
                    inputs, feature_mask, **kwargs
                )
                total_forwards = (
                    sum(
                        math.ceil(count / perturbations_per_eval)
                        for count in feature_counts
                    )
                    + 1
                )  # add 1 for the initial eval
                attr_progress = progress(
                    desc=f"{self.get_name()} attribution", total=total_forwards
                )
                attr_progress.update(0)

            # Computes initial evaluation with all features, which is compared
            # to each ablated result.
            initial_eval = self._strict_run_forward(
                self.forward_func, inputs, target, additional_forward_args
            )

            if attr_progress is not None:
                attr_progress.update()

            # number of elements in the output of forward_func
            # since our _strict_run_forward will always return a tensor with shape (batch_size, targets, ...)
            # the output shape is (batch_size, targets)
            output_shape = initial_eval.shape

            # flatten eval outputs into 1D (n_outputs)
            # add the leading dim for n_feature_perturbed
            flattened_initial_eval = initial_eval.reshape(1, -1)

            # Initialize attribution totals and counts
            attrib_type = cast(dtype, flattened_initial_eval.dtype)

            # for multi-target case, we generate output attributions of size batch_size * targets for each input
            total_attrib = [
                # attribute w.r.t each output element
                torch.zeros(
                    (output_shape[0] * output_shape[1],) + input.shape[1:],
                    dtype=attrib_type,
                    device=input.device,
                )
                for input in inputs
            ]

            # Weights are used in cases where ablations may be overlapping.
            weights = []
            if self.use_weights:
                weights = [
                    torch.zeros(
                        (output_shape[0],) + input.shape[1:], device=input.device
                    ).float()
                    for input in inputs
                ]

            # Iterate through each feature tensor for ablation
            for i in range(len(inputs)):
                # Skip any empty input tensors
                if torch.numel(inputs[i]) == 0:
                    continue

                for (
                    current_inputs,
                    current_add_args,
                    current_target,
                    current_mask,
                ) in self._ith_input_ablation_generator(
                    i,
                    inputs,
                    additional_forward_args,
                    target,
                    baselines,
                    feature_mask,
                    perturbations_per_eval,
                    **kwargs,
                ):
                    # modified_eval has (n_feature_perturbed * n_outputs) elements
                    # shape:
                    #   agg mode: (*initial_eval.shape)
                    #   non-agg mode:
                    #     (feature_perturbed * batch_size, *initial_eval.shape[1:])
                    modified_eval = self._strict_run_forward(
                        self.forward_func,
                        current_inputs,
                        current_target,
                        current_add_args,
                    )

                    if attr_progress is not None:
                        attr_progress.update()

                    # if perturbations_per_eval > 1, the output shape must grow with
                    # input and not be aggregated
                    if perturbations_per_eval > 1 and not self._is_output_shape_valid:
                        current_batch_size = current_inputs[0].shape[0]

                        # number of perturbation, which is not the same as
                        # perturbations_per_eval when not enough features to perturb
                        n_perturb = current_batch_size / num_examples
                        current_output_shape = modified_eval.shape

                        # use initial_eval as the forward of perturbations_per_eval = 1
                        initial_output_shape = initial_eval.shape

                        assert (
                            # check if the output is not a scalar
                            current_output_shape
                            and initial_output_shape
                            # check if the output grow in same ratio, i.e., not agg
                            and current_output_shape[0]
                            == n_perturb * initial_output_shape[0]
                        ), (
                            "When perturbations_per_eval > 1, forward_func's output "
                            "should be a tensor whose 1st dim grow with the input "
                            f"batch size: when input batch size is {num_examples}, "
                            f"the output shape is {initial_output_shape}; "
                            f"when input batch size is {current_batch_size}, "
                            f"the output shape is {current_output_shape}"
                        )

                        self._is_output_shape_valid = True

                    # reshape the leading dim for n_feature_perturbed
                    # flatten each feature's eval outputs into 1D of (n_outputs)
                    modified_eval = modified_eval.reshape(
                        -1, output_shape[0] * output_shape[1]
                    )

                    # eval_diff in shape (n_feature_perturbed, n_outputs)
                    eval_diff = flattened_initial_eval - modified_eval

                    # append the shape of one input example
                    # to make it broadcastable to mask
                    # at this point the eval diff looks something like this:
                    # (perturbation_steps, batch_size * n_targets)
                    eval_diff = eval_diff.reshape(
                        eval_diff.shape + (inputs[i].dim() - 1) * (1,)
                    )
                    eval_diff = eval_diff.to(total_attrib[i].device)

                    if self.use_weights:
                        weights[i] += (
                            current_mask.float().sum(dim=0)
                            * current_mask[0][0].float().sum()
                        )

                    # This (eval_diff * current_mask.to(attrib_type)).sum(dim=0) is of shape
                    # (perturbation_steps, batch_size * n_targets, input_shape)
                    # where each perturbation_step dimension is for a single feature
                    # so for each feature the target attribution is stored as (batch_size * n_targets, input_shape)
                    # note that for each perturbation step output, all attributions will be zero except for the
                    # feature that was perturbed in that step. In this manner the final attribution is obtained
                    # by summing over the first dimension, so all the attributions for each feature are summed
                    # independently.
                    total_attrib[i] += (
                        eval_diff
                        * current_mask.to(attrib_type).repeat(
                            (
                                1,
                                output_shape[1],
                            )  # since the current_mask is for a single feature, we repeat it for all targets
                            + (inputs[i].dim() - 1) * (1,)
                        )
                    ).sum(dim=0)
            if attr_progress is not None:
                attr_progress.close()

            if self.use_weights:
                attrib = tuple(
                    single_attrib.float()
                    / weight.repeat((output_shape[1],) + (weight.dim() - 1) * (1,))
                    for single_attrib, weight in zip(
                        total_attrib, weights, strict=False
                    )
                )
            else:
                attrib = tuple(total_attrib)

            attrib = tuple(
                single_attrib.reshape(
                    (output_shape[0], output_shape[1]) + single_attrib.shape[1:]
                )
                for single_attrib in attrib
            )

            attrib_list_with_tuples = [
                tuple(single_attrib[:, idx] for single_attrib in attrib)
                for idx in range(output_shape[1])
            ]

            _result = [
                _format_output(is_inputs_tuple, single_atrib)
                for single_atrib in attrib_list_with_tuples
            ]
        return _result

    def _strict_run_forward(self, *args, **kwargs) -> Tensor:
        """
        A temp wrapper for global _run_forward util to force forward output
        type assertion & conversion.
        Remove after the strict logic is supported by all attr classes
        """
        forward_output = _run_forward_multi_target(*args, **kwargs)
        if isinstance(forward_output, Tensor):
            if len(forward_output.shape) == 1:
                return forward_output.unsqueeze(-1)
            return forward_output

        output_type = type(forward_output)
        assert output_type is int or output_type is float, (
            "the return of forward_func must be a tensor, int, or float,"
            f" received: {forward_output}"
        )

        # using python built-in type as torch dtype
        # int -> torch.int64, float -> torch.float64
        # ref: https://github.com/pytorch/pytorch/pull/21215
        forward_output = torch.tensor(forward_output, dtype=output_type)
        if len(forward_output.shape) == 1:
            return forward_output.unsqueeze(-1)
        return forward_output

    def _ith_input_ablation_generator(
        self,
        i,
        inputs,
        additional_args,
        target,
        baselines,
        input_mask,
        perturbations_per_eval,
        **kwargs,
    ):
        """
        This method returns a generator of ablation perturbations of the i-th input

        Returns:
            ablation_iter (Generator): yields each perturbation to be evaluated
                        as a tuple (inputs, additional_forward_args, targets, mask).
        """
        extra_args = {}
        for key, value in kwargs.items():
            # For any tuple argument in kwargs, we choose index i of the tuple.
            if isinstance(value, tuple):
                extra_args[key] = value[i]
            else:
                extra_args[key] = value

        input_mask = input_mask[i] if input_mask is not None else None
        min_feature, num_features, input_mask = self._get_feature_range_and_mask(
            inputs[i], input_mask, **extra_args
        )
        num_examples = inputs[0].shape[0]
        if input_mask is not None and input_mask.shape[0] != num_examples:
            input_mask = input_mask.expand(num_examples, *input_mask.shape[1:])

        perturbations_per_eval = int(min(perturbations_per_eval, num_features))
        baseline = baselines[i] if isinstance(baselines, tuple) else baselines
        if isinstance(baseline, torch.Tensor):
            baseline = baseline.reshape((1,) + baseline.shape)

        if perturbations_per_eval > 1:
            # Repeat features and additional args for batch size.
            all_features_repeated = [
                torch.cat([inputs[j]] * perturbations_per_eval, dim=0)
                for j in range(len(inputs))
            ]
            additional_args_repeated = (
                _expand_additional_forward_args(additional_args, perturbations_per_eval)
                if additional_args is not None
                else None
            )
            if isinstance(target, list):
                target_repeated = [
                    _expand_target(t, perturbations_per_eval) for t in target
                ]
            else:
                target_repeated = _expand_target(target, perturbations_per_eval)
        else:
            all_features_repeated = list(inputs)
            additional_args_repeated = additional_args
            target_repeated = target

        num_features_processed = min_feature
        while num_features_processed < num_features:
            current_num_ablated_features = int(
                min(perturbations_per_eval, num_features - num_features_processed)
            )

            # Store appropriate inputs and additional args based on batch size.
            if current_num_ablated_features != perturbations_per_eval:
                current_features = [
                    feature_repeated[0 : current_num_ablated_features * num_examples]
                    for feature_repeated in all_features_repeated
                ]
                current_additional_args = (
                    _expand_additional_forward_args(
                        additional_args, current_num_ablated_features
                    )
                    if additional_args is not None
                    else None
                )
                if isinstance(target, list):
                    current_target = [
                        _expand_target(t, current_num_ablated_features) for t in target
                    ]
                else:
                    current_target = _expand_target(
                        target, current_num_ablated_features
                    )
            else:
                current_features = all_features_repeated
                current_additional_args = additional_args_repeated
                current_target = target_repeated

            # Store existing tensor before modifying
            original_tensor = current_features[i]
            # Construct ablated batch for features in range num_features_processed
            # to num_features_processed + current_num_ablated_features and return
            # mask with same size as ablated batch. ablated_features has dimension
            # (current_num_ablated_features, num_examples, inputs[i].shape[1:])
            # Note that in the case of sparse tensors, the second dimension
            # may not necessarilly be num_examples and will match the first
            # dimension of this tensor.
            current_reshaped = current_features[i].reshape(
                (current_num_ablated_features, -1) + current_features[i].shape[1:]
            )
            ablated_features, current_mask = self._construct_ablated_input(
                current_reshaped,
                input_mask,
                baseline,
                num_features_processed,
                num_features_processed + current_num_ablated_features,
                **extra_args,
            )

            # current_features[i] has dimension
            # (current_num_ablated_features * num_examples, inputs[i].shape[1:]),
            # which can be provided to the model as input.
            current_features[i] = ablated_features.reshape(
                (-1,) + ablated_features.shape[2:]
            )
            yield (
                tuple(current_features),
                current_additional_args,
                current_target,
                current_mask,
            )
            # Replace existing tensor at index i.
            current_features[i] = original_tensor
            num_features_processed += current_num_ablated_features


class FeatureAblationExplainer(FeatureAttributionExplainer):
    """Feature Ablation explainer for computing systematic feature removal attributions.

    This explainer computes attributions using Feature Ablation, which systematically
    removes features or feature groups and measures the resulting change in model output.
    This direct approach provides intuitive explanations by showing exactly how much
    each feature contributes to the prediction. The method supports feature grouping
    through masks, allowing for hierarchical ablation studies. Supports both single-target
    and multi-target modes with structured input/output.

    Feature Ablation provides the most direct measure of feature importance through
    systematic removal and impact measurement.

    Args:
        model: The PyTorch model whose output is to be explained.
        multi_target: Whether to use multi-target mode. When True, can compute
            attributions for multiple targets simultaneously. Defaults to False.
        internal_batch_size: Batch size for internal computations (perturbations
            per evaluation). Defaults to 64.
        weight_attributions: Whether to weight attributions by feature group sizes
            when using feature masks. Defaults to False.

    Examples:
        Single-target usage for tabular data:
        >>> import torch
        >>> from collections import OrderedDict
        >>> from torchxai.data_types import ExplanationInputs
        >>>
        >>> model = torch.nn.Sequential(
        ...     torch.nn.Linear(10, 5), torch.nn.ReLU(), torch.nn.Linear(5, 2)
        ... )
        >>> explainer = FeatureAblationExplainer(model, internal_batch_size=32)
        >>>
        >>> explanation_inputs = ExplanationInputs(
        ...     inputs=OrderedDict({"features": torch.randn(2, 10)}),
        ...     target=torch.tensor([0, 1]),
        ...     baselines=OrderedDict({"features": torch.zeros(2, 10)}),
        ... )
        >>> attributions = explainer.explain(explanation_inputs)
        >>> # Returns: OrderedDict({"features": torch.Tensor})

        Multi-target usage with feature grouping:
        >>> explainer_mt = FeatureAblationExplainer(
        ...     model, multi_target=True, weight_attributions=True
        ... )
        >>> feature_mask = torch.tensor([[0, 0, 1, 1, 2, 2, 2, 3, 3, 4]])
        >>> explanation_inputs_mt = ExplanationInputs(
        ...     inputs=OrderedDict({"features": torch.randn(2, 10)}),
        ...     target=[torch.tensor([0, 1]), torch.tensor([1, 0])],
        ...     baselines=OrderedDict({"features": torch.zeros(2, 10)}),
        ...     feature_mask=feature_mask,
        ... )
        >>> mt_attributions = explainer_mt.explain(explanation_inputs_mt)
        >>> # Returns: [OrderedDict({"features": torch.Tensor}), OrderedDict({"features": torch.Tensor})]
    """

    __repr_attrs__ = [
        "_multi_target",
        "_internal_batch_size",
        "_weight_attributions",
        "_show_progress",
    ]

    def __init__(
        self,
        model: torch.nn.Module,
        multi_target: bool = False,
        internal_batch_size: int = 64,
        weight_attributions: bool = False,
        show_progress: bool = False,
    ) -> None:
        """Initialize the FeatureAblationExplainer.

        Args:
            model: The model whose output is to be explained.
            multi_target: Whether to use multi-target mode. Defaults to False.
            internal_batch_size: Batch size for internal computations. Defaults to 64.
            weight_attributions: Whether to weight attributions by feature groups. Defaults to False.
        """
        self._weight_attributions = weight_attributions
        self._show_progress = show_progress

        super().__init__(model, multi_target, internal_batch_size)

    def _init_single_target_explanation_fn(self) -> Callable:
        """Initialize single-target Feature Ablation attribution function.

        Returns:
            FeatureAblation attribution function for single targets.
        """
        expl_func = partial(
            FeatureAblation(self._model).attribute,
            perturbations_per_eval=self._internal_batch_size,
            show_progress=self._show_progress,
        )
        return self._expl_fn_with_post_process(expl_func)

    def _init_multi_target_explanation_fn(self) -> Callable:
        """Initialize multi-target Feature Ablation attribution function.

        Returns:
            MultiTargetFeatureAblation attribution function for multiple targets.
        """
        expl_func = partial(
            MultiTargetFeatureAblation(self._model).attribute,
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
        target: ExplanationTargetType | list[ExplanationTargetType],
        baselines: TensorOrTupleOfTensorsGeneric | None = None,
        feature_mask: TensorOrTupleOfTensorsGeneric | None = None,
        additional_forward_args: tuple[Any, ...] | None = None,
    ) -> TensorOrTupleOfTensorsGeneric | list[TensorOrTupleOfTensorsGeneric]:
        """Compute Feature Ablation attributions for the given inputs.

        This method provides a backward-compatible interface that accepts individual
        parameters and constructs ExplanationInputs internally before calling the
        parent class explain method.

        Args:
            inputs: Input tensors for attribution computation. Should be an OrderedDict
                mapping feature names to tensors when used with this explainer.
            target: Target indices for attribution computation. Can be a tensor
                (single-target) or list of tensors (multi-target).
            baselines: Baseline tensors for ablation (typically zeros). If None,
                uses zero baselines matching input shape.
            feature_mask: Masks representing feature groups for ablation. Features
                with the same mask value are ablated together as a group.
            additional_forward_args: Additional arguments for model forward pass.

        Returns:
            For single-target mode: OrderedDict mapping feature names to attribution tensors.
            For multi-target mode: List of OrderedDicts, one per target.

        Note:
            Feature Ablation directly measures feature importance through systematic removal.
            The internal_batch_size parameter controls how many features are ablated
            simultaneously for computational efficiency.

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
        )
