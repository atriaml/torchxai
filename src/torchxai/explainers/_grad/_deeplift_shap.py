import typing
from collections.abc import Callable
from functools import partial
from typing import Any, Literal, cast

import torch
import tqdm
from captum._utils.common import (
    ExpansionTypes,
    _expand_additional_forward_args,
    _expand_target,
    _format_additional_forward_args,
    _format_output,
    _format_tensor_into_tuples,
    _is_tuple,
)
from captum.attr import DeepLift
from captum.attr._utils.common import _format_callable_baseline
from torch import Tensor
from torch.nn import Module

from torchxai.data_types import (
    ExplanationTargetType,
    TargetType,
    TensorOrTupleOfTensorsGeneric,
)
from torchxai.explainers._explainer import FeatureAttributionExplainer
from torchxai.explainers._grad._deeplift import MultiTargetDeepLift
from torchxai.explainers._utils import (
    _compute_gradients_sequential_autograd,
    _compute_gradients_vmap_autograd,
    _verify_target_for_multi_target_impl,
)


class DeepLiftShapBatched(DeepLift):
    """Batched DeepLIFT SHAP implementation.

    This class extends Captum's DeepLift to provide DeepLIFT SHAP functionality
    with improved memory efficiency through batching. DeepLIFT SHAP computes
    Shapley values using DeepLIFT with multiple reference baselines from training data.

    Args:
        model: The PyTorch model instance.
        multiply_by_inputs: Whether to multiply gradients by inputs. Defaults to True.
    """

    def __init__(self, model: Module, multiply_by_inputs: bool = True) -> None:
        DeepLift.__init__(self, model, multiply_by_inputs=multiply_by_inputs)

    # There's a mismatch between the signatures of DeepLift.attribute and
    # DeepLiftShap.attribute, so we ignore typing here
    @typing.overload  # type: ignore
    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines: TensorOrTupleOfTensorsGeneric
        | Callable[..., TensorOrTupleOfTensorsGeneric],
        target: TargetType = None,
        additional_forward_args: Any = None,
        return_convergence_delta: Literal[False] = False,
        custom_attribution_func: None | Callable[..., tuple[Tensor, ...]] = None,
    ) -> TensorOrTupleOfTensorsGeneric: ...

    @typing.overload
    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines: TensorOrTupleOfTensorsGeneric
        | Callable[..., TensorOrTupleOfTensorsGeneric],
        target: TargetType = None,
        additional_forward_args: Any = None,
        *,
        return_convergence_delta: Literal[True],
        custom_attribution_func: None | Callable[..., tuple[Tensor, ...]] = None,
    ) -> tuple[TensorOrTupleOfTensorsGeneric, Tensor]: ...

    def attribute(  # type: ignore
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines: TensorOrTupleOfTensorsGeneric
        | Callable[..., TensorOrTupleOfTensorsGeneric],
        target: TargetType = None,
        additional_forward_args: Any = None,
        return_convergence_delta: bool = False,
        custom_attribution_func: None | Callable[..., tuple[Tensor, ...]] = None,
        internal_batch_size: int | None = None,
        quiet: bool = True,
    ) -> TensorOrTupleOfTensorsGeneric | tuple[TensorOrTupleOfTensorsGeneric, Tensor]:
        baselines = _format_callable_baseline(baselines, inputs)

        assert isinstance(baselines[0], torch.Tensor) and baselines[0].shape[0] > 1, (
            "Baselines distribution has to be provided in form of a torch.Tensor"
            f" with more than one example but found: {baselines[0]}."
            " If baselines are provided in shape of scalars or with a single"
            " baseline example, `DeepLift`"
            " approach can be used instead."
        )

        # Keeps track whether original input is a tuple or not before
        # converting it into a tuple.
        is_inputs_tuple = _is_tuple(inputs)

        inputs = _format_tensor_into_tuples(inputs)
        additional_forward_args = _format_additional_forward_args(
            additional_forward_args
        )

        # batch sizes
        inp_bsz = inputs[0].shape[0]
        base_bsz = baselines[0].shape[0]

        (exp_inp, exp_base, exp_tgt, exp_addit_args) = (
            self._expand_inputs_baselines_targets(
                baselines, inputs, target, additional_forward_args
            )
        )

        if internal_batch_size is not None:
            num_examples = exp_inp[0].shape[0]
            agg_attributions = None
            delta = None
            for batch_idx in tqdm.tqdm(
                range(0, num_examples, internal_batch_size),
                desc="Computing DeepLiftShap attributions...",
                disable=quiet,
            ):
                batch_attributions = super().attribute(  # type: ignore
                    tuple(
                        x[batch_idx : batch_idx + internal_batch_size] for x in exp_inp
                    ),
                    tuple(
                        x[batch_idx : batch_idx + internal_batch_size] for x in exp_base
                    ),
                    target=(
                        exp_tgt[batch_idx : batch_idx + internal_batch_size]
                        if (
                            isinstance(exp_tgt, torch.Tensor)
                            and exp_tgt.shape[0] == num_examples
                        )
                        or isinstance(exp_tgt, list)
                        else exp_tgt
                    ),
                    additional_forward_args=(
                        tuple(
                            (
                                x[batch_idx : batch_idx + internal_batch_size]
                                if isinstance(x, torch.Tensor)
                                else x
                            )
                            for x in exp_addit_args
                        )
                        if additional_forward_args is not None
                        else None
                    ),
                    return_convergence_delta=cast(
                        Literal[True, False], return_convergence_delta
                    ),
                    custom_attribution_func=custom_attribution_func,
                )
                if return_convergence_delta:
                    batch_attributions, batch_delta = cast(
                        tuple[tuple[Tensor, ...], Tensor], batch_attributions
                    )
                    delta = (
                        torch.cat((delta, batch_delta), dim=0)
                        if delta is not None
                        else batch_delta
                    )
                agg_attributions = (
                    tuple(
                        torch.cat((agg_attribution, batch_attribution), dim=0)  # type: ignore
                        for agg_attribution, batch_attribution in zip(
                            agg_attributions, batch_attributions, strict=False
                        )
                    )
                    if agg_attributions is not None
                    else batch_attributions
                )
            assert agg_attributions is not None, (
                "No attributions were computed in batching."
            )
            attributions = agg_attributions
        else:
            attributions = super().attribute(  # type: ignore
                exp_inp,
                exp_base,
                target=exp_tgt,
                additional_forward_args=exp_addit_args,
                return_convergence_delta=cast(
                    Literal[True, False], return_convergence_delta
                ),
                custom_attribution_func=custom_attribution_func,
            )

            if return_convergence_delta:
                attributions, delta = cast(
                    tuple[tuple[Tensor, ...], Tensor], attributions
                )

        attributions = tuple(
            self._compute_mean_across_baselines(
                inp_bsz, base_bsz, cast(Tensor, attribution)
            )
            for attribution in attributions
        )

        if return_convergence_delta:
            return _format_output(is_inputs_tuple, attributions), delta  # type: ignore
        else:
            return _format_output(is_inputs_tuple, attributions)

    def _expand_inputs_baselines_targets(
        self,
        baselines: tuple[Tensor, ...],
        inputs: tuple[Tensor, ...],
        target: TargetType,
        additional_forward_args: Any,
    ) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...], TargetType, Any]:
        inp_bsz = inputs[0].shape[0]
        base_bsz = baselines[0].shape[0]

        expanded_inputs = tuple(
            [
                input.repeat_interleave(base_bsz, dim=0).requires_grad_()
                for input in inputs
            ]
        )
        expanded_baselines = tuple(
            [
                baseline.repeat(
                    (inp_bsz,) + tuple([1] * (len(baseline.shape) - 1))
                ).requires_grad_()
                for baseline in baselines
            ]
        )
        expanded_target = _expand_target(
            target, base_bsz, expansion_type=ExpansionTypes.repeat_interleave
        )
        input_additional_args = (
            _expand_additional_forward_args(
                additional_forward_args,
                base_bsz,
                expansion_type=ExpansionTypes.repeat_interleave,
            )
            if additional_forward_args is not None
            else None
        )
        return (
            expanded_inputs,
            expanded_baselines,
            expanded_target,
            input_additional_args,
        )

    def _compute_mean_across_baselines(
        self, inp_bsz: int, base_bsz: int, attribution: Tensor
    ) -> Tensor:
        # Average for multiple references
        attr_shape: tuple = (inp_bsz, base_bsz)
        if len(attribution.shape) > 1:
            attr_shape += attribution.shape[1:]
        return torch.mean(attribution.view(attr_shape), dim=1, keepdim=False)


class MultiTargetDeepLiftShapBatched(MultiTargetDeepLift):
    """Multi-target DeepLIFT SHAP with batching support.

    This class extends MultiTargetDeepLift to support DeepLIFT SHAP computations
    for multiple targets simultaneously, with efficient memory management through
    internal batching of baseline samples.

    Args:
        model: The PyTorch model instance.
        multiply_by_inputs: Whether to multiply gradients by inputs. Defaults to True.
        gradient_func: Function for computing gradients. Automatically selects
            between vmap and sequential methods based on PyTorch version.
        grad_batch_size: Batch size for gradient computations. Defaults to 10.
    """

    def __init__(
        self,
        model: Module,
        multiply_by_inputs: bool = True,
        gradient_func=(
            _compute_gradients_vmap_autograd
            if torch.__version__ >= "2.1.0"
            else _compute_gradients_sequential_autograd
        ),
        grad_batch_size: int = 10,
    ) -> None:
        MultiTargetDeepLift.__init__(
            self,
            model,
            multiply_by_inputs=multiply_by_inputs,
            gradient_func=gradient_func,
            grad_batch_size=grad_batch_size,
        )

    def attribute(  # type: ignore
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines: TensorOrTupleOfTensorsGeneric
        | Callable[..., TensorOrTupleOfTensorsGeneric],
        target: list[TargetType],
        additional_forward_args: Any = None,
        return_convergence_delta: bool = False,
        custom_attribution_func: None | Callable[..., tuple[Tensor, ...]] = None,
        internal_batch_size: int | None = None,
    ) -> (
        tuple[list[TensorOrTupleOfTensorsGeneric], list[Tensor]]
        | list[TensorOrTupleOfTensorsGeneric]
    ):
        baselines = _format_callable_baseline(baselines, inputs)

        assert isinstance(baselines[0], torch.Tensor) and baselines[0].shape[0] > 1, (
            "Baselines distribution has to be provided in form of a torch.Tensor"
            f" with more than one example but found: {baselines[0]}."
            " If baselines are provided in shape of scalars or with a single"
            " baseline example, `DeepLift`"
            " approach can be used instead."
        )

        # Keeps track whether original input is a tuple or not before
        # converting it into a tuple.
        is_inputs_tuple = _is_tuple(inputs)

        inputs = _format_tensor_into_tuples(inputs)
        additional_forward_args = _format_additional_forward_args(
            additional_forward_args
        )

        # verify that the target is valid
        _verify_target_for_multi_target_impl(inputs, target)

        # batch sizes
        inp_bsz = inputs[0].shape[0]
        base_bsz = baselines[0].shape[0]

        (exp_inp, exp_base, exp_tgt, exp_addit_args) = (
            self._expand_inputs_baselines_targets(
                baselines, inputs, target, additional_forward_args
            )
        )

        if internal_batch_size is not None:
            with torch.no_grad():
                num_examples = exp_inp[0].shape[0]
                output_sample_indices = [x // base_bsz for x in range(num_examples)]
                if isinstance(target, list):
                    multi_target_attributions = [
                        [
                            torch.zeros_like(input, requires_grad=False)
                            for input in inputs
                        ]
                        for _ in range(len(target))
                    ]
                else:
                    multi_target_attributions = [
                        [
                            torch.zeros_like(input, requires_grad=False)
                            for input in inputs
                        ]
                    ]

            multi_target_delta = None
            for batch_idx in range(0, num_examples, internal_batch_size):
                batch_inputs = tuple(
                    x[batch_idx : batch_idx + internal_batch_size] for x in exp_inp
                )
                batch_baslines = tuple(
                    x[batch_idx : batch_idx + internal_batch_size] for x in exp_base
                )
                batch_targets = []
                for tgt in exp_tgt:
                    if (
                        isinstance(tgt, torch.Tensor) and tgt.shape[0] == num_examples
                    ) or isinstance(tgt, list):
                        batch_targets.append(
                            tgt[batch_idx : batch_idx + internal_batch_size]
                        )
                    else:
                        batch_targets = exp_tgt

                batch_additional_args = None
                if additional_forward_args is not None:
                    batch_additional_args = tuple(
                        (
                            x[batch_idx : batch_idx + internal_batch_size]
                            if isinstance(x, torch.Tensor)
                            else x
                        )
                        for x in exp_addit_args
                    )

                multi_target_batch_attributions = super().attribute(  # type: ignore
                    inputs=batch_inputs,
                    baselines=batch_baslines,
                    target=batch_targets,
                    additional_forward_args=batch_additional_args,
                    return_convergence_delta=return_convergence_delta,
                    custom_attribution_func=custom_attribution_func,
                )

                if return_convergence_delta:
                    multi_target_batch_attributions, batch_delta = cast(
                        tuple[list[tuple[Tensor, ...]], list[Tensor]],
                        multi_target_batch_attributions,
                    )
                    multi_target_delta = (
                        [
                            torch.cat((agg, curr), dim=0)
                            for agg, curr in zip(
                                multi_target_delta, batch_delta, strict=False
                            )
                        ]
                        if multi_target_delta is not None
                        else batch_delta
                    )
                else:
                    multi_target_batch_attributions = cast(
                        list[tuple[Tensor, ...]], multi_target_batch_attributions
                    )

                # get the output attribution indices of this batch
                output_indices = output_sample_indices[
                    batch_idx : batch_idx + internal_batch_size
                ]

                # update attributions sum batch-wise. The sum is taken across the baselines given the output index
                for target_idx in range(len(multi_target_attributions)):
                    for idx in range(len(multi_target_attributions[target_idx])):
                        multi_target_attributions[target_idx][idx].index_add_(
                            0,
                            torch.tensor(
                                output_indices,
                                device=exp_inp[0].device,
                                requires_grad=False,
                            ),
                            multi_target_batch_attributions[target_idx][idx],
                        )

            # now find the average
            multi_target_attributions_average = [
                tuple([x / base_bsz for x in attrib_single_target])
                for attrib_single_target in multi_target_attributions
            ]
        else:
            multi_target_attributions = super().attribute(  # type: ignore
                inputs=exp_inp,
                baselines=exp_base,
                target=exp_tgt,
                additional_forward_args=exp_addit_args,
                return_convergence_delta=return_convergence_delta,
                custom_attribution_func=custom_attribution_func,
            )

            if return_convergence_delta:
                multi_target_batch_attributions, multi_target_delta = cast(
                    tuple[list[tuple[Tensor, ...]], list[Tensor]],
                    multi_target_attributions,
                )

            def process_per_target_attributions(per_target_attributions):
                return tuple(
                    self._compute_mean_across_baselines(
                        inp_bsz, base_bsz, cast(Tensor, attribution)
                    )
                    for attribution in per_target_attributions
                )

            multi_target_attributions_average = [
                process_per_target_attributions(per_target_attributions)
                for per_target_attributions in multi_target_attributions
            ]

        if return_convergence_delta:
            return [
                _format_output(is_inputs_tuple, per_target_attributions)
                for per_target_attributions in multi_target_attributions_average
            ], multi_target_delta  # type: ignore
        else:
            return [
                _format_output(is_inputs_tuple, per_target_attributions)
                for per_target_attributions in multi_target_attributions_average
            ]

    def _expand_inputs_baselines_targets(
        self,
        baselines: tuple[Tensor, ...],
        inputs: tuple[Tensor, ...],
        target: list[TargetType],
        additional_forward_args: Any,
    ) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...], list[TargetType], Any]:
        inp_bsz = inputs[0].shape[0]
        base_bsz = baselines[0].shape[0]

        expanded_inputs = tuple(
            [
                input.repeat_interleave(base_bsz, dim=0).requires_grad_()
                for input in inputs
            ]
        )
        expanded_baselines = tuple(
            [
                baseline.repeat(
                    (inp_bsz,) + tuple([1] * (len(baseline.shape) - 1))
                ).requires_grad_()
                for baseline in baselines
            ]
        )
        expanded_target = [
            _expand_target(t, base_bsz, expansion_type=ExpansionTypes.repeat_interleave)
            for t in target
        ]

        input_additional_args = (
            _expand_additional_forward_args(
                additional_forward_args,
                base_bsz,
                expansion_type=ExpansionTypes.repeat_interleave,
            )
            if additional_forward_args is not None
            else None
        )
        return (
            expanded_inputs,
            expanded_baselines,
            expanded_target,
            input_additional_args,
        )

    def _compute_mean_across_baselines(
        self, inp_bsz: int, base_bsz: int, attribution: Tensor
    ) -> Tensor:
        # Average for multiple references
        attr_shape: tuple = (inp_bsz, base_bsz)
        if len(attribution.shape) > 1:
            attr_shape += attribution.shape[1:]
        return torch.mean(attribution.view(attr_shape), dim=1, keepdim=False)


class DeepLiftShapExplainer(FeatureAttributionExplainer):
    """DeepLIFT SHAP explainer for computing Shapley values with DeepLIFT.

    This explainer computes attributions using DeepLIFT SHAP, which combines DeepLIFT
    with Shapley value computation by using a distribution of training baselines.
    This approach provides theoretically grounded attributions that satisfy Shapley
    value axioms while leveraging DeepLIFT's efficient computation. Supports both
    single-target and multi-target modes with structured input/output.

    DeepLIFT SHAP is particularly effective when you have representative training
    baselines, as it averages attributions across multiple reference points.

    Args:
        model: The PyTorch model whose output is to be explained.
        multi_target: Whether to use multi-target mode. When True, can compute
            attributions for multiple targets simultaneously. Defaults to False.
        internal_batch_size: Batch size for internal computations. Defaults to 64.
        grad_batch_size: Batch size for gradient computations. Defaults to 64.
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
        >>> explainer = DeepLiftShapExplainer(model)
        >>>
        >>> # Training baselines from representative training samples
        >>> baselines = torch.randn(50, 10)  # 50 training samples
        >>> explanation_inputs = ExplanationInputs(
        ...     inputs=OrderedDict({"input": torch.randn(2, 10)}),
        ...     target=torch.tensor([0, 1]),
        ...     baselines=OrderedDict({"input": baselines}),
        ... )
        >>> attributions = explainer.explain(explanation_inputs)
        >>> # Returns: OrderedDict({"input": torch.Tensor})

        Multi-target usage:
        >>> explainer_mt = DeepLiftShapExplainer(model, multi_target=True)
        >>> explanation_inputs_mt = ExplanationInputs(
        ...     inputs=OrderedDict({"input": torch.randn(2, 10)}),
        ...     target=[torch.tensor([0]), torch.tensor([1])],
        ...     baselines=OrderedDict({"input": baselines}),
        ... )
        >>> mt_attributions = explainer_mt.explain(explanation_inputs_mt)
        >>> # Returns: [OrderedDict({"input": torch.Tensor}), OrderedDict({"input": torch.Tensor})]
    """

    __repr_attrs__ = [
        "_multi_target",
        "_internal_batch_size",
        "_grad_batch_size",
        "return_convergence_delta",
    ]

    def __init__(
        self,
        model: Module,
        multi_target: bool = False,
        internal_batch_size: int = 6,
        grad_batch_size: int = 16,
        return_convergence_delta: bool = False,
    ) -> None:
        """Initialize the DeepLiftShapExplainer.

        Args:
            model: The model whose output is to be explained.
            multi_target: Whether to use multi-target mode. Defaults to False.
            internal_batch_size: Batch size for internal computations. Defaults to 64.
            grad_batch_size: Batch size for gradient computations. Defaults to 64.
            return_convergence_delta: Whether to return convergence delta for
                completeness check. Defaults to False.
        """
        self.return_convergence_delta = return_convergence_delta

        super().__init__(
            model, multi_target, internal_batch_size, grad_batch_size=grad_batch_size
        )

    def _init_single_target_explanation_fn(self) -> Callable:
        """Initialize single-target DeepLIFT SHAP attribution function.

        Returns:
            DeepLiftShapBatched attribution function for single targets.
        """
        return partial(
            DeepLiftShapBatched(self._model).attribute,
            return_convergence_delta=self.return_convergence_delta,
            internal_batch_size=self._internal_batch_size,  # type: ignore
        )

    def _init_multi_target_explanation_fn(self) -> Callable:
        """Initialize multi-target DeepLIFT SHAP attribution function.

        Returns:
            MultiTargetDeepLiftShapBatched attribution function for multiple targets.
        """
        return partial(
            MultiTargetDeepLiftShapBatched(
                self._model, grad_batch_size=self._grad_batch_size
            ).attribute,
            return_convergence_delta=self.return_convergence_delta,
            internal_batch_size=self._internal_batch_size,
        )

    def explain(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: ExplanationTargetType,
        baselines: TensorOrTupleOfTensorsGeneric | None = None,
        additional_forward_args: tuple[Any, ...] | None = None,
    ) -> TensorOrTupleOfTensorsGeneric | list[TensorOrTupleOfTensorsGeneric]:
        """Compute DeepLIFT SHAP attributions for the given inputs.

        This method provides a backward-compatible interface that accepts individual
        parameters and constructs ExplanationInputs internally before calling the
        parent class explain method.

        Args:
            inputs: Input tensors for attribution computation. Should be an OrderedDict
                mapping feature names to tensors when used with this explainer.
            target: Target indices for attribution computation. Can be a tensor
                (single-target) or list of tensors (multi-target).
            baselines: Training baseline distribution for DeepLIFT SHAP.
                Must be provided as a tensor distribution representing training samples.
                The method averages attributions across these baselines.
            additional_forward_args: Additional arguments for model forward pass.

        Returns:
            For single-target mode: OrderedDict mapping feature names to attribution tensors.
            For multi-target mode: List of OrderedDicts, one per target.

        Note:
            DeepLIFT SHAP requires multiple baseline samples (typically from training data)
            rather than a single baseline. The convergence delta behavior is controlled
            by initialization settings.

        Examples:
            >>> # With training baselines
            >>> baselines = torch.randn(100, 10)  # 100 training samples
            >>> attributions = explainer.explain(
            ...     inputs=OrderedDict({"input": torch.randn(2, 10)}),
            ...     target=torch.tensor([0, 1]),
            ...     baselines=OrderedDict({"input": baselines}),
            ... )
        """
        return self._default_explain(
            inputs=inputs,
            target=target,
            baselines=baselines,
            additional_forward_args=additional_forward_args,
        )
