from collections import OrderedDict
from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np
import torch
from captum._utils.common import _format_tensor_into_tuples
from captum.attr._utils.common import (
    _format_and_verify_sliding_window_shapes,
    _format_and_verify_strides,
)
from torch import Tensor
from torch.nn.modules import Module

from torchxai.data_types.common import (
    BaselineType,
    TargetType,
    TensorOrTupleOfTensorsGeneric,
)
from torchxai.explainers._perturbation._feature_ablation import (
    FeatureAblation,
    MultiTargetFeatureAblation,
)
from torchxai.explainers.explainer import Explainer


class Occlusion(FeatureAblation):
    """Occlusion attribution method using sliding windows.

    This implementation extends FeatureAblation to provide occlusion-based
    attributions by systematically replacing rectangular regions of the input
    with baseline values and measuring the impact on model output.
    """

    def __init__(self, forward_func: Callable) -> None:
        FeatureAblation.__init__(self, forward_func)
        self.use_weights = True

    def attribute(  # type: ignore
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        sliding_window_shapes: tuple[int, ...] | tuple[tuple[int, ...], ...],
        strides: None
        | int
        | tuple[int, ...]
        | tuple[int | tuple[int, ...], ...] = None,
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Any = None,
        perturbations_per_eval: int = 1,
        show_progress: bool = False,
    ) -> TensorOrTupleOfTensorsGeneric:
        formatted_inputs = _format_tensor_into_tuples(inputs)

        # Formatting strides
        strides = _format_and_verify_strides(strides, formatted_inputs)

        # Formatting sliding window shapes
        sliding_window_shapes = _format_and_verify_sliding_window_shapes(
            sliding_window_shapes, formatted_inputs
        )

        # Construct tensors from sliding window shapes
        sliding_window_tensors = tuple(
            torch.ones(window_shape, device=formatted_inputs[i].device)
            for i, window_shape in enumerate(sliding_window_shapes)
        )

        # Construct counts, defining number of steps to make of occlusion block in
        # each dimension.
        shift_counts = []
        for i, inp in enumerate(formatted_inputs):
            current_shape = np.subtract(inp.shape[1:], sliding_window_shapes[i])
            # Verify sliding window doesn't exceed input dimensions.
            assert (np.array(current_shape) >= 0).all(), (
                f"Sliding window dimensions {sliding_window_shapes[i]} cannot exceed input dimensions"
                f"{tuple(inp.shape[1:])}."
            )
            # Stride cannot be larger than sliding window for any dimension where
            # the sliding window doesn't cover the entire input.
            assert np.logical_or(
                np.array(current_shape) == 0,
                np.array(strides[i]) <= sliding_window_shapes[i],
            ).all(), (
                f"Stride dimension {strides[i]} cannot be larger than sliding window "
                f"shape dimension {sliding_window_shapes[i]}."
            )
            shift_counts.append(
                tuple(
                    np.add(np.ceil(np.divide(current_shape, strides[i])).astype(int), 1)
                )
            )

        # Use ablation attribute method
        return super().attribute(
            inputs,
            baselines=baselines,
            target=target,
            additional_forward_args=additional_forward_args,
            perturbations_per_eval=perturbations_per_eval,
            sliding_window_tensors=sliding_window_tensors,
            shift_counts=tuple(shift_counts),
            strides=strides,
            show_progress=show_progress,
        )

    def _construct_ablated_input(
        self,
        expanded_input: Tensor,
        input_mask: None | Tensor,
        baseline: Tensor | int | float,
        start_feature: int,
        end_feature: int,
        **kwargs: Any,
    ) -> tuple[Tensor, Tensor]:
        input_mask = torch.stack(
            [
                self._occlusion_mask(
                    expanded_input,
                    j,
                    kwargs["sliding_window_tensors"],
                    kwargs["strides"],
                    kwargs["shift_counts"],
                )
                for j in range(start_feature, end_feature)
            ],
            dim=0,
        ).long()
        ablated_tensor = (
            expanded_input
            * (
                torch.ones(1, dtype=torch.long, device=expanded_input.device)
                - input_mask
            ).to(expanded_input.dtype)
        ) + (baseline * input_mask.to(expanded_input.dtype))
        return ablated_tensor, input_mask

    def _occlusion_mask(
        self,
        expanded_input: Tensor,
        ablated_feature_num: int,
        sliding_window_tsr: Tensor,
        strides: int | tuple[int, ...],
        shift_counts: tuple[int, ...],
    ) -> Tensor:
        remaining_total = ablated_feature_num
        current_index = []
        for i, shift_count in enumerate(shift_counts):
            stride = strides[i] if isinstance(strides, tuple) else strides
            current_index.append((remaining_total % shift_count) * stride)
            remaining_total = remaining_total // shift_count

        remaining_padding = np.subtract(
            expanded_input.shape[2:], np.add(current_index, sliding_window_tsr.shape)
        )
        pad_values = [
            val
            for pair in zip(remaining_padding, current_index, strict=False)
            for val in pair
        ]
        pad_values.reverse()
        padded_tensor = torch.nn.functional.pad(
            sliding_window_tsr,
            tuple(pad_values),  # type: ignore
        )
        return padded_tensor.reshape((1,) + padded_tensor.shape)

    def _get_feature_range_and_mask(
        self, input: Tensor, input_mask: Tensor, **kwargs: Any
    ) -> tuple[int, int, None]:
        feature_max = np.prod(kwargs["shift_counts"])
        return 0, feature_max, None

    def _get_feature_counts(self, inputs, feature_mask, **kwargs):
        """return the numbers of possible input features"""
        return tuple(np.prod(counts).astype(int) for counts in kwargs["shift_counts"])


class MultiTargetOcclusion(MultiTargetFeatureAblation):
    """Multi-target Occlusion attribution method.

    This class extends MultiTargetFeatureAblation to support computing
    occlusion attributions for multiple targets simultaneously using
    sliding window perturbations.
    """

    def __init__(self, forward_func: Callable) -> None:
        r"""
        Args:

            forward_func (Callable): The forward function of the model or
                        any modification of it.
        """
        FeatureAblation.__init__(self, forward_func)
        self.use_weights = True

    def attribute(  # type: ignore
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        sliding_window_shapes: tuple[int, ...] | tuple[tuple[int, ...], ...],
        strides: None
        | int
        | tuple[int, ...]
        | tuple[int | tuple[int, ...], ...] = None,
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Any = None,
        perturbations_per_eval: int = 1,
        show_progress: bool = False,
    ) -> TensorOrTupleOfTensorsGeneric:
        formatted_inputs = _format_tensor_into_tuples(inputs)

        # Formatting strides
        strides = _format_and_verify_strides(strides, formatted_inputs)

        # Formatting sliding window shapes
        sliding_window_shapes = _format_and_verify_sliding_window_shapes(
            sliding_window_shapes, formatted_inputs
        )

        # Construct tensors from sliding window shapes
        sliding_window_tensors = tuple(
            torch.ones(window_shape, device=formatted_inputs[i].device)
            for i, window_shape in enumerate(sliding_window_shapes)
        )

        # Construct counts, defining number of steps to make of occlusion block in
        # each dimension.
        shift_counts = []
        for i, inp in enumerate(formatted_inputs):
            current_shape = np.subtract(inp.shape[1:], sliding_window_shapes[i])
            # Verify sliding window doesn't exceed input dimensions.
            assert (np.array(current_shape) >= 0).all(), (
                f"Sliding window dimensions {sliding_window_shapes[i]} cannot exceed input dimensions"
                f"{tuple(inp.shape[1:])}."
            )
            # Stride cannot be larger than sliding window for any dimension where
            # the sliding window doesn't cover the entire input.
            assert np.logical_or(
                np.array(current_shape) == 0,
                np.array(strides[i]) <= sliding_window_shapes[i],
            ).all(), (
                f"Stride dimension {strides[i]} cannot be larger than sliding window "
                f"shape dimension {sliding_window_shapes[i]}."
            )
            shift_counts.append(
                tuple(
                    np.add(np.ceil(np.divide(current_shape, strides[i])).astype(int), 1)
                )
            )

        # Use ablation attribute method
        return super().attribute(
            inputs,
            baselines=baselines,
            target=target,
            additional_forward_args=additional_forward_args,
            perturbations_per_eval=perturbations_per_eval,
            sliding_window_tensors=sliding_window_tensors,
            shift_counts=tuple(shift_counts),
            strides=strides,
            show_progress=show_progress,
        )

    def _construct_ablated_input(
        self,
        expanded_input: Tensor,
        input_mask: None | Tensor,
        baseline: Tensor | int | float,
        start_feature: int,
        end_feature: int,
        **kwargs: Any,
    ) -> tuple[Tensor, Tensor]:
        r"""
        Ablates given expanded_input tensor with given feature mask, feature range,
        and baselines, and any additional arguments.
        expanded_input shape is (num_features, num_examples, ...)
        with remaining dimensions corresponding to remaining original tensor
        dimensions and num_features = end_feature - start_feature.

        input_mask is None for occlusion, and the mask is constructed
        using sliding_window_tensors, strides, and shift counts, which are provided in
        kwargs. baseline is expected to
        be broadcastable to match expanded_input.

        This method returns the ablated input tensor, which has the same
        dimensionality as expanded_input as well as the corresponding mask with
        either the same dimensionality as expanded_input or second dimension
        being 1. This mask contains 1s in locations which have been ablated (and
        thus counted towards ablations for that feature) and 0s otherwise.
        """
        input_mask = torch.stack(
            [
                self._occlusion_mask(
                    expanded_input,
                    j,
                    kwargs["sliding_window_tensors"],
                    kwargs["strides"],
                    kwargs["shift_counts"],
                )
                for j in range(start_feature, end_feature)
            ],
            dim=0,
        ).long()
        ablated_tensor = (
            expanded_input
            * (
                torch.ones(1, dtype=torch.long, device=expanded_input.device)
                - input_mask
            ).to(expanded_input.dtype)
        ) + (baseline * input_mask.to(expanded_input.dtype))

        # we expand the input mask to shape (perturbation steps, batch size, ...)
        input_mask = input_mask.expand_as(ablated_tensor)

        return ablated_tensor, input_mask

    def _occlusion_mask(
        self,
        expanded_input: Tensor,
        ablated_feature_num: int,
        sliding_window_tsr: Tensor,
        strides: int | tuple[int, ...],
        shift_counts: tuple[int, ...],
    ) -> Tensor:
        """
        This constructs the current occlusion mask, which is the appropriate
        shift of the sliding window tensor based on the ablated feature number.
        The feature number ranges between 0 and the product of the shift counts
        (# of times the sliding window should be shifted in each dimension).

        First, the ablated feature number is converted to the number of steps in
        each dimension from the origin, based on shift counts. This procedure
        is similar to a base conversion, with the position values equal to shift
        counts. The feature number is first taken modulo shift_counts[0] to
        get the number of shifts in the first dimension (each shift
        by shift_count[0]), and then divided by shift_count[0].
        The procedure is then continued for each element of shift_count. This
        computes the total shift in each direction for the sliding window.

        We then need to compute the padding required after the window in each
        dimension, which is equal to the total input dimension minus the sliding
        window dimension minus the (left) shift amount. We construct the
        array pad_values which contains the left and right pad values for each
        dimension, in reverse order of dimensions, starting from the last one.

        Once these padding values are computed, we pad the sliding window tensor
        of 1s with 0s appropriately, which is the corresponding mask,
        and the result will match the input shape.
        """
        remaining_total = ablated_feature_num
        current_index = []
        for i, shift_count in enumerate(shift_counts):
            stride = strides[i] if isinstance(strides, tuple) else strides
            current_index.append((remaining_total % shift_count) * stride)
            remaining_total = remaining_total // shift_count

        remaining_padding = np.subtract(
            expanded_input.shape[2:], np.add(current_index, sliding_window_tsr.shape)
        )
        pad_values = [
            val
            for pair in zip(remaining_padding, current_index, strict=False)
            for val in pair
        ]
        pad_values.reverse()
        padded_tensor = torch.nn.functional.pad(
            sliding_window_tsr,
            tuple(pad_values),  # type: ignore
        )
        return padded_tensor.reshape((1,) + padded_tensor.shape)

    def _get_feature_range_and_mask(
        self, input: Tensor, input_mask: Tensor, **kwargs: Any
    ) -> tuple[int, int, None]:
        feature_max = np.prod(kwargs["shift_counts"])
        return 0, int(feature_max), None

    def _get_feature_counts(self, inputs, feature_mask, **kwargs):
        """return the numbers of possible input features"""
        return tuple(np.prod(counts).astype(int) for counts in kwargs["shift_counts"])


class OcclusionExplainer(Explainer):
    """Occlusion explainer for computing sliding-window perturbation attributions.

    This explainer computes attributions using the Occlusion method, which systematically
    replaces rectangular regions of the input with baseline values (typically zeros)
    and measures the resulting change in model output. This approach is particularly
    effective for image data where spatial regions can be meaningfully occluded.
    Supports both single-target and multi-target modes with structured input/output.

    The Occlusion method provides intuitive attributions by directly measuring
    the importance of spatial regions through systematic perturbation.

    Args:
        model: The PyTorch model whose output is to be explained.
        sliding_window_shapes: Shape of the occlusion window for each input tensor.
            Can be a single tuple for all inputs or tuple of tuples for each input.
        strides: Stride for sliding the occlusion window. Can be int, tuple, or
            tuple of tuples. If None, defaults to sliding_window_shapes.
        multi_target: Whether to use multi-target mode. When True, can compute
            attributions for multiple targets simultaneously. Defaults to False.
        internal_batch_size: Batch size for internal computations (perturbations
            per evaluation). Defaults to 1.

    Example:
        Single-target usage for image data:
        >>> import torch
        >>> from collections import OrderedDict
        >>> from torchxai.data_types import ExplanationInputs
        >>>
        >>> # CNN model for image classification
        >>> model = torch.nn.Sequential(
        ...     torch.nn.Conv2d(3, 16, 3),
        ...     torch.nn.ReLU(),
        ...     torch.nn.AdaptiveAvgPool2d(1),
        ...     torch.nn.Flatten(),
        ...     torch.nn.Linear(16, 10),
        ... )
        >>> # 8x8 sliding window with 4x4 stride
        >>> explainer = OcclusionExplainer(
        ...     model, sliding_window_shapes=(8, 8), strides=(4, 4)
        ... )
        >>>
        >>> explanation_inputs = ExplanationInputs(
        ...     inputs=OrderedDict({"image": torch.randn(1, 3, 32, 32)}),
        ...     target=torch.tensor([5]),
        ...     baselines=OrderedDict({"image": torch.zeros(1, 3, 32, 32)}),
        ... )
        >>> attributions = explainer.explain(explanation_inputs)
        >>> # Returns: OrderedDict({"image": torch.Tensor})

        Multi-target usage:
        >>> explainer_mt = OcclusionExplainer(
        ...     model, sliding_window_shapes=(8, 8), strides=(4, 4), multi_target=True
        ... )
        >>> explanation_inputs_mt = ExplanationInputs(
        ...     inputs=OrderedDict({"image": torch.randn(1, 3, 32, 32)}),
        ...     target=[torch.tensor([5]), torch.tensor([2])],
        ...     baselines=OrderedDict({"image": torch.zeros(1, 3, 32, 32)}),
        ... )
        >>> mt_attributions = explainer_mt.explain(explanation_inputs_mt)
        >>> # Returns: [OrderedDict({"image": torch.Tensor}), OrderedDict({"image": torch.Tensor})]
    """

    def __init__(
        self,
        model: Module,
        sliding_window_shapes: tuple[int, ...] | tuple[tuple[int, ...], ...],
        strides: None
        | int
        | tuple[int, ...]
        | tuple[int | tuple[int, ...], ...] = None,
        multi_target: bool = False,
        internal_batch_size: int = 1,
        show_progress: bool = False,
    ) -> None:
        """Initialize the OcclusionExplainer.

        Args:
            model: The model whose output is to be explained.
            sliding_window_shapes: Shape of the occlusion window for each input tensor.
            strides: Stride for sliding the occlusion window. If None, uses sliding_window_shapes.
            multi_target: Whether to use multi-target mode. Defaults to False.
            internal_batch_size: Batch size for internal computations. Defaults to 1.
        """
        self._sliding_window_shapes = sliding_window_shapes
        self._strides = strides if strides is not None else sliding_window_shapes
        self._show_progress = show_progress

        super().__init__(model, multi_target, internal_batch_size)

    def _init_single_target_explanation_fn(self) -> Callable:
        """Initialize single-target Occlusion attribution function.

        Returns:
            Occlusion attribution function for single targets.
        """
        return partial(
            Occlusion(self._model).attribute,
            sliding_window_shapes=self._sliding_window_shapes,
            strides=self._strides,
            perturbations_per_eval=self._internal_batch_size,
            show_progress=self._show_progress,
        )

    def _init_multi_target_explanation_fn(self) -> Callable:
        """Initialize multi-target Occlusion attribution function.

        Returns:
            MultiTargetOcclusion attribution function for multiple targets.
        """
        return partial(
            MultiTargetOcclusion(self._model).attribute,
            sliding_window_shapes=self._sliding_window_shapes,
            strides=self._strides,
            perturbations_per_eval=self._internal_batch_size,
            show_progress=self._show_progress,
        )

    def explain(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: TargetType,
        baselines: BaselineType = None,
        additional_forward_args: Any = None,
    ) -> OrderedDict[str, torch.Tensor] | list[OrderedDict[str, torch.Tensor]]:
        """Compute Occlusion attributions for the given inputs.

        This method provides a backward-compatible interface that accepts individual
        parameters and constructs ExplanationInputs internally before calling the
        parent class explain method.

        Args:
            inputs: Input tensors for attribution computation. Should be an OrderedDict
                mapping feature names to tensors when used with this explainer.
            target: Target indices for attribution computation. Can be a tensor
                (single-target) or list of tensors (multi-target).
            baselines: Baseline tensors for occlusion (typically zeros). If None,
                uses zero baselines matching input shape.
            additional_forward_args: Additional arguments for model forward pass.

        Returns:
            For single-target mode: OrderedDict mapping feature names to attribution tensors.
            For multi-target mode: List of OrderedDicts, one per target.

        Note:
            The sliding window and stride parameters are set during initialization.
            Attribution values represent the importance of each spatial region
            as measured by occlusion impact.

        Example:
            >>> # For image data with 8x8 occlusion windows
            >>> attributions = explainer.explain(
            ...     inputs=OrderedDict({"image": torch.randn(1, 3, 224, 224)}),
            ...     target=torch.tensor([285]),  # ImageNet class
            ...     baselines=OrderedDict({"image": torch.zeros(1, 3, 224, 224)}),
            ... )
        """
        return super().explain(
            inputs=inputs,
            target=target,
            baselines=baselines,
            additional_forward_args=additional_forward_args,
        )
