from collections.abc import Callable
from typing import Any

import torch
from captum._utils.common import _format_tensor_into_tuples

from torchxai.data_types import TargetType, TensorOrTupleOfTensorsGeneric
from torchxai.explainers._explainer import FeatureAttributionExplainer


class RandomExplainer(FeatureAttributionExplainer):
    """Random explainer for generating baseline attributions using random noise.

    This explainer generates random attributions with the same shape as the input tensors.
    It serves as a baseline method for comparison with other attribution techniques,
    helping to establish whether other methods provide meaningful signal above random
    noise. This is particularly useful for sanity checks and statistical significance
    testing of attribution methods. Supports both single-target and multi-target modes
    with structured input/output.

    Random attributions help establish baseline performance and can be used to
    validate that other attribution methods provide meaningful explanations.

    Args:
        model: The PyTorch model whose output is to be explained (not used for computation
            but required for API consistency).
        multi_target: Whether to use multi-target mode. When True, can generate
            random attributions for multiple targets simultaneously. Defaults to False.
        internal_batch_size: Batch size for internal computations (not used but
            maintained for API consistency). Defaults to 64.
        random_seed: Random seed for reproducible random attributions. If None,
            uses PyTorch's current random state. Defaults to None.

    Examples:
        Single-target usage:
        >>> import torch
        >>> from collections import OrderedDict
        >>> from torchxai.data_types import ExplanationInputs
        >>>
        >>> model = torch.nn.Sequential(
        ...     torch.nn.Linear(10, 5), torch.nn.ReLU(), torch.nn.Linear(5, 2)
        ... )
        >>> explainer = RandomExplainer(model, random_seed=42)
        >>>
        >>> explanation_inputs = ExplanationInputs(
        ...     inputs=OrderedDict({"features": torch.randn(2, 10)}),
        ...     target=torch.tensor([0, 1]),
        ... )
        >>> attributions = explainer.explain(explanation_inputs)
        >>> # Returns: OrderedDict({"features": torch.Tensor}) with random values

        Multi-target usage:
        >>> explainer_mt = RandomExplainer(model, multi_target=True, random_seed=42)
        >>> explanation_inputs_mt = ExplanationInputs(
        ...     inputs=OrderedDict({"features": torch.randn(2, 10)}),
        ...     target=[torch.tensor([0, 1]), torch.tensor([1, 0])],
        ... )
        >>> mt_attributions = explainer_mt.explain(explanation_inputs_mt)
        >>> # Returns: [OrderedDict({"features": torch.Tensor}), OrderedDict({"features": torch.Tensor})]
    """

    def __init__(
        self,
        model: torch.nn.Module,
        multi_target: bool = False,
        internal_batch_size: int = 64,
        random_seed: int | None = None,
    ) -> None:
        """Initialize the RandomExplainer.

        Args:
            model: The model whose output is to be explained (for API consistency).
            multi_target: Whether to use multi-target mode. Defaults to False.
            internal_batch_size: Batch size for internal computations. Defaults to 64.
            random_seed: Random seed for reproducible results. Defaults to None.
        """
        self._random_seed = random_seed
        super().__init__(model, multi_target, internal_batch_size)

    def _init_single_target_explanation_fn(self) -> Callable:
        """Initialize single-target random attribution function.

        Returns:
            Function that generates random attributions for single targets.
        """

        def explanation_fn(inputs, *args, **kwargs):
            if self._random_seed is not None:
                torch.manual_seed(self._random_seed)
            inputs = _format_tensor_into_tuples(inputs)
            return tuple(torch.randn_like(input) for input in inputs)

        return explanation_fn

    def _init_multi_target_explanation_fn(self) -> Callable:
        """Initialize multi-target random attribution function.

        Returns:
            Function that generates random attributions for multiple targets.
        """

        def mt_explanation_fn(inputs, target, *args, **kwargs):
            if self._random_seed is not None:
                torch.manual_seed(self._random_seed)
            inputs = _format_tensor_into_tuples(inputs)
            # Generate the same random attributions for each target
            base_attributions = tuple(torch.randn_like(input) for input in inputs)
            # Return list with one attribution set per target
            num_targets = len(target) if isinstance(target, list) else 1
            return [base_attributions for _ in range(num_targets)]

        return mt_explanation_fn

    def explain(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: TargetType,
        additional_forward_args: tuple[Any, ...] | None = None,
    ) -> TensorOrTupleOfTensorsGeneric | list[TensorOrTupleOfTensorsGeneric]:
        """Generate random attributions for the given inputs.

        This method provides a backward-compatible interface that accepts individual
        parameters and constructs ExplanationInputs internally before calling the
        parent class explain method.

        Args:
            inputs: Input tensors for attribution computation. Should be an OrderedDict
                mapping feature names to tensors when used with this explainer.
            target: Target indices for attribution computation. Can be a tensor
                (single-target) or list of tensors (multi-target).
            additional_forward_args: Additional arguments for model forward pass (ignored).

        Returns:
            For single-target mode: OrderedDict mapping feature names to random attribution tensors.
            For multi-target mode: List of OrderedDicts, one per target.

        Note:
            Random attributions are generated with the same shape as input tensors.
            If a random seed was provided during initialization, the results will be
            reproducible across calls with the same inputs.

        Examples:
            >>> # Generate random baseline for comparison
            >>> random_attributions = explainer.explain(
            ...     inputs=OrderedDict({"features": torch.randn(1, 10)}),
            ...     target=torch.tensor([1]),
            ... )
            >>> print(
            ...     f"Random attribution range: {random_attributions['features'].min():.3f} to {random_attributions['features'].max():.3f}"
            ... )
        """
        return super().explain(
            inputs=inputs,
            target=target,
            additional_forward_args=additional_forward_args,
        )
