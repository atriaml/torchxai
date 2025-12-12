import typing
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable

import torch

from torchxai.data_types import ExplanationInputs
from torchxai.data_types.common import TensorOrTupleOfTensorsGeneric


class Explainer(ABC):
    """Abstract base class for TorchXAI explainers.

    This class provides a common interface for all explanation methods in TorchXAI.
    It supports both single-target and multi-target explanations with automatic
    function inspection and structured input handling via ExplanationInputs.

    Args:
        model: The PyTorch model for which explanations will be computed.
        multi_target: Whether to use the multi-target version of the explainer.
            When True, the explainer can compute attributions for multiple targets
            simultaneously. Defaults to False.
        internal_batch_size: Batch size used internally for attribution computation.
            Defaults to 64.
        grad_batch_size: Batch size used for gradient computation operations.
            Defaults to 64.

    Attributes:
        model: The model used for attribution computation.
        multi_target: Flag controlling which explainer version is loaded.
        internal_batch_size: Internal batch size for computations.
        grad_batch_size: Batch size for gradient operations.

    Examples:
        >>> import torch
        >>> from collections import OrderedDict
        >>> from torchxai.data_types import ExplanationInputs
        >>>
        >>> model = torch.nn.Linear(10, 2)
        >>> explainer = MyExplainer(model, multi_target=False)
        >>>
        >>> inputs = torch.randn(2, 10)
        >>> explanation_inputs = ExplanationInputs(
        ...     inputs=OrderedDict({"features": inputs}), target=torch.tensor([0, 1])
        ... )
        >>> attributions = explainer.explain(explanation_inputs)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        multi_target: bool = False,
        internal_batch_size: int = 64,
        grad_batch_size: int = 64,
    ) -> None:
        self._model = model
        self._multi_target = multi_target
        self._internal_batch_size = internal_batch_size
        self._grad_batch_size = grad_batch_size
        self._explanation_fn = self._init_explanation_fn()

    @property
    def model(self) -> torch.nn.Module:
        """The model used for attribution computation."""
        return self._model

    @model.setter
    def model(self, model: torch.nn.Module) -> None:
        """Set the model and reinitialize the explanation function.

        Args:
            model: The new PyTorch model to use for explanations.
        """
        self._model = model
        self._explanation_fn = self._init_explanation_fn()

    @property
    def multi_target(self) -> bool:
        """Whether the explainer uses multi-target mode."""
        return self._multi_target

    @multi_target.setter
    def multi_target(self, multi_target: bool) -> None:
        """Set multi-target mode and reinitialize the explanation function.

        Args:
            multi_target: Whether to enable multi-target mode.
        """
        self._multi_target = multi_target
        self._explanation_fn = self._init_explanation_fn()

    def _init_explanation_fn(self) -> Callable:
        """Initialize the appropriate explanation function based on mode.

        Returns:
            A callable that computes attributions for the configured model.
        """
        if self._multi_target:
            return self._init_multi_target_explanation_fn()
        else:
            return self._init_single_target_explanation_fn()

    @abstractmethod
    def _init_single_target_explanation_fn(self) -> Callable:
        """Initialize the single-target attribution generation callable.

        This method must be implemented by subclasses to provide the specific
        single-target attribution computation logic.

        Returns:
            A callable that computes single-target attributions.
        """

    def _init_multi_target_explanation_fn(self) -> Callable:
        """Initialize the multi-target attribution generation callable.

        This method provides a default implementation that falls back to iterative
        calls to the single-target explainer if not overridden by subclasses.
        Subclasses should override this method for more efficient multi-target implementations.

        Returns:
            A callable that computes multi-target attributions by iterating over targets.
        """
        # Fix: Call single-target method directly to avoid infinite recursion
        single_target_fn = self._init_single_target_explanation_fn()

        def _default_multi_target_explanation_fn(*args, **kwargs):
            targets = kwargs.pop("target", [])
            return [
                single_target_fn(*args, target=target, **kwargs) for target in targets
            ]

        return _default_multi_target_explanation_fn

    def _single_target_forward(
        self, explanation_inputs
    ) -> dict[str, TensorOrTupleOfTensorsGeneric]:
        """Forward pass for single-target explanations.

        Args:
            explanation_inputs: Structured explanation inputs containing features and target.

        Returns:
            Dictionary mapping feature names to their corresponding attributions.
        """
        # inspect explainer signature and save
        fields_set = explanation_inputs.model_fields_set
        explanation_tuple_inputs = explanation_inputs.to_explanation_tuple_inputs()
        explanations = self._explanation_fn(
            **explanation_tuple_inputs.model_dump(include=fields_set)
        )
        return OrderedDict(
            {
                key: explanations[idx]
                for idx, key in enumerate(explanation_inputs.inputs.keys())
            }
        )

    def _multi_target_forward(
        self, explanation_inputs
    ) -> list[OrderedDict[str, torch.Tensor]]:
        """Forward pass for multi-target explanations.

        Args:
            explanation_inputs: Structured explanation inputs with list of targets.

        Returns:
            List of dictionaries, one per target, mapping feature names to attributions.

        Raises:
            AssertionError: If targets are not properly formatted for multi-target mode.
        """
        # inspect explainer signature and save
        fields_set = explanation_inputs.model_fields_set
        explanation_tuple_inputs = explanation_inputs.to_explanation_tuple_inputs()

        assert isinstance(explanation_tuple_inputs.target, list), (
            "Target must be a list for multi-target explanation."
        )
        per_target_explanations = self._explanation_fn(
            **explanation_tuple_inputs.model_dump(include=fields_set)
        )
        assert isinstance(per_target_explanations, list), (
            "Explanations must be a list for multi-target explanation."
        )
        assert len(per_target_explanations) == len(explanation_tuple_inputs.target), (
            "Number of explanations must match number of targets."
        )

        # convert the list[tuple[tensors]] -> list[dict[tensors]] to -> tuples -> list of targets
        def _tuples_to_dict(
            exp_tuples: tuple, keys: list[str]
        ) -> OrderedDict[str, torch.Tensor]:
            return OrderedDict(zip(keys, exp_tuples, strict=True))

        feature_keys = list(explanation_inputs.inputs.keys())
        per_target_explanations = [
            _tuples_to_dict(exp, feature_keys) for exp in per_target_explanations
        ]

        return per_target_explanations

    @abstractmethod
    def _build_inputs(self, *args, **kwargs) -> ExplanationInputs:
        """Build ExplanationInputs from args and kwargs.

        This method must be implemented by subclasses to construct an
        ExplanationInputs object from the provided arguments.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        Returns:
            ExplanationInputs: The constructed ExplanationInputs object.
        """
        pass

    def explain(
        self, *args, **kwargs
    ) -> OrderedDict[str, torch.Tensor] | list[OrderedDict[str, torch.Tensor]]:
        """Compute attributions for the given structured inputs.

        This is the main method for generating explanations. It automatically handles
        both single-target and multi-target modes based on the explainer configuration.
        Accepts either ExplanationInputs object or individual keyword arguments.

        Args:
            *args: Positional arguments (typically ExplanationInputs object).
            **kwargs: Keyword arguments that can include:
                - inputs: OrderedDict mapping feature names to tensors
                - target: Target tensor (single-target) or list of tensors (multi-target)
                - additional_forward_args: Optional additional arguments for model forward pass
                - baselines: Optional baseline tensors for attribution methods
                - feature_mask: Optional feature masks for attribution computation

        Returns:
            For single-target mode: OrderedDict mapping feature names to attribution tensors.
            For multi-target mode: List of OrderedDicts, one per target.

        Examples:
            Using ExplanationInputs object:
            >>> from torchxai.data_types import ExplanationInputs
            >>> explanation_inputs = ExplanationInputs(
            ...     inputs=OrderedDict({"input": torch.randn(2, 10)}),
            ...     target=torch.tensor([0, 1]),
            ... )
            >>> attributions = explainer.explain(explanation_inputs)
            >>> # Returns: OrderedDict({"input": torch.Tensor})

            Using keyword arguments:
            >>> attributions = explainer.explain(
            ...     inputs=OrderedDict({"input": torch.randn(2, 10)}),
            ...     target=torch.tensor([0, 1]),
            ... )

        Raises:
            AssertionError: If target format doesn't match the explainer mode or batch size requirements.
        """

        # we internally use ExplanationInputs for structured validation of the explainer inputs before passing to the explainer function
        explanation_inputs = self._build_inputs(*args, **kwargs)

        if self._multi_target:
            return typing.cast(
                list[OrderedDict[str, torch.Tensor]],
                self._multi_target_forward(explanation_inputs),
            )
        else:
            return typing.cast(
                OrderedDict[str, torch.Tensor],
                self._single_target_forward(explanation_inputs),
            )
