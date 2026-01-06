from abc import ABC, abstractmethod
from collections.abc import Callable

import torch

from torchxai.data_types import TensorOrTupleOfTensorsGeneric
from torchxai.data_types._target import ExplanationTarget


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

    __repr_attrs__ = ["_multi_target", "_internal_batch_size", "_grad_batch_size"]

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
        self, **explanation_inputs
    ) -> TensorOrTupleOfTensorsGeneric:
        """Forward pass for single-target explanations.

        Args:
            explanation_inputs: Structured explanation inputs containing features and target.

        Returns:
            Dictionary mapping feature names to their corresponding attributions.
        """
        # inspect explainer signature and save
        target = explanation_inputs.pop("target")
        assert isinstance(target, ExplanationTarget), (
            "Explainer explain method must be called with target of type ExplanationTarget."
        )
        return self._explanation_fn(**explanation_inputs, target=target.value)

    def _multi_target_forward(
        self, **explanation_inputs
    ) -> list[TensorOrTupleOfTensorsGeneric]:
        """Forward pass for multi-target explanations.

        Args:
            explanation_inputs: Structured explanation inputs with list of targets.

        Returns:
            List of dictionaries, one per target, mapping feature names to attributions.

        Raises:
            AssertionError: If targets are not properly formatted for multi-target mode.
        """
        # inspect explainer signature and saveexplanation_inputs
        target = explanation_inputs.pop("target")
        assert isinstance(target, list), (
            "Target must be a list for multi-target explanation."
        )
        target_validated = []
        for t in target:
            assert isinstance(t, ExplanationTarget), (
                "Each target in multi-target explanation must be of type ExplanationTarget."
            )
            target_validated.append(t.value)
        per_target_explanations = self._explanation_fn(
            **explanation_inputs, target=target_validated
        )
        assert isinstance(per_target_explanations, list), (
            "Explanations must be a list for multi-target explanation."
        )
        assert len(per_target_explanations) == len(target), (
            "Number of explanations must match number of targets."
        )

        return per_target_explanations

    def _default_explain(
        self, **kwargs
    ) -> TensorOrTupleOfTensorsGeneric | list[TensorOrTupleOfTensorsGeneric]:
        # get the explanation_outputs
        # get value of the target

        if self._multi_target:
            return self._multi_target_forward(**kwargs)
        else:
            return self._single_target_forward(**kwargs)

    @abstractmethod
    def explain(
        self, *args, **kwargs
    ) -> TensorOrTupleOfTensorsGeneric | list[TensorOrTupleOfTensorsGeneric]:
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

    def __repr__(self) -> str:
        attr_str = ", ".join(
            f"{attr.lstrip('_')}={getattr(self, attr)}" for attr in self.__repr_attrs__
        )
        return f"{self.__class__.__name__}({attr_str})"
