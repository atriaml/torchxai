from abc import ABC, abstractmethod
from collections.abc import Callable

import torch

from torchxai.data_types import TensorOrTupleOfTensorsGeneric
from torchxai.data_types._target import ExplanationTarget


class Explainer(ABC):
    """Abstract base class for TorchXAI explainers.

    Provides a common interface for all explanation methods in TorchXAI,
    supporting both single-target and multi-target attribution.

    Args:
        model: The PyTorch model for which explanations will be computed.

    Attributes:
        model: The model used for attribution computation.
    """

    __repr_attrs__: list[str] = []

    def __init__(self, model: torch.nn.Module) -> None:
        self._model = model

    @abstractmethod
    def explain(
        self, *args, **kwargs
    ) -> TensorOrTupleOfTensorsGeneric | list[TensorOrTupleOfTensorsGeneric]:
        """Compute attributions for the given inputs.

        Args:
            *args: Positional arguments passed to the underlying attribution method.
            **kwargs: Keyword arguments including `inputs`, `target`, and method-specific
                parameters such as `baselines`, `feature_mask`, and `sliding_window_shapes`.

        Returns:
            Tensor in single-target mode. List of Tensors, one per target, in multi-target mode.

        Raises:
            AssertionError: If target format doesn't match the explainer mode.
        """

    def __repr__(self) -> str:
        attr_str = ", ".join(
            f"{attr.lstrip('_')}={getattr(self, attr)}" for attr in self.__repr_attrs__
        )
        return f"{self.__class__.__name__}({attr_str})"


class FeatureAttributionExplainer(Explainer):
    """Concrete base class for all TorchXAI feature-attribution explainers.

    Extends `Explainer` with `multi_target` support, configurable batch sizes, and
    the plumbing that routes `explain()` calls to single-target or multi-target
    Captum attribution functions.

    All concrete explainers (`SaliencyExplainer`, `IntegratedGradientsExplainer`,
    etc.) inherit from this class.

    Args:
        model: The PyTorch model for which explanations will be computed.
        multi_target: When ``True`` the explainer accepts a list of
            `ExplanationTargetType` objects as ``target`` and returns a
            ``list[Tensor]``. Defaults to ``False``.
        internal_batch_size: Batch size used internally for attribution computation.
            Defaults to 64.
        grad_batch_size: Batch size used for gradient computation operations.
            Defaults to 64.

    Attributes:
        model: The model used for attribution computation.
        multi_target: Flag controlling single- vs. multi-target mode.
        internal_batch_size: Internal batch size for computations.
        grad_batch_size: Batch size for gradient operations.

    Examples:
        >>> import torch
        >>> from torchxai.explainers import SaliencyExplainer
        >>> from torchxai.data_types import SingleTargetAcrossBatch
        >>>
        >>> model = torch.nn.Linear(10, 3)
        >>> inputs = torch.randn(1, 10)
        >>>
        >>> # Single-target
        >>> explainer = SaliencyExplainer(model, multi_target=False)
        >>> attrs = explainer.explain(inputs=inputs, target=SingleTargetAcrossBatch(index=0))
        >>> attrs.shape   # (1, 10)
        >>>
        >>> # Multi-target
        >>> explainer_mt = SaliencyExplainer(model, multi_target=True)
        >>> targets = [SingleTargetAcrossBatch(index=i) for i in range(3)]
        >>> attrs_list = explainer_mt.explain(inputs=inputs, target=targets)
        >>> len(attrs_list)   # 3
    """

    __repr_attrs__ = ["_multi_target", "_internal_batch_size", "_grad_batch_size"]

    def __init__(
        self,
        model: torch.nn.Module,
        multi_target: bool = False,
        internal_batch_size: int = 64,
        grad_batch_size: int = 64,
    ) -> None:
        super().__init__(model)
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
            List of Tensors, one per target.

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
        """Compute attributions for the given inputs.

        Args:
            inputs: Input tensor(s) for which attributions are computed.
            target: An `ExplanationTargetType` (e.g. `SingleTargetAcrossBatch`) for
                single-target mode, or a list of them for multi-target mode.
            **kwargs: Additional method-specific arguments (`baselines`,
                `feature_mask`, `sliding_window_shapes`, etc.).

        Returns:
            Tensor in single-target mode. List of Tensors, one per target, in multi-target mode.

        Raises:
            AssertionError: If target format doesn't match the explainer mode.
        """

    def __repr__(self) -> str:
        attr_str = ", ".join(
            f"{attr.lstrip('_')}={getattr(self, attr)}" for attr in self.__repr_attrs__
        )
        return f"{self.__class__.__name__}({attr_str})"
