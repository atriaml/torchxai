from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Self, cast

import torch
from pydantic import (
    BaseModel,
    ConfigDict,
    field_serializer,
    field_validator,
    model_validator,
)

from torchxai.ignite._utilities import _tensor_or_tensor_dict_as_detached_tuple

BaselineType = torch.Tensor | tuple[torch.Tensor]


def _convert_scalar(v):
    if isinstance(v, (int, float)):
        return torch.tensor(v)
    return v


def _normalize_dictlike(v, allow_none=True):
    if v is None:
        return None if allow_none else OrderedDict()

    # Convert scalar → tensor
    v = _convert_scalar(v)

    # Already a dict → keep order
    if isinstance(v, dict):
        return OrderedDict(v)

    # Sequence of items → positional OrderedDict
    if isinstance(v, (list, tuple)):
        return OrderedDict((str(i), _convert_scalar(x)) for i, x in enumerate(v))

    # Single tensor → wrap
    if isinstance(v, torch.Tensor):
        return OrderedDict({"0": v})

    # Fallback for arbitrary single values
    return OrderedDict({"0": _convert_scalar(v)})


def _to_device(obj, device):
    if obj is None:
        return None
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, OrderedDict):
        return OrderedDict({k: _to_device(v, device) for k, v in obj.items()})
    if isinstance(obj, tuple):
        return tuple(_to_device(v, device) for v in obj)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_device(v, device) for v in obj]
    return obj


def _match_keys(dict1, dict2):
    if dict1 is None or dict2 is None:
        return
    if set(dict1.keys()) != set(dict2.keys()):
        raise ValueError("Keys of the two dictionaries do not match.")


class ExplanationTarget(BaseModel, ABC):
    """Base class for all target types with Pydantic validation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def value(self) -> Any:
        """Return the raw value of the target."""
        raise NotImplementedError()

    @classmethod
    def from_raw_input(cls, raw_target: Any) -> ExplanationTargetType:
        """Create appropriate TargetType from raw input.

        Args:
            raw_target: Raw target input (int, list, tuple, etc.)
        Returns:
            Appropriate TargetType instance.
        """
        if raw_target is None:
            return NoTarget()
        if isinstance(raw_target, torch.Tensor):
            raw_target = (
                raw_target.item() if raw_target.numel() == 1 else raw_target.tolist()
            )
        if isinstance(raw_target, int):
            return SingleTargetAcrossBatch(index=raw_target)
        if isinstance(raw_target, tuple) and all(
            isinstance(i, int) for i in raw_target
        ):
            return MultiIndexTargetAcrossBatch(indices=raw_target)
        if isinstance(raw_target, list) and all(isinstance(i, int) for i in raw_target):
            return SingleTargetPerSample(indices=raw_target)
        if isinstance(raw_target, list) and all(
            isinstance(i, tuple) and all(isinstance(j, int) for j in i)
            for i in raw_target
        ):
            return MultiIndexTargetPerSample(indices=raw_target)
        raise ValueError(f"Cannot convert raw target of type {type(raw_target)}.")

    @abstractmethod
    def select(self, output: torch.Tensor) -> torch.Tensor:
        """Select the appropriate part of the model output."""
        pass

    def __repr__(self) -> str:
        params_str = ", ".join(f"{k}={v!r}" for k, v in self.model_dump().items())
        return f"{self.__class__.__name__}({params_str})"

    def __str__(self) -> str:
        return super().__repr__()


class NoTarget(ExplanationTarget):
    """No target selection - return the full model output."""

    def select(self, output: torch.Tensor) -> torch.Tensor:
        return output

    @property
    def value(self):
        return None


class SingleTargetAcrossBatch(ExplanationTarget):
    """Single target index applied to all examples in the batch."""

    index: int

    @field_validator("index")
    @classmethod
    def validate_index(cls, v):
        if v < 0:
            raise ValueError("Target index must be non-negative")
        return v

    def select(self, output: torch.Tensor) -> torch.Tensor:
        """Select output[:, self.index] or output[..., self.index]"""
        assert self.index < output.shape[-1], (
            f"Target index {self.index} out of bounds for output shape {output.shape}"
        )
        return output[..., self.index]

    def is_multi_target(self) -> bool:
        return False

    @property
    def value(self) -> int:
        return self.index


class MultiIndexTargetAcrossBatch(ExplanationTarget):
    """Multiple indices for selecting nested dimensions."""

    indices: tuple[int, ...]

    def select(self, output: torch.Tensor) -> torch.Tensor:
        """Select output[:, indices[0], indices[1], ...]"""
        assert len(self.indices) <= len(output.shape) - 1, (
            f"Cannot choose target column with output shape {output.shape!r}."
        )
        selection = (slice(None),) + self.indices
        return output[selection]

    @property
    def value(self) -> tuple[int, ...]:
        return self.indices


class SingleTargetPerSample(ExplanationTarget):
    indices: list[int]

    def select(self, output: torch.Tensor) -> torch.Tensor:
        """Use torch.gather to select different index per example."""
        assert len(self.indices) == output.shape[0], (
            "Target list length does not match output!"
        )
        return torch.gather(
            output,
            1,
            torch.tensor(self.indices, device=output.device).reshape(len(output), 1),
        ).squeeze(-1)

    @property
    def value(self) -> list[int]:
        return self.indices


class MultiIndexTargetPerSample(ExplanationTarget):
    indices: list[tuple[int, ...]]

    def select(self, output: torch.Tensor) -> torch.Tensor:
        """Use torch.gather to select different index per example."""
        assert len(self.indices) == output.shape[0], (
            "Target list length does not match output!"
        )
        return torch.stack(
            [
                output[(i,) + cast(tuple, targ_elem)]
                for i, targ_elem in enumerate(self.indices)
            ]
        )

    @property
    def value(self) -> list[tuple[int, ...]]:
        return self.indices


ExplanationTargetType = (
    NoTarget
    | SingleTargetAcrossBatch
    | MultiIndexTargetAcrossBatch
    | SingleTargetPerSample
    | MultiIndexTargetPerSample
)


class ExplanationInputs(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        frozen=True,
        extra="forbid",
        revalidate_instances="always",
    )
    sample_id: list[str] | None = None
    inputs: OrderedDict[str, torch.Tensor] | torch.Tensor
    additional_forward_args: tuple[Any, ...] | None = None
    baselines: OrderedDict[str, torch.Tensor] | torch.Tensor | None = None
    feature_mask: OrderedDict[str, torch.Tensor] | torch.Tensor | None = None
    target: ExplanationTargetType | list[ExplanationTargetType] = NoTarget()
    frozen_features: list[torch.Tensor] | None = None

    @property
    def batch_size(self) -> int:
        first_key = next(iter(self.inputs))
        return self.inputs[first_key].shape[0]

    @field_validator("target", mode="before")
    @classmethod
    def convert_target(cls, v):
        if (
            isinstance(v, ExplanationTarget)
            or isinstance(v, list)
            and all(isinstance(t, ExplanationTarget) for t in v)
        ):
            return v
        validated = ExplanationTarget.from_raw_input(v)
        return validated

    @field_serializer("target")
    def serialize_target(self, v: ExplanationTarget | list[ExplanationTarget]) -> Any:
        if isinstance(v, list):
            return [t.value for t in v]
        return v.value

    @field_validator("additional_forward_args", mode="before")
    @classmethod
    def to_tuple(cls, v):
        if v is None:
            return None
        if isinstance(v, tuple):
            return v
        if isinstance(v, list):
            return tuple(v)
        return (v,)

    @field_validator("baselines", "inputs", "feature_mask", mode="before")
    @classmethod
    def normalize_inputs(cls, v):
        return _normalize_dictlike(v)

    @model_validator(mode="after")
    def validate_inputs(self):
        assert isinstance(self.inputs, OrderedDict), (
            "inputs must be an OrderedDict internally."
        )
        _match_keys(self.baselines, self.inputs)
        _match_keys(self.feature_mask, self.inputs)
        return self

    @field_validator("frozen_features", mode="before")
    @classmethod
    def normalize_frozen_features(cls, v):
        if v is None:
            return None
        if isinstance(v, torch.Tensor):
            return [v]
        if isinstance(v, (list, tuple)):
            return list(v)
        return [v]

    # ------------------------------
    # Move everything to device
    # ------------------------------
    def to(self, device: str | torch.device = "cpu") -> Self:
        return self.model_copy(
            update={
                "baselines": _to_device(self.baselines, device),
                "feature_mask": _to_device(self.feature_mask, device),
                "inputs": _to_device(self.inputs, device),
                "additional_forward_args": _to_device(
                    self.additional_forward_args, device
                ),
                "target": _to_device(self.target, device),
                "frozen_features": _to_device(self.frozen_features, device),
            }
        )

    @property
    def model_inputs(self) -> tuple[torch.Tensor, ...]:
        assert isinstance(self.inputs, OrderedDict), (
            f"Expected OrderedDict, got {type(self.inputs)}"
        )
        return tuple(self.inputs.values()) + (
            self.additional_forward_args
            if self.additional_forward_args is not None
            else ()
        )

    def to_explanation_tuple_inputs(self) -> ExplanationTupleInputs:
        return ExplanationTupleInputs(
            sample_id=self.sample_id,
            inputs=_tensor_or_tensor_dict_as_detached_tuple(self.inputs),
            baselines=_tensor_or_tensor_dict_as_detached_tuple(self.baselines),
            feature_mask=_tensor_or_tensor_dict_as_detached_tuple(self.feature_mask),
            additional_forward_args=_tensor_or_tensor_dict_as_detached_tuple(
                self.additional_forward_args
            ),
            target=self.target,
            feature_keys=list(self.inputs.keys()),
            frozen_features=self.frozen_features,
        )


class ExplanationTupleInputs(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        frozen=True,
        extra="forbid",
        revalidate_instances="always",
    )
    feature_keys: list[str]
    sample_id: list[str] | None = None
    inputs: tuple[torch.Tensor, ...]
    additional_forward_args: tuple[Any, ...] | None = None
    baselines: tuple[torch.Tensor, ...] | None = None
    feature_mask: tuple[torch.Tensor, ...] | None = None
    target: ExplanationTargetType | list[ExplanationTargetType] = NoTarget()
    frozen_features: list[torch.Tensor] | None = None

    @field_serializer("target")
    def serialize_target(
        self, v: ExplanationTargetType | list[ExplanationTargetType]
    ) -> Any:
        if isinstance(v, list):
            return [t.value for t in v]
        return v.value

    def to_explanation_inputs(self) -> ExplanationInputs:
        return ExplanationInputs(
            sample_id=self.sample_id,
            inputs=OrderedDict(
                (self.feature_keys[i], tensor) for i, tensor in enumerate(self.inputs)
            ),
            baselines=OrderedDict(
                (self.feature_keys[i], tensor)
                for i, tensor in enumerate(self.baselines)
            )
            if self.baselines is not None
            else None,
            feature_mask=OrderedDict(
                (self.feature_keys[i], tensor)
                for i, tensor in enumerate(self.feature_mask)
            )
            if self.feature_mask is not None
            else None,
            additional_forward_args=self.additional_forward_args,
            target=self.target,
            frozen_features=self.frozen_features,
        )


class ExplanationState(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        frozen=True,
        extra="forbid",
        revalidate_instances="always",
    )
    explanation_inputs: ExplanationInputs
    model_outputs: torch.Tensor | None = None

    # explanations
    explanations: OrderedDict[str, torch.Tensor]
    reduced_explanations: OrderedDict[str, torch.Tensor] | None = None

    @property
    def explanations_as_tuple(self) -> tuple[torch.Tensor, ...]:
        return tuple(self.explanations.values())

    # ------------------------------
    # Normalize explanations to OrderedDict
    # ------------------------------
    @field_validator("explanations", mode="before")
    @classmethod
    def normalize_explanations(cls, v):
        return _normalize_dictlike(v)

    @model_validator(mode="after")
    def validate_all(self):
        explained = self.explanation_inputs.inputs
        feature_keys = list(explained.keys())

        # ----- Validate explanations shape -----
        for name in ["explanations", "reduced_explanations"]:
            exp = getattr(self, name)
            if exp is not None:
                if not isinstance(exp, OrderedDict):
                    raise ValueError(f"{name} must be an OrderedDict internally.")
                exp_keys = list(exp.keys())
                if exp_keys != feature_keys:
                    raise ValueError(
                        f"{name} must have same number of feature types as inputs "
                        f"({len(exp_keys)} vs {len(feature_keys)})"
                    )

        # ----- Validate sample_id count -----
        if len(self.explanation_inputs.sample_id) != self.model_outputs.shape[0]:
            raise ValueError(
                f"sample_id count mismatch: got {len(self.explanation_inputs.sample_id)} "
                f"but model_outputs batch is {self.model_outputs.shape[0]}"
            )

        return self

    # ------------------------------
    # Move everything to device
    # ------------------------------
    def to(self, device: str | torch.device = "cpu") -> ExplanationState:
        return self.model_copy(
            update={
                "explainer_step_inputs": self.explanation_inputs.to(device),
                "model_outputs": self.model_outputs.to(device),
                "explanations": _to_device(self.explanations, device),
            }
        )


class MultiTargetExplanationState(ExplanationState):
    explanations: list[OrderedDict[str, torch.Tensor]]  # type: ignore[override]
    reduced_explanations: list[OrderedDict[str, torch.Tensor]] | None = None  # type: ignore[override]

    @property
    def explanations_as_tuple(self) -> list[tuple[torch.Tensor, ...]]:  # type: ignore[override]
        return [tuple(x.values()) for x in self.explanations]

    # ------------------------------
    # Normalize explanations to OrderedDict
    # ------------------------------
    @field_validator("explanations", mode="before")
    @classmethod
    def normalize_explanations(cls, v):  # type: ignore[override]
        return [_normalize_dictlike(x) for x in v]

    @model_validator(mode="after")
    def validate_all(self):
        explained = self.explanation_inputs.inputs
        num_feature_types = len(explained)

        # ----- Validate explanations shape -----
        for name in ["explanations", "reduced_explanations"]:
            exp_list = getattr(self, name)
            if exp_list is not None:
                if not isinstance(exp_list, list) and not all(
                    isinstance(e, OrderedDict) for e in exp_list
                ):
                    raise ValueError(
                        f"{name} must be a list of OrderedDict internally."
                    )

                for exp in exp_list:
                    if len(exp) != num_feature_types:
                        raise ValueError(
                            f"{name} must have same number of feature types as inputs "
                            f"({len(exp)} vs {num_feature_types})"
                        )

        # ----- Validate sample_id count -----
        if len(self.explanation_inputs.sample_id) != self.model_outputs.shape[0]:
            raise ValueError(
                f"sample_id count mismatch: got {len(self.explanation_inputs.sample_id)} "
                f"but model_outputs batch is {self.model_outputs.shape[0]}"
            )

        return self


class MetricInputs(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        frozen=True,
        extra="forbid",
        revalidate_instances="always",
    )
    baselines: OrderedDict[str, torch.Tensor] | Any | None = None
    shift_baselines: OrderedDict[str, torch.Tensor] | Any | None = None
    feature_mask: OrderedDict[str, torch.Tensor] | Any | None = None
    constant_shifts: OrderedDict[str, torch.Tensor] | Any | None = None
    input_layer_names: list[str] | None = None

    # -------------------------
    # Validators (normalize all)
    # -------------------------

    @field_validator(
        "baselines", "shift_baselines", "feature_mask", "constant_shifts", mode="before"
    )
    @classmethod
    def normalize_inputs(cls, v):
        return _normalize_dictlike(v)

    # -------------------------
    # Device movement
    # -------------------------

    def to(self, device: str | torch.device = "cpu") -> MetricInputs:
        updates = self.model_copy(
            update={
                "baselines": _to_device(self.baselines, device),
                "shift_baselines": _to_device(self.shift_baselines, device),
                "feature_mask": _to_device(self.feature_mask, device),
                "constant_shifts": _to_device(self.constant_shifts, device),
            }
        )
        return updates


class ExplanationStepOutputs(BaseModel):
    explanation_state: ExplanationState
    metric_inputs: MetricInputs | None = None

    def to(self, device: str | torch.device = "cpu") -> ExplanationStepOutputs:
        return self.model_copy(
            update={
                "explanation_state": self.explanation_state.to(device),
                "metric_inputs": self.metric_inputs.to(device)
                if self.metric_inputs is not None
                else None,
            }
        )

    @property
    def batch_size(self) -> int:
        return self.inputs[0].shape[0]

    @property
    def inputs(self) -> tuple[torch.Tensor, ...]:
        return _tensor_or_tensor_dict_as_detached_tuple(
            self.explanation_state.explanation_inputs.inputs
        )

    @property
    def additional_forward_args(self) -> tuple[Any, ...] | None:
        return _tensor_or_tensor_dict_as_detached_tuple(
            self.explanation_state.explanation_inputs.additional_forward_args
        )

    @property
    def feature_mask(self) -> tuple[torch.Tensor, ...] | None:
        return _tensor_or_tensor_dict_as_detached_tuple(
            self.explanation_state.explanation_inputs.feature_mask
        )

    @property
    def attributions(self) -> tuple[torch.Tensor, ...]:
        return _tensor_or_tensor_dict_as_detached_tuple(
            self.explanation_state.explanations
        )

    @property
    def explainer_baselines(self) -> tuple[torch.Tensor, ...] | None:
        return _tensor_or_tensor_dict_as_detached_tuple(
            self.explanation_state.explanation_inputs.baselines
        )

    @property
    def metric_baselines(self) -> tuple[torch.Tensor, ...] | None:
        if self.metric_inputs is None:
            return None
        return _tensor_or_tensor_dict_as_detached_tuple(self.metric_inputs.baselines)

    @property
    def metric_shift_baselines(self) -> tuple[torch.Tensor, ...] | None:
        if self.metric_inputs is None:
            return None
        return _tensor_or_tensor_dict_as_detached_tuple(
            self.metric_inputs.shift_baselines
        )

    @property
    def constant_shifts(self) -> tuple[torch.Tensor, ...] | None:
        if self.metric_inputs is None:
            return None
        return _tensor_or_tensor_dict_as_detached_tuple(
            self.metric_inputs.constant_shifts
        )

    @property
    def input_layer_names(self) -> list[str] | None:
        if self.metric_inputs is None:
            return None
        return self.metric_inputs.input_layer_names

    @property
    def target(self) -> list[torch.Tensor] | torch.Tensor | None:
        return self.explanation_state.explanation_inputs.target


class MultiTargetExplanationStepOutputs(ExplanationStepOutputs):
    explanation_state: MultiTargetExplanationState  # type: ignore[override]
    metric_inputs: MetricInputs | None = None

    @property
    def attributions(self) -> list[tuple[torch.Tensor, ...]]:  # type: ignore[override]
        # flatten list of OrderedDicts into tuple of tensors
        attributions_list = []
        for explanation_dict in self.explanation_state.explanations:
            attributions_list.append(
                _tensor_or_tensor_dict_as_detached_tuple(explanation_dict)
            )
        return attributions_list

    @property
    def target(self) -> list[torch.Tensor] | None:  # type: ignore[override]
        targets = self.explanation_state.explanation_inputs.target
        if targets is None:
            return None
        assert isinstance(targets, list), "targets must be a list of targets"
        return targets
