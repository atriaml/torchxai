from __future__ import annotations

from collections import OrderedDict
from typing import Any, Self

import torch
from pydantic import BaseModel, ConfigDict, field_validator

from torchxai.data_types import ExplanationTarget, ExplanationTargetType, NoTarget


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


class ExplanationInputs(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        frozen=True,
        extra="forbid",
        revalidate_instances="always",
    )
    sample_id: list[str] | None = None
    inputs: tuple[torch.Tensor, ...]
    additional_forward_args: tuple[Any, ...] | None = None
    baselines: tuple[torch.Tensor, ...] | None = None
    feature_mask: tuple[torch.Tensor, ...] | None = None
    target: ExplanationTargetType | list[ExplanationTargetType] = NoTarget()
    frozen_features: list[torch.Tensor] | None = None
    strides: Any | None = None
    sliding_window_shapes: Any | None = None

    @property
    def model_inputs(self) -> tuple[Any, ...]:
        return self.inputs + (
            self.additional_forward_args
            if self.additional_forward_args is not None
            else ()
        )

    @field_validator("inputs", "baselines", "feature_mask", mode="plain")
    @classmethod
    def validate_to_tuple(cls, v):
        if isinstance(v, torch.Tensor):
            return (v,)
        if v is None:
            return None
        assert isinstance(v, tuple), "Expected inputs to be a tuple of tensors."
        return v

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


class MetricInputs(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        frozen=True,
        extra="forbid",
        revalidate_instances="always",
    )
    baselines: tuple[torch.Tensor, ...] | Any | None = None
    shift_baselines: tuple[torch.Tensor, ...] | Any | None = None
    feature_mask: tuple[torch.Tensor, ...] | Any | None = None
    constant_shifts: tuple[torch.Tensor, ...] | Any | None = None
    input_layer_names: list[str] | None = None

    @field_validator(
        "baselines", "shift_baselines", "feature_mask", "constant_shifts", mode="before"
    )
    @classmethod
    def normalize_inputs(cls, v):
        if isinstance(v, torch.Tensor):
            return (v,)
        assert isinstance(v, tuple) or v is None, (
            "Expected baselines, shift_baselines, feature_mask, constant_shifts to be a tuple of tensors or None."
        )
        return v

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
    model_config = ConfigDict(arbitrary_types_allowed=True)

    explanation_inputs: ExplanationInputs
    model_outputs: torch.Tensor
    explanations: tuple[torch.Tensor, ...] | list[tuple[torch.Tensor, ...]]

    def to(self, device: str | torch.device = "cpu") -> ExplanationStepOutputs:
        return self.model_copy(
            update={
                "explanation_inputs": self.explanation_inputs.to(device),
                "model_outputs": _to_device(self.model_outputs, device),
                "explanations": _to_device(self.explanations, device),
            }
        )
