from __future__ import annotations

from collections import OrderedDict
from typing import Any, Self

import torch
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from torchxai.ignite._utilities import _as_detached_tuple


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
    )
    sample_id: list[str]
    explained_features: OrderedDict[str, torch.Tensor] | Any
    additional_forward_args: tuple[Any, ...] | None = None
    baselines: OrderedDict[str, torch.Tensor] | Any | None = None
    train_baselines: OrderedDict[str, torch.Tensor] | Any | None = None
    feature_masks: OrderedDict[str, torch.Tensor] | Any | None = None
    target: list[torch.Tensor] | torch.Tensor | None = None

    @field_validator(
        "baselines",
        "train_baselines",
        "explained_features",
        "feature_masks",
        mode="before",
    )
    @classmethod
    def normalize_baselines(cls, v):
        return _normalize_dictlike(v)

    @model_validator(mode="after")
    def validate_explained_features(self):
        assert isinstance(self.explained_features, OrderedDict)
        _match_keys(self.baselines, self.explained_features)
        _match_keys(self.train_baselines, self.explained_features)
        _match_keys(self.feature_masks, self.explained_features)
        if not isinstance(self.explained_features, dict):
            raise ValueError("explained_features must be an OrderedDict internally.")
        return self

    # ------------------------------
    # Move everything to device
    # ------------------------------
    def to(self, device: str | torch.device = "cpu") -> Self:
        return self.model_copy(
            update={
                "baselines": _to_device(self.baselines, device),
                "train_baselines": _to_device(self.train_baselines, device),
                "feature_masks": _to_device(self.feature_masks, device),
                "explained_features": _to_device(self.explained_features, device),
                "additional_forward_args": _to_device(
                    self.additional_forward_args, device
                ),
            }
        )

    @property
    def model_inputs(self) -> tuple[torch.Tensor, ...]:
        assert isinstance(self.explained_features, OrderedDict)
        return tuple(self.explained_features.values()) + (
            self.additional_forward_args
            if self.additional_forward_args is not None
            else ()
        )

    def to_explainer_kwargs(self) -> dict[str, Any]:
        return {
            "inputs": _as_detached_tuple(self.explained_features),
            "baselines": _as_detached_tuple(self.baselines),
            "train_baselines": _as_detached_tuple(self.train_baselines),
            "feature_mask": _as_detached_tuple(self.feature_masks),
            "additional_forward_args": _as_detached_tuple(self.additional_forward_args),
            "target": self.target,
        }


class MetricInputs(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        frozen=True,
        extra="forbid",
    )
    baselines: OrderedDict[str, torch.Tensor] | Any | None = None
    shift_baselines: OrderedDict[str, torch.Tensor] | Any | None = None
    feature_masks: OrderedDict[str, torch.Tensor] | Any | None = None
    constant_shifts: OrderedDict[str, torch.Tensor] | Any | None = None
    input_layer_names: list[str] | None = None
    frozen_features: list[torch.Tensor] | None = None

    # -------------------------
    # Validators (normalize all)
    # -------------------------

    @field_validator("baselines", mode="before")
    @classmethod
    def normalize_baselines(cls, v):
        return _normalize_dictlike(v)

    @field_validator("feature_masks", "constant_shifts", mode="before")
    @classmethod
    def normalize_dictlike(cls, v):
        return _normalize_dictlike(v)

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

    # -------------------------
    # Device movement
    # -------------------------

    def to(self, device: str | torch.device = "cpu") -> MetricInputs:
        return self.model_copy(
            update={
                "baselines": _to_device(self.baselines, device),
                "shift_baselines": _to_device(self.shift_baselines, device),
                "feature_masks": _to_device(self.feature_masks, device),
                "constant_shifts": _to_device(self.constant_shifts, device),
                "frozen_features": _to_device(self.frozen_features, device),
            }
        )


class ExplanationState(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        frozen=True,
        extra="forbid",
    )
    explanation_inputs: ExplanationInputs
    metric_inputs: MetricInputs | None = None
    model_outputs: torch.Tensor

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

    # -------------------------
    # Cross-field validation
    # -------------------------

    @model_validator(mode="after")
    def validate_input_layer_names(self):
        if (
            self.metric_inputs is not None
            and self.metric_inputs.input_layer_names is not None
        ):
            if len(self.metric_inputs.input_layer_names) != len(
                self.explanation_inputs.explained_features
            ):
                raise ValueError(
                    "input_layer_names must match number of explained_features"
                )
        return self

    @model_validator(mode="after")
    def validate_all(self):
        explained = self.explanation_inputs.explained_features
        num_feature_types = len(explained)

        # ----- Validate explanations shape -----
        for name in ["explanations", "reduced_explanations"]:
            exp = getattr(self, name)
            if exp is not None:
                if not isinstance(exp, dict):
                    raise ValueError(f"{name} must be an OrderedDict internally.")
                if len(exp) != num_feature_types:
                    raise ValueError(
                        f"{name} must have same number of feature types as explained_features "
                        f"({len(exp)} vs {num_feature_types})"
                    )

        # ----- Validate sample_id count -----
        if len(self.explanation_inputs.sample_id) != self.model_outputs.shape[0]:
            raise ValueError(
                f"sample_id count mismatch: got {len(self.explanation_inputs.sample_id)} "
                f"but model_outputs batch is {self.model_outputs.shape[0]}"
            )

        # ----- Validate frozen_features count -----
        if self.metric_inputs is not None:
            ff = self.metric_inputs.frozen_features
            if ff is not None and len(ff) != len(self.explanation_inputs.sample_id):
                raise ValueError(
                    f"frozen_features must match number of samples "
                    f"({len(ff)} vs {len(self.explanation_inputs.sample_id)})"
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
