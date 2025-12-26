from __future__ import annotations

from collections.abc import Callable
from typing import Any, Self

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator

from tests.utils.types import ExplanationInputs, MetricInputs


class BaseTestConfig(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
        extra="forbid",
        revalidate_instances="always",
    )
    explanation_inputs: ExplanationInputs
    model: torch.nn.Module
    metric_inputs: MetricInputs = MetricInputs()
    multiply_by_inputs: bool = False
    n_features: int

    def to(self, device: str | torch.device = "cpu") -> Self:
        return self.model_copy(
            update={
                "model": self.model.to(device),
                "explanation_inputs": self.explanation_inputs.to(device),
                "metric_inputs": self.metric_inputs.to(device),
            }
        )


class RuntimeTestConfig(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
        extra="forbid",
        revalidate_instances="always",
    )
    test_name: str | None = None
    explainer: str = "integrated_gradients"
    target_fixture: str | None = None
    explainer_kwargs: dict | None = Field(default_factory=dict)
    expected: Any = None
    delta: float = 1e-4
    mode: str = "sum"
    override_target: Any = None
    throws_exception: bool = False
    device: str = Field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )
    multi_target: bool = False
    set_image_feature_mask: bool = False
    image_feature_mask_cell_size: int = 32
    model_type: str = "linear"
    train_and_eval_model: bool = False
    set_baselines_to_type: str | None = None

    n_perturbations_per_feature: int | list[int | None] | None = None
    max_features_processed_per_batch: int | list[int | None] | None = None
    perturb_func: Callable | None = None
    percentage_feature_removal_per_step: float = 0.0
    zero_variance_threshold: float = 1e-5
    return_ratio: bool = False

    @model_validator(mode="before")
    @classmethod
    def validate_and_set_defaults(cls, values) -> dict:
        target_fixture = values.get("target_fixture", None)
        explainer = values.get("explainer", None)
        assert target_fixture is not None or explainer is not None, (
            "Either target_fixture or explainer name must be provided"
        )

        test_name = f"fixture={target_fixture}_explainer={explainer}"

        if "test_name" in values:
            values["test_name"] = test_name + "_" + values["test_name"]
        else:
            values["test_name"] = test_name

        if "set_baselines_to_type" in values:
            assert values["set_baselines_to_type"] in ["zero", "black", None], (
                "set_baselines_to_type must be one of 'zero', 'black', or None"
            )

        return values


class ExplainersTestRuntimeConfig(RuntimeTestConfig):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
        extra="forbid",
        revalidate_instances="always",
    )
    grad_batch_size: int = 64
    visualize: bool = False
    check_multi_target_list_against_single_target: bool = True
