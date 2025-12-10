from __future__ import annotations

from typing import Any, Self

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator

from torchxai.data_types import ExplanationInputs, MetricInputs


class TestBaseConfig(BaseModel):
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


class TestRuntimeConfig(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
        extra="forbid",
        revalidate_instances="always",
    )
    test_name: str
    explainer: str
    target_fixture: str | None = None
    explainer_kwargs: dict | None = Field(default_factory=dict)
    use_captum_explainer: bool = False
    expected: Any = None
    delta: float = 1e-4
    override_target: Any = None
    throws_exception: bool = False
    device: str = Field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )

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

        return values


class ExplainersTestRuntimeConfig(TestRuntimeConfig):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
        extra="forbid",
        revalidate_instances="always",
    )
    is_multi_target: bool = False
    grad_batch_size: int = 64
    visualize: bool = False
    check_multi_target_list_against_single_target: bool = True
