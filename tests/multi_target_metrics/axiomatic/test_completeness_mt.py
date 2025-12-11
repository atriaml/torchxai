from dataclasses import field

import pytest
import torch  # noqa

from tests.utils.common import _assert_tensor_almost_equal
from tests.utils.configs import TestBaseConfig, TestRuntimeConfig
from torchxai.data_types import MultiTargetExplanationStepOutputs
from torchxai.metrics.axiomatic.completeness import completeness


class MetricTestRuntimeConfig(TestRuntimeConfig):
    test_name: str | None = "compare_multi_target_to_single_target"
    explainer: str = "saliency"
    override_target: list[int] = field(default_factory=lambda: [0, 1, 2])
    expected: torch.Tensor | None = None
    explainer_kwargs: dict | None = field(
        default_factory=lambda: {"is_multi_target": True}
    )
    delta: float = 1e-8
    is_multi_target: bool = True


test_configurations = [
    MetricTestRuntimeConfig(target_fixture="classification_alexnet_model_config"),
    MetricTestRuntimeConfig(
        target_fixture="classification_alexnet_model_single_sample_config"
    ),
    MetricTestRuntimeConfig(
        target_fixture="classification_alexnet_model_real_images_config",
        explainer="integrated_gradients",
    ),
    MetricTestRuntimeConfig(
        target_fixture="classification_alexnet_model_real_images_single_sample_config",
        explainer="integrated_gradients",
    ),
]


@pytest.mark.metrics
@pytest.mark.parametrize(
    "metrics_runtime_test_configuration",
    test_configurations,
    ids=[f"{idx}_{config.test_name}" for idx, config in enumerate(test_configurations)],
    indirect=True,
)
def test_completeness_multi_target(metrics_runtime_test_configuration):
    base_config, runtime_config, explanation_step_outputs = (
        metrics_runtime_test_configuration
    )
    base_config: TestBaseConfig
    runtime_config: MetricTestRuntimeConfig
    explanation_step_outputs: MultiTargetExplanationStepOutputs
    target = base_config.explanation_inputs.target

    per_target_completeness = []
    for explanation, t in zip(
        explanation_step_outputs.attributions, target, strict=True
    ):
        output = completeness(
            forward_func=base_config.model,
            inputs=explanation_step_outputs.inputs,
            attributions=explanation,
            baselines=explanation_step_outputs.metric_baselines,
            additional_forward_args=explanation_step_outputs.additional_forward_args,
            target=t,
        )
        per_target_completeness.append(output)

    multi_target_completeness_output = completeness(
        forward_func=base_config.model,
        inputs=explanation_step_outputs.inputs,
        attributions=explanation_step_outputs.attributions,
        baselines=explanation_step_outputs.metric_baselines,
        additional_forward_args=explanation_step_outputs.additional_forward_args,
        target=target,
        is_multi_target=True,
    )

    assert len(per_target_completeness) == len(multi_target_completeness_output)
    for output, expected in zip(
        multi_target_completeness_output, per_target_completeness, strict=True
    ):
        _assert_tensor_almost_equal(output, expected, delta=runtime_config.delta)
