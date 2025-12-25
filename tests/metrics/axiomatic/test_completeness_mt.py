from dataclasses import field

import pytest
import torch  # noqa

from tests.utils.common import _assert_tensor_almost_equal, _run_metric_via_ignite
from tests.utils.configs import BaseTestConfig, RuntimeTestConfig
from torchxai.data_types import (
    MultiTargetExplanationStepOutputs,
    SingleTargetAcrossBatch,
)
from torchxai.ignite._axiomatic import CompletenessMetric
from torchxai.metrics.axiomatic.completeness import completeness


class MetricTestRuntimeConfig(RuntimeTestConfig):
    test_name: str | None = "compare_multi_target_to_single_target"
    explainer: str = "saliency"
    override_target: list[int] = field(
        default_factory=lambda: [
            SingleTargetAcrossBatch(index=0),
            SingleTargetAcrossBatch(index=1),
            SingleTargetAcrossBatch(index=2),
        ]
    )
    expected: torch.Tensor | None = None
    explainer_kwargs: dict | None = field(
        default_factory=lambda: {"multi_target": True}
    )
    delta: float = 1e-8
    multi_target: bool = True


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
    base_config: BaseTestConfig
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

    multi_target_completeness = completeness(
        forward_func=base_config.model,
        inputs=explanation_step_outputs.inputs,
        attributions=explanation_step_outputs.attributions,
        baselines=explanation_step_outputs.metric_baselines,
        additional_forward_args=explanation_step_outputs.additional_forward_args,
        target=target,
        multi_target=True,
    )

    assert len(per_target_completeness) == len(multi_target_completeness)
    for output, expected in zip(
        multi_target_completeness, per_target_completeness, strict=True
    ):
        _assert_tensor_almost_equal(output, expected, delta=runtime_config.delta)

    # test via ignite metric interface
    ignite_metric = CompletenessMetric(
        model=base_config.model, device=runtime_config.device
    )
    multi_target_completeness_ignite = _run_metric_via_ignite(
        metric=ignite_metric, explanation_step_outputs=explanation_step_outputs
    )["score"]
    for output, expected in zip(
        multi_target_completeness, multi_target_completeness_ignite, strict=True
    ):
        _assert_tensor_almost_equal(output, expected, delta=runtime_config.delta)
