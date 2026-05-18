import pytest
import torch  # noqa

from tests.fixtures._metric import _run_metric_test_simple_mt
from tests.utils.common import _assert_tensor_almost_equal
from tests.utils.configs import RuntimeTestConfig
from torchxai.data_types._target import SingleTargetAcrossBatch
from torchxai.metrics import complexity_entropy
from torchxai.metrics.complexity.complexity_entropy import (
    complexity_entropy_feature_grouped,
)


def setup_test_config_for_explainer(**kwargs):
    return RuntimeTestConfig(
        test_name="compare_multi_target_to_single_target",
        explainer=kwargs.pop("explainer", "saliency"),
        override_target=[
            SingleTargetAcrossBatch(index=0),
            SingleTargetAcrossBatch(index=1),
            SingleTargetAcrossBatch(index=2),
        ],
        expected=None,
        delta=1e-8,
        multi_target=True,
        **kwargs,
    )


test_configurations = [
    setup_test_config_for_explainer(
        target_fixture="classification_alexnet_model_config"
    ),
    setup_test_config_for_explainer(
        target_fixture="classification_alexnet_model_single_sample_config"
    ),
    setup_test_config_for_explainer(
        target_fixture="classification_alexnet_model_real_images_config",
        explainer="integrated_gradients",
    ),
    setup_test_config_for_explainer(
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
def test_complexity_entropy_mt(metrics_runtime_test_configuration):
    base_config, runtime_config, explanation_step_outputs = (
        metrics_runtime_test_configuration
    )

    def comparison_func(output: tuple, expected: tuple):
        for x, y in zip(output, expected, strict=True):
            _assert_tensor_almost_equal(x, y, delta=runtime_config.delta)

    _run_metric_test_simple_mt(
        base_config=base_config,
        runtime_config=runtime_config,
        explanation_step_outputs=explanation_step_outputs,
        metric_func=complexity_entropy,
        comparison_func=comparison_func,
    )

    _run_metric_test_simple_mt(
        base_config=base_config,
        runtime_config=runtime_config,
        explanation_step_outputs=explanation_step_outputs,
        metric_func=complexity_entropy_feature_grouped,
        comparison_func=comparison_func,
    )
