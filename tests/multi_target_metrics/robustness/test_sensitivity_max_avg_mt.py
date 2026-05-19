import pytest
import torch  # noqa

from tests.fixtures._metric import _run_metric_test_simple_mt
from tests.utils.common import _assert_tensor_almost_equal
from tests.utils.configs import RuntimeTestConfig
from torchxai.data_types._target import SingleTargetAcrossBatch
from torchxai.metrics import sensitivity_max_and_avg


class MetricTestRuntimeConfig(RuntimeTestConfig):
    explainer: str = "saliency"
    delta: float = 1e-5
    multi_target: bool = True


test_configurations = [
    MetricTestRuntimeConfig(
        target_fixture="classification_alexnet_model_config",
        override_target=[
            SingleTargetAcrossBatch(index=0),
            SingleTargetAcrossBatch(index=1),
            SingleTargetAcrossBatch(index=2),
        ],
    ),
    MetricTestRuntimeConfig(
        target_fixture="classification_alexnet_model_single_sample_config",
        override_target=[
            SingleTargetAcrossBatch(index=0),
            SingleTargetAcrossBatch(index=1),
            SingleTargetAcrossBatch(index=2),
        ],
    ),
    MetricTestRuntimeConfig(
        target_fixture="classification_alexnet_model_real_images_config",
        explainer="integrated_gradients",
        override_target=[
            SingleTargetAcrossBatch(index=0),
            SingleTargetAcrossBatch(index=1),
            SingleTargetAcrossBatch(index=2),
        ],
    ),
    MetricTestRuntimeConfig(
        target_fixture="classification_alexnet_model_real_images_single_sample_config",
        explainer="integrated_gradients",
        override_target=[
            SingleTargetAcrossBatch(index=0),
            SingleTargetAcrossBatch(index=1),
            SingleTargetAcrossBatch(index=2),
        ],
    ),
]


@pytest.mark.metrics
@pytest.mark.parametrize(
    "explainer_based_metrics_runtime_test_configuration",
    test_configurations,
    ids=[f"{idx}_{config.test_name}" for idx, config in enumerate(test_configurations)],
    indirect=True,
)
def test_sensitivity_max_avg_mt(explainer_based_metrics_runtime_test_configuration):
    base_config, runtime_config, explainer = (
        explainer_based_metrics_runtime_test_configuration
    )

    def comparison_func(output: list, expected: list):
        for to, te in zip(output, expected, strict=True):
            _assert_tensor_almost_equal(to, te, delta=runtime_config.delta, mode="mean")

    _run_metric_test_simple_mt(
        base_config=base_config,
        runtime_config=runtime_config,
        metric_func=sensitivity_max_and_avg,
        comparison_func=comparison_func,
        explainer=explainer,
    )
