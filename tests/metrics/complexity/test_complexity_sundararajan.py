import pytest  # noqa
import torch

from tests.fixtures._metric import _run_metric_test_simple
from tests.utils.common import _assert_tensor_almost_equal
from tests.utils.configs import RuntimeTestConfig
from torchxai.metrics import complexity_sundararajan
from torchxai.metrics.complexity.complexity_sundararajan import (
    complexity_sundararajan_feature_grouped,
)

test_configurations = [
    # the park function is taken from the paper: https://arxiv.org/pdf/2007.07584
    # and the complexity results match from the paper
    RuntimeTestConfig(
        test_name="park_function_configuration_saliency",
        target_fixture="park_function_configuration",
        explainer="saliency",
        expected=torch.tensor([0.6667]),
    ),
    RuntimeTestConfig(
        test_name="park_function_configuration_input_x_gradient",
        target_fixture="park_function_configuration",
        explainer="input_x_gradient",
        expected=torch.tensor([0.6667]),
    ),
    RuntimeTestConfig(
        test_name="park_function_configuration_integrated_gradients",
        target_fixture="park_function_configuration",
        explainer="integrated_gradients",
        expected=torch.tensor([0.6667]),
    ),
    RuntimeTestConfig(
        test_name="basic_model_single_input_integrated_gradients",
        target_fixture="basic_model_single_input_config",
        explainer="integrated_gradients",
        expected=torch.tensor([1]),
    ),
    RuntimeTestConfig(
        test_name="basic_model_batch_input_integrated_gradients",
        target_fixture="basic_model_batch_input_config",
        explainer="integrated_gradients",
        expected=torch.tensor([1] * 3),
    ),
    RuntimeTestConfig(
        test_name="basic_model_batch_input_with_additional_forward_args_integrated_gradients",
        target_fixture="basic_model_batch_input_with_additional_forward_args_config",
        explainer="integrated_gradients",
        expected=torch.tensor([0]),
    ),
    RuntimeTestConfig(
        test_name="classification_convnet_model_with_multiple_targets_deep_lift",
        target_fixture="classification_convnet_model_with_multiple_targets_config",
        explainer="integrated_gradients",
        expected=torch.tensor([1] * 20),
    ),
    RuntimeTestConfig(
        test_name="classification_multilayer_model_with_tuple_targets_integrated_gradients",
        target_fixture="classification_multilayer_model_with_tuple_targets_config",
        explainer="integrated_gradients",
        expected=torch.tensor([1] * 4),
    ),
    RuntimeTestConfig(
        test_name="classification_multilayer_model_with_baseline_and_tuple_targets_integrated_gradients",
        target_fixture="classification_multilayer_model_with_baseline_and_tuple_targets_config",
        explainer="integrated_gradients",
        expected=torch.tensor([0.6667, 1, 1, 1]),
    ),
]


@pytest.mark.metrics
@pytest.mark.parametrize(
    "metrics_runtime_test_configuration",
    test_configurations,
    ids=[f"{idx}_{config.test_name}" for idx, config in enumerate(test_configurations)],
    indirect=True,
)
def test_complexity_sundararajan(metrics_runtime_test_configuration):
    base_config, runtime_config, explanation_step_outputs = (
        metrics_runtime_test_configuration
    )

    def comparison_func(output: torch.Tensor, expected: torch.Tensor):
        _assert_tensor_almost_equal(
            output, expected, delta=runtime_config.delta, mode="mean"
        )

    _run_metric_test_simple(
        base_config=base_config,
        runtime_config=runtime_config,
        explanation_step_outputs=explanation_step_outputs,
        metric_func=complexity_sundararajan,
        comparison_func=comparison_func,
    )

    _run_metric_test_simple(
        base_config=base_config,
        runtime_config=runtime_config,
        explanation_step_outputs=explanation_step_outputs,
        metric_func=complexity_sundararajan_feature_grouped,
        comparison_func=comparison_func,
    )
