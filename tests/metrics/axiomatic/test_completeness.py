import pytest  # noqa
import torch

from tests.utils.common import _assert_tensor_almost_equal, _run_metric_via_ignite
from tests.utils.configs import RuntimeTestConfig
from torchxai.ignite._axiomatic import CompletenessMetric
from torchxai.metrics.axiomatic.completeness import completeness

test_configurations = [
    RuntimeTestConfig(
        target_fixture="multi_modal_sequence_sum",
        explainer="integrated_gradients",
        expected=torch.tensor(
            [0.0]
        ),  # integrated gradients completeness should be 0 for this case
    ),
    RuntimeTestConfig(
        target_fixture="multi_modal_sequence_sum",
        explainer="saliency",
        expected=torch.tensor([116]),  # saliency completeness is not so great
    ),
    # the park function is taken from the paper: https://arxiv.org/pdf/2007.07584
    RuntimeTestConfig(
        target_fixture="park_function_configuration",
        explainer="saliency",
        expected=torch.tensor([1.6322]),  # saliency completeness is not so great
    ),
    RuntimeTestConfig(
        target_fixture="park_function_configuration",
        explainer="input_x_gradient",
        expected=torch.tensor(
            [0.1865]
        ),  # input_x_gradient results in better completeness
    ),
    RuntimeTestConfig(
        target_fixture="park_function_configuration",
        explainer="integrated_gradients",
        expected=torch.tensor(
            [1.3856e-08]
        ),  # integrated_gradients results in full completeness
    ),
    RuntimeTestConfig(
        target_fixture="basic_model_single_input_config",
        explainer="integrated_gradients",
        expected=torch.zeros(1),
    ),
    RuntimeTestConfig(
        target_fixture="basic_model_batch_input_config",
        explainer="integrated_gradients",
        expected=torch.zeros(3),
    ),
    RuntimeTestConfig(
        target_fixture="basic_model_batch_input_with_additional_forward_args_config",
        explainer="integrated_gradients",
        expected=torch.zeros(1),
    ),
    RuntimeTestConfig(
        target_fixture="classification_convnet_model_with_multiple_targets_config",
        explainer="deep_lift",
        expected=torch.zeros(20),
    ),
    RuntimeTestConfig(
        target_fixture="classification_convnet_model_with_multiple_targets_config",
        explainer="integrated_gradients",
        expected=torch.tensor([1.7565] * 20),
        delta=1e-3,
    ),
    RuntimeTestConfig(
        target_fixture="classification_multilayer_model_with_tuple_targets_config",
        explainer="integrated_gradients",
        expected=torch.tensor([0.6538, 0.0, 0.0, 0.0]),
    ),
    RuntimeTestConfig(
        target_fixture="classification_multilayer_model_with_baseline_and_tuple_targets_config",
        explainer="integrated_gradients",
        expected=torch.tensor([0.3269, 0.0, 0.0, 0.0]),
    ),
]


@pytest.mark.metrics
@pytest.mark.parametrize(
    "metrics_runtime_test_configuration",
    test_configurations,
    ids=[f"{idx}_{config.test_name}" for idx, config in enumerate(test_configurations)],
    indirect=True,
)
def test_completeness(metrics_runtime_test_configuration):
    base_config, runtime_config, explanation_step_outputs = (
        metrics_runtime_test_configuration
    )

    # test direct function call
    metric_output = completeness(
        forward_func=base_config.model,
        inputs=explanation_step_outputs.inputs,
        attributions=explanation_step_outputs.attributions,
        baselines=explanation_step_outputs.metric_baselines,
        additional_forward_args=explanation_step_outputs.additional_forward_args,
        target=explanation_step_outputs.target,  # type: ignore
    )
    _assert_tensor_almost_equal(
        metric_output, runtime_config.expected, delta=runtime_config.delta
    )

    # test via ignite metric interface
    metric = CompletenessMetric(model=base_config.model, device=runtime_config.device)
    metric_output = _run_metric_via_ignite(
        metric=metric, explanation_step_outputs=explanation_step_outputs
    )["score"]
    _assert_tensor_almost_equal(
        metric_output, runtime_config.expected, delta=runtime_config.delta
    )
