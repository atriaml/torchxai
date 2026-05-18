import inspect
import itertools

import pytest  # noqa
import torch

from tests.fixtures._metric import _get_metric_inputs
from tests.utils.common import (
    _assert_all_tensors_almost_equal,
    _assert_tensor_almost_equal,
    _set_all_random_seeds,
)
from tests.utils.configs import RuntimeTestConfig
from torchxai.metrics import faithfulness_estimate


def _format_to_list(value):
    if not isinstance(value, list):
        return [value]
    return value


class MetricTestRuntimeConfig_(RuntimeTestConfig):
    max_features_processed_per_batch: int | list[int | None] | None = None


test_configurations = [
    MetricTestRuntimeConfig_(
        test_name="basic_model_single_input_config",
        target_fixture="basic_model_single_input_config",
        expected=torch.ones(1).unsqueeze(0),
        max_features_processed_per_batch=[5, 1, 40],
    ),
    MetricTestRuntimeConfig_(
        test_name="basic_model_batch_input_config",
        target_fixture="basic_model_batch_input_config",
        expected=torch.ones(3).unsqueeze(0),
        max_features_processed_per_batch=[5, 1, 40],
    ),
    MetricTestRuntimeConfig_(
        test_name="basic_model_batch_input_with_additional_forward_args_config",
        target_fixture="basic_model_batch_input_with_additional_forward_args_config",
        expected=torch.tensor([torch.nan]).unsqueeze(0),
        max_features_processed_per_batch=[5, 1, 40],
    ),
    MetricTestRuntimeConfig_(
        test_name="classification_convnet_model_with_multiple_targets_config",
        target_fixture="classification_convnet_model_with_multiple_targets_config",
        explainer="deep_lift",
        expected=[
            torch.tensor([0.4150] * 20),
            torch.tensor([0.4150] * 20),
            torch.tensor([0.4150] * 20),
        ],
        max_features_processed_per_batch=[5, 1, 40],
    ),
    MetricTestRuntimeConfig_(
        test_name="classification_multilayer_model_with_tuple_targets_config",
        target_fixture="classification_multilayer_model_with_tuple_targets_config",
        expected=torch.tensor([0.9966, 1.0000, 1.0000, 1.0000]).unsqueeze(0),
        max_features_processed_per_batch=[5, 1, 40],
    ),
    MetricTestRuntimeConfig_(
        test_name="classification_multilayer_model_with_baseline_and_tuple_targets_config",
        target_fixture="classification_multilayer_model_with_baseline_and_tuple_targets_config",
        expected=torch.tensor([1.0000, 1.0000, 1.0000, 1.0000]).unsqueeze(0),
        max_features_processed_per_batch=[5, 1, 40],
    ),
]


@pytest.mark.metrics
@pytest.mark.parametrize(
    "metrics_runtime_test_configuration",
    test_configurations,
    ids=[f"{idx}_{config.test_name}" for idx, config in enumerate(test_configurations)],
    indirect=True,
)
def test_faithfulness_estimate(metrics_runtime_test_configuration):
    base_config, runtime_config, explanation_step_outputs = (
        metrics_runtime_test_configuration
    )
    max_features_processed_per_batch = _format_to_list(
        runtime_config.max_features_processed_per_batch
    )
    expected = _format_to_list(runtime_config.expected)

    assert len(max_features_processed_per_batch) == len(expected) or len(expected) == 1

    attributions_sum_perturbed_list = []
    inputs_perturbed_fwd_diffs_list = []
    for max_features, curr_expected in zip(
        runtime_config.max_features_processed_per_batch,
        itertools.cycle(runtime_config.expected),
    ):
        _set_all_random_seeds(1234)
        kwargs = _get_metric_inputs(
            base_config, runtime_config, explanation_step_outputs
        )
        kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in inspect.signature(faithfulness_estimate).parameters
        }
        (
            faithfulness_estimate_score,
            attributions_sum_perturbed,
            inputs_perturbed_fwd_diffs,
        ) = faithfulness_estimate(
            **kwargs,
            max_features_processed_per_batch=max_features,
            return_intermediate_results=True,
        )
        _assert_tensor_almost_equal(
            faithfulness_estimate_score,
            curr_expected,
            delta=runtime_config.delta,
            mode="mean",
        )
        attributions_sum_perturbed_list.append(torch.cat(attributions_sum_perturbed))
        inputs_perturbed_fwd_diffs_list.append(torch.cat(inputs_perturbed_fwd_diffs))
    _assert_all_tensors_almost_equal(attributions_sum_perturbed_list)
    _assert_all_tensors_almost_equal(inputs_perturbed_fwd_diffs_list)
