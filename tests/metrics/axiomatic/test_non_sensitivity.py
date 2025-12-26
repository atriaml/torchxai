import math
from collections.abc import Callable
from typing import Any

import pytest  # noqa
import torch

from tests.fixtures._metric import _run_metric_test_looped
from tests.utils.common import _assert_tensor_almost_equal
from tests.utils.configs import RuntimeTestConfig
from torchxai.metrics import monotonicity_corr_and_non_sens
from torchxai.metrics._utils.perturbation import (
    default_fixed_baseline_perturb_func,
    default_random_perturb_func,
)


def _format_to_list(value):
    if not isinstance(value, list):
        return [value]
    return value


class MetricTestRuntimeConfig_(RuntimeTestConfig):
    perturb_func: Callable = default_random_perturb_func()
    n_perturbations_per_feature: int | list[int | None] = 100
    max_features_processed_per_batch: int | list[int | None] | None = None
    zero_attribution_threshold: float = 1e-5
    zero_variance_threshold: float = 1e-5
    use_percentage_attribution_threshold: bool = False
    percentage_feature_removal_per_step: float = 0.0


test_configurations = [
    MetricTestRuntimeConfig_(
        test_name="n_perturbations_per_feature_1",
        target_fixture="multi_modal_sequence_relu",
        explainer="integrated_gradients",
        expected=torch.tensor(
            # in this test set first 5 features have variance below 1e-4 and
            # and first first attributions have values below 0.01
            [0.0]
        ),  # integrated gradients completeness should be 0 for this case
        perturb_func=default_fixed_baseline_perturb_func(),
        n_perturbations_per_feature=1,
        use_percentage_attribution_threshold=False,
        zero_variance_threshold=1e-4,
        zero_attribution_threshold=0.01,
    ),
    MetricTestRuntimeConfig_(
        test_name="n_perturbations_per_feature_10",
        target_fixture="multi_modal_sequence_relu",
        explainer="integrated_gradients",
        expected=torch.tensor(
            # in this test set first 5 features have variance below 1e-4 and
            # and first first attributions have values below 0.01
            [0.0]
        ),  # integrated gradients completeness should be 0 for this case
        perturb_func=default_fixed_baseline_perturb_func(),
        n_perturbations_per_feature=10,
        use_percentage_attribution_threshold=False,
        zero_variance_threshold=1e-4,
        zero_attribution_threshold=0.01,
    ),
    MetricTestRuntimeConfig_(
        test_name="n_perturbations_per_feature_1_percentage_feature_removal_per_step_0.1",
        target_fixture="multi_modal_sequence_relu",
        explainer="integrated_gradients",
        expected=torch.tensor(
            # in this test set first 5 features have variance below 1e-4 and
            # and first first attributions have values below 0.01
            [0.0]
        ),  # integrated gradients completeness should be 0 for this case
        perturb_func=default_fixed_baseline_perturb_func(),
        n_perturbations_per_feature=1,
        use_percentage_attribution_threshold=False,
        zero_variance_threshold=1e-4,
        zero_attribution_threshold=0.01,
        percentage_feature_removal_per_step=0.1,
    ),
    MetricTestRuntimeConfig_(
        test_name="n_perturbations_per_feature_10_percentage_feature_removal_per_step_0.1",
        target_fixture="multi_modal_sequence_relu",
        explainer="integrated_gradients",
        expected=torch.tensor(
            # in this test set first 5 features have variance below 1e-4 and
            # and first first attributions have values below 0.01
            [0.0]
        ),  # integrated gradients completeness should be 0 for this case
        perturb_func=default_fixed_baseline_perturb_func(),
        n_perturbations_per_feature=10,
        use_percentage_attribution_threshold=False,
        zero_variance_threshold=1e-4,
        zero_attribution_threshold=0.01,
        percentage_feature_removal_per_step=0.1,
    ),
    MetricTestRuntimeConfig_(
        test_name="n_perturbations_per_feature_1",
        target_fixture="multi_modal_sequence_sum",
        explainer="integrated_gradients",
        expected=torch.tensor(
            # in this test set first 5 features have variance below 1e-4 and
            # and first first attributions have values below 0.01
            [0.0]
        ),  # integrated gradients completeness should be 0 for this case
        perturb_func=default_fixed_baseline_perturb_func(),
        n_perturbations_per_feature=1,
        use_percentage_attribution_threshold=False,
        zero_variance_threshold=1e-4,
        zero_attribution_threshold=0.01,
    ),
    MetricTestRuntimeConfig_(
        test_name="n_perturbations_per_feature_10",
        target_fixture="multi_modal_sequence_sum",
        explainer="saliency",
        # in this test set first 5 features have variance below 1e-4 and
        # and all attributions are the same as importance is assigned 1 to all features by Saliency
        expected=torch.tensor([5]),  # saliency completeness is not so great
        perturb_func=default_fixed_baseline_perturb_func(),
        n_perturbations_per_feature=10,
        use_percentage_attribution_threshold=False,
        zero_variance_threshold=1e-4,
        zero_attribution_threshold=0.01,
    ),
    MetricTestRuntimeConfig_(
        test_name="n_perturbations_per_feature_1_percentage_feature_removal_per_step_0.1",
        target_fixture="multi_modal_sequence_sum",
        explainer="integrated_gradients",
        expected=torch.tensor(
            # in this test set first 5 features have variance below 1e-4 and
            # and first first attributions have values below 0.01
            [0.0]
        ),  # integrated gradients completeness should be 0 for this case
        perturb_func=default_fixed_baseline_perturb_func(),
        n_perturbations_per_feature=1,
        use_percentage_attribution_threshold=False,
        zero_variance_threshold=1e-4,
        zero_attribution_threshold=0.01,
        percentage_feature_removal_per_step=0.1,
    ),
    MetricTestRuntimeConfig_(
        test_name="n_perturbations_per_feature_10_percentage_feature_removal_per_step_0.1",
        target_fixture="multi_modal_sequence_sum",
        explainer="integrated_gradients",
        expected=torch.tensor(
            # in this test set first 5 features have variance below 1e-4 and
            # and first first attributions have values below 0.01
            [0.0]
        ),  # integrated gradients completeness should be 0 for this case
        perturb_func=default_fixed_baseline_perturb_func(),
        n_perturbations_per_feature=10,
        use_percentage_attribution_threshold=False,
        zero_variance_threshold=1e-4,
        zero_attribution_threshold=0.01,
        percentage_feature_removal_per_step=0.1,
    ),
    # the park function is taken from the paper: https://arxiv.org/pdf/2007.07584
    MetricTestRuntimeConfig_(
        test_name="park_function_configuration_saliency",
        target_fixture="park_function_configuration",
        explainer="saliency",
        expected=torch.tensor([0]),
        perturb_func=default_random_perturb_func(noise_scale=1.0),
    ),
    MetricTestRuntimeConfig_(
        test_name="park_function_configuration_input_x_gradient",
        target_fixture="park_function_configuration",
        explainer="input_x_gradient",
        expected=torch.tensor([0]),
        perturb_func=default_random_perturb_func(noise_scale=1.0),
    ),
    MetricTestRuntimeConfig_(
        test_name="park_function_configuration_integrated_gradients",
        target_fixture="park_function_configuration",
        explainer="integrated_gradients",
        expected=torch.tensor([0]),
        perturb_func=default_random_perturb_func(noise_scale=1.0),
    ),
    MetricTestRuntimeConfig_(
        test_name="basic_model_single_input_config_integrated_gradients",
        target_fixture="basic_model_single_input_config",
        expected=torch.zeros(1),
        n_perturbations_per_feature=[10, 10, 20],
        max_features_processed_per_batch=[None, 1, 40],
    ),
    MetricTestRuntimeConfig_(
        test_name="basic_model_batch_input_config_integrated_gradients",
        target_fixture="basic_model_batch_input_config",
        expected=torch.zeros(3),
        n_perturbations_per_feature=[10, 10, 20],
        max_features_processed_per_batch=[None, 1, 40],
    ),
    MetricTestRuntimeConfig_(
        test_name="basic_model_batch_input_with_additional_forward_args_config_integrated_gradients",
        target_fixture="basic_model_batch_input_with_additional_forward_args_config",
        expected=torch.ones(1),
        n_perturbations_per_feature=[10, 10, 20],
        max_features_processed_per_batch=[None, 1, 40],
    ),
    MetricTestRuntimeConfig_(
        test_name="classification_convnet_model_with_multiple_targets_config_integrated_gradients",
        target_fixture="classification_convnet_model_with_multiple_targets_config",
        expected=torch.tensor([4] * 20),
        n_perturbations_per_feature=[10, 10, 20],
        max_features_processed_per_batch=[None, 1, 40],
    ),
    MetricTestRuntimeConfig_(
        test_name="classification_multilayer_model_with_tuple_targets_config_integrated_gradients",
        target_fixture="classification_multilayer_model_with_tuple_targets_config",
        expected=torch.tensor([0.0, 0.0, 0.0, 0.0]),
        n_perturbations_per_feature=[10, 10, 20],
        max_features_processed_per_batch=[None, 1, 40],
    ),
    MetricTestRuntimeConfig_(
        test_name="classification_multilayer_model_with_baseline_and_tuple_targets_config_integrated_gradients",
        target_fixture="classification_multilayer_model_with_baseline_and_tuple_targets_config",
        expected=torch.tensor([1, 0, 0, 0]),
        n_perturbations_per_feature=[10, 10, 20],
        max_features_processed_per_batch=[None, 1, 40],
    ),
]


@pytest.mark.metrics
@pytest.mark.parametrize(
    "metrics_runtime_test_configuration",
    test_configurations,
    ids=[f"{idx}_{config.test_name}" for idx, config in enumerate(test_configurations)],
    indirect=True,
)
def test_non_sensitivity(metrics_runtime_test_configuration):
    base_config, runtime_config, explanation_step_outputs = (
        metrics_runtime_test_configuration
    )

    def comparison_func(output: Any, expected: torch.Tensor):
        (_, non_sensitivity, n_features_found, _, _) = output
        print(
            "non_sensitivity",
            non_sensitivity,
            "expected",
            expected,
            "n_features_found",
            n_features_found,
        )
        _assert_tensor_almost_equal(
            non_sensitivity, expected, delta=runtime_config.delta
        )
        target_n_features = (
            base_config.n_features
            if runtime_config.percentage_feature_removal_per_step == 0.0
            else base_config.n_features
            // math.ceil(
                base_config.n_features
                * runtime_config.percentage_feature_removal_per_step
            )
        )
        assert n_features_found[0].item() == target_n_features, (
            f"{n_features_found} != {target_n_features}"
        )

    _run_metric_test_looped(
        base_config=base_config,
        runtime_config=runtime_config,
        explanation_step_outputs=explanation_step_outputs,
        metric_func=monotonicity_corr_and_non_sens,
        comparison_func=comparison_func,
        use_percentage_attribution_threshold=runtime_config.use_percentage_attribution_threshold,
        zero_attribution_threshold=runtime_config.zero_attribution_threshold,
        zero_variance_threshold=runtime_config.zero_variance_threshold,
        percentage_feature_removal_per_step=runtime_config.percentage_feature_removal_per_step,
        perturb_func=runtime_config.perturb_func,
        return_intermediate_results=True,
        return_ratio=False,
    )
