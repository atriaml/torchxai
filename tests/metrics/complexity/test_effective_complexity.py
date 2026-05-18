import math
from typing import Any

import pytest  # noqa
import torch

from tests.fixtures._metric import _run_metric_test_looped
from tests.utils.common import _assert_tensor_almost_equal
from tests.utils.configs import RuntimeTestConfig
from torchxai.metrics._utils.perturbation import (
    default_fixed_baseline_perturb_func,
    default_random_perturb_func,
)
from torchxai.metrics.complexity.effective_complexity import effective_complexity


def _format_to_list(value):
    if not isinstance(value, list):
        return [value]
    return value


def setup_test_config_for_explainer(**kwargs):
    return RuntimeTestConfig(
        perturb_func=kwargs.pop("perturb_func", default_random_perturb_func()),
        n_perturbations_per_feature=kwargs.pop("n_perturbations_per_feature", 10),
        zero_variance_threshold=kwargs.pop("zero_variance_threshold", 1e-5),
        percentage_feature_removal_per_step=kwargs.pop(
            "percentage_feature_removal_per_step", 0.0
        ),
        return_ratio=kwargs.pop("return_ratio", False),
        **kwargs,
    )


test_configurations = [
    setup_test_config_for_explainer(
        test_name="n_perturbations_per_feature_1",
        target_fixture="multi_modal_sequence_relu",
        explainer="integrated_gradients",
        expected=torch.tensor(
            # in this test set first 4 features have variance above 0.01
            [0.1481]
        ),
        perturb_func=default_fixed_baseline_perturb_func(),
        n_perturbations_per_feature=1,
        zero_variance_threshold=0.01,
        return_ratio=True,
    ),
    setup_test_config_for_explainer(
        test_name="n_perturbations_per_feature_10",
        target_fixture="multi_modal_sequence_relu",
        explainer="integrated_gradients",
        expected=torch.tensor(
            # in this test set first 4 features have variance above 0.01
            [0.1481]
        ),
        perturb_func=default_fixed_baseline_perturb_func(),
        n_perturbations_per_feature=10,
        zero_variance_threshold=1e-4,
        return_ratio=True,
    ),
    setup_test_config_for_explainer(
        test_name="n_perturbations_per_feature_1_percentage_feature_removal_per_step_0.1",
        target_fixture="multi_modal_sequence_relu",
        explainer="integrated_gradients",
        expected=torch.tensor(
            # in this test set first 4 features have variance above 0.01
            [0.2222]
        ),
        perturb_func=default_fixed_baseline_perturb_func(),
        n_perturbations_per_feature=1,
        zero_variance_threshold=1e-4,
        percentage_feature_removal_per_step=0.1,
        return_ratio=True,
    ),
    setup_test_config_for_explainer(
        test_name="n_perturbations_per_feature_10_percentage_feature_removal_per_step_0.1",
        target_fixture="multi_modal_sequence_relu",
        explainer="integrated_gradients",
        expected=torch.tensor(
            # in this test set first 4 features have variance above 0.01
            [0.2222]
        ),
        perturb_func=default_fixed_baseline_perturb_func(),
        n_perturbations_per_feature=10,
        zero_variance_threshold=1e-4,
        percentage_feature_removal_per_step=0.1,
        return_ratio=True,
    ),
    setup_test_config_for_explainer(
        test_name="n_perturbations_per_feature_1",
        target_fixture="multi_modal_sequence_sum",
        explainer="integrated_gradients",
        expected=torch.tensor([0.4074]),
        perturb_func=default_fixed_baseline_perturb_func(),
        n_perturbations_per_feature=1,
        zero_variance_threshold=1e-1,
        return_ratio=True,
    ),
    setup_test_config_for_explainer(
        test_name="n_perturbations_per_feature_10",
        target_fixture="multi_modal_sequence_sum",
        explainer="saliency",
        expected=torch.tensor([0.7037]),  # saliency completeness is not so great
        perturb_func=default_fixed_baseline_perturb_func(),
        n_perturbations_per_feature=10,
        zero_variance_threshold=1e-1,
        return_ratio=True,
    ),
    setup_test_config_for_explainer(
        test_name="n_perturbations_per_feature_1_percentage_feature_removal_per_step_0.1",
        target_fixture="multi_modal_sequence_sum",
        explainer="integrated_gradients",
        expected=torch.tensor([0.4444]),
        perturb_func=default_fixed_baseline_perturb_func(),
        n_perturbations_per_feature=1,
        zero_variance_threshold=1e-1,
        percentage_feature_removal_per_step=0.1,
        return_ratio=True,
    ),
    setup_test_config_for_explainer(
        test_name="n_perturbations_per_feature_10_percentage_feature_removal_per_step_0.1",
        target_fixture="multi_modal_sequence_sum",
        explainer="integrated_gradients",
        expected=torch.tensor([0.4444]),
        perturb_func=default_fixed_baseline_perturb_func(),
        n_perturbations_per_feature=10,
        zero_variance_threshold=1e-1,
        percentage_feature_removal_per_step=0.1,
        return_ratio=True,
    ),
    # the park function is taken from the paper: https://arxiv.org/pdf/2007.07584
    setup_test_config_for_explainer(
        test_name="park_function_configuration_saliency",
        target_fixture="park_function_configuration",
        explainer="saliency",
        expected=torch.tensor([3]),
        perturb_func=default_random_perturb_func(noise_scale=1.0),
        zero_variance_threshold=1e-2,
    ),
    setup_test_config_for_explainer(
        test_name="park_function_configuration_input_x_gradient",
        target_fixture="park_function_configuration",
        explainer="input_x_gradient",
        expected=torch.tensor([3]),
        perturb_func=default_random_perturb_func(noise_scale=1.0),
        zero_variance_threshold=1e-2,
    ),
    setup_test_config_for_explainer(
        test_name="park_function_configuration_integrated_gradients",
        target_fixture="park_function_configuration",
        explainer="integrated_gradients",
        expected=torch.tensor([4]),
        perturb_func=default_random_perturb_func(noise_scale=1.0),
        zero_variance_threshold=1e-2,
    ),
    setup_test_config_for_explainer(
        test_name="basic_model_single_input_integrated_gradients",
        target_fixture="basic_model_single_input_config",
        expected=torch.tensor([2]),
        n_perturbations_per_feature=[10, 10, 20],
        max_features_processed_per_batch=[None, 1, 40],
    ),
    setup_test_config_for_explainer(
        test_name="basic_model_batch_input_integrated_gradients",
        target_fixture="basic_model_batch_input_config",
        expected=torch.tensor([2, 2, 2]),
        n_perturbations_per_feature=[10, 10, 20],
        max_features_processed_per_batch=[None, 1, 40],
    ),
    setup_test_config_for_explainer(
        test_name="basic_model_batch_input_with_additional_forward_args_integrated_gradients",
        target_fixture="basic_model_batch_input_with_additional_forward_args_config",
        expected=torch.tensor([0]),
        n_perturbations_per_feature=[10, 10, 20],
        max_features_processed_per_batch=[None, 1, 40],
    ),
    setup_test_config_for_explainer(
        test_name="classification_convnet_model_with_multiple_targets_deep_lift",
        target_fixture="classification_convnet_model_with_multiple_targets_config",
        expected=torch.tensor([16] * 20),
        n_perturbations_per_feature=[10, 10, 20],
        max_features_processed_per_batch=[None, 1, 40],
    ),
    setup_test_config_for_explainer(
        test_name="classification_convnet_model_with_multiple_targets_deep_lift_zero_variance_threshold_1e-2",
        target_fixture="classification_convnet_model_with_multiple_targets_config",
        expected=torch.tensor([14] * 20),
        n_perturbations_per_feature=[10, 10, 20],
        max_features_processed_per_batch=[None, 1, 40],
        zero_variance_threshold=1e-2,
    ),
    setup_test_config_for_explainer(
        test_name="classification_convnet_model_with_multiple_targets_deep_lift_zero_variance_threshold_1e-1",
        target_fixture="classification_convnet_model_with_multiple_targets_config",
        expected=torch.tensor([10] * 20),
        n_perturbations_per_feature=[10, 10, 20],
        max_features_processed_per_batch=[None, 1, 40],
        zero_variance_threshold=1e-1,
    ),
    setup_test_config_for_explainer(
        test_name="classification_multilayer_model_with_tuple_targets_integrated_gradients",
        target_fixture="classification_multilayer_model_with_tuple_targets_config",
        expected=torch.tensor([3, 3, 3, 3]),
        n_perturbations_per_feature=[10, 10, 20],
        max_features_processed_per_batch=[None, 1, 40],
    ),
    setup_test_config_for_explainer(
        test_name="classification_multilayer_model_with_tuple_targets_integrated_gradients_zero_variance_threshold_1e-1",
        target_fixture="classification_multilayer_model_with_tuple_targets_config",
        expected=torch.tensor([1, 2, 2, 2]),
        n_perturbations_per_feature=[10, 10, 20],
        max_features_processed_per_batch=[None, 1, 40],
        zero_variance_threshold=1e-1,
    ),
    setup_test_config_for_explainer(
        test_name="classification_multilayer_model_with_baseline_and_tuple_targets_integrated_gradients",
        target_fixture="classification_multilayer_model_with_baseline_and_tuple_targets_config",
        expected=torch.tensor([3, 3, 3, 3]),
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
def test_effective_complexity(metrics_runtime_test_configuration):
    base_config, runtime_config, explanation_step_outputs = (
        metrics_runtime_test_configuration
    )

    def comparison_func(output: Any, expected: torch.Tensor):
        (
            effective_complexity_score,
            k_features_perturbed_fwd_diff_vars,
            n_features_found,
        ) = output
        target_n_features = (
            base_config.n_features
            if runtime_config.percentage_feature_removal_per_step == 0.0
            else base_config.n_features
            // math.ceil(
                base_config.n_features
                * runtime_config.percentage_feature_removal_per_step
            )
        )
        _assert_tensor_almost_equal(
            effective_complexity_score, expected, delta=runtime_config.delta
        )
        assert n_features_found[0] == target_n_features, (
            f"{n_features_found[0]} != {target_n_features}"
        )

    _run_metric_test_looped(
        base_config=base_config,
        runtime_config=runtime_config,
        explanation_step_outputs=explanation_step_outputs,
        metric_func=effective_complexity,
        comparison_func=comparison_func,
        zero_variance_threshold=runtime_config.zero_variance_threshold,
        percentage_feature_removal_per_step=runtime_config.percentage_feature_removal_per_step,
        perturb_func=runtime_config.perturb_func,
        return_intermediate_results=True,
        return_ratio=runtime_config.return_ratio,
        show_progress=False,
    )
