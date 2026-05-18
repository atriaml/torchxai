import pytest
import torch

from tests.fixtures._metric import _run_metric_test_looped_mt
from tests.utils.common import _assert_tensor_almost_equal
from tests.utils.configs import RuntimeTestConfig
from torchxai.data_types import SingleTargetAcrossBatch
from torchxai.metrics import monotonicity_corr_and_non_sens
from torchxai.metrics._utils.perturbation import default_random_perturb_func


def _format_to_list(value):
    if not isinstance(value, list):
        return [value]
    return value


def _make_test_config_for_explainer(**kwargs):
    return RuntimeTestConfig(
        test_name="compare_multi_target_to_single_target",
        explainer=kwargs.pop("explainer", "saliency"),
        train_and_eval_model=True,
        override_target=[
            SingleTargetAcrossBatch(index=0),
            SingleTargetAcrossBatch(index=1),
            SingleTargetAcrossBatch(index=2),
        ],
        expected=None,
        delta=1e-6,
        perturb_func=default_random_perturb_func(),
        n_perturbations_per_feature=[10, 10, 20],
        max_features_processed_per_batch=[1, None, 40],
        set_image_feature_mask=True,
        percentage_feature_removal_per_step=kwargs.pop(
            "percentage_feature_removal_per_step", 0.0
        ),
        multi_target=True,
        mode="mean",
        **kwargs,
    )


test_configurations = [
    _make_test_config_for_explainer(
        target_fixture="classification_alexnet_model_config",
        percentage_feature_removal_per_step=0.1,
    ),
    _make_test_config_for_explainer(
        target_fixture="classification_alexnet_model_single_sample_config",
        percentage_feature_removal_per_step=0.1,
    ),
    _make_test_config_for_explainer(
        target_fixture="classification_alexnet_model_real_images_config",
        explainer="integrated_gradients",
        percentage_feature_removal_per_step=0.1,
    ),
    _make_test_config_for_explainer(
        target_fixture="classification_alexnet_model_real_images_single_sample_config",
        explainer="integrated_gradients",
        percentage_feature_removal_per_step=0.1,
    ),
    _make_test_config_for_explainer(
        target_fixture="classification_alexnet_model_config"
    ),
    _make_test_config_for_explainer(
        target_fixture="classification_alexnet_model_single_sample_config"
    ),
    _make_test_config_for_explainer(
        target_fixture="classification_alexnet_model_real_images_config",
        explainer="integrated_gradients",
    ),
    _make_test_config_for_explainer(
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
def test_non_sensitivity_multi_target(metrics_runtime_test_configuration):
    base_config, runtime_config, explanation_step_outputs = (
        metrics_runtime_test_configuration
    )

    def comparison_func(output: list, expected: list):
        for to, te in zip(output[1], expected[1], strict=True):
            _assert_tensor_almost_equal(to, te, delta=runtime_config.delta, mode="mean")

        perturbed_fwd_diffs_relative_vars_batch_list_1 = output[3]
        perturbed_fwd_diffs_relative_vars_batch_list_2 = expected[3]
        for x, y in zip(
            perturbed_fwd_diffs_relative_vars_batch_list_1,
            perturbed_fwd_diffs_relative_vars_batch_list_2,
            strict=True,
        ):
            for xx, yy in zip(x, y, strict=True):
                xx = xx / torch.max(xx)
                yy = yy / torch.max(yy)
                _assert_tensor_almost_equal(
                    xx.float(), yy.float(), delta=runtime_config.delta, mode="mean"
                )

        feature_group_attribution_scores_batch_list_1 = output[4]
        feature_group_attribution_scores_batch_list_2 = expected[4]
        for x, y in zip(
            feature_group_attribution_scores_batch_list_1,
            feature_group_attribution_scores_batch_list_2,
            strict=True,
        ):
            for xx, yy in zip(x, y, strict=True):
                _assert_tensor_almost_equal(
                    xx.float(), yy.float(), delta=runtime_config.delta, mode="mean"
                )

    _run_metric_test_looped_mt(
        base_config=base_config,
        runtime_config=runtime_config,
        explanation_step_outputs=explanation_step_outputs,
        metric_func=monotonicity_corr_and_non_sens,
        comparison_func=comparison_func,
        percentage_feature_removal_per_step=runtime_config.percentage_feature_removal_per_step,
        perturb_func=runtime_config.perturb_func,
        show_progress=True,
        return_intermediate_results=True,
        return_ratio=False,
    )
