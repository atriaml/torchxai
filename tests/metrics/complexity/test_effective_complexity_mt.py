import pytest
import torch  # noqa

from tests.fixtures._metric import _run_metric_test_looped_mt
from tests.utils.common import _assert_tensor_almost_equal
from tests.utils.configs import RuntimeTestConfig
from torchxai.data_types._target import SingleTargetAcrossBatch
from torchxai.metrics import effective_complexity
from torchxai.metrics._utils.perturbation import default_random_perturb_func


def _format_to_list(value):
    if not isinstance(value, list):
        return [value]
    return value


def _make_test_config_for_explainer(**kwargs):
    return RuntimeTestConfig(
        test_name="compare_multi_target_to_single_target",
        explainer=kwargs.pop("explainer", "saliency"),
        override_target=[
            SingleTargetAcrossBatch(index=0),
            SingleTargetAcrossBatch(index=1),
            SingleTargetAcrossBatch(index=2),
        ],
        delta=1e-8,
        perturb_func=default_random_perturb_func(),
        n_perturbations_per_feature=[10, 10, 20],
        max_features_processed_per_batch=[1, None, 40],
        set_image_feature_mask=True,
        multi_target=True,
        mode="mean",
        **kwargs,
    )


test_configurations = [
    _make_test_config_for_explainer(
        target_fixture="classification_alexnet_model_single_sample_config"
    ),
    _make_test_config_for_explainer(
        target_fixture="classification_alexnet_model_config"
    ),
    _make_test_config_for_explainer(
        target_fixture="classification_alexnet_model_real_images_single_sample_config",
        explainer="integrated_gradients",
    ),
    _make_test_config_for_explainer(
        target_fixture="classification_alexnet_model_real_images_config",
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
def test_effective_complexity_multi_target(metrics_runtime_test_configuration):
    base_config, runtime_config, explanation_step_outputs = (
        metrics_runtime_test_configuration
    )

    def comparison_func(output: list, expected: list):
        effective_complexity_score_batch_list_1 = output[0]
        effective_complexity_score_batch_list_2 = expected[0]
        perturbed_fwd_diffs_rel_vars_batch_list_1 = output[1]
        perturbed_fwd_diffs_rel_vars_batch_list_2 = expected[1]
        for x, y in zip(
            effective_complexity_score_batch_list_1,
            effective_complexity_score_batch_list_2,
            strict=True,
        ):
            _assert_tensor_almost_equal(
                x.float(), y.float(), delta=runtime_config.delta, mode="mean"
            )
        for (
            perturbed_fwd_diffs_rel_vars_batch_1,
            perturbed_fwd_diffs_rel_vars_batch_2,
        ) in zip(
            perturbed_fwd_diffs_rel_vars_batch_list_1,
            perturbed_fwd_diffs_rel_vars_batch_list_2,
            strict=True,
        ):
            for x, y in zip(
                perturbed_fwd_diffs_rel_vars_batch_1,
                perturbed_fwd_diffs_rel_vars_batch_2,
                strict=True,
            ):
                _assert_tensor_almost_equal(
                    x.float(), y.float(), delta=runtime_config.delta, mode="mean"
                )

    _run_metric_test_looped_mt(
        base_config=base_config,
        runtime_config=runtime_config,
        explanation_step_outputs=explanation_step_outputs,
        metric_func=effective_complexity,
        comparison_func=comparison_func,
        perturb_func=runtime_config.perturb_func,
        show_progress=True,
        return_intermediate_results=True,
        seed_outside_loop=True,
    )
