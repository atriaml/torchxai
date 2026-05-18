import inspect
from collections.abc import Callable

import pytest  # noqa

from tests.fixtures._metric import _get_metric_inputs
from tests.utils.common import (
    _assert_tensor_almost_equal,
    _grid_segmenter,
    _set_all_random_seeds,
)
from tests.utils.configs import RuntimeTestConfig
from torchxai.data_types._target import SingleTargetAcrossBatch
from torchxai.metrics import monotonicity_corr_and_non_sens
from torchxai.metrics._utils.perturbation import default_random_perturb_func


def _format_to_list(value):
    if not isinstance(value, list):
        return [value]
    return value


class MetricTestRuntimeConfig_(RuntimeTestConfig):
    perturb_func: Callable = default_random_perturb_func()
    n_perturbations_per_feature: int | list[int | None] | None = 100
    max_features_processed_per_batch: int | list[int | None] | None = None
    set_image_feature_mask: bool = False
    multi_target: bool = True


test_configurations = [
    MetricTestRuntimeConfig_(
        test_name="classification_alexnet_model",
        target_fixture="classification_alexnet_model_single_sample_config",
        explainer="saliency",
        override_target=[
            SingleTargetAcrossBatch(index=0),
            SingleTargetAcrossBatch(index=1),
            SingleTargetAcrossBatch(index=2),
        ],
        expected=None,
        delta=1e-6,
        n_perturbations_per_feature=[10, 10, 20],
        max_features_processed_per_batch=[1, None, 40],
        set_image_feature_mask=True,
    ),
    MetricTestRuntimeConfig_(
        test_name="classification_alexnet_model",
        target_fixture="classification_alexnet_model_real_images_single_sample_config",
        explainer="saliency",
        override_target=[
            SingleTargetAcrossBatch(index=0),
            SingleTargetAcrossBatch(index=1),
            SingleTargetAcrossBatch(index=2),
        ],
        expected=None,
        delta=1e-6,
        n_perturbations_per_feature=[10, 10, 20],
        max_features_processed_per_batch=[1, None, 40],
        set_image_feature_mask=True,
    ),
]


@pytest.mark.metrics
@pytest.mark.parametrize(
    "metrics_runtime_test_configuration",
    test_configurations,
    ids=[f"{idx}_{config.test_name}" for idx, config in enumerate(test_configurations)],
    indirect=True,
)
def test_monotonicity_corr_multi_target(metrics_runtime_test_configuration):
    base_config, runtime_config, explanation_step_outputs = (
        metrics_runtime_test_configuration
    )
    assert len(explanation_step_outputs.explanations) == len(
        runtime_config.override_target
    ), "Number of explanations should be equal to the number of targets"

    kwargs = _get_metric_inputs(base_config, runtime_config, explanation_step_outputs)
    kwargs = {
        k: v
        for k, v in kwargs.items()
        if k in inspect.signature(monotonicity_corr_and_non_sens).parameters
    }

    if runtime_config.set_image_feature_mask:
        kwargs["feature_mask"] = _grid_segmenter(
            base_config.explanation_inputs.inputs[0], cell_size=32
        ).expand_as(base_config.explanation_inputs.inputs[0])

    n_perturbations_per_feature = _format_to_list(
        runtime_config.n_perturbations_per_feature
    )
    max_features_processed_per_batch = _format_to_list(
        runtime_config.max_features_processed_per_batch
    )
    expected = _format_to_list(runtime_config.expected)

    assert len(n_perturbations_per_feature) == len(max_features_processed_per_batch)
    assert len(n_perturbations_per_feature) == len(expected) or len(expected) == 1

    attributions_list = kwargs.pop("attributions")
    targets_list = kwargs.pop("target")
    for n_perturbs, max_features in zip(
        n_perturbations_per_feature, max_features_processed_per_batch, strict=True
    ):
        _set_all_random_seeds(1234)
        (
            monotonicity_corr_score_batch_list_1,
            _,
            _,
            perturbed_fwd_diffs_relative_vars_batch_list_1,
            feature_group_attribution_scores_batch_list_1,
        ) = monotonicity_corr_and_non_sens(
            attributions=attributions_list,
            target=targets_list,
            **kwargs,
            perturb_func=runtime_config.perturb_func,
            n_perturbations_per_feature=n_perturbs,
            max_features_processed_per_batch=max_features,
            show_progress=False,
            multi_target=True,
            return_intermediate_results=True,
            return_ratio=False,
        )
        monotonicity_corr_score_batch_list_2 = []
        perturbed_fwd_diffs_relative_vars_batch_list_2 = []
        feature_group_attribution_scores_batch_list_2 = []
        for attributions, target in zip(attributions_list, targets_list, strict=True):
            _set_all_random_seeds(1234)
            # for this test we take the sum of the explanations over channel dimension to match the feature dimension
            # of the feature mask
            (
                monotonicity_corr_score_batch,
                _,
                _,
                perturbed_fwd_diffs_relative_vars_batch,
                feature_group_attribution_scores_batch,
            ) = monotonicity_corr_and_non_sens(
                attributions=attributions,
                target=target,
                **kwargs,
                perturb_func=runtime_config.perturb_func,
                n_perturbations_per_feature=n_perturbs,
                max_features_processed_per_batch=max_features,
                show_progress=False,
                return_intermediate_results=True,
                return_ratio=False,
            )
            monotonicity_corr_score_batch_list_2.append(monotonicity_corr_score_batch)
            perturbed_fwd_diffs_relative_vars_batch_list_2.append(
                perturbed_fwd_diffs_relative_vars_batch
            )
            feature_group_attribution_scores_batch_list_2.append(
                feature_group_attribution_scores_batch
            )
        assert len(monotonicity_corr_score_batch_list_1) == len(
            monotonicity_corr_score_batch_list_2
        )
        assert len(perturbed_fwd_diffs_relative_vars_batch_list_1) == len(
            perturbed_fwd_diffs_relative_vars_batch_list_2
        )
        assert len(feature_group_attribution_scores_batch_list_1) == len(
            feature_group_attribution_scores_batch_list_2
        )
        import torch

        for x, y in zip(
            monotonicity_corr_score_batch_list_1,
            monotonicity_corr_score_batch_list_2,
            strict=True,
        ):
            _assert_tensor_almost_equal(
                x.float(), y.float(), delta=runtime_config.delta, mode="mean"
            )
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
        for x, y in zip(
            feature_group_attribution_scores_batch_list_1,
            feature_group_attribution_scores_batch_list_2,
            strict=True,
        ):
            for xx, yy in zip(x, y, strict=True):
                _assert_tensor_almost_equal(
                    xx.float(), yy.float(), delta=runtime_config.delta, mode="mean"
                )
