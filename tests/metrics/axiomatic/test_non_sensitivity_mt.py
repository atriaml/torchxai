from collections.abc import Callable
from dataclasses import field

import pytest
import torch

from tests.utils.common import _assert_tensor_almost_equal, _set_all_random_seeds
from tests.utils.configs import TestBaseConfig, TestRuntimeConfig
from torchxai.data_types import (
    MultiTargetExplanationStepOutputs,
    SingleTargetAcrossBatch,
)
from torchxai.metrics import monotonicity_corr_and_non_sens
from torchxai.metrics._utils.perturbation import default_random_perturb_func


def _format_to_list(value):
    if not isinstance(value, list):
        return [value]
    return value


class MetricTestRuntimeConfig(TestRuntimeConfig):
    test_name: str | None = "compare_multi_target_to_single_target"
    expainer: str = "saliency"
    override_target: list[int] = field(
        default_factory=lambda: [
            SingleTargetAcrossBatch(index=0),
            SingleTargetAcrossBatch(index=1),
            SingleTargetAcrossBatch(index=2),
        ]
    )
    expected: torch.Tensor | None = None
    explainer_kwargs: dict | None = field(
        default_factory=lambda: {"multi_target": True}
    )
    delta: float = 1e-8
    perturb_func: Callable = default_random_perturb_func()
    n_perturbations_per_feature: list[int] = field(default_factory=lambda: [10, 10, 20])
    max_features_processed_per_batch: list[int | None] = field(
        default_factory=lambda: [1, None, 40]
    )
    set_image_feature_mask: bool = True
    percentage_feature_removal_per_step: float = 0.0
    multi_target: bool = True


test_configurations = [
    MetricTestRuntimeConfig(
        target_fixture="classification_alexnet_model_config",
        percentage_feature_removal_per_step=0.1,
    ),
    MetricTestRuntimeConfig(
        target_fixture="classification_alexnet_model_single_sample_config",
        percentage_feature_removal_per_step=0.1,
    ),
    MetricTestRuntimeConfig(
        target_fixture="classification_alexnet_model_real_images_config",
        explainer="integrated_gradients",
        percentage_feature_removal_per_step=0.1,
    ),
    MetricTestRuntimeConfig(
        target_fixture="classification_alexnet_model_real_images_single_sample_config",
        explainer="integrated_gradients",
        percentage_feature_removal_per_step=0.1,
    ),
    MetricTestRuntimeConfig(target_fixture="classification_alexnet_model_config"),
    MetricTestRuntimeConfig(
        target_fixture="classification_alexnet_model_single_sample_config"
    ),
    MetricTestRuntimeConfig(
        target_fixture="classification_alexnet_model_real_images_config",
        explainer="integrated_gradients",
    ),
    MetricTestRuntimeConfig(
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
    base_config: TestBaseConfig
    runtime_config: MetricTestRuntimeConfig
    explanation_step_outputs: MultiTargetExplanationStepOutputs

    n_perturbations_per_feature = _format_to_list(
        runtime_config.n_perturbations_per_feature
    )
    max_features_processed_per_batch = _format_to_list(
        runtime_config.max_features_processed_per_batch
    )
    expected = _format_to_list(runtime_config.expected)

    assert len(n_perturbations_per_feature) == len(max_features_processed_per_batch)
    assert len(n_perturbations_per_feature) == len(expected) or len(expected) == 1
    for n_perturbs, max_features in zip(
        n_perturbations_per_feature, max_features_processed_per_batch, strict=True
    ):
        _set_all_random_seeds(1234)
        (
            _,
            non_sensitivity_score_batch_list_1,
            _,
            perturbed_fwd_diffs_relative_vars_batch_list_1,
            feature_group_attribution_scores_batch_list_1,
        ) = monotonicity_corr_and_non_sens(
            forward_func=base_config.model,
            inputs=explanation_step_outputs.inputs,
            attributions=explanation_step_outputs.attributions,  # type: ignore
            feature_mask=explanation_step_outputs.feature_mask,
            additional_forward_args=explanation_step_outputs.additional_forward_args,
            target=explanation_step_outputs.target,
            perturb_func=runtime_config.perturb_func,
            n_perturbations_per_feature=n_perturbs,
            max_features_processed_per_batch=max_features,
            percentage_feature_removal_per_step=runtime_config.percentage_feature_removal_per_step,
            show_progress=True,
            multi_target=True,
            return_intermediate_results=True,
            return_ratio=False,
        )
        non_sensitivity_score_batch_list_2 = []
        perturbed_fwd_diffs_relative_vars_batch_list_2 = []
        feature_group_attribution_scores_batch_list_2 = []
        for explanation, target in zip(
            explanation_step_outputs.attributions,
            explanation_step_outputs.target,
            strict=True,  # type: ignore
        ):
            _set_all_random_seeds(1234)
            # for this test we take the sum of the explanations over channel dimension to match the feature dimension
            # of the feature mask
            (
                _,
                non_sensitivity_score_batch,
                _,
                perturbed_fwd_diffs_relative_vars_batch,
                feature_group_attribution_scores_batch,
            ) = monotonicity_corr_and_non_sens(
                forward_func=base_config.model,
                inputs=explanation_step_outputs.inputs,
                attributions=explanation,
                feature_mask=explanation_step_outputs.feature_mask,
                additional_forward_args=explanation_step_outputs.additional_forward_args,
                target=target,
                perturb_func=runtime_config.perturb_func,
                n_perturbations_per_feature=n_perturbs,
                max_features_processed_per_batch=max_features,
                percentage_feature_removal_per_step=runtime_config.percentage_feature_removal_per_step,
                show_progress=True,
                return_intermediate_results=True,
                return_ratio=False,
            )
            non_sensitivity_score_batch_list_2.append(non_sensitivity_score_batch)
            perturbed_fwd_diffs_relative_vars_batch_list_2.append(
                perturbed_fwd_diffs_relative_vars_batch
            )
            feature_group_attribution_scores_batch_list_2.append(
                feature_group_attribution_scores_batch
            )
        assert len(non_sensitivity_score_batch_list_1) == len(
            non_sensitivity_score_batch_list_2
        )
        assert len(perturbed_fwd_diffs_relative_vars_batch_list_1) == len(
            perturbed_fwd_diffs_relative_vars_batch_list_2
        )
        assert len(feature_group_attribution_scores_batch_list_1) == len(
            feature_group_attribution_scores_batch_list_2
        )
        import torch

        for x, y in zip(
            non_sensitivity_score_batch_list_1,
            non_sensitivity_score_batch_list_2,
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


# def test_multi_target_metric(metrics_runtime_test_configuration, metric_func: partial):
#     base_config, runtime_config, explanation_step_outputs = (
#         metrics_runtime_test_configuration
#     )
#     base_config: TestBaseConfig
#     runtime_config: MetricTestRuntimeConfig
#     explanation_step_outputs: MultiTargetExplanationStepOutputs

#     n_perturbations_per_feature = _format_to_list(
#         runtime_config.n_perturbations_per_feature
#     )
#     max_features_processed_per_batch = _format_to_list(
#         runtime_config.max_features_processed_per_batch
#     )
#     expected = _format_to_list(runtime_config.expected)

#     assert len(n_perturbations_per_feature) == len(max_features_processed_per_batch)
#     assert len(n_perturbations_per_feature) == len(expected) or len(expected) == 1
#     for n_perturbs, max_features in zip(
#         n_perturbations_per_feature, max_features_processed_per_batch, strict=True
#     ):
#         _set_all_random_seeds(1234)
#         outputs_list_1 = metric_func(
#             multi_target=False,
#             n_perturbs=n_perturbs,
#             max_features=max_features,
#             explanation=explanation_step_outputs.attributions,
#             target=explanation_step_outputs.target,
#         )
#         outputs_list_2 = []
#         for explanation, target in zip(
#             explanation_step_outputs.attributions,
#             explanation_step_outputs.target,
#             strict=True,  # type: ignore
#         ):
#             _set_all_random_seeds(1234)
#             outputs = metric_func(
#                 multi_target=False,
#                 n_perturbs=n_perturbs,
#                 max_features=max_features,
#                 explanation=explanation,
#                 target=target,
#             )
#             outputs_list_2.append(outputs)

#         # list of dict to dict of list
#         outputs_list_2 = {
#             key: [output[key] for output in outputs_list_2] for key in outputs_list_2[0]
#         }

#         for key in list(outputs_list_1.keys()):
#             output_1 = outputs_list_1[key]
#             output_2 = outputs_list_2[key]
#             assert len(output_1) == len(output_2)
#             _assert_tensor_almost_equal(
#                 torch.stack(output_1).float(),
#                 torch.stack(output_2).float(),
#                 delta=runtime_config.delta,
#                 mode="mean",
#             )

#             for x, y in zip(
#                 non_sensitivity_score_batch_list_1,
#                 non_sensitivity_score_batch_list_2,
#                 strict=True,
#             ):
#                 _assert_tensor_almost_equal(
#                     x.float(), y.float(), delta=runtime_config.delta, mode="mean"
#                 )
#             for x, y in zip(
#                 perturbed_fwd_diffs_relative_vars_batch_list_1,
#                 perturbed_fwd_diffs_relative_vars_batch_list_2,
#                 strict=True,
#             ):
#                 for xx, yy in zip(x, y, strict=True):
#                     xx = xx / torch.max(xx)
#                     yy = yy / torch.max(yy)
#                     _assert_tensor_almost_equal(
#                         xx.float(), yy.float(), delta=runtime_config.delta, mode="mean"
#                     )
#             for x, y in zip(
#                 feature_group_attribution_scores_batch_list_1,
#                 feature_group_attribution_scores_batch_list_2,
#                 strict=True,
#             ):
#                 for xx, yy in zip(x, y, strict=True):
#                     _assert_tensor_almost_equal(
#                         xx.float(), yy.float(), delta=runtime_config.delta, mode="mean"
#                     )
