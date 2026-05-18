import inspect

import pytest  # noqa

from tests.fixtures._metric import _get_metric_inputs
from tests.utils.common import (
    _assert_tensor_almost_equal,
    _grid_segmenter,
    _set_all_random_seeds,
)
from tests.utils.configs import RuntimeTestConfig
from torchxai.data_types._target import SingleTargetAcrossBatch
from torchxai.metrics import faithfulness_estimate


def _format_to_list(value):
    if not isinstance(value, list):
        return [value]
    return value


class MetricTestRuntimeConfig_(RuntimeTestConfig):
    max_features_processed_per_batch: int | list[int | None] | None = None
    set_image_feature_mask: bool = True
    multi_target: bool = True


test_configurations = [
    MetricTestRuntimeConfig_(
        target_fixture="classification_alexnet_model_config",
        explainer="saliency",
        override_target=[
            SingleTargetAcrossBatch(index=0),
            SingleTargetAcrossBatch(index=1),
            SingleTargetAcrossBatch(index=2),
        ],
        delta=1e-8,
        max_features_processed_per_batch=[5, 1, 40],
    ),
    MetricTestRuntimeConfig_(
        target_fixture="classification_alexnet_model_config",
        explainer="integrated_gradients",
        override_target=[
            SingleTargetAcrossBatch(index=0),
            SingleTargetAcrossBatch(index=1),
            SingleTargetAcrossBatch(index=2),
        ],
        delta=1e-8,
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
def test_faithfulness_estimate_multi_target(metrics_runtime_test_configuration):
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
        if k in inspect.signature(faithfulness_estimate).parameters
    }

    if runtime_config.set_image_feature_mask:
        kwargs["feature_mask"] = _grid_segmenter(
            base_config.explanation_inputs.inputs[0], cell_size=32
        ).expand_as(base_config.explanation_inputs.inputs[0])

    max_features_processed_per_batch = _format_to_list(
        runtime_config.max_features_processed_per_batch
    )

    attributions_list = kwargs.pop("attributions")
    targets_list = kwargs.pop("target")
    for max_features in max_features_processed_per_batch:
        _set_all_random_seeds(1234)
        (
            faithfulness_estimate_batch_list_1,
            attributions_sum_perturbed_batch_list_1,
            inputs_perturbed_fwd_diffs_batch_list_1,
        ) = faithfulness_estimate(
            attributions=attributions_list,
            target=targets_list,
            **kwargs,
            max_features_processed_per_batch=max_features,
            multi_target=True,
            return_intermediate_results=True,
        )

        _set_all_random_seeds(1234)
        faithfulness_estimate_batch_list_2 = []
        attributions_sum_perturbed_batch_list_2 = []
        inputs_perturbed_fwd_diffs_batch_list_2 = []
        for attributions, target in zip(attributions_list, targets_list, strict=True):
            (
                faithfulness_estimate_batch,
                attributions_sum_perturbed_batch,
                inputs_perturbed_fwd_diffs_batch,
            ) = faithfulness_estimate(
                attributions=attributions,
                target=target,
                **kwargs,
                max_features_processed_per_batch=max_features,
                return_intermediate_results=True,
            )
            faithfulness_estimate_batch_list_2.append(faithfulness_estimate_batch)
            attributions_sum_perturbed_batch_list_2.append(
                attributions_sum_perturbed_batch
            )
            inputs_perturbed_fwd_diffs_batch_list_2.append(
                inputs_perturbed_fwd_diffs_batch
            )

        assert len(faithfulness_estimate_batch_list_1) == len(
            faithfulness_estimate_batch_list_2
        )
        assert len(attributions_sum_perturbed_batch_list_1) == len(
            attributions_sum_perturbed_batch_list_2
        )
        assert len(inputs_perturbed_fwd_diffs_batch_list_1) == len(
            inputs_perturbed_fwd_diffs_batch_list_2
        )

        for x, y in zip(
            faithfulness_estimate_batch_list_1,
            faithfulness_estimate_batch_list_2,
            strict=True,
        ):
            for xx, yy in zip(x, y, strict=True):
                _assert_tensor_almost_equal(
                    xx.float(), yy.float(), delta=runtime_config.delta, mode="mean"
                )
        for x, y in zip(
            attributions_sum_perturbed_batch_list_1,
            attributions_sum_perturbed_batch_list_2,
            strict=True,
        ):
            for xx, yy in zip(x, y, strict=True):
                _assert_tensor_almost_equal(
                    xx.float(), yy.float(), delta=runtime_config.delta, mode="mean"
                )
        for x, y in zip(
            inputs_perturbed_fwd_diffs_batch_list_1,
            inputs_perturbed_fwd_diffs_batch_list_2,
            strict=True,
        ):
            for xx, yy in zip(x, y, strict=True):
                _assert_tensor_almost_equal(
                    xx.float(), yy.float(), delta=runtime_config.delta, mode="mean"
                )
