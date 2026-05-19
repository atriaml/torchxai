import inspect

import pytest
import torch  # noqa

from tests.fixtures._metric import _get_metric_inputs
from tests.utils.common import (
    _assert_tensor_almost_equal,
    _grid_segmenter,
    _set_all_random_seeds,
)
from tests.utils.configs import RuntimeTestConfig
from torchxai.data_types._target import SingleTargetAcrossBatch
from torchxai.metrics import sensitivity_n


def _format_to_list(value):
    if not isinstance(value, list):
        return [value]
    return value


class MetricTestRuntimeConfig_(RuntimeTestConfig):
    set_image_feature_mask: bool = True
    n_perturb_samples: int = 10
    max_examples_per_batch: int | list[int | None] | None = None
    normalize: bool = True
    sensitivity_n: int = 1
    multi_target: bool = True


test_configurations = [
    *[
        MetricTestRuntimeConfig_(
            test_name="classification_alexnet_model",
            target_fixture="classification_alexnet_model_config",
            explainer="saliency",
            override_target=[
                SingleTargetAcrossBatch(index=0),
                SingleTargetAcrossBatch(index=1),
                SingleTargetAcrossBatch(index=2),
            ],
            delta=1e-8,
            max_examples_per_batch=[40],
            sensitivity_n=n,
        )
        for n in [1, 10, 20]
    ],
    *[
        MetricTestRuntimeConfig_(
            test_name="classification_alexnet_model",
            target_fixture="classification_alexnet_model_config",
            explainer="integrated_gradients",
            override_target=[
                SingleTargetAcrossBatch(index=0),
                SingleTargetAcrossBatch(index=1),
                SingleTargetAcrossBatch(index=2),
            ],
            delta=1e-8,
            max_examples_per_batch=[40],
            sensitivity_n=n,
        )
        for n in [1, 10, 20]
    ],
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
    assert len(explanation_step_outputs.explanations) == len(
        runtime_config.override_target
    ), "Number of explanations should be equal to the number of targets"

    kwargs = _get_metric_inputs(base_config, runtime_config, explanation_step_outputs)
    kwargs = {
        k: v
        for k, v in kwargs.items()
        if k in inspect.signature(sensitivity_n).parameters
    }
    kwargs["baselines"] = (
        kwargs["baselines"] if kwargs["baselines"] is not None else 0,
    )

    if runtime_config.set_image_feature_mask:
        kwargs["feature_mask"] = _grid_segmenter(
            base_config.explanation_inputs.inputs[0], cell_size=32
        ).expand_as(base_config.explanation_inputs.inputs[0])

    max_examples_per_batch = _format_to_list(runtime_config.max_examples_per_batch)

    attributions_list = kwargs.pop("attributions")
    targets_list = kwargs.pop("target")
    for max_examples in max_examples_per_batch:
        _set_all_random_seeds(1234)
        sensitivity_n_batch_list_1 = sensitivity_n(
            attributions=attributions_list,
            target=targets_list,
            **kwargs,
            n_features_perturbed=runtime_config.sensitivity_n,
            n_perturb_samples=runtime_config.n_perturb_samples,
            max_examples_per_batch=max_examples,
            normalize=runtime_config.normalize,
            multi_target=True,
        )

        sensitivity_n_batch_list_2 = []
        for attributions, target in zip(attributions_list, targets_list, strict=True):
            _set_all_random_seeds(1234)
            sensitivity_n_batch = sensitivity_n(
                attributions=attributions,
                target=target,
                **kwargs,
                n_features_perturbed=runtime_config.sensitivity_n,
                n_perturb_samples=runtime_config.n_perturb_samples,
                max_examples_per_batch=max_examples,
                normalize=runtime_config.normalize,
            )
            sensitivity_n_batch_list_2.append(sensitivity_n_batch)

        assert len(sensitivity_n_batch_list_1) == len(sensitivity_n_batch_list_2)
        for x, y in zip(
            sensitivity_n_batch_list_1, sensitivity_n_batch_list_2, strict=True
        ):
            for xx, yy in zip(x, y, strict=True):
                _assert_tensor_almost_equal(
                    xx.float(), yy.float(), delta=runtime_config.delta, mode="mean"
                )
