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
from torchxai.metrics import infidelity


def _format_to_list(value):
    if not isinstance(value, list):
        return [value]
    return value


class MetricTestRuntimeConfig_(RuntimeTestConfig):
    set_image_feature_mask: bool = True
    n_perturb_samples: int | list[int | None] | None = 10
    max_examples_per_batch: int | list[int | None] | None = None
    normalize: bool = True
    multi_target: bool = True


test_configurations = [
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
        max_examples_per_batch=[5, 1, 40],
    ),
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
        max_examples_per_batch=[5, 1, 40],
    ),
]


@pytest.mark.metrics
@pytest.mark.parametrize(
    "metrics_runtime_test_configuration",
    test_configurations,
    ids=[f"{idx}_{config.test_name}" for idx, config in enumerate(test_configurations)],
    indirect=True,
)
def test_infidelity_multi_target(metrics_runtime_test_configuration):
    base_config, runtime_config, explanation_step_outputs = (
        metrics_runtime_test_configuration
    )
    assert len(explanation_step_outputs.explanations) == len(
        runtime_config.override_target
    ), "Number of explanations should be equal to the number of targets"

    kwargs = _get_metric_inputs(base_config, runtime_config, explanation_step_outputs)
    kwargs = {
        k: v for k, v in kwargs.items() if k in inspect.signature(infidelity).parameters
    }

    if runtime_config.set_image_feature_mask:
        kwargs["feature_mask"] = _grid_segmenter(
            base_config.explanation_inputs.inputs[0], cell_size=32
        ).expand_as(base_config.explanation_inputs.inputs[0])

    max_examples_per_batch = _format_to_list(runtime_config.max_examples_per_batch)

    def perturb_fn(inputs, baselines=None, **kwargs):
        is_input_tuple = isinstance(inputs, tuple)
        if not isinstance(inputs, tuple):
            inputs = (inputs,)

        noise = tuple(torch.randn_like(x, device=x.device) * 0.1 for x in inputs)

        if is_input_tuple:
            return noise, tuple(x - y for x, y in zip(inputs, noise, strict=True))
        else:
            return noise[0], inputs[0] - noise[0]

    attributions_list = kwargs.pop("attributions")
    targets_list = kwargs.pop("target")
    for batch_size in max_examples_per_batch:
        _set_all_random_seeds(1234)
        infidelity_batch_list_1 = infidelity(
            attributions=attributions_list,
            target=targets_list,
            **kwargs,
            perturb_func=perturb_fn,
            n_perturb_samples=runtime_config.n_perturb_samples,
            max_examples_per_batch=batch_size,
            normalize=runtime_config.normalize,
            multi_target=True,
        )

        infidelity_batch_list_2 = []
        for attributions, target in zip(attributions_list, targets_list, strict=True):
            _set_all_random_seeds(1234)
            infidelity_batch = infidelity(
                attributions=attributions,
                target=target,
                **kwargs,
                perturb_func=perturb_fn,
                n_perturb_samples=runtime_config.n_perturb_samples,
                max_examples_per_batch=batch_size,
                normalize=runtime_config.normalize,
            )
            infidelity_batch_list_2.append(infidelity_batch)

        assert len(infidelity_batch_list_1) == len(infidelity_batch_list_2)
        for x, y in zip(infidelity_batch_list_1, infidelity_batch_list_2, strict=True):
            for xx, yy in zip(x, y, strict=True):
                _assert_tensor_almost_equal(
                    xx.float(), yy.float(), delta=runtime_config.delta, mode="mean"
                )
