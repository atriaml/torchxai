import logging
from logging import getLogger

import pytest
import torch  # noqa

from tests.fixtures._metric import _run_metric_test_simple
from tests.utils.common import _assert_tensor_almost_equal
from tests.utils.configs import RuntimeTestConfig
from torchxai.metrics._utils.perturbation import default_random_perturb_func
from torchxai.metrics.localization.attribution_localization import (
    attribution_localization,
)

logging.basicConfig(level=logging.INFO)
logger = getLogger(__name__)


def _make_test_config_for_explainer(**kwargs):
    return RuntimeTestConfig(
        explainer=kwargs.pop("explainer", "saliency"),
        # override_target=[
        #     SingleTargetAcrossBatch(index=0),
        #     SingleTargetAcrossBatch(index=1),
        #     SingleTargetAcrossBatch(index=2),
        # ],
        delta=1e-8,
        perturb_func=default_random_perturb_func(),
        n_perturbations_per_feature=[10, 10, 20],
        max_features_processed_per_batch=[1, None, 40],
        set_image_feature_mask=True,
        **kwargs,
    )


test_configurations = [
    _make_test_config_for_explainer(
        target_fixture="classification_alexnet_model_single_sample_config",
        expected=(torch.tensor([1.0]),),
    ),
    _make_test_config_for_explainer(
        target_fixture="classification_alexnet_model_config",
        expected=(torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),),
    ),
    _make_test_config_for_explainer(
        target_fixture="classification_alexnet_model_real_images_single_sample_config",
        explainer="integrated_gradients",
        expected=(torch.tensor([1.0]),),
    ),
    _make_test_config_for_explainer(
        target_fixture="classification_alexnet_model_real_images_config",
        explainer="integrated_gradients",
        expected=(torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),),
    ),
]


@pytest.mark.metrics
@pytest.mark.parametrize(
    "metrics_runtime_test_configuration",
    test_configurations,
    ids=[f"{idx}_{config.test_name}" for idx, config in enumerate(test_configurations)],
    indirect=True,
)
def test_attribution_localization(metrics_runtime_test_configuration):
    base_config, runtime_config, explanation_step_outputs = (
        metrics_runtime_test_configuration
    )

    def comparison_func(output: tuple, expected: tuple):
        _assert_tensor_almost_equal(
            output, runtime_config.expected, delta=runtime_config.delta, mode="mean"
        )

    _run_metric_test_simple(
        base_config=base_config,
        runtime_config=runtime_config,
        explanation_step_outputs=explanation_step_outputs,
        metric_func=attribution_localization,
        comparison_func=comparison_func,
        feature_mask=tuple(
            torch.ones_like(input).bool()
            for input in base_config.explanation_inputs.inputs
        ),
    )
