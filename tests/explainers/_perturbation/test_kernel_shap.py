import pytest
import torch  # noqa

from tests.explainers.utils import (
    ExplainersTestRuntimeConfig,
    run_explainer_test_with_config,
)
from tests.utils.common import _grid_segmenter
from torchxai.data_types import ExplanationTarget, SingleTargetAcrossBatch


class ExplainersTestRuntimeConfig_(ExplainersTestRuntimeConfig):
    set_image_feature_mask: bool = False


def _make_config_for_explainer(*args, **kwargs):
    return [
        ExplainersTestRuntimeConfig_(
            *args,
            **kwargs,
            explainer="kernel_shap",
            explainer_kwargs={"internal_batch_size": internal_batch_size},
            test_name=f"internal_batch_size_{internal_batch_size}",
        )
        for internal_batch_size in [1, 20, 100]
    ]


test_configurations = [
    *_make_config_for_explainer(
        target_fixture="basic_model_single_input_config",
        expected=(torch.tensor([1.4898]), torch.tensor([-0.4898])),
    ),
    *_make_config_for_explainer(
        target_fixture="basic_model_single_input_config",
        expected=(torch.tensor([1.4898]), torch.tensor([-0.4898])),
        override_target=0,
        throws_exception=True,
    ),
    *_make_config_for_explainer(
        target_fixture="basic_model_single_batched_input_config",
        expected=(torch.tensor([[1.4898]]), torch.tensor([[-0.4898]])),
        override_target=0,
    ),
    *_make_config_for_explainer(
        target_fixture="basic_model_batch_input_config",
        expected=(
            torch.tensor([1.4898, 1.5510, 1.3980]),
            torch.tensor([-0.4898, -0.5510, -0.3980]),
        ),
    ),
    *_make_config_for_explainer(
        target_fixture="basic_model_batch_input_with_additional_forward_args_config",
        expected=(
            torch.tensor([[0.2364, 0.0028, -0.0453]]),
            torch.tensor([[-0.2154, -0.0592, 0.0807]]),
        ),
    ),
    *_make_config_for_explainer(
        target_fixture="classification_convnet_model_with_multiple_targets_config",
        expected=None,
    ),
    *_make_config_for_explainer(
        target_fixture="classification_multilayer_model_with_tuple_targets_config",
        expected=[
            torch.tensor(
                [
                    [27.5679, 54.6671, 77.7651],
                    [127.9998, 160.0000, 192.0000],
                    [223.9999, 256.0000, 288.0000],
                    [320.0004, 352.0001, 384.0001],
                ]
            ),
            torch.tensor(
                [
                    [3.4460, 6.8334, 9.7206],
                    [127.9998, 160.0000, 192.0000],
                    [223.9999, 256.0000, 288.0000],
                    [320.0004, 352.0001, 384.0001],
                ]
            ),
        ],
        override_target=[
            ExplanationTarget.from_raw_input(
                [(0, 1, 1), (0, 1, 1), (1, 1, 1), (0, 1, 1)]
            ),
            ExplanationTarget.from_raw_input(
                [(0, 0, 0), (0, 1, 1), (1, 1, 1), (0, 1, 1)]
            ),
        ],
        delta=1e-2,
    ),
    # *_make_config_for
    *_make_config_for_explainer(
        target_fixture="classification_sigmoid_model_single_input_single_target_config",
        expected=[
            torch.tensor(
                [
                    [
                        0.0103,
                        0.0172,
                        0.0015,
                        -0.0075,
                        0.0151,
                        -0.0255,
                        0.0054,
                        -0.0079,
                        -0.0029,
                        -0.0353,
                    ]
                ]
            ),
            torch.tensor(
                [
                    [
                        0.0016,
                        -0.0094,
                        -0.0061,
                        -0.0157,
                        -0.0110,
                        0.0095,
                        0.0145,
                        -0.0037,
                        -0.0229,
                        0.0248,
                    ]
                ]
            ),
        ],
        override_target=[
            ExplanationTarget.from_raw_input(torch.tensor([0])),
            ExplanationTarget.from_raw_input(torch.tensor([1])),
        ],
    ),
    *_make_config_for_explainer(
        target_fixture="classification_alexnet_model_single_sample_config",
        override_target=[
            SingleTargetAcrossBatch(index=0),
            SingleTargetAcrossBatch(index=1),
            SingleTargetAcrossBatch(index=2),
        ],
        expected=[None] * 3,
        set_image_feature_mask=True,
        visualize=False,
    ),
    *_make_config_for_explainer(
        target_fixture="classification_alexnet_model_real_images_single_sample_config",
        override_target=[
            SingleTargetAcrossBatch(index=0),
            SingleTargetAcrossBatch(index=1),
            SingleTargetAcrossBatch(index=2),
        ],
        expected=[None] * 3,
        set_image_feature_mask=True,
        visualize=False,
    ),
]


@pytest.mark.explainers
@pytest.mark.parametrize(
    "explainer_runtime_test_configuration",
    test_configurations,
    ids=[f"{idx}_{config.test_name}" for idx, config in enumerate(test_configurations)],
    indirect=True,
)
def test_kernel_shap(explainer_runtime_test_configuration):
    base_config, runtime_config = explainer_runtime_test_configuration

    if runtime_config.set_image_feature_mask:
        base_config = base_config.model_copy(
            update={
                "explanation_inputs": base_config.explanation_inputs.model_copy(
                    update={
                        "feature_mask": _grid_segmenter(
                            base_config.explanation_inputs.inputs[0], cell_size=32
                        )
                    }
                )
            }
        )

    run_explainer_test_with_config(
        base_config=base_config, runtime_config=runtime_config
    )
