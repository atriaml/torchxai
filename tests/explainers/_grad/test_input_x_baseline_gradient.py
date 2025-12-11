import pytest  # noqa
import torch

from tests.explainers.utils import (
    make_config_for_explainer_with_grad_batch_size,
    run_explainer_test_with_config,
)

test_configurations = [
    *make_config_for_explainer_with_grad_batch_size(
        target_fixture="basic_model_single_input_config",
        explainer="input_x_baseline_gradient",
        expected=(torch.tensor([0.0]), torch.tensor([0.0])),
    ),
    *make_config_for_explainer_with_grad_batch_size(
        target_fixture="basic_model_single_input_config",
        explainer="input_x_baseline_gradient",
        expected=(torch.tensor([0.0]), torch.tensor([0.0])),
        override_target=0,
        throws_exception=True,
    ),
    *make_config_for_explainer_with_grad_batch_size(
        target_fixture="basic_model_single_batched_input_config",
        explainer="input_x_baseline_gradient",
        expected=(torch.tensor([[0.0]]), torch.tensor([[0]])),
        override_target=0,
    ),
    *make_config_for_explainer_with_grad_batch_size(
        target_fixture="basic_model_batch_input_config",
        explainer="input_x_baseline_gradient",
        expected=(torch.tensor([0, 3, 0]), torch.tensor([0, -1, 0])),
    ),
    *make_config_for_explainer_with_grad_batch_size(
        target_fixture="basic_model_batch_input_with_additional_forward_args_config",
        explainer="input_x_baseline_gradient",
        expected=(torch.tensor([[0, 0, 0]]), torch.tensor([[0, 0, 0]])),
    ),
    *make_config_for_explainer_with_grad_batch_size(
        target_fixture="classification_convnet_model_with_multiple_targets_config",
        explainer="input_x_baseline_gradient",
        expected=None,
    ),
    *make_config_for_explainer_with_grad_batch_size(
        target_fixture="classification_multilayer_model_with_tuple_targets_config",
        explainer="input_x_baseline_gradient",
        expected=[
            torch.tensor(
                [
                    [24.0, 48.0, 72.0],
                    [128.0, 160.0, 192.0],
                    [224.0, 256.0, 288.0],
                    [320.0, 352.0, 384.0],
                ]
            ),
            torch.tensor(
                [
                    [3.0, 6.0, 9.0],
                    [128.0, 160.0, 192.0],
                    [224.0, 256.0, 288.0],
                    [320.0, 352.0, 384.0],
                ]
            ),
        ],
        override_target=[
            [(0, 1, 1), (0, 1, 1), (1, 1, 1), (0, 1, 1)],
            [(0, 0, 0), (0, 1, 1), (1, 1, 1), (0, 1, 1)],
        ],
    ),
    *make_config_for_explainer_with_grad_batch_size(
        target_fixture="classification_sigmoid_model_single_input_single_target_config",
        explainer="input_x_baseline_gradient",
        expected=[
            (
                torch.tensor(
                    [
                        [
                            0.0074,
                            0.0288,
                            -0.0109,
                            -0.0016,
                            0.0274,
                            -0.0158,
                            0.0009,
                            -0.0071,
                            0.0154,
                            -0.0439,
                        ]
                    ]
                ),
            ),
            (
                torch.tensor(
                    [
                        [
                            -0.0018,
                            -0.0026,
                            -0.0207,
                            -0.0197,
                            0.0049,
                            0.0157,
                            0.0118,
                            -0.0028,
                            -0.0151,
                            0.0201,
                        ]
                    ]
                ),
            ),
        ],
        override_target=[torch.tensor([0]), torch.tensor([1])],
    ),
    *make_config_for_explainer_with_grad_batch_size(
        target_fixture="classification_softmax_model_single_input_single_target_config",
        explainer="input_x_baseline_gradient",
        expected=[
            (
                torch.tensor(
                    [
                        [
                            -0.0019,
                            0.0005,
                            -0.0020,
                            -0.0023,
                            -0.0013,
                            -0.0019,
                            0.0011,
                            0.0017,
                            -0.0002,
                            0.0013,
                        ]
                    ]
                ),
            ),
            (
                torch.tensor(
                    [
                        [
                            3.9293e-03,
                            2.7826e-03,
                            -1.2806e-03,
                            -1.9040e-03,
                            6.3413e-05,
                            1.8654e-03,
                            1.6219e-03,
                            -1.7387e-04,
                            -3.2753e-03,
                            2.2756e-03,
                        ]
                    ]
                ),
            ),
        ],
        override_target=[torch.tensor([0]), torch.tensor([1])],
    ),
    *make_config_for_explainer_with_grad_batch_size(
        target_fixture="classification_alexnet_model_config",
        explainer="input_x_baseline_gradient",
        override_target=[0, 1, 2],
        expected=[None] * 3,
    ),
    *make_config_for_explainer_with_grad_batch_size(
        target_fixture="classification_alexnet_model_config",
        explainer="input_x_baseline_gradient",
        override_target=[
            [0] * 10,
            [1] * 10,
            list(range(10)),
        ],  # take all the outputs at 0th index as target
        expected=[None] * 3,
    ),
]


@pytest.mark.explainers
@pytest.mark.parametrize(
    "explainer_runtime_test_configuration",
    test_configurations,
    ids=[f"{idx}_{config.test_name}" for idx, config in enumerate(test_configurations)],
    indirect=True,
)
def test_input_x_baseline_gradient(explainer_runtime_test_configuration):
    base_config, runtime_config = explainer_runtime_test_configuration
    run_explainer_test_with_config(
        base_config=base_config, runtime_config=runtime_config
    )
