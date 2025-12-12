from collections import OrderedDict

import pytest  # noqa
import torch

from tests.explainers.utils import (
    make_config_for_explainer_with_internal_and_grad_batch_size,
    run_explainer_test_with_config,
)
from tests.utils.configs import ExplainersTestRuntimeConfig, TestBaseConfig
from torchxai.data_types import (
    ExplanationTarget,
    SingleTargetAcrossBatch,
    SingleTargetPerSample,
)

test_configurations = [
    *make_config_for_explainer_with_internal_and_grad_batch_size(
        target_fixture="basic_model_single_input_config",
        explainer="deep_lift_shap",
        expected=(torch.tensor([3.2823]), torch.tensor([-1.1275])),
        internal_batch_sizes=[None, 1, 4],
    ),
    *make_config_for_explainer_with_internal_and_grad_batch_size(
        target_fixture="basic_model_single_input_config",
        explainer="deep_lift_shap",
        expected=None,
        override_target=0,
        throws_exception=True,
    ),
    *make_config_for_explainer_with_internal_and_grad_batch_size(
        target_fixture="basic_model_single_batched_input_config",
        explainer="deep_lift_shap",
        expected=(torch.tensor([[3.2823]]), torch.tensor([[-1.1275]])),
        override_target=0,
        internal_batch_sizes=[None, 1, 4],
    ),
    *make_config_for_explainer_with_internal_and_grad_batch_size(
        target_fixture="basic_model_batch_input_config",
        explainer="deep_lift_shap",
        expected=(
            torch.tensor([3.2823, 3.2823, 3.2823]),
            torch.tensor([-1.1275, -1.1275, -1.1275]),
        ),
        internal_batch_sizes=[None, 1, 4],
    ),
    *make_config_for_explainer_with_internal_and_grad_batch_size(
        target_fixture="basic_model_batch_input_with_additional_forward_args_config",
        explainer="deep_lift_shap",
        expected=(torch.tensor([[0, 0, 0]]), torch.tensor([[0, 0, 0]])),
        internal_batch_sizes=[None, 1, 4],
    ),
    *make_config_for_explainer_with_internal_and_grad_batch_size(
        target_fixture="classification_convnet_model_with_multiple_targets_config",
        explainer="deep_lift_shap",
        expected=None,
        internal_batch_sizes=[None, 1, 4],
    ),
    *make_config_for_explainer_with_internal_and_grad_batch_size(
        target_fixture="classification_multilayer_model_with_tuple_targets_config",
        explainer="deep_lift_shap",
        expected=[
            (
                torch.tensor(
                    [
                        [27.1948, 57.8040, 70.4866],
                        [128.7149, 164.9986, 180.2669],
                        [224.7149, 260.9985, 276.2669],
                        [320.7149, 356.9985, 372.2669],
                    ]
                ),
            ),
            (
                torch.tensor(
                    [
                        [3.3993, 7.2255, 8.8108],
                        [128.7149, 164.9986, 180.2669],
                        [224.7149, 260.9985, 276.2669],
                        [320.7149, 356.9985, 372.2669],
                    ]
                ),
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
        internal_batch_sizes=[None, 1, 4],
    ),
    *make_config_for_explainer_with_internal_and_grad_batch_size(
        target_fixture="classification_sigmoid_model_single_input_single_target_config",
        explainer="deep_lift_shap",
        expected=[
            torch.tensor(
                [
                    [
                        0.0121,
                        0.0323,
                        0.0076,
                        -0.0103,
                        0.0134,
                        -0.0246,
                        0.0078,
                        -0.0021,
                        -0.0052,
                        -0.0457,
                    ]
                ]
            ),
            torch.tensor(
                [
                    [
                        0.0016,
                        0.0025,
                        -0.0060,
                        -0.0198,
                        -0.0048,
                        0.0100,
                        0.0052,
                        -0.0043,
                        -0.0185,
                        0.0302,
                    ]
                ]
            ),
        ],
        override_target=[
            ExplanationTarget.from_raw_input(torch.tensor([0])),
            ExplanationTarget.from_raw_input(torch.tensor([1])),
        ],
        internal_batch_sizes=[None, 1, 4],
    ),
    *make_config_for_explainer_with_internal_and_grad_batch_size(
        target_fixture="classification_softmax_model_single_input_single_target_config",
        explainer="deep_lift_shap",
        expected=[
            torch.tensor(
                [
                    [
                        0.0002,
                        -0.0022,
                        -0.0012,
                        0.0004,
                        -0.0003,
                        0.0011,
                        0.0003,
                        -0.0005,
                        0.0001,
                        -0.0008,
                    ]
                ]
            ),
            torch.tensor(
                [
                    [
                        2.9974e-03,
                        -1.6590e-03,
                        2.7560e-03,
                        9.3551e-04,
                        4.0538e-05,
                        3.3316e-03,
                        -1.1196e-03,
                        -1.8362e-03,
                        -1.8250e-03,
                        1.7030e-03,
                    ]
                ]
            ),
        ],
        override_target=[
            ExplanationTarget.from_raw_input(torch.tensor([0])),
            ExplanationTarget.from_raw_input(torch.tensor([1])),
        ],
        internal_batch_sizes=[None, 1, 4],
    ),
    *make_config_for_explainer_with_internal_and_grad_batch_size(
        target_fixture="classification_alexnet_model_config",
        explainer="deep_lift_shap",
        override_target=[
            SingleTargetAcrossBatch(index=0),
            SingleTargetAcrossBatch(index=1),
            SingleTargetAcrossBatch(index=2),
        ],
        expected=[None] * 3,
        internal_batch_sizes=[4, 16],
    ),
    *make_config_for_explainer_with_internal_and_grad_batch_size(
        target_fixture="classification_alexnet_model_config",
        explainer="deep_lift_shap",
        override_target=[
            SingleTargetPerSample(indices=[0] * 10),
            SingleTargetPerSample(indices=[1] * 10),
            SingleTargetPerSample(indices=list(range(10))),
        ],  # take all the outputs at 0th index as target
        expected=[None] * 3,
        internal_batch_sizes=[4, 16],
    ),
]


@pytest.mark.explainers
@pytest.mark.parametrize(
    "explainer_runtime_test_configuration",
    test_configurations,
    ids=[f"{idx}_{config.test_name}" for idx, config in enumerate(test_configurations)],
    indirect=True,
)
def test_deep_lift_shap(explainer_runtime_test_configuration):
    base_config, runtime_config = explainer_runtime_test_configuration
    base_config: TestBaseConfig
    runtime_config: ExplainersTestRuntimeConfig

    # deeplift shap always requires a random train baselines
    baselines = OrderedDict(
        {
            k: torch.randn((20, *v.shape[1:]))
            for k, v in base_config.explanation_inputs.inputs.items()
        }
    )

    base_config = base_config.model_copy(
        update={
            "explanation_inputs": base_config.explanation_inputs.model_copy(
                update={"baselines": baselines}
            )
        }
    )

    run_explainer_test_with_config(
        base_config=base_config, runtime_config=runtime_config
    )
