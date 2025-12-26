import pytest
import torch

from tests.fixtures._metric import _run_metric_test
from tests.utils.common import _assert_tensor_almost_equal
from tests.utils.configs import RuntimeTestConfig
from torchxai.metrics._utils.visualization import visualize_attribution
from torchxai.metrics.axiomatic.input_invariance import input_invariance


def setup_test_config_for_explainer(explainer, **kwargs):
    return [
        RuntimeTestConfig(
            test_name=f"{explainer}",
            model_type="linear",
            target_fixture="mnist_train_configuration",
            explainer=explainer,
            train_and_eval_model=True,
            image_feature_mask_cell_size=4,
            mode="mean",
            **kwargs,
        )
    ]


test_configurations = [
    # this setup is exactly the same as in the paper: https://arxiv.org/pdf/1711.00867
    # a 3-layer linear model is trained on MNIST, input invariance is computed for saliency maps
    # on 4 input samples. The expected output is [True, True, True, True]
    *setup_test_config_for_explainer(
        explainer="saliency", expected=torch.tensor([0.0, 0.0, 0.0, 0.0])
    ),
    *setup_test_config_for_explainer(
        explainer="input_x_gradient",
        expected=torch.tensor(
            [0.0886, 0.0753, 0.0749, 0.0829]
        ),  # these results might not be exactly reproducible across machines
        delta=1e-3,
    ),
    # this setup is exactly the same as in the paper: https://arxiv.org/pdf/1711.00867
    # a 3-layer linear model is trained on MNIST, input invariance is computed for integrated_gradients
    # on 4 input samples. The expected output is [False, False, False, False] with zero_baseline
    *setup_test_config_for_explainer(
        explainer="integrated_gradients",
        expected=torch.tensor(
            [0.1054, 0.0862, 0.0843, 0.0868]
        ),  # these results might not be exactly reproducible across machines
        set_baselines_to_type="zero",
        explainer_kwargs={"n_steps": 200},
        delta=1e-3,
    ),
    # this setup is exactly the same as in the paper: https://arxiv.org/pdf/1711.00867
    # a 3-layer linear model is trained on MNIST, input invariance is computed for integrated_gradients
    # on 4 input samples. The expected output is [True, True, True, True] with black_baseline
    *setup_test_config_for_explainer(
        explainer="integrated_gradients",
        expected=torch.tensor([0.0, 0.0, 0.0, 0.0]),
        set_baselines_to_type="black",
        explainer_kwargs={"n_steps": 200},
    ),
    # here apply the same logic as in the paper: https://arxiv.org/pdf/1711.00867
    # a 3-layer linear model is trained on MNIST, input invariance is computed for occlusion
    # on 4 input samples. The expected output is [True, True, True, True] with black_baseline and
    # delta=1e-3. Note that these results were not in the paper, so this shows how the implementation
    # can be used for other explainers
    *setup_test_config_for_explainer(
        explainer="occlusion",
        expected=torch.tensor([0.0, 0.0, 0.0, 0.0]),
        set_baselines_to_type="black",
        explainer_kwargs={"sliding_window_shapes": (1, 4, 4), "strides": None},
    ),
    # here apply the same logic as in the paper: https://arxiv.org/pdf/1711.00867
    # a 3-layer linear model is trained on MNIST, input invariance is computed for lime
    # on 4 input samples. The expected output is [True, True, True, True] with black_baseline and
    # delta=1e-1. Note that these results were not in the paper, so this shows how the implementation
    # can be used for other explainers
    *setup_test_config_for_explainer(
        explainer="lime",
        expected=torch.tensor(
            [0.0151, 0.0157, 0.0138, 0.0280]
        ),  # these results might not be exactly reproducible across machines
        set_baselines_to_type="black",
        explainer_kwargs={"n_samples": 200, "weight_attributions": False},
        set_image_feature_mask=True,
        delta=1e-3,
    ),
]


@pytest.mark.metrics
@pytest.mark.parametrize(
    "explainer_based_metrics_runtime_test_configuration",
    test_configurations,
    ids=[f"{idx}_{config.test_name}" for idx, config in enumerate(test_configurations)],
    indirect=True,
)
def test_input_invariance(explainer_based_metrics_runtime_test_configuration):
    base_config, runtime_config, explainer = (
        explainer_based_metrics_runtime_test_configuration
    )

    viz = False

    def comparison_func(output: tuple, expected: tuple):
        output_invariance, expl_inputs, expl_shifted_inputs = output
        if viz:
            # here explanations can be visualized for debugging purposes
            for input, expl_input, expl_shifted_input in zip(
                base_config.inputs, expl_inputs, expl_shifted_inputs, strict=True
            ):
                visualize_attribution(input, expl_input, "Original")
                visualize_attribution(input, expl_shifted_input, "Shifted")
        _assert_tensor_almost_equal(
            output_invariance.float(),
            runtime_config.expected.float(),
            delta=runtime_config.delta,
            mode="mean",
        )

    _run_metric_test(
        base_config=base_config,
        runtime_config=runtime_config,
        explainer=explainer,
        metric_func=input_invariance,
        comparison_func=comparison_func,
        return_intermediate_results=True,
    )
