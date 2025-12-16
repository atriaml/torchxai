import pytest
import torch
from pydantic import model_validator

from tests.utils.common import _assert_tensor_almost_equal
from tests.utils.configs import TestRuntimeConfig
from torchxai.data_types import ExplanationStepOutputs
from torchxai.metrics.axiomatic.input_invariance import input_invariance


class MetricTestRuntimeConfig_(TestRuntimeConfig):
    model_type: str = "linear"
    train_and_eval_model: bool = False
    constant_shifts: tuple[torch.Tensor, ...] | None = None
    shifted_baselines: torch.Tensor | None = None
    set_baselines_to_type: str | None = None
    generate_feature_mask: bool = False
    visualize: bool = False
    image_feature_mask_cell_size: int = 4

    @model_validator(mode="before")
    @classmethod
    def validate_fields(cls, values):
        if "constant_shifts" not in values or values["constant_shifts"] is None:
            raise ValueError("constant_shifts must be provided")
        if isinstance(values["constant_shifts"], torch.Tensor):
            values["constant_shifts"] = (values["constant_shifts"],)
        if "set_baselines_to_type" in values:
            assert values["set_baselines_to_type"] in ["zero", "black", None], (
                "set_baselines_to_type must be one of 'zero', 'black', or None"
            )
        return values


def setup_test_config_for_explainer(explainer, **kwargs):
    return [
        MetricTestRuntimeConfig_(
            test_name=f"{explainer}",
            target_fixture="mnist_train_configuration",
            explainer=explainer,
            train_and_eval_model=True,
            constant_shifts=torch.ones(1, 28, 28).unsqueeze(0),
            use_captum_explainer=False,
            **kwargs,
        )
        # MetricTestRuntimeConfig_(
        #     test_name=f"captum_{explainer}",
        #     target_fixture="mnist_train_configuration",
        #     explainer=explainer,
        #     train_and_eval_model=True,
        #     constant_shifts=torch.ones(1, 28, 28).unsqueeze(0),
        #     use_captum_explainer=True,
        #     **kwargs,
        # ),
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
        generate_feature_mask=True,
        delta=1e-3,
    ),
]


@pytest.mark.metrics
@pytest.mark.parametrize(
    "metrics_runtime_test_configuration",
    test_configurations,
    ids=[f"{idx}_{config.test_name}" for idx, config in enumerate(test_configurations)],
    indirect=True,
)
def test_input_invariance(metrics_runtime_test_configuration):
    base_config, runtime_config, explanation_step_outputs = (
        metrics_runtime_test_configuration
    )
    explanation_step_outputs: ExplanationStepOutputs

    # device = base_config.inputs.device
    # kwargs = {"target": base_config.target}
    # if base_config.feature_mask is not None:
    #     kwargs["feature_mask"] = base_config.feature_mask.to(device)
    output_invariance, expl_inputs, expl_shifted_inputs = input_invariance(
        explainer=explanation_step_outputs.explainer,
        inputs=explanation_step_outputs.inputs,
        constant_shifts=explanation_step_outputs.constant_shifts,
        input_layer_names=explanation_step_outputs.input_layer_names,
        shifted_baselines=explanation_step_outputs.metric_shift_baselines,
        return_intermediate_results=True,
        **explanation_step_outputs.explanation_state.explanation_inputs.model_dump(),
    )

    # if runtime_config.visualize:
    #     # here explanations can be visualized for debugging purposes
    #     for input, expl_input, expl_shifted_input in zip(
    #         base_config.inputs, expl_inputs, expl_shifted_inputs, strict=True
    #     ):
    #         visualize_attribution(input, expl_input, "Original")
    #         visualize_attribution(input, expl_shifted_input, "Shifted")

    _assert_tensor_almost_equal(
        output_invariance.float(),
        runtime_config.expected.float(),
        delta=runtime_config.delta,
        mode="mean",
    )

    # # first prepare the metric
    # metric = InputInvarianceMetric(model=model, explainer=explainer, device=device)

    # # now test via the Ignite Metric interface
    # explanation_state.to(device)
    # input_invarance_score = _run_metric_via_ignite(metric, explanation_state)[
    #     "input_invarance_score"
    # ]
    # _assert_tensor_almost_equal(
    #     input_invarance_score,
    #     runtime_config.expected.float(),
    #     delta=runtime_config.delta,
    #     mode="mean",
    # )
