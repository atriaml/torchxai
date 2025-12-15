import pytest
import torch
from pydantic import model_validator

from tests.utils.common import _assert_tensor_almost_equal, _grid_segmenter
from tests.utils.configs import TestBaseConfig, TestRuntimeConfig
from torchxai.data_types import ExplainerInputs, ExplanationState, ModelInputs
from torchxai.explainers.factory import ExplainerFactory
from torchxai.ignite._axiomatic import InputInvarianceMetric
from torchxai.metrics._utils.visualization import visualize_attribution
from torchxai.metrics.axiomatic.input_invariance import input_invariance


class MetricTestRuntimeConfig_(TestRuntimeConfig):
    model_type: str = "linear"
    train_and_eval_model: bool = False
    constant_shifts: torch.Tensor | None = None
    shifted_baselines: torch.Tensor | None = None
    set_baselines_to_type: str | None = None
    generate_feature_mask: bool = False
    visualize: bool = False

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


@pytest.fixture()
def metrics_runtime_test_configuration_with_explanation_state(request):
    runtime_config: MetricTestRuntimeConfig_ = request.param
    base_config: TestBaseConfig = request.getfixturevalue(
        runtime_config.target_fixture
    )(runtime_config.model_type, runtime_config.train_and_eval_model)
    assert base_config.model is not None
    assert base_config.inputs is not None
    if runtime_config.override_target is not None:
        base_config.target = runtime_config.override_target
    base_config.model.eval()
    base_config.put_to_device(runtime_config.device)
    explainer = ExplainerFactory.create(
        runtime_config.explainer, base_config.model, **runtime_config.explainer_kwargs
    )
    if runtime_config.use_captum_explainer:
        explainer = explainer._explanation_fn
    if runtime_config.generate_feature_mask:
        base_config.feature_mask = _grid_segmenter(base_config.inputs, 4)
    if runtime_config.set_baselines_to_type == "zero":
        base_config.baselines = 0
        runtime_config.shifted_baselines = 0
    elif runtime_config.set_baselines_to_type == "black":
        base_config.baselines = 0
        runtime_config.shifted_baselines = -1
    else:
        base_config.baselines = None
        runtime_config.shifted_baselines = None
    with torch.no_grad():
        model_inputs = base_config.inputs
        additional_forward_args = base_config.additional_forward_args
        if not isinstance(base_config.inputs, tuple):
            model_inputs = (base_config.inputs,)
        if additional_forward_args is not None:
            if not isinstance(additional_forward_args, tuple):
                additional_forward_args = additional_forward_args
        else:
            additional_forward_args = ()
        batch_size = model_inputs[0].shape[0]
        model_outputs = base_config.model(*model_inputs, *additional_forward_args)

    yield (
        base_config,
        runtime_config,
        base_config.model,
        explainer,
        ExplanationState(
            sample_id=[str(x) for x in range(batch_size)],
            explainer_inputs=ExplainerInputs(
                model_inputs=ModelInputs(
                    inputs=model_inputs, additional_forward_args=additional_forward_args
                ),
                explainer_baselines=base_config.baselines,
                metric_baselines=base_config.baselines,
                feature_mask=base_config.feature_mask,
                input_layer_names=base_config.input_layer_names,
                frozen_features=base_config.frozen_features,
                train_baselines=base_config.train_baselines,
                constant_shifts=runtime_config.constant_shifts
                if hasattr(runtime_config, "constant_shifts")
                else None,  # type: ignore
            ),
            target=base_config.target,  # type: ignore
            model_outputs=model_outputs,
            explanations=None,
        ),
    )


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
        ),
        MetricTestRuntimeConfig_(
            test_name=f"captum_{explainer}",
            target_fixture="mnist_train_configuration",
            explainer=explainer,
            train_and_eval_model=True,
            constant_shifts=torch.ones(1, 28, 28).unsqueeze(0),
            use_captum_explainer=True,
            **kwargs,
        ),
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
def test_non_sensitivity(metrics_runtime_test_configuration_with_explanation_state):
    base_config, runtime_config, model, explainer, explanation_state = (
        metrics_runtime_test_configuration_with_explanation_state
    )

    device = base_config.inputs.device
    kwargs = {"target": base_config.target}
    if base_config.feature_mask is not None:
        kwargs["feature_mask"] = base_config.feature_mask.to(device)
    if base_config.baselines is not None:
        kwargs["baselines"] = (
            base_config.baselines.to(device)
            if isinstance(base_config.baselines, torch.Tensor)
            else base_config.baselines
        )
    if runtime_config.shifted_baselines is not None:
        kwargs["shifted_baselines"] = (
            runtime_config.shifted_baselines.to(device)
            if isinstance(runtime_config.shifted_baselines, torch.Tensor)
            else runtime_config.shifted_baselines
        )
    if runtime_config.use_captum_explainer:
        runtime_config.explainer_kwargs.pop(
            "weight_attributions", None
        )  # this is only available in our implementation
    runtime_config.constant_shifts = tuple(
        x.to(device) for x in runtime_config.constant_shifts
    )
    output_invariance, expl_inputs, expl_shifted_inputs = input_invariance(
        explainer=explainer,
        inputs=base_config.inputs,
        constant_shifts=runtime_config.constant_shifts,
        input_layer_names=base_config.input_layer_names,
        return_intermediate_results=True,
        **kwargs,
        **(
            runtime_config.explainer_kwargs
            if runtime_config.use_captum_explainer
            else {}
        ),
    )

    if runtime_config.visualize:
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

    # first prepare the metric
    metric = InputInvarianceMetric(model=model, explainer=explainer, device=device)

    # now test via the Ignite Metric interface
    explanation_state.to(device)
    input_invarance_score = _run_metric_via_ignite(metric, explanation_state)[
        "input_invarance_score"
    ]
    _assert_tensor_almost_equal(
        input_invarance_score,
        runtime_config.expected.float(),
        delta=runtime_config.delta,
        mode="mean",
    )
