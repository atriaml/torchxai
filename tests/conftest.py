import logging

import pytest
import torch

from tests.helpers.basic_models import (
    BasicModel7_ReluMultiTensor,
    BasicModel7_SumMultiTensor,
)
from tests.utils.common import (
    _run_explainer_forward,
    mnist_trainer,
    set_all_random_seeds,
)
from tests.utils.configs import TestBaseConfig, TestRuntimeConfig
from torchxai.data_types import ExplanationState, MetricInputs
from torchxai.explainers.factory import ExplainerFactory

from .fixtures._models import *  # noqa: F403, F401

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def pytest_runtest_setup():
    set_all_random_seeds(1234)


@pytest.fixture()
def mnist_train_configuration():
    def _mnist_train_configuration(model_type: str, train_and_eval_model: bool):
        return mnist_trainer(model_type, train_and_eval_model)

    yield _mnist_train_configuration


@pytest.fixture()
def metrics_runtime_test_configuration(request):
    runtime_config: TestRuntimeConfig = request.param
    base_config: TestBaseConfig = request.getfixturevalue(runtime_config.target_fixture)
    if runtime_config.override_target is not None:
        base_config.target = runtime_config.override_target
    base_config.model.eval()
    base_config.put_to_device(runtime_config.device)
    explainer = ExplainerFactory.create(
        runtime_config.explainer, base_config.model, **runtime_config.explainer_kwargs
    )
    explanations = _run_explainer_forward(
        explainer=explainer,
        inputs=base_config.inputs,
        additional_forward_args=base_config.additional_forward_args,
        baselines=base_config.baselines,
        train_baselines=base_config.train_baselines,
        feature_mask=base_config.feature_mask,
        target=base_config.target,
        multiply_by_inputs=base_config.multiply_by_inputs,
        use_captum_explainer=runtime_config.use_captum_explainer,
        **runtime_config.explainer_kwargs,
    )
    yield base_config, runtime_config, explanations


@pytest.fixture()
def metrics_runtime_test_configuration_with_explanation_state(request):
    # get the configs from the fixture request
    runtime_config: TestRuntimeConfig = request.param
    base_config: TestBaseConfig = request.getfixturevalue(runtime_config.target_fixture)

    # validate configs
    assert base_config.model is not None
    assert base_config.model_inputs.explained_features is not None

    # override target if specified
    if runtime_config.override_target is not None:
        base_config = base_config.model_copy(
            update={"target": runtime_config.override_target}
        )

    # run _run_model_forward
    model_outputs = _run_model_forward(
        model=base_config.model,
        model_inputs=base_config.model_inputs,
        device=runtime_config.device,
    )

    # set up explainer
    explainer = ExplainerFactory.create(
        runtime_config.explainer, base_config.model, **runtime_config.explainer_kwargs
    )
    explanations = _run_explainer_forward(
        explainer=explainer,
        inputs=base_config.model_inputs.explained_features,
        additional_forward_args=base_config.model_inputs.additional_forward_args,
        baselines=base_config.explainer_step_inputs.baselines,
        train_baselines=base_config.explainer_step_inputs.train_baselines,
        feature_mask=base_config.explainer_step_inputs.feature_masks,
        target=base_config.target,
        multiply_by_inputs=base_config.multiply_by_inputs,
        use_captum_explainer=runtime_config.use_captum_explainer,
        **runtime_config.explainer_kwargs,
    )
    if isinstance(explanations, torch.Tensor):
        explanations = (explanations,)

    yield (
        base_config,
        runtime_config,
        base_config.model,
        explainer,
        ExplanationState(
            sample_id=[str(x) for x in range(batch_size)],
            model_inputs=ModelInputs(
                explained_features=model_inputs,
                additional_forward_args=additional_forward_args,
            ),
            explainer_inputs=ExplainerInputs(
                baselines=base_config.baselines,
                feature_masks=base_config.feature_mask,
                train_baselines=base_config.train_baselines,
            ),
            metric_inputs=MetricInputs(
                baselines=base_config.baselines,
                shift_baselines=runtime_config.metric_shift_baselines
                if hasattr(runtime_config, "metric_shift_baselines")
                else None,  # type: ignore,
                feature_masks=base_config.feature_mask,
                input_layer_names=base_config.input_layer_names,
                frozen_features=base_config.frozen_features,
                train_baselines=base_config.train_baselines,
                constant_shifts=runtime_config.constant_shifts
                if hasattr(runtime_config, "constant_shifts")
                else None,  # type: ignore
            ),
            target=base_config.target,  # type: ignore
            model_outputs=model_outputs,
            explanations=explanations,
        ),
    )


@pytest.fixture()
def explainer_metrics_runtime_test_configuration(request):
    runtime_config: TestRuntimeConfig = request.param
    base_config: TestBaseConfig = request.getfixturevalue(runtime_config.target_fixture)
    if runtime_config.override_target is not None:
        base_config = base_config.model_copy(
            update={"target": runtime_config.override_target}
        )
    base_config.model.eval()
    base_config.put_to_device(runtime_config.device)
    explainer = ExplainerFactory.create(
        runtime_config.explainer, base_config.model, **runtime_config.explainer_kwargs
    )
    yield base_config, runtime_config, explainer


@pytest.fixture()
def explainer_runtime_test_configuration(request):
    runtime_config: TestRuntimeConfig = request.param
    base_config: TestBaseConfig = request.getfixturevalue(runtime_config.target_fixture)
    if runtime_config.override_target is not None:
        base_config = base_config.model_copy(
            update={"target": runtime_config.override_target}
        )

    yield base_config, runtime_config


@pytest.fixture()
def multi_modal_sequence_sum():
    def test_sequence_tensor(size=12, embedding_size=3):
        return (
            torch.tensor([0] + list(range(3, size + 3)) + [1, 2])
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(1, embedding_size, size + 3)
            .repeat(1, 1, 1)
            .permute(0, 2, 1)
        ).float()

    def test_image(size=9):
        return (
            torch.arange(size)
            .view(1, 1, 3, 3)
            .repeat_interleave(2, dim=-1)
            .repeat_interleave(2, dim=-2)
            .float()
        )

    size = 6
    sequence1 = test_sequence_tensor(size)
    sequence2 = test_sequence_tensor(size) + size + 3
    sequence3 = test_sequence_tensor(size) + (size + 3) * 2
    image1 = test_image() + (size + 3) * 3
    feature_mask = (
        sequence1.clone().long(),
        sequence2.clone().long(),
        sequence3.clone().long(),
        image1.clone().long(),
    )
    frozen_features = torch.tensor([0, 1, 2, 9, 10, 11, 18, 19, 20])
    n_features = (
        torch.cat([x.flatten() for x in feature_mask]).unique().numel()
        - frozen_features.numel()
    )
    inputs = (sequence1, sequence2, sequence3, image1)
    total_sum = sum(x.sum() for x in inputs)
    inputs = tuple(x / total_sum for x in inputs)
    target = None

    yield TestBaseConfig(
        model=BasicModel7_SumMultiTensor(),
        model_inputs=ModelInputs(explained_features=inputs),
        target=target,
        explainer_step_inputs=ExplainerInputs(
            baselines=tuple(torch.zeros_like(x) for x in inputs),
            feature_masks=feature_mask,
        ),
        metric_inputs=MetricInputs(
            baselines=tuple(torch.zeros_like(x) for x in inputs),
            frozen_features=[torch.tensor([0, 1, 2, 9, 10, 11, 18, 19, 20])],
        ),
    )


@pytest.fixture()
def multi_modal_sequence_relu():
    def test_sequence_tensor(size=12, embedding_size=4):
        return (
            torch.tensor([0] + list(range(3, size + 3)) + [1, 2])
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(1, embedding_size, size + 3)
            .repeat(1, 1, 1)
            .permute(0, 2, 1)
        ).float()

    def test_image(size=9):
        return (
            torch.arange(size)
            .view(1, 1, 3, 3)
            .repeat_interleave(2, dim=-1)
            .repeat_interleave(2, dim=-2)
            .float()
        )

    size = 6
    sequence1 = test_sequence_tensor(size)
    sequence2 = test_sequence_tensor(size) + size + 3
    sequence3 = test_sequence_tensor(size) + (size + 3) * 2
    image1 = test_image() + (size + 3) * 3
    feature_mask = (
        sequence1.clone().long(),
        sequence2.clone().long(),
        sequence3.clone().long(),
        image1.clone().long(),
    )
    frozen_features = torch.tensor([0, 1, 2, 9, 10, 11, 18, 19, 20])
    n_features = (
        torch.cat([x.flatten() for x in feature_mask]).unique().numel()
        - frozen_features.numel()
    )
    inputs = (sequence1, sequence2, sequence3, image1)
    mean = torch.cat(tuple(x.flatten() for x in inputs)).mean()
    std = torch.cat(tuple(x.flatten() for x in inputs)).std()
    inputs = tuple((x - mean) / std for x in inputs)
    target = None

    yield TestBaseConfig(
        model=BasicModel7_ReluMultiTensor(),
        inputs=inputs,
        target=target,
        feature_mask=feature_mask,
        baselines=tuple(torch.zeros_like(x) for x in inputs),
        frozen_features=[torch.tensor([0, 1, 2, 9, 10, 11, 18, 19, 20])],
        n_features=n_features,
    )


def _run_metric_via_ignite(metric, explanation_state):
    """Helper function to run a metric via the Ignite Engine interface.

    Args:
        metric: The Ignite metric to evaluate
        explanation_state: The explanation state to process

    Returns:
        The metric output from the engine state
    """
    from ignite.engine import Engine

    def explanation_step(engine, batch):
        return explanation_state

    engine = Engine(explanation_step)
    metric.attach(engine, "metric")
    state = engine.run([None], max_epochs=1)
    print("Engine state metrics:", state.metrics)
    return state.metrics["metric"][0]
