import inspect
from collections.abc import Callable

import pytest
import torch  # noqa

from tests.metrics.utils import prepare_explainer, prepare_explanations
from tests.utils.common import _grid_segmenter, _set_all_random_seeds
from tests.utils.configs import BaseTestConfig, RuntimeTestConfig
from tests.utils.types import ExplanationStepOutputs
from torchxai.explainers._explainer import Explainer


def _get_metric_inputs(
    base_config: BaseTestConfig,
    runtime_config: RuntimeTestConfig,
    explanation_step_outputs: ExplanationStepOutputs | None = None,
    explainer: Explainer | None = None,
):
    kwargs = {
        "forward_func": base_config.model,
        "inputs": base_config.explanation_inputs.inputs,
        "attributions": explanation_step_outputs.explanations
        if explanation_step_outputs
        else None,
        "feature_mask": base_config.metric_inputs.feature_mask,
        "baselines": base_config.metric_inputs.baselines,
        "shift_baselines": base_config.metric_inputs.shift_baselines,
        "additional_forward_args": base_config.explanation_inputs.additional_forward_args,
        "target": base_config.explanation_inputs.target,
        "frozen_features": base_config.explanation_inputs.frozen_features,
        "explainer": explainer,
        "constant_shifts": base_config.metric_inputs.constant_shifts,
        "input_layer_names": base_config.metric_inputs.input_layer_names,
    }
    return kwargs


def _run_metric_test_simple(
    base_config: BaseTestConfig,
    runtime_config: RuntimeTestConfig,
    metric_func: Callable,
    comparison_func: Callable,
    explanation_step_outputs: ExplanationStepOutputs | None = None,
    explainer: Explainer | None = None,
    **other_kwargs,
):
    # yield base_config, runtime_config, explanation_step_outputs
    kwargs = _get_metric_inputs(base_config, runtime_config, explanation_step_outputs)
    kwargs = {
        k: v
        for k, v in kwargs.items()
        if k in inspect.signature(metric_func).parameters
    }
    if explainer is not None:
        kwargs["explainer"] = explainer
    kwargs.update(other_kwargs)
    metric_output = metric_func(**kwargs)
    comparison_func(metric_output, runtime_config.expected)


def _run_metric_test_looped(
    base_config: BaseTestConfig,
    runtime_config: RuntimeTestConfig,
    metric_func: Callable,
    comparison_func: Callable,
    explanation_step_outputs: ExplanationStepOutputs | None = None,
    explainer: Explainer | None = None,
    **other_kwargs,
):
    n_perturbations_per_feature = _format_to_list(
        runtime_config.n_perturbations_per_feature
    )
    max_features_processed_per_batch = _format_to_list(
        runtime_config.max_features_processed_per_batch
    )
    expected = _format_to_list(runtime_config.expected)

    assert len(n_perturbations_per_feature) == len(max_features_processed_per_batch)
    assert len(n_perturbations_per_feature) == len(expected) or len(expected) == 1
    for n_perturbs, max_features in zip(
        n_perturbations_per_feature, max_features_processed_per_batch, strict=True
    ):
        kwargs = _get_metric_inputs(
            base_config, runtime_config, explanation_step_outputs
        )
        kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in inspect.signature(metric_func).parameters
        }
        if explainer is not None:
            kwargs["explainer"] = explainer
        kwargs.update(other_kwargs)
        kwargs.update(
            {
                "n_perturbations_per_feature": n_perturbs,
                "max_features_processed_per_batch": max_features,
            }
        )

        metric_output = metric_func(**kwargs)
        comparison_func(metric_output, runtime_config.expected)


def _run_metric_test_simple_mt(
    base_config: BaseTestConfig,
    runtime_config: RuntimeTestConfig,
    metric_func: Callable,
    comparison_func: Callable,
    explanation_step_outputs: ExplanationStepOutputs | None = None,
    explainer: Explainer | None = None,
    **other_kwargs,
):
    kwargs = _get_metric_inputs(base_config, runtime_config, explanation_step_outputs)
    kwargs = {
        k: v
        for k, v in kwargs.items()
        if k in inspect.signature(metric_func).parameters
    }
    if explainer is not None:
        kwargs["explainer"] = explainer
    kwargs.update(other_kwargs)

    # multi target metric output
    if explainer is not None:
        explainer.multi_target = True
    multi_target_score = metric_func(**kwargs, multi_target=True)
    if not isinstance(multi_target_score, tuple):
        multi_target_score = (multi_target_score,)

    # per target metric outputs
    if explainer is not None:
        per_target_scores = []
        explainer.multi_target = False
        for tgt in kwargs.pop("target"):
            output = metric_func(target=tgt, **kwargs)
            per_target_scores.append(output)

    else:
        target = kwargs.pop("target", None)
        if target is not None:
            per_target_scores = []
            for attr, tgt in zip(kwargs.pop("attributions"), target, strict=True):
                output = metric_func(attributions=attr, target=tgt, **kwargs)
                per_target_scores.append(output)
        else:
            per_target_scores = []
            for attr in kwargs.pop("attributions"):
                output = metric_func(attributions=attr, **kwargs)
                per_target_scores.append(output)

    per_target_scores = [
        score if isinstance(score, tuple) else (score,) for score in per_target_scores
    ]

    # map list of tuples to tuple of lists
    per_target_scores = tuple(map(list, zip(*per_target_scores, strict=True)))

    comparison_func(per_target_scores, multi_target_score)


def _format_to_list(value):
    if not isinstance(value, list):
        return [value]
    return value


def _run_metric_test_looped_mt(
    base_config: BaseTestConfig,
    runtime_config: RuntimeTestConfig,
    metric_func: Callable,
    comparison_func: Callable,
    explanation_step_outputs: ExplanationStepOutputs | None = None,
    explainer: Explainer | None = None,
    seed_outside_loop: bool = False,
    **other_kwargs,
):
    n_perturbations_per_feature = _format_to_list(
        runtime_config.n_perturbations_per_feature
    )
    max_features_processed_per_batch = _format_to_list(
        runtime_config.max_features_processed_per_batch
    )
    expected = _format_to_list(runtime_config.expected)

    assert len(n_perturbations_per_feature) == len(max_features_processed_per_batch)
    assert len(n_perturbations_per_feature) == len(expected) or len(expected) == 1
    for n_perturbs, max_features in zip(
        n_perturbations_per_feature, max_features_processed_per_batch, strict=True
    ):
        kwargs = _get_metric_inputs(
            base_config, runtime_config, explanation_step_outputs
        )
        kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in inspect.signature(metric_func).parameters
        }
        if explainer is not None:
            kwargs["explainer"] = explainer
        kwargs.update(other_kwargs)
        kwargs.update(
            {
                "n_perturbations_per_feature": n_perturbs,
                "max_features_processed_per_batch": max_features,
            }
        )

        # multi target metric output
        if explainer is not None:
            explainer.multi_target = True
        _set_all_random_seeds(1234)
        multi_target_score = metric_func(**kwargs, multi_target=True)

        # per target metric outputs
        if explainer is not None:
            per_target_scores = []
            explainer.multi_target = False
            if seed_outside_loop:
                _set_all_random_seeds(1234)
            for tgt in kwargs.pop("target"):
                if not seed_outside_loop:
                    _set_all_random_seeds(1234)
                output = metric_func(target=tgt, **kwargs)
                per_target_scores.append(output)

        else:
            if seed_outside_loop:
                _set_all_random_seeds(1234)
            per_target_scores = []
            for attr, tgt in zip(
                kwargs.pop("attributions"), kwargs.pop("target"), strict=True
            ):
                if not seed_outside_loop:
                    _set_all_random_seeds(1234)
                output = metric_func(attributions=attr, target=tgt, **kwargs)
                per_target_scores.append(output)

        per_target_scores = [
            score if isinstance(score, tuple) else (score,)
            for score in per_target_scores
        ]

        # map list of tuples to tuple of lists
        per_target_scores = tuple(map(list, zip(*per_target_scores, strict=True)))

        # compare
        comparison_func(per_target_scores, multi_target_score)


def metrics_runtime_test_configuration_base(request):
    # get the configs from the fixture request
    runtime_config: RuntimeTestConfig = request.param
    if runtime_config.target_fixture == "mnist_train_configuration":
        # special case to train and eval model
        base_config: BaseTestConfig = request.getfixturevalue(
            "mnist_train_configuration"
        )(
            train_and_eval_model=runtime_config.train_and_eval_model,
            model_type=runtime_config.model_type,
        )
    else:
        base_config: BaseTestConfig = request.getfixturevalue(
            runtime_config.target_fixture
        )
    base_config = base_config.to(runtime_config.device)
    if runtime_config.override_target is not None:
        explanation_inputs = base_config.explanation_inputs.model_copy(
            update={"target": runtime_config.override_target}
        )
        explanation_inputs = explanation_inputs.model_validate(
            explanation_inputs, strict=True
        )
        base_config = base_config.model_copy(
            update={"explanation_inputs": explanation_inputs}
        )
        base_config = base_config.model_validate(base_config, strict=True)

    if runtime_config.set_baselines_to_type is not None:
        if runtime_config.set_baselines_to_type == "zero":
            metric_inputs = base_config.metric_inputs.model_copy(
                update={
                    "baselines": tuple(
                        torch.zeros_like(input)
                        for input in base_config.explanation_inputs.inputs
                    ),
                    "shift_baselines": tuple(
                        torch.zeros_like(input)
                        for input in base_config.explanation_inputs.inputs
                    ),
                }
            )
        elif runtime_config.set_baselines_to_type == "black":
            metric_inputs = base_config.metric_inputs.model_copy(
                update={
                    "baselines": tuple(
                        torch.zeros_like(input)
                        for input in base_config.explanation_inputs.inputs
                    ),
                    "shift_baselines": tuple(
                        torch.ones_like(input) * -1
                        for input in base_config.explanation_inputs.inputs
                    ),
                }
            )
        else:
            raise ValueError(
                f"Unknown set_baselines_to_type: {runtime_config.set_baselines_to_type}"
            )
        base_config = base_config.model_copy(update={"metric_inputs": metric_inputs})
        base_config = base_config.model_validate(base_config, strict=True)

    if runtime_config.set_image_feature_mask:
        metric_inputs = base_config.metric_inputs.model_copy(
            update={
                "feature_mask": _grid_segmenter(
                    base_config.explanation_inputs.inputs[0],
                    cell_size=runtime_config.image_feature_mask_cell_size,
                )
            }
        )
        base_config = base_config.model_copy(update={"metric_inputs": metric_inputs})
        base_config.model_validate(base_config, strict=True)
    return base_config, runtime_config


@pytest.fixture()
def metrics_runtime_test_configuration(request):
    # get the configs from the fixture request
    base_config, runtime_config = metrics_runtime_test_configuration_base(request)
    explanation_step_outputs = prepare_explanations(
        base_config=base_config, runtime_config=runtime_config
    )
    if runtime_config.multi_target:
        targets_list = base_config.explanation_inputs.target
        assert len(explanation_step_outputs.explanations) == len(targets_list), (  # type: ignore
            "Number of explanations should be equal to the number of targets"
        )
    explanation_step_outputs = explanation_step_outputs.to(runtime_config.device)
    yield base_config, runtime_config, explanation_step_outputs


@pytest.fixture()
def explainer_based_metrics_runtime_test_configuration(request):
    base_config, runtime_config = metrics_runtime_test_configuration_base(request)
    explainer = prepare_explainer(
        base_config=base_config, runtime_config=runtime_config
    )

    yield base_config, runtime_config, explainer
