import pytest  # noqa

from tests.utils.common import _compare_explanation_per_target, _set_all_random_seeds
from tests.utils.configs import ExplainersTestRuntimeConfig, BaseTestConfig
from torchxai.data_types import ExplanationTarget
from torchxai.explainers._factory import ExplainerFactory
from tests.utils.explanation_steps import ExplanationStep, MultiTargetExplanationStep


def make_config_for_explainer_with_grad_batch_size(*args, **kwargs):
    return [
        ExplainersTestRuntimeConfig(
            *args,
            **kwargs,
            explainer_kwargs={"grad_batch_size": grad_batch_size},
            test_name=f"grad_batch_size{grad_batch_size}",
        )
        for grad_batch_size in [1, 10, 64]
    ]


def make_config_for_explainer_with_internal_and_grad_batch_size(*args, **kwargs):
    internal_batch_sizes = kwargs.pop("internal_batch_sizes", [None])
    return [
        ExplainersTestRuntimeConfig(
            *args,
            **kwargs,
            explainer_kwargs={
                "internal_batch_size": internal_batch_size,
                "grad_batch_size": grad_batch_size,
            },
            test_name=f"internal_batch_size_{internal_batch_size}_grad_batch_size_{grad_batch_size}",
        )
        for grad_batch_size in [1, 10, 64]
        for internal_batch_size in internal_batch_sizes
    ]


def make_config_for_explainer_with_internal_batch_size(*args, **kwargs):
    internal_batch_sizes = kwargs.pop("internal_batch_sizes", [None])
    return [
        ExplainersTestRuntimeConfig(
            *args,
            **kwargs,
            explainer_kwargs={"internal_batch_size": internal_batch_size},
            test_name=f"internal_batch_size_{internal_batch_size}",
        )
        for internal_batch_size in internal_batch_sizes
    ]


def _format_to_list_if_not_list(obj):
    if not isinstance(obj, list):
        return [obj]
    return obj


def run_explainer_test_with_config(
    base_config: BaseTestConfig, runtime_config: ExplainersTestRuntimeConfig
):
    # perform basic validation
    expected = _format_to_list_if_not_list(runtime_config.expected)
    target = _format_to_list_if_not_list(base_config.explanation_inputs.target)
    assert len(target) == len(expected), (
        "The number of targets must be equal to the number of expected outputs. Found "
        f"{target} targets and {expected} expected outputs."
    )

    # in the first pass we compute explanations for each target separately using single-target explainer
    single_target_explanations = []
    for curr_target, curr_expected in zip(target, expected, strict=True):
        assert isinstance(curr_target, ExplanationTarget), (
            f"The target must be of type BaseTarget, Got: {curr_target}"
        )

        explanations = run_single_test(
            base_config.model_copy(
                update={
                    "explanation_inputs": base_config.explanation_inputs.model_copy(
                        update={"target": curr_target}
                    )
                }
            ),
            runtime_config.model_copy(update={"expected": curr_expected}),
        )

        if runtime_config.check_multi_target_list_against_single_target:
            multi_target_explanations_2 = run_single_test(
                base_config.model_copy(
                    update={
                        "explanation_inputs": base_config.explanation_inputs.model_copy(
                            update={"target": [curr_target]}
                        )
                    }
                ),
                runtime_config.model_copy(update={"expected": [curr_expected]}),
                is_multi_target=True,
            )
            if multi_target_explanations_2 is None:
                assert explanations is None
            else:
                _compare_explanation_per_target(
                    multi_target_explanations_2[0],
                    explanations,  # type: ignore
                    delta=runtime_config.delta,
                    visualize=runtime_config.visualize,
                )

        single_target_explanations.append(explanations)

    if len(single_target_explanations) > 1:
        multi_target_explanations = run_single_test(
            base_config, runtime_config, is_multi_target=True
        )

        for multi_target_explanation, single_target_explanation in zip(
            multi_target_explanations,  # type: ignore
            single_target_explanations,
            strict=True,
        ):
            # target explanation in the list should match the single target explanations at the same index
            _compare_explanation_per_target(
                multi_target_explanation,
                single_target_explanation,
                delta=runtime_config.delta,
                visualize=runtime_config.visualize,
            )


def run_single_test(
    base_config: BaseTestConfig,
    runtime_config: ExplainersTestRuntimeConfig,
    is_multi_target: bool = False,
):
    _set_all_random_seeds(1234)

    explainer = ExplainerFactory.create(
        runtime_config.explainer,
        base_config.model,
        **(runtime_config.explainer_kwargs or {}),
    )
    if is_multi_target:
        explanation_step = MultiTargetExplanationStep(
            model=base_config.model, explainer=explainer, device=runtime_config.device
        )
    else:
        explanation_step = ExplanationStep(
            model=base_config.model, explainer=explainer, device=runtime_config.device
        )
    if runtime_config.throws_exception:
        with pytest.raises(Exception) as _:
            explanation_step(explanation_inputs=base_config.explanation_inputs)
        return

    explanations = explanation_step(
        explanation_inputs=base_config.explanation_inputs
    ).explanations

    has_expected = (
        runtime_config.expected is not None
        if not isinstance(runtime_config.expected, list)
        else all(v is not None for v in runtime_config.expected)
    )
    if has_expected:
        if is_multi_target:
            assert isinstance(explanations, list), (
                "The output explanations must be a list when is_multi_target is True"
            )
            assert isinstance(runtime_config.expected, list), (
                "The expected explanations must be a list when is_multi_target is True"
            )
            assert len(explanations) == len(runtime_config.expected), (
                "The number of output explanations must be equal to the number of expected outputs "
                "when is_multi_target is True"
            )

            for output_per_target, expected_per_target in zip(
                explanations, runtime_config.expected, strict=True
            ):
                _compare_explanation_per_target(
                    output_per_target, expected_per_target, delta=runtime_config.delta
                )
        else:
            _compare_explanation_per_target(
                explanations,
                runtime_config.expected,  # type: ignore
                delta=runtime_config.delta,  # type: ignore
            )
    return explanations
