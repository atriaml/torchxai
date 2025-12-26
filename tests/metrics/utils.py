import pytest  # noqa

from tests.utils.common import _set_all_random_seeds
from tests.utils.configs import BaseTestConfig, RuntimeTestConfig
from tests.utils.explanation_steps import ExplanationStep, MultiTargetExplanationStep
from torchxai.explainers._factory import ExplainerFactory


def prepare_explanations(
    base_config: BaseTestConfig, runtime_config: RuntimeTestConfig
):
    _set_all_random_seeds(1234)
    assert isinstance(runtime_config.explainer_kwargs, dict), (
        "runtime_config.explainer_kwargs must be a dict"
    )
    explainer = ExplainerFactory.create(
        runtime_config.explainer,
        base_config.model,
        multi_target=runtime_config.multi_target,
        **runtime_config.explainer_kwargs,
    )
    if runtime_config.multi_target:
        explanation_step = MultiTargetExplanationStep(
            model=base_config.model, explainer=explainer, device=runtime_config.device
        )
    else:
        explanation_step = ExplanationStep(
            model=base_config.model, explainer=explainer, device=runtime_config.device
        )
    return explanation_step(explanation_inputs=base_config.explanation_inputs)


def prepare_explainer(base_config: BaseTestConfig, runtime_config: RuntimeTestConfig):
    assert isinstance(runtime_config.explainer_kwargs, dict), (
        "runtime_config.explainer_kwargs must be a dict"
    )
    explainer = ExplainerFactory.create(
        runtime_config.explainer,
        base_config.model,
        multi_target=runtime_config.multi_target,
        **runtime_config.explainer_kwargs,
    )
    return explainer
