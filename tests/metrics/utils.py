import pytest  # noqa

from tests.utils.common import _set_all_random_seeds
from tests.utils.configs import TestBaseConfig, TestRuntimeConfig
from torchxai.explainers.factory import ExplainerFactory
from torchxai.ignite._explanation_step import (
    ExplanationStep,
    MultiTargetExplanationStep,
)


def prepare_explanations(
    base_config: TestBaseConfig,
    runtime_config: TestRuntimeConfig,
    is_multi_target: bool = False,
):
    _set_all_random_seeds(1234)
    assert isinstance(runtime_config.explainer_kwargs, dict), (
        "runtime_config.explainer_kwargs must be a dict"
    )
    explainer = ExplainerFactory.create(
        runtime_config.explainer, base_config.model, **runtime_config.explainer_kwargs
    )
    if is_multi_target:
        explanation_step = MultiTargetExplanationStep(
            model=base_config.model, explainer=explainer, device=runtime_config.device
        )
    else:
        explanation_step = ExplanationStep(
            model=base_config.model,
            explainer=explainer,
            device=runtime_config.device,
            use_captum_explainer=runtime_config.use_captum_explainer,
        )
    return explanation_step(
        explanation_inputs=base_config.explanation_inputs,
        metric_inputs=base_config.metric_inputs,
    )
