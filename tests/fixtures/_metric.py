import pytest  # noqa

from tests.metrics.utils import prepare_explanations
from tests.utils.configs import TestBaseConfig, TestRuntimeConfig


@pytest.fixture()
def metrics_runtime_test_configuration(request):
    # get the configs from the fixture request
    runtime_config: TestRuntimeConfig = request.param
    base_config: TestBaseConfig = request.getfixturevalue(runtime_config.target_fixture)
    if runtime_config.override_target is not None:
        base_config = base_config.model_copy(
            update={
                "explanation_inputs": base_config.explanation_inputs.model_copy(
                    update={"target": runtime_config.override_target}
                )
            }
        )
    explanation_step_outputs = prepare_explanations(
        base_config=base_config, runtime_config=runtime_config
    )
    explanation_step_outputs = explanation_step_outputs.to(runtime_config.device)
    yield base_config, runtime_config, explanation_step_outputs
