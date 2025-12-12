import pytest  # noqa

from tests.utils.configs import TestBaseConfig, TestRuntimeConfig


@pytest.fixture()
def explainer_runtime_test_configuration(request):
    runtime_config: TestRuntimeConfig = request.param
    base_config: TestBaseConfig = request.getfixturevalue(runtime_config.target_fixture)
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
        base_config.model_validate(base_config, strict=True)
    yield base_config, runtime_config
