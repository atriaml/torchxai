import pytest  # noqa

from tests.utils.configs import TestBaseConfig, TestRuntimeConfig


@pytest.fixture()
def explainer_runtime_test_configuration(request):
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
    yield base_config, runtime_config
