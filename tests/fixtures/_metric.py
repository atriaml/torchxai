import pytest  # noqa

from tests.metrics.utils import prepare_explanations
from tests.utils.common import _grid_segmenter
from tests.utils.configs import TestBaseConfig, TestRuntimeConfig


@pytest.fixture()
def metrics_runtime_test_configuration(request):
    # get the configs from the fixture request
    runtime_config: TestRuntimeConfig = request.param
    base_config: TestBaseConfig = request.getfixturevalue(runtime_config.target_fixture)
    is_multi_target = runtime_config.is_multi_target
    if runtime_config.override_target is not None:
        base_config = base_config.model_copy(
            update={
                "explanation_inputs": base_config.explanation_inputs.model_copy(
                    update={"target": runtime_config.override_target}
                )
            }
        )
    explanation_step_outputs = prepare_explanations(
        base_config=base_config,
        runtime_config=runtime_config,
        is_multi_target=is_multi_target,
    )
    explanation_step_outputs = explanation_step_outputs.to(runtime_config.device)
    if is_multi_target:
        targets_list = base_config.explanation_inputs.target
        assert len(explanation_step_outputs.attributions) == len(targets_list), (  # type: ignore
            "Number of explanations should be equal to the number of targets"
        )
    if runtime_config.set_image_feature_mask:
        base_config = base_config.model_copy(
            update={
                "explanation_inputs": base_config.explanation_inputs.model_copy(
                    update={
                        "feature_mask": _grid_segmenter(
                            base_config.explanation_inputs.inputs["0"], cell_size=32
                        )
                    }
                )
            }
        )
    yield base_config, runtime_config, explanation_step_outputs
