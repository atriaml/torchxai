import pytest  # noqa

from tests.metrics.utils import prepare_explanations
from tests.utils.common import _grid_segmenter
from tests.utils.configs import TestBaseConfig, TestRuntimeConfig


@pytest.fixture()
def metrics_runtime_test_configuration(request):
    # get the configs from the fixture request
    runtime_config: TestRuntimeConfig = request.param
    base_config: TestBaseConfig = request.getfixturevalue(runtime_config.target_fixture)
    multi_target = runtime_config.multi_target
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
    explanation_step_outputs = prepare_explanations(
        base_config=base_config,
        runtime_config=runtime_config,
        multi_target=multi_target,
    )
    if multi_target:
        targets_list = base_config.explanation_inputs.target
        assert len(explanation_step_outputs.attributions) == len(targets_list), (  # type: ignore
            "Number of explanations should be equal to the number of targets"
        )
    if runtime_config.set_image_feature_mask:
        metric_inputs = base_config.metric_inputs.model_copy(
            update={
                "feature_mask": _grid_segmenter(
                    base_config.explanation_inputs.inputs["0"],
                    cell_size=runtime_config.image_feature_mask_cell_size,
                )
            }
        )
        base_config = base_config.model_copy(update={"metric_inputs": metric_inputs})
        base_config.model_validate(base_config, strict=True)
        explanation_step_outputs = explanation_step_outputs.model_copy(
            update={"metric_inputs": metric_inputs}
        )
    explanation_step_outputs = explanation_step_outputs.to(runtime_config.device)
    yield base_config, runtime_config, explanation_step_outputs
