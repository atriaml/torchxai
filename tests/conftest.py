import logging

from tests.utils.common import _set_all_random_seeds

from .fixtures._explainer_fixtures import *  # noqa: F403, F401
from .fixtures._models import *  # noqa: F403, F401
from .fixtures._trainers import *  # noqa: F403, F401

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def pytest_runtest_setup():
    _set_all_random_seeds(1234)


# @pytest.fixture()
# def metrics_runtime_test_configuration(request):
#     runtime_config: TestRuntimeConfig = request.param
#     base_config: TestBaseConfig = request.getfixturevalue(runtime_config.target_fixture)
#     if runtime_config.override_target is not None:
#         base_config.target = runtime_config.override_target
#     base_config.model.eval()
#     base_config.put_to_device(runtime_config.device)
#     explainer = ExplainerFactory.create(
#         runtime_config.explainer, base_config.model, **runtime_config.explainer_kwargs
#     )
#     explanations = _run_explainer_forward(
#         explainer=explainer,
#         inputs=base_config.inputs,
#         additional_forward_args=base_config.additional_forward_args,
#         baselines=base_config.baselines,
#         train_baselines=base_config.train_baselines,
#         feature_mask=base_config.feature_mask,
#         target=base_config.target,
#         multiply_by_inputs=base_config.multiply_by_inputs,
#         use_captum_explainer=runtime_config.use_captum_explainer,
#         **runtime_config.explainer_kwargs,
#     )
#     yield base_config, runtime_config, explanations


# @pytest.fixture()
# def metrics_runtime_test_configuration_with_explanation_state(request):
#     # get the configs from the fixture request
#     runtime_config: TestRuntimeConfig = request.param
#     base_config: TestBaseConfig = request.getfixturevalue(runtime_config.target_fixture)

#     # validate configs
#     assert base_config.model is not None
#     assert base_config.model_inputs.explained_features is not None

#     # override target if specified
#     if runtime_config.override_target is not None:
#         base_config = base_config.model_copy(
#             update={"target": runtime_config.override_target}
#         )

#     # run _run_model_forward
#     model_outputs = _run_model_forward(
#         model=base_config.model,
#         model_inputs=base_config.model_inputs,
#         device=runtime_config.device,
#     )

#     # set up explainer
#     explainer = ExplainerFactory.create(
#         runtime_config.explainer, base_config.model, **runtime_config.explainer_kwargs
#     )
#     explanations = _run_explainer_forward(
#         explainer=explainer,
#         inputs=base_config.model_inputs.explained_features,
#         additional_forward_args=base_config.model_inputs.additional_forward_args,
#         baselines=base_config.explainer_step_inputs.baselines,
#         train_baselines=base_config.explainer_step_inputs.train_baselines,
#         feature_mask=base_config.explainer_step_inputs.feature_masks,
#         target=base_config.target,
#         multiply_by_inputs=base_config.multiply_by_inputs,
#         use_captum_explainer=runtime_config.use_captum_explainer,
#         **runtime_config.explainer_kwargs,
#     )
#     if isinstance(explanations, torch.Tensor):
#         explanations = (explanations,)

#     yield (
#         base_config,
#         runtime_config,
#         base_config.model,
#         explainer,
#         ExplanationState(
#             sample_id=[str(x) for x in range(batch_size)],
#             model_inputs=ModelInputs(
#                 explained_features=model_inputs,
#                 additional_forward_args=additional_forward_args,
#             ),
#             explainer_inputs=ExplainerInputs(
#                 baselines=base_config.baselines,
#                 feature_masks=base_config.feature_mask,
#                 train_baselines=base_config.train_baselines,
#             ),
#             metric_inputs=MetricInputs(
#                 baselines=base_config.baselines,
#                 shift_baselines=runtime_config.metric_shift_baselines
#                 if hasattr(runtime_config, "metric_shift_baselines")
#                 else None,  # type: ignore,
#                 feature_masks=base_config.feature_mask,
#                 input_layer_names=base_config.input_layer_names,
#                 frozen_features=base_config.frozen_features,
#                 train_baselines=base_config.train_baselines,
#                 constant_shifts=runtime_config.constant_shifts
#                 if hasattr(runtime_config, "constant_shifts")
#                 else None,  # type: ignore
#             ),
#             target=base_config.target,  # type: ignore
#             model_outputs=model_outputs,
#             explanations=explanations,
#         ),
#     )
