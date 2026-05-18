import pytest

from tests.fixtures._metric import _run_metric_test_simple_mt
from tests.utils.common import _assert_tensor_almost_equal
from tests.utils.configs import RuntimeTestConfig
from torchxai.data_types._target import SingleTargetAcrossBatch
from torchxai.metrics.axiomatic.input_invariance import input_invariance


def setup_test_config_for_explainer(explainer, **kwargs):
    return [
        RuntimeTestConfig(
            test_name="compare_multi_target_to_single_target",
            target_fixture="mnist_train_configuration",
            explainer=explainer,
            train_and_eval_model=True,
            override_target=[
                SingleTargetAcrossBatch(index=0),
                SingleTargetAcrossBatch(index=1),
                SingleTargetAcrossBatch(index=2),
            ],
            expected=None,
            delta=1e-8,
            multi_target=True,
            **kwargs,
        )
    ]


test_configurations = [
    *setup_test_config_for_explainer(explainer="saliency"),
    *setup_test_config_for_explainer(explainer="input_x_gradient"),
    *setup_test_config_for_explainer(
        explainer="integrated_gradients",
        set_baselines_to_type="zero",
        explainer_kwargs={"n_steps": 200},
    ),
    *setup_test_config_for_explainer(
        explainer="integrated_gradients",
        set_baselines_to_type="black",
        explainer_kwargs={"n_steps": 200},
    ),
    *setup_test_config_for_explainer(
        explainer="occlusion",
        set_baselines_to_type="black",
        sliding_window_shapes=(1, 4, 4),
        strides=None,
    ),
]


@pytest.mark.metrics
@pytest.mark.parametrize(
    "explainer_based_metrics_runtime_test_configuration",
    test_configurations,
    ids=[f"{idx}_{config.test_name}" for idx, config in enumerate(test_configurations)],
    indirect=True,
)
def test_input_invariance(explainer_based_metrics_runtime_test_configuration):
    base_config, runtime_config, explainer = (
        explainer_based_metrics_runtime_test_configuration
    )

    def comparison_func(output: list, expected: list):
        for to, te in zip(output, expected, strict=True):
            _assert_tensor_almost_equal(to, te, delta=runtime_config.delta, mode="mean")

    _run_metric_test_simple_mt(
        base_config=base_config,
        runtime_config=runtime_config,
        metric_func=input_invariance,
        comparison_func=comparison_func,
        explainer=explainer,
        return_intermediate_results=True,
    )
