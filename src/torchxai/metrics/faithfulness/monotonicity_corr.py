from collections.abc import Callable
from typing import Any

import torch

from torchxai.data_types import (
    BaselineType,
    ExplanationTarget,
    NoTarget,
    TensorOrTupleOfTensorsGeneric,
)
from torchxai.metrics._utils.perturbation import default_fixed_baseline_perturb_func
from torchxai.metrics.axiomatic.monotonicity_corr_and_non_sens import (
    monotonicity_corr_and_non_sens,
)


def monotonicity_corr(
    forward_func: Callable,
    inputs: TensorOrTupleOfTensorsGeneric,
    attributions: list[TensorOrTupleOfTensorsGeneric] | TensorOrTupleOfTensorsGeneric,
    baselines: BaselineType = None,
    feature_mask: TensorOrTupleOfTensorsGeneric | None = None,
    additional_forward_args: Any = None,
    target: ExplanationTarget | list[ExplanationTarget] = NoTarget(),
    frozen_features: list[torch.Tensor] | None = None,
    perturb_func: Callable = default_fixed_baseline_perturb_func(),
    n_perturbations_per_feature: int = 10,
    max_features_processed_per_batch: int | None = None,
    percentage_feature_removal_per_step: float = 0.0,
    zero_attribution_threshold: float = 1e-5,
    zero_variance_threshold: float = 1e-5,
    use_percentage_attribution_threshold: bool = False,
    return_ratio: bool = True,
    show_progress: bool = True,
    multi_target: bool = False,
) -> torch.Tensor | list[torch.Tensor]:
    """Spearman correlation between attribution magnitudes and output variance under unordered perturbations. ↑ better.

    Faithfulness metric: measures whether features with higher attribution magnitudes actually produce
    larger changes in model output when perturbed. Alias for the first return value of
    `monotonicity_corr_and_non_sens`. See that function for full argument documentation.
    """
    result = monotonicity_corr_and_non_sens(
        forward_func=forward_func,
        inputs=inputs,
        attributions=attributions,
        baselines=baselines,
        feature_mask=feature_mask,
        additional_forward_args=additional_forward_args,
        target=target,
        frozen_features=frozen_features,
        perturb_func=perturb_func,
        n_perturbations_per_feature=n_perturbations_per_feature,
        max_features_processed_per_batch=max_features_processed_per_batch,
        percentage_feature_removal_per_step=percentage_feature_removal_per_step,
        zero_attribution_threshold=zero_attribution_threshold,
        zero_variance_threshold=zero_variance_threshold,
        use_percentage_attribution_threshold=use_percentage_attribution_threshold,
        return_ratio=return_ratio,
        show_progress=show_progress,
        multi_target=multi_target,
    )
    return result[0]
