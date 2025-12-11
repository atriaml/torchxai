import typing

import torch
from torch.nn.modules import Module

from torchxai.data_types import ExplanationStepOutputs
from torchxai.ignite._base import TorchXAIMetricBase
from torchxai.metrics._utils.perturbation import default_fixed_baseline_perturb_func
from torchxai.metrics.complexity.complexity_entropy import (
    complexity_entropy,
    complexity_entropy_feature_grouped,
)
from torchxai.metrics.complexity.complexity_sundararajan import (
    complexity_sundararajan,
    complexity_sundararajan_feature_grouped,
)
from torchxai.metrics.complexity.effective_complexity import effective_complexity
from torchxai.metrics.complexity.sparseness import (
    sparseness,
    sparseness_feature_grouped,
)


class ComplexityEntropy(TorchXAIMetricBase):
    def __init__(
        self,
        model: Module,
        with_amp: bool = False,
        device="cpu",
        group_features: bool = False,
    ):
        super().__init__(model, with_amp, device)
        self._group_features = group_features

    def _update(
        self, output: ExplanationStepOutputs, is_multi_target: bool = False
    ) -> dict[str, torch.Tensor]:
        if self._group_features:
            return typing.cast(
                dict,
                complexity_entropy_feature_grouped(
                    attributions=output.attributions,
                    feature_mask=output.feature_masks,
                    is_multi_target=is_multi_target,
                    return_dict=True,
                ),
            )
        else:
            return typing.cast(
                dict,
                complexity_entropy(
                    attributions=output.attributions,
                    is_multi_target=is_multi_target,
                    return_dict=True,
                ),
            )


class ComplexitySundranajan(TorchXAIMetricBase):
    def __init__(
        self,
        model: Module,
        with_amp: bool = False,
        device="cpu",
        group_features: bool = False,
        eps: float = 0.00001,
        normalize_attribution: bool = True,
    ):
        super().__init__(model, with_amp, device)
        self._group_features = group_features
        self._eps = eps
        self._normalize_attribution = normalize_attribution

    def _update(
        self, output: ExplanationStepOutputs, is_multi_target: bool = False
    ) -> dict[str, torch.Tensor]:
        if self._group_features:
            return typing.cast(
                dict,
                complexity_sundararajan_feature_grouped(
                    attributions=output.attributions,
                    feature_mask=output.feature_masks,
                    eps=self._eps,
                    normalize_attribution=self._normalize_attribution,
                    is_multi_target=is_multi_target,
                    return_dict=True,
                ),
            )
        else:
            return typing.cast(
                dict,
                complexity_sundararajan(
                    attributions=output.attributions,
                    eps=self._eps,
                    normalize_attribution=self._normalize_attribution,
                    is_multi_target=is_multi_target,
                    return_dict=True,
                ),
            )


class Sparseness(TorchXAIMetricBase):
    def __init__(
        self,
        model: Module,
        with_amp: bool = False,
        device="cpu",
        group_features: bool = False,
    ):
        super().__init__(model, with_amp, device)
        self._group_features = group_features

    def _update(
        self, output: ExplanationStepOutputs, is_multi_target: bool = False
    ) -> dict[str, torch.Tensor]:
        if self._group_features:
            return typing.cast(
                dict,
                sparseness_feature_grouped(
                    attributions=output.attributions,
                    feature_mask=output.feature_masks,
                    is_multi_target=is_multi_target,
                    return_dict=True,
                ),
            )
        else:
            return typing.cast(
                dict,
                sparseness(
                    attributions=output.attributions,
                    is_multi_target=is_multi_target,
                    return_dict=True,
                ),
            )


class EffectiveComplexity(TorchXAIMetricBase):
    def __init__(
        self,
        model: torch.nn.Module,
        with_amp: bool = False,
        device="cpu",
        n_perturbations_per_feature: int = 10,
        max_features_processed_per_batch: int | None = None,
        percentage_feature_removal_per_step: float = 0,
        zero_variance_threshold: float = 0.01,
        perturb_func: typing.Callable[
            ..., typing.Any
        ] = default_fixed_baseline_perturb_func(),
        return_ratio: bool = False,
    ):
        self._n_perturbations_per_feature = n_perturbations_per_feature
        self._max_features_processed_per_batch = max_features_processed_per_batch
        self._percentage_feature_removal_per_step = percentage_feature_removal_per_step
        self._zero_variance_threshold = zero_variance_threshold
        self._perturb_func = perturb_func
        self._return_ratio = return_ratio

        super().__init__(model=model, with_amp=with_amp, device=device)

    def _update(
        self, output: ExplanationStepOutputs, is_multi_target: bool = False
    ) -> dict[str, torch.Tensor]:
        return effective_complexity(
            forward_func=self._model,
            inputs=output.inputs,
            attributions=output.attributions,
            # NOTE:
            # notice metric baselines, explainer baselines must not be passed here
            # this baseline is used to compute the completeness score wrt to a baseline against already computed attributions
            # these contributions may be computed wrt different explainer baselines
            baselines=output.metric_baselines,
            feature_mask=output.feature_masks,
            additional_forward_args=output.additional_forward_args,
            target=output.target,  # type: ignore
            perturb_func=self._perturb_func,
            n_perturbations_per_feature=self._n_perturbations_per_feature,
            max_features_processed_per_batch=self._max_features_processed_per_batch,  # type: ignore
            percentage_feature_removal_per_step=self._percentage_feature_removal_per_step,
            frozen_features=output.frozen_features,
            zero_variance_threshold=self._zero_variance_threshold,
            return_ratio=self._return_ratio,
            is_multi_target=is_multi_target,
            show_progress=False,
            return_intermediate_results=False,
            return_dict=True,
        )
