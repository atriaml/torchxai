import typing

import torch
from captum.attr import Attribution
from torch.nn.modules import Module

from torchxai.data_types import ExplanationStepOutputs
from torchxai.explainers import Explainer
from torchxai.ignite._base import TorchXAIMetricBase
from torchxai.metrics._utils.perturbation import default_fixed_baseline_perturb_func
from torchxai.metrics.axiomatic.completeness import completeness
from torchxai.metrics.axiomatic.input_invariance import input_invariance
from torchxai.metrics.axiomatic.monotonicity_corr_and_non_sens import (
    monotonicity_corr_and_non_sens,
)


class CompletenessMetric(TorchXAIMetricBase):
    def _update(
        self, output: ExplanationStepOutputs, is_multi_target: bool = False
    ) -> dict[str, torch.Tensor]:
        return typing.cast(
            dict,
            completeness(
                forward_func=self._model,
                inputs=output.inputs,
                attributions=output.attributions,
                # NOTE:
                # notice metric baselines, explainer baselines must not be passed here
                # this baseline is used to compute the completeness score wrt to a baseline against already computed attributions
                # these contributions may be computed wrt different explainer baselines
                baselines=output.metric_baselines,
                additional_forward_args=output.additional_forward_args,
                target=output.target,  # type: ignore
                is_multi_target=is_multi_target,
                return_dict=True,
            ),
        )


class InputInvarianceMetric(TorchXAIMetricBase):
    def __init__(
        self,
        model: Module,
        explainer: Explainer | Attribution,
        with_amp: bool = False,
        device="cpu",
    ):
        self._explainer = explainer
        if isinstance(self._explainer, Explainer):
            assert self._explainer.model is model, (
                "Explainer model and metric model must be the same"
            )
        elif isinstance(self._explainer, Attribution):
            assert self._explainer.forward_func is model, (
                "Explainer forward function and metric model must be the same"
            )
        super().__init__(model, with_amp, device)

    def _update(
        self, output: ExplanationStepOutputs, is_multi_target: bool = False
    ) -> dict[str, torch.Tensor]:
        assert output.constant_shifts is not None, (
            "Constant shifts must be provided for input invariance metric"
        )
        assert output.input_layer_names is not None, (
            "Input layer names must be provided for input invariance metric"
        )
        return input_invariance(
            explainer=self._explainer,
            inputs=output.inputs,
            constant_shifts=output.constant_shifts,
            input_layer_names=output.input_layer_names,  # type: ignore
            is_multi_target=is_multi_target,
            return_intermediate_results=False,
            return_dict=True,
            # these are additionall explainer forward call args
            # NOTE:
            # notice explainer baselines here
            # this is used to compute attributions on the go during metric computation
            # this metric does not use metric baselines
            target=output.target,
            baselines=output.explainer_baselines,  # notice explainer baselines, this is different from metric baselines
            feature_mask=output.feature_mask,
            additional_forward_args=output.additional_forward_args,
        )


class MonotonicityCorrAndNonSensMetric(TorchXAIMetricBase):
    def __init__(
        self,
        model: torch.nn.Module,
        with_amp: bool = False,
        device="cpu",
        n_perturbations_per_feature: int = 10,
        max_features_processed_per_batch: int | None = None,
        percentage_feature_removal_per_step: float = 0,
        zero_attribution_threshold: float = 0.00001,
        zero_variance_threshold: float = 0.00001,
        use_percentage_attribution_threshold: bool = False,
        perturb_func: typing.Callable[
            ..., typing.Any
        ] = default_fixed_baseline_perturb_func(),
        return_ratio: bool = False,
    ):
        self._n_perturbations_per_feature = n_perturbations_per_feature
        self._max_features_processed_per_batch = max_features_processed_per_batch
        self._percentage_feature_removal_per_step = percentage_feature_removal_per_step
        self._zero_attribution_threshold = zero_attribution_threshold
        self._zero_variance_threshold = zero_variance_threshold
        self._use_percentage_attribution_threshold = (
            use_percentage_attribution_threshold
        )
        self._perturb_func = perturb_func
        self._return_ratio = return_ratio

        super().__init__(model=model, with_amp=with_amp, device=device)

    def _update(
        self, output: ExplanationStepOutputs, is_multi_target: bool = False
    ) -> dict[str, torch.Tensor]:
        return monotonicity_corr_and_non_sens(
            forward_func=self._model,
            inputs=output.inputs,
            attributions=output.attributions,
            # NOTE:
            # notice metric baselines, explainer baselines must not be passed here
            # this baseline is used to compute the completeness score wrt to a baseline against already computed attributions
            # these contributions may be computed wrt different explainer baselines
            baselines=output.metric_baselines,
            feature_mask=output.feature_mask,
            additional_forward_args=output.additional_forward_args,
            target=output.target,  # type: ignore
            frozen_features=output.frozen_features,
            perturb_func=self._perturb_func,
            n_perturbations_per_feature=self._n_perturbations_per_feature,
            max_features_processed_per_batch=self._max_features_processed_per_batch,  # type: ignore
            percentage_feature_removal_per_step=self._percentage_feature_removal_per_step,
            zero_attribution_threshold=self._zero_attribution_threshold,
            zero_variance_threshold=self._zero_variance_threshold,
            use_percentage_attribution_threshold=self._use_percentage_attribution_threshold,
            return_ratio=self._return_ratio,
            show_progress=False,
            return_intermediate_results=False,
            is_multi_target=is_multi_target,
            return_dict=True,
        )
