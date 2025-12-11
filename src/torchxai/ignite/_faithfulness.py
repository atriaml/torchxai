import torch

from torchxai.data_types import ExplanationStepOutputs
from torchxai.ignite._base import TorchXAIMetricBase
from torchxai.metrics.faithfulness.aopc import aopc


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
        total_feature_bins: int | None = None,
        n_random_perms: int = 10,
        seed: int = 42,
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
        self._total_feature_bins = total_feature_bins
        self._n_random_perms = n_random_perms
        self._seed = seed
        self._return_ratio = return_ratio

        super().__init__(model=model, with_amp=with_amp, device=device)

    def _update(
        self, output: ExplanationStepOutputs, is_multi_target: bool = False
    ) -> dict[str, torch.Tensor]:
        return aopc(
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
            max_features_processed_per_batch=self._max_features_processed_per_batch,  # type: ignore
            total_feature_bins=self._total_feature_bins,
            frozen_features=output.frozen_features,
            n_random_perms=self._n_random_perms,
            seed=self._seed,
            show_progress=False,
            return_intermediate_results=False,
            is_multi_target=is_multi_target,
            return_dict=True,
        )
