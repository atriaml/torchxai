import typing

import torch
from torch.nn.modules import Module

from torchxai.data_types import ExplanationStepOutputs
from torchxai.ignite._base import TorchXAIMetricBase
from torchxai.metrics.localization.attribution_localization import (
    attribution_localization,
)


class AttributionLocalization(TorchXAIMetricBase):
    def __init__(
        self,
        model: Module,
        with_amp: bool = False,
        device="cpu",
        positive_attributions: bool = True,
        weighted: bool = False,
    ):
        super().__init__(model, with_amp, device)
        self._positive_attributions = positive_attributions
        self._weighted = weighted

    def _update(self, output: ExplanationStepOutputs) -> dict[str, torch.Tensor]:
        assert output.feature_mask is not None, (
            "Feature mask is required for AttributionLocalization metric."
        )
        return typing.cast(
            dict,
            attribution_localization(
                attributions=output.attributions,
                feature_mask=output.feature_mask,
                weighted=self._weighted,
                positive_attributions=self._positive_attributions,
                return_dict=True,
            ),
        )
