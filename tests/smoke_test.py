"""Minimal smoke test — run after building the wheel/sdist to verify the package installs correctly."""
import torchxai
import torchxai.explainers
from torchxai.explainers import (
    DeepLiftExplainer,
    DeepLiftShapExplainer,
    FeatureAblationExplainer,
    GradientShapExplainer,
    GuidedBackpropExplainer,
    InputXBaselineGradientExplainer,
    InputXGradientExplainer,
    IntegratedGradientsExplainer,
    KernelShapExplainer,
    LimeExplainer,
    OcclusionExplainer,
    RandomExplainer,
    SaliencyExplainer,
)
from torchxai.data_types import SingleTargetAcrossBatch

import torch
import torch.nn as nn

model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
model.eval()
inputs = torch.randn(1, 4)
target = SingleTargetAcrossBatch(index=0)

attrs = SaliencyExplainer(model).explain(inputs=inputs, target=target)
assert attrs.shape == inputs.shape, f"unexpected shape {attrs.shape}"

attrs_list = SaliencyExplainer(model, multi_target=True).explain(
    inputs=inputs,
    target=[SingleTargetAcrossBatch(index=i) for i in range(2)],
)
assert len(attrs_list) == 2

print("smoke test passed")
