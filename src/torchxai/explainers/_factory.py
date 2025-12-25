from __future__ import annotations

from torch import nn

from torchxai.explainers._explainer import Explainer
from torchxai.explainers._grad._deeplift import DeepLiftExplainer
from torchxai.explainers._grad._deeplift_shap import DeepLiftShapExplainer
from torchxai.explainers._grad._gradient_shap import GradientShapExplainer
from torchxai.explainers._grad._guided_backprop import GuidedBackpropExplainer
from torchxai.explainers._grad._input_x_baseline_gradient import (
    InputXBaselineGradientExplainer,
)
from torchxai.explainers._grad._input_x_gradient import InputXGradientExplainer
from torchxai.explainers._grad._integrated_gradients import IntegratedGradientsExplainer
from torchxai.explainers._grad._saliency import SaliencyExplainer
from torchxai.explainers._perturbation._feature_ablation import FeatureAblationExplainer
from torchxai.explainers._perturbation._kernel_shap import KernelShapExplainer
from torchxai.explainers._perturbation._lime import LimeExplainer
from torchxai.explainers._perturbation._occlusion import OcclusionExplainer
from torchxai.explainers._random import RandomExplainer

AVAILABLE_EXPLAINERS = {
    "random": RandomExplainer,
    "saliency": SaliencyExplainer,
    "integrated_gradients": IntegratedGradientsExplainer,
    "deep_lift": DeepLiftExplainer,
    "deep_lift_shap": DeepLiftShapExplainer,
    "gradient_shap": GradientShapExplainer,
    "input_x_gradient": InputXGradientExplainer,
    "input_x_baseline_gradient": InputXBaselineGradientExplainer,
    "guided_backprop": GuidedBackpropExplainer,
    "feature_ablation": FeatureAblationExplainer,
    "occlusion": OcclusionExplainer,
    "lime": LimeExplainer,
    "kernel_shap": KernelShapExplainer,
}


class ExplainerFactory:
    @staticmethod
    def create(explanation_method: str, model: nn.Module, **kwargs) -> Explainer:
        """
        Creates an explainer object based on the given explanation method.
        Args:
            explanation_method (str): The explanation method to be used.
        Returns:
            CaptumExplainerBase: The created CaptumExplainerBase object.
        Raises:
            ValueError: If the given explanation method is not supported.
        """

        explainer_class = AVAILABLE_EXPLAINERS.get(explanation_method, None)
        if explainer_class is None:
            raise ValueError(
                f"Attribution method [{explanation_method}] is not supported. Supported methods are: {AVAILABLE_EXPLAINERS.keys()}."
            )
        return explainer_class(model, **kwargs)
