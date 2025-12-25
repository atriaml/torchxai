from __future__ import annotations

import inspect

import torch
import tqdm

from tests.utils.types import ExplanationInputs, ExplanationStepOutputs
from torchxai.data_types import ExplanationTarget
from torchxai.explainers._explainer import Explainer


class ExplanationStep:
    def __init__(
        self,
        model: torch.nn.Module,
        explainer: Explainer,
        device: str | torch.device,
        with_amp: bool = False,
        only_allow_tensors_as_targets: bool = False,
    ):
        self._model = model
        self._explainer = explainer
        self._device = torch.device(device)
        self._with_amp = with_amp
        self._only_allow_tensors_as_targets = only_allow_tensors_as_targets
        self._explainer._model = self._model

    def _run_model_forward(self, explanation_input: ExplanationInputs) -> torch.Tensor:
        self._model.eval()
        self._model.to(self._device)
        with torch.no_grad():
            return self._model(*explanation_input.model_inputs)

    def _run_explainer_forward(
        self, explainer: Explainer, explanation_inputs: ExplanationInputs
    ) -> tuple[torch.Tensor, ...]:
        # filster args here so there is no error on fowrard
        # verify that impossible args are not set
        kwargs = {}
        signature = inspect.signature(explainer.explain).parameters.keys()
        for arg in signature:
            kwargs[arg] = getattr(explanation_inputs, arg)

        # if target is in
        target = kwargs.pop("target", None)
        assert isinstance(target, ExplanationTarget), (
            "Explainer explain method must be called with target of type ExplanationTarget."
        )
        explanations = explainer.explain(**kwargs, target=target)
        assert isinstance(explanations, tuple), (
            f"Explainer explain method must return a tuple, got {type(explanations)}"
        )
        return explanations

    def __call__(self, explanation_inputs: ExplanationInputs) -> ExplanationStepOutputs:
        explanation_inputs = explanation_inputs.to(self._device)
        model_outputs = self._run_model_forward(explanation_inputs)
        explanation = self._run_explainer_forward(
            explainer=self._explainer, explanation_inputs=explanation_inputs
        )
        return ExplanationStepOutputs(
            explanation_inputs=explanation_inputs,
            model_outputs=model_outputs,
            explanations=explanation,
        )


class MultiTargetExplanationStep(ExplanationStep):
    def __init__(
        self,
        model: torch.nn.Module,
        explainer: Explainer,
        device: str | torch.device,
        with_amp: bool = False,
        iterative_computation: bool = False,
        only_allow_tensors_as_targets: bool = False,
    ):
        super().__init__(
            model=model,
            explainer=explainer,
            device=device,
            with_amp=with_amp,
            only_allow_tensors_as_targets=only_allow_tensors_as_targets,
        )
        self._iterative_computation = iterative_computation
        self._explainer.multi_target = True

    def _run_explainer_forward(  # type: ignore
        self, explainer: Explainer, explanation_inputs: ExplanationInputs
    ) -> list[tuple[torch.Tensor, ...]]:
        assert isinstance(explainer.multi_target, bool) and explainer.multi_target, (
            "Explainer must be set to multi-target mode for MultiTargetExplanationStep."
        )
        # filster args here so there is no error on fowrard
        # verify that impossible args are not set
        kwargs = {}
        signature = inspect.signature(explainer.explain).parameters.keys()
        for arg in signature:
            kwargs[arg] = getattr(explanation_inputs, arg)

        if self._iterative_computation:
            # disable multi-target for iterative computation
            explainer.multi_target = False

            target = explanation_inputs.target
            per_target_explanations = []
            for t in tqdm.tqdm(target, desc="Computing explanations per target"):
                kwargs = {**kwargs, "target": t}
                curr_explanation = explainer.explain(**kwargs)
                assert len(curr_explanation) == 1, (
                    "Expected single explanation per target in iterative computation."
                )
                per_target_explanations.append(curr_explanation)

            # re-enable multi-target
            explainer.multi_target = True
            return per_target_explanations
        else:
            explanations = explainer.explain(**kwargs)
            assert isinstance(explanations, list), (
                f"Explainer explain method must return a list, got {type(explanations)}"
            )
            return explanations  # type: ignore

    def __call__(  # type: ignore
        self, explanation_inputs: ExplanationInputs
    ) -> ExplanationStepOutputs:
        assert isinstance(self._explainer, Explainer), (
            "Multi-target explainer must be an instance of Explainer."
        )
        explanation_inputs = explanation_inputs.to(self._device)
        model_outputs = self._run_model_forward(explanation_inputs)
        explanation = self._run_explainer_forward(
            explainer=self._explainer, explanation_inputs=explanation_inputs
        )
        return ExplanationStepOutputs(
            explanation_inputs=explanation_inputs,
            model_outputs=model_outputs,
            explanations=explanation,
        )
