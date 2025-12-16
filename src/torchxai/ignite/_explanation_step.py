from __future__ import annotations

import inspect
import typing
from collections import OrderedDict

import torch
import tqdm

from torchxai.data_types import (
    ExplanationInputs,
    ExplanationState,
    ExplanationStepOutputs,
    MetricInputs,
    MultiTargetExplanationState,
    MultiTargetExplanationStepOutputs,
)
from torchxai.explainers.explainer import Explainer


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
    ) -> OrderedDict[str, torch.Tensor]:
        # filster args here so there is no error on fowrard
        # verify that impossible args are not set
        kwargs = {}
        signature = inspect.signature(explainer.explain).parameters.keys()
        for arg in signature:
            kwargs[arg] = getattr(explanation_inputs, arg)

        return typing.cast(OrderedDict[str, torch.Tensor], explainer.explain(**kwargs))

    def __call__(
        self,
        explanation_inputs: ExplanationInputs,
        metric_inputs: MetricInputs | None = None,
    ) -> ExplanationStepOutputs:
        explanation_inputs = explanation_inputs.to(self._device)
        metric_inputs = (
            metric_inputs.to(self._device) if metric_inputs is not None else None
        )
        model_outputs = self._run_model_forward(explanation_inputs)
        explanation = self._run_explainer_forward(
            explainer=self._explainer, explanation_inputs=explanation_inputs
        )
        expl_state = ExplanationState(
            explanation_inputs=explanation_inputs,
            model_outputs=model_outputs,
            explanations=explanation,
        )
        return ExplanationStepOutputs(
            explanation_state=expl_state, metric_inputs=metric_inputs
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
    ) -> list[OrderedDict[str, torch.Tensor]]:
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
            return typing.cast(
                list[OrderedDict[str, torch.Tensor]], explainer.explain(**kwargs)
            )

    def __call__(  # type: ignore
        self,
        explanation_inputs: ExplanationInputs,
        metric_inputs: MetricInputs | None = None,
    ) -> MultiTargetExplanationStepOutputs:
        assert isinstance(self._explainer, Explainer), (
            "Multi-target explainer must be an instance of Explainer."
        )
        explanation_inputs = explanation_inputs.to(self._device)
        metric_inputs = (
            metric_inputs.to(self._device) if metric_inputs is not None else None
        )
        model_outputs = self._run_model_forward(explanation_inputs)
        explanation = self._run_explainer_forward(
            explainer=self._explainer, explanation_inputs=explanation_inputs
        )
        expl_state = MultiTargetExplanationState(
            explanation_inputs=explanation_inputs,
            model_outputs=model_outputs,
            explanations=explanation,
        )
        return MultiTargetExplanationStepOutputs(
            explanation_state=expl_state, metric_inputs=metric_inputs
        )
