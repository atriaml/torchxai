from __future__ import annotations

import inspect
from collections import OrderedDict

import torch
import tqdm
from captum.attr import Attribution

from torchxai.data_types import ExplanationInputs, ExplanationState, MetricInputs
from torchxai.explainers.explainer import Explainer


class ExplanationStep:
    def __init__(
        self,
        model: torch.nn.Module,
        explainer: Explainer | Attribution,
        device: str | torch.device,
        with_amp: bool = False,
        use_captum_explainer: bool = False,
        only_allow_tensors_as_targets: bool = False,
    ):
        self._model = model
        self._explainer = explainer
        self._device = torch.device(device)
        self._with_amp = with_amp
        self._use_captum_explainer = use_captum_explainer
        self._only_allow_tensors_as_targets = only_allow_tensors_as_targets

        if self._use_captum_explainer and isinstance(explainer, Explainer):
            explainer = explainer._explanation_fn

        # inspect explainer signature and save
        method = (
            self._explainer.attribute
            if isinstance(self._explainer, Attribution)
            else self._explainer.explain
        )
        self._explainer_possible_kwargs = inspect.signature(method).parameters

        # reassign model to explainer
        if isinstance(self._explainer, Explainer):
            self._explainer._model = self._model
        else:
            self._explainer.forward_func = self._model

    # def _wrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
    #     class SoftMaxOutputWrapper(torch.nn.Module):
    #         def __init__(self, model: torch.nn.Module):
    #             super().__init__()
    #             self.model = model
    #             self.softmax = torch.nn.Softmax(dim=1)

    #         def forward(self, *args, **kwargs):
    #             logits = self.model(*args, **kwargs)
    #             return self.softmax(logits)

    #     return SoftMaxOutputWrapper(model)

    def _run_model_forward(self, explanation_input: ExplanationInputs) -> torch.Tensor:
        self._model.eval()
        self._model.to(self._device)
        with torch.no_grad():
            return self._model(*explanation_input.model_inputs)

    def _run_explainer_forward(
        self, explainer: Explainer | Attribution, explanation_inputs: ExplanationInputs
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        if self._use_captum_explainer and isinstance(explainer, Explainer):
            explainer = explainer._explanation_fn

        # inspect explainer signature and save
        explainer_kwargs = explanation_inputs.to_explainer_kwargs()

        # for our case, we keep target simple as tensor with equal batch size to inputs
        if self._only_allow_tensors_as_targets:
            assert isinstance(explanation_inputs.target, torch.Tensor), (
                "Target must be a torch.Tensor for single-target explanation."
            )
            assert (
                explanation_inputs.target.shape[0]
                == explainer_kwargs["inputs"][0].shape[0]
            ), "Target batch size must match inputs batch size."

        # filter out only possible kwargs
        explainer_kwargs = {
            key: value
            for key, value in explainer_kwargs.items()
            if key in self._explainer_possible_kwargs
        }

        if isinstance(explainer, Explainer):
            explanations = explainer.explain(**explainer_kwargs)
        elif isinstance(explainer, Attribution):
            explanations = explainer.attribute(**explainer_kwargs)
        else:
            raise AssertionError(
                "Explainer must be an instance of Explainer or Attribution."
            )
        return explanations

    def __call__(
        self, explanation_inputs: ExplanationInputs, metric_inputs: MetricInputs
    ) -> ExplanationState:
        model_outputs = self._run_model_forward(explanation_inputs)
        explanation = self._run_explainer_forward(
            explainer=self._explainer, explanation_inputs=explanation_inputs
        )
        return ExplanationState(
            explanation_inputs=explanation_inputs,
            metric_inputs=metric_inputs,
            model_outputs=model_outputs,
            explanations=explanation,
        )


class MultiTargetExplanationStep(ExplanationStep):
    def __init__(
        self,
        model: torch.nn.Module,
        explainer: Explainer | Attribution,
        device: str | torch.device,
        with_amp: bool = False,
        iterative_computation: bool = False,
    ):
        super().__init__(
            model=model,
            explainer=explainer,
            device=device,
            with_amp=with_amp,
            use_captum_explainer=False,
        )
        self._iterative_computation = iterative_computation

    def _run_explainer_forward(
        self, explainer: Explainer | Attribution, explanation_inputs: ExplanationInputs
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        print("explanation_inputs", explanation_inputs)
        assert isinstance(explainer, Explainer), (
            "Multi-target explainer must be an instance of Explainer."
        )
        assert isinstance(explanation_inputs.target, list), (
            "Target must be a list for multi-target explanation."
        )
        # inputs: dict[str, torch.Tensor],
        # additional_forward_args: tuple[torch.Tensor, ...] | None = None,
        # baselines: dict[str, torch.Tensor] | None = None,
        # train_baselines: dict[str, torch.Tensor] | None = None,
        # feature_mask: dict[str, torch.Tensor] | None = None,
        # target: list[torch.Tensor] | None = None,

        # target for each input in the batch can be variable length. However, since that will only work for 1 batch-size
        # we only currently support that target is the same length for all inputs in the batch.
        # # e.g. target = [[0,1], [0,1], [0,1]] for batch-size 3 with 2 targets each
        # validate this here
        # also just in our case we only support list of lists
        for t in explanation_inputs.target:
            assert isinstance(t, torch.Tensor | None), (
                "Each target must be a torch.Tensor."
            )
            if t is None:
                continue
            assert t.shape == explanation_inputs.target[0].shape, (
                "All targets must have the same shape."
            )
            assert t.ndim == 1, "Each target tensor must be 1-dimensional."

        # inspect explainer signature and save
        explainer_kwargs = explanation_inputs.to_explainer_kwargs()

        # filter out only possible kwargs
        explainer_kwargs = {
            key: value
            for key, value in explainer_kwargs.items()
            if key in self._explainer_possible_kwargs
        }

        per_target_explanations = []
        if self._iterative_computation:
            # disable multi-target for iterative computation
            explainer._is_multi_target = False

            target = explainer_kwargs.pop("target")
            for t in tqdm.tqdm(target, desc="Computing explanations per target"):
                curr_explanation = explainer.explain(**explainer_kwargs, target=t)
                assert len(curr_explanation) == 1, (
                    "Expected single explanation per target in iterative computation."
                )
                per_target_explanations.append(curr_explanation[0])

            # re-enable multi-target
            explainer._is_multi_target = True
        else:
            per_target_explanations = explainer.explain(**explainer_kwargs)
            assert len(per_target_explanations) == len(explainer_kwargs["target"]), (
                "Number of explanations must match number of targets."
            )

        # convert the list[tuple[tensors]] -> list[dict[tensors]] to -> tuples -> list of targets
        def _tuples_to_dict(
            exp_tuples: tuple, keys: list[str]
        ) -> dict[str, torch.Tensor]:
            return dict(zip(keys, exp_tuples, strict=True))

        feature_keys = list(explanation_inputs.explained_features.keys())
        per_target_explanations = [
            _tuples_to_dict(exp, feature_keys) for exp in per_target_explanations
        ]

        # next we aggregate per target explanations into per input explanations
        aggregated_explanations = OrderedDict()
        for key in feature_keys:
            aggregated_explanations[key] = [
                per_target_explanation[key]
                for per_target_explanation in per_target_explanations
            ]
            aggregated_explanations[key] = torch.stack(
                aggregated_explanations[key], dim=1
            )  # shape: (batch_size, num_targets, ...)
        return aggregated_explanations

    def __call__(
        self, explanation_inputs: ExplanationInputs, metric_inputs: MetricInputs
    ) -> ExplanationState:
        model_outputs = self._run_model_forward(explanation_inputs)
        explanation = self._run_explainer_forward(
            explainer=self._explainer, explanation_inputs=explanation_inputs
        )
        return ExplanationState(
            explanation_inputs=explanation_inputs,
            metric_inputs=metric_inputs,
            model_outputs=model_outputs,
            explanations=explanation,
        )
