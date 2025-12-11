from typing import Any

import torch
from captum._utils.common import _format_output, _format_tensor_into_tuples, _is_tuple
from torch import Tensor

from torchxai.data_types.common import TensorOrTupleOfTensorsGeneric
from torchxai.explainers.explainer import Explainer
from torchxai.metrics.axiomatic.utilities import (
    _create_shifted_expainer,
    _prepare_kwargs_for_base_and_shifted_inputs,
)


def _format_output_list_of_tensor_tuples(
    is_inputs_tuple: bool, output_list: list
) -> list[tuple[Tensor, ...]]:
    formatted_output_list = []
    for output in output_list:
        assert isinstance(output, tuple), (
            "Each explanation in the list must be a tuple of tensors."
        )
        formatted_output = _format_output(is_inputs_tuple, output)
        formatted_output_list.append(formatted_output)
    return formatted_output_list


def _multi_target_input_invariance(
    explainer: Explainer,
    inputs: TensorOrTupleOfTensorsGeneric,
    constant_shifts: TensorOrTupleOfTensorsGeneric,
    input_layer_names: tuple[str],
    **kwargs: Any,
) -> tuple[list[float], list[tuple[Tensor, ...]], list[tuple[Tensor, ...]]]:
    assert isinstance(explainer, Explainer), (
        "The explainer must be an instance of Explainer."
    )
    assert explainer._is_multi_target, "The explainer must be a multi-target explainer."

    target = kwargs.get("target", None)
    assert isinstance(target, list), "targets must be a list of targets"
    assert all(isinstance(x, (tuple, int)) for x in target), (
        "targets must be a list of ints"
    )

    # Keeps track whether original input is a tuple or not before
    # converting it into a tuple.
    is_inputs_tuple = _is_tuple(inputs)

    kwargs_copy, shifted_kwargs_copy = _prepare_kwargs_for_base_and_shifted_inputs(
        kwargs
    )
    inputs = _format_tensor_into_tuples(inputs)  # type: ignore
    constant_shifts = _format_tensor_into_tuples(constant_shifts)  # type: ignore

    assert len(input_layer_names) == len(set(input_layer_names)), (
        "Each input layer must be unique for each input constant shift tensor."
    )

    assert len(input_layer_names) == len(constant_shifts), (
        "The number of input layer names should be the same as the number of constant shifts. "
    )

    assert (
        len(inputs) == len(constant_shifts)
        and inputs[0].shape[1:] == constant_shifts[0].shape[1:]
        and constant_shifts[0].shape[0] == 1
    ), (
        "The number of inputs should be the same as the number of constant shifts and the batch size of the "
        "constant shifts should be 1. Current shapes are: "
        f"{inputs[0].shape} and {constant_shifts[0].shape}"
    )

    shifted_explainer = _create_shifted_expainer(
        explainer=explainer,
        input_layer_names=input_layer_names,
        constant_shifts=constant_shifts,
        **kwargs,
    )

    # create shifted inputs
    constant_shift_expanded = tuple(
        constant_shift.expand_as(input)
        for input, constant_shift in zip(inputs, constant_shifts, strict=True)
    )
    shifted_inputs = tuple(
        input - constant_shift
        for input, constant_shift in zip(inputs, constant_shift_expanded, strict=True)
    )

    with torch.no_grad():
        if isinstance(explainer, Explainer):
            inputs_expl_list = explainer.explain(inputs, **kwargs_copy)
            shifted_inputs_expl_list = shifted_explainer.explain(
                shifted_inputs, **shifted_kwargs_copy
            )
            assert isinstance(inputs_expl_list, list), (
                "For multi-target explainers, the output must be a list of explanations."
            )
            assert isinstance(shifted_inputs_expl_list, list), (
                "For multi-target explainers, the output must be a list of explanations."
            )
        else:
            raise ValueError(
                "Explanation function must be an instance of Attribution or FusionExplainer"
            )

        for inputs_expl in inputs_expl_list:
            assert isinstance(inputs_expl, tuple), (
                "Each explanation in the list must be a tuple of tensors."
            )
        for shifted_inputs_expl in shifted_inputs_expl_list:
            assert isinstance(shifted_inputs_expl, tuple), (
                "Each shifted explanation in the list must be a tuple of tensors."
            )

        # calculate the difference between the two explanations step by step
        input_invarance_score_list = []
        for inputs_expl, shifted_inputs_expl in zip(
            inputs_expl_list, shifted_inputs_expl_list, strict=True
        ):
            # For each target, compute differences across all input layers
            layer_scores = []
            for input_expl, shifted_input_expl in zip(
                inputs_expl, shifted_inputs_expl, strict=True
            ):
                # For each layer, compute mean absolute difference per sample
                sample_scores = []
                for per_sample_input_expl, per_sample_shifted_input_expl in zip(
                    input_expl, shifted_input_expl, strict=True
                ):
                    diff = torch.abs(
                        per_sample_input_expl - per_sample_shifted_input_expl
                    )
                    mean_diff = torch.mean(diff).item()
                    sample_scores.append(mean_diff)

                layer_score = torch.tensor(sample_scores, device=inputs[0].device)
                layer_scores.append(layer_score)

            # Sum scores across all layers for this target
            total_score = sum(layer_scores)
            input_invarance_score_list.append(total_score)

        return (
            input_invarance_score_list,
            _format_output_list_of_tensor_tuples(is_inputs_tuple, inputs_expl_list),
            _format_output_list_of_tensor_tuples(
                is_inputs_tuple, shifted_inputs_expl_list
            ),
        )
