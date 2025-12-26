from __future__ import annotations

import logging
from typing import Any, Literal

import torch


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


def _tensor_or_tensor_dict_as_detached_tuple(
    tensor_or_mapping: Any,
) -> tuple[torch.Tensor, ...] | None:
    if tensor_or_mapping is None:
        return None
    if isinstance(tensor_or_mapping, torch.Tensor):
        return (tensor_or_mapping,)
    elif isinstance(tensor_or_mapping, dict):
        return tuple(
            x.detach() if isinstance(x, torch.Tensor) else x
            for x in tensor_or_mapping.values()
        )
    elif isinstance(tensor_or_mapping, tuple):
        return tuple(
            x.detach() if isinstance(x, torch.Tensor) else x for x in tensor_or_mapping
        )
    else:
        raise TypeError("Input must be a torch.Tensor or a dict of torch.Tensors.")


def _prepare_baselines_from_type(
    inputs: torch.Tensor | tuple[torch.Tensor, ...],
    baselines_type: Literal["zeros", "ones", "batch_mean", "random", "fixed"],
    fixed_value: float = 0.5,
) -> torch.Tensor:
    is_inputs_tuple = isinstance(inputs, tuple)
    if isinstance(inputs, torch.Tensor):
        inputs = (inputs,)
    if baselines_type == "zeros":
        baselines = tuple(torch.zeros_like(inp) for inp in inputs)
    elif baselines_type == "ones":
        baselines = tuple(torch.ones_like(inp) for inp in inputs)
    elif baselines_type == "batch_mean":
        baselines = tuple(torch.full_like(inp, inp.mean().item()) for inp in inputs)
    elif baselines_type == "fixed":
        baselines = tuple(torch.full_like(inp, fixed_value) for inp in inputs)
    elif baselines_type == "random":
        baselines = tuple(torch.rand_like(inp) for inp in inputs)
    else:
        raise ValueError(
            f"Unsupported baselines_type: {baselines_type}. Supported types are 'zeros' and 'random'."
        )
    if not is_inputs_tuple:
        baselines = baselines[0]
    return baselines
