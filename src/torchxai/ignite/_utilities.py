from __future__ import annotations

import logging
from typing import Any

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


def _as_detached_tuple(tensor_or_mapping: Any) -> tuple[torch.Tensor, ...]:
    if isinstance(tensor_or_mapping, torch.Tensor):
        return (tensor_or_mapping,)
    elif isinstance(tensor_or_mapping, dict):
        return tuple(
            x.detach() if isinstance(x, torch.Tensor) else x
            for x in tensor_or_mapping.values()
        )
    else:
        return tensor_or_mapping
