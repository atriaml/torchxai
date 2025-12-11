from typing import TypeVar

from torch import Tensor
from torch.nn import Module

ModuleOrModuleList = TypeVar("ModuleOrModuleList", Module, list[Module])
TargetType = (
    None
    | int
    | tuple[int, ...]
    | Tensor
    | list[tuple[int, ...]]
    | list[int]
    | list[Tensor]
)
BaselineType = None | Tensor | tuple[Tensor, ...]
TensorOrTupleOfTensorsGeneric = Tensor | tuple[Tensor, ...]
TensorOrTupleOfTensorsOrListOfTensorsGeneric = (
    Tensor | tuple[Tensor, ...] | list[Tensor]
)
