from typing import TypeVar

# from captum._utils.typing import BaselineType, TargetType
from torch import Tensor
from torch.nn import Module

ModuleOrModuleList = TypeVar("ModuleOrModuleList", Module, list[Module])
TargetType = (
    None | int | tuple[int, ...] | Tensor | list[tuple[int, ...]] | list[int]
)  # same as captum
BaselineType = (
    None | Tensor | int | float | tuple[Tensor | int | float, ...]
)  # same as captum
TensorOrTupleOfTensorsGeneric = Tensor | tuple[Tensor, ...]
TensorOrTupleOfTensorsOrListOfTensorsGeneric = (
    Tensor | tuple[Tensor, ...] | list[Tensor] | list[tuple[Tensor, ...]]
)

TupleOfTensorsOrListOfTuplesOfTensors = tuple[Tensor, ...] | list[tuple[Tensor, ...]]
