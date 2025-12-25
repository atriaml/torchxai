from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, cast

import torch
from pydantic import BaseModel, ConfigDict, field_validator


class ExplanationTarget(BaseModel, ABC):
    """Base class for all target types with Pydantic validation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def value(self) -> Any:
        """Return the raw value of the target."""
        raise NotImplementedError()

    @classmethod
    def from_raw_input(cls, raw_target: Any) -> ExplanationTargetType:
        """Create appropriate TargetType from raw input.

        Args:
            raw_target: Raw target input (int, list, tuple, etc.)
        Returns:
            Appropriate TargetType instance.
        """
        if raw_target is None:
            return NoTarget()
        if isinstance(raw_target, torch.Tensor):
            raw_target = (
                raw_target.item() if raw_target.numel() == 1 else raw_target.tolist()
            )
        if isinstance(raw_target, int):
            return SingleTargetAcrossBatch(index=raw_target)
        if isinstance(raw_target, tuple) and all(
            isinstance(i, int) for i in raw_target
        ):
            return MultiIndexTargetAcrossBatch(indices=raw_target)
        if isinstance(raw_target, list) and all(isinstance(i, int) for i in raw_target):
            return SingleTargetPerSample(indices=raw_target)
        if isinstance(raw_target, list) and all(
            isinstance(i, tuple) and all(isinstance(j, int) for j in i)
            for i in raw_target
        ):
            return MultiIndexTargetPerSample(indices=raw_target)
        raise ValueError(f"Cannot convert raw target of type {type(raw_target)}.")

    @abstractmethod
    def select(self, output: torch.Tensor) -> torch.Tensor:
        """Select the appropriate part of the model output."""
        pass

    def __repr__(self) -> str:
        params_str = ", ".join(f"{k}={v!r}" for k, v in self.model_dump().items())
        return f"{self.__class__.__name__}({params_str})"

    def __str__(self) -> str:
        return super().__repr__()


class NoTarget(ExplanationTarget):
    """No target selection - return the full model output."""

    def select(self, output: torch.Tensor) -> torch.Tensor:
        return output

    @property
    def value(self):
        return None


class SingleTargetAcrossBatch(ExplanationTarget):
    """Single target index applied to all examples in the batch."""

    index: int

    @field_validator("index")
    @classmethod
    def validate_index(cls, v):
        if v < 0:
            raise ValueError("Target index must be non-negative")
        return v

    def select(self, output: torch.Tensor) -> torch.Tensor:
        """Select output[:, self.index] or output[..., self.index]"""
        assert self.index < output.shape[-1], (
            f"Target index {self.index} out of bounds for output shape {output.shape}"
        )
        return output[..., self.index]

    def is_multi_target(self) -> bool:
        return False

    @property
    def value(self) -> int:
        return self.index


class MultiIndexTargetAcrossBatch(ExplanationTarget):
    """Multiple indices for selecting nested dimensions."""

    indices: tuple[int, ...]

    def select(self, output: torch.Tensor) -> torch.Tensor:
        """Select output[:, indices[0], indices[1], ...]"""
        assert len(self.indices) <= len(output.shape) - 1, (
            f"Cannot choose target column with output shape {output.shape!r}."
        )
        selection = (slice(None),) + self.indices
        return output[selection]

    @property
    def value(self) -> tuple[int, ...]:
        return self.indices


class SingleTargetPerSample(ExplanationTarget):
    indices: list[int]

    def select(self, output: torch.Tensor) -> torch.Tensor:
        """Use torch.gather to select different index per example."""
        assert len(self.indices) == output.shape[0], (
            "Target list length does not match output!"
        )
        return torch.gather(
            output,
            1,
            torch.tensor(self.indices, device=output.device).reshape(len(output), 1),
        ).squeeze(-1)

    @property
    def value(self) -> list[int]:
        return self.indices


class MultiIndexTargetPerSample(ExplanationTarget):
    indices: list[tuple[int, ...]]

    def select(self, output: torch.Tensor) -> torch.Tensor:
        """Use torch.gather to select different index per example."""
        assert len(self.indices) == output.shape[0], (
            "Target list length does not match output!"
        )
        return torch.stack(
            [
                output[(i,) + cast(tuple, targ_elem)]
                for i, targ_elem in enumerate(self.indices)
            ]
        )

    @property
    def value(self) -> list[tuple[int, ...]]:
        return self.indices


ExplanationTargetType = (
    NoTarget
    | SingleTargetAcrossBatch
    | MultiIndexTargetAcrossBatch
    | SingleTargetPerSample
    | MultiIndexTargetPerSample
)
