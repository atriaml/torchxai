from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, TypeAlias, cast

from pydantic import BaseModel, ConfigDict, field_validator

if TYPE_CHECKING:
    import torch

RawTargetType: TypeAlias = (
    int | list[int] | tuple[int, ...] | list[tuple[int, ...]] | None
)


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
        import torch

        if raw_target is None:
            return NoTarget()

        def _sanitize_tensor(value: RawTargetType | torch.Tensor) -> RawTargetType:
            """Convert torch.Tensor to Python native types."""

            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    scalar = value.item()
                    assert isinstance(scalar, int), (
                        "Expected single integer value from tensor."
                    )
                    return scalar
                else:
                    result: list[Any] = value.tolist()  # type: ignore[attr-defined]
                    assert all(isinstance(i, int) for i in result), (
                        "Expected list of integers from tensor."
                    )
                    return result
            return value

        sanitized: RawTargetType = _sanitize_tensor(raw_target)
        if isinstance(sanitized, int):
            return SingleTargetAcrossBatch(index=sanitized)
        if isinstance(sanitized, tuple) and all(isinstance(i, int) for i in sanitized):
            return MultiIndexTargetAcrossBatch(indices=sanitized)
        if isinstance(sanitized, list) and all(isinstance(i, int) for i in sanitized):
            return SingleTargetPerSample(indices=cast(list[int], sanitized))
        if isinstance(sanitized, list) and all(
            isinstance(i, tuple) and all(isinstance(j, int) for j in i)
            for i in sanitized
        ):
            return MultiIndexTargetPerSample(
                indices=cast(list[tuple[int, ...]], sanitized)
            )
        raise ValueError(f"Cannot convert raw target of type {type(sanitized)}.")

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
    def value(self) -> None:
        return None


class SingleTargetAcrossBatch(ExplanationTarget):
    """Single target index applied to all examples in the batch."""

    index: int

    @field_validator("index")
    @classmethod
    def validate_index(cls, v: int) -> int:
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
            [output[(i,) + targ_elem] for i, targ_elem in enumerate(self.indices)]
        )

    @property
    def value(self) -> list[tuple[int, ...]]:
        return self.indices


NO_TARGET = NoTarget()
ExplanationTargetType: TypeAlias = (
    NoTarget
    | SingleTargetAcrossBatch
    | MultiIndexTargetAcrossBatch
    | SingleTargetPerSample
    | MultiIndexTargetPerSample
)
