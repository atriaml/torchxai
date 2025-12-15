import logging
import random

import numpy as np
import torch
from torch import Tensor

from torchxai.data_types import ExplanationStepOutputs

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _grid_segmenter(images: torch.Tensor, cell_size: int = 16) -> torch.Tensor:
    feature_mask = []
    for image in images:
        # image dimensions are C x H x H
        c = image.shape[0]
        dim_x, dim_y = image.shape[1] // cell_size, image.shape[2] // cell_size
        mask = (
            torch.arange(dim_x * dim_y, device=images.device)
            .view((dim_x, dim_y))
            .repeat_interleave(cell_size, dim=0)
            .repeat_interleave(cell_size, dim=1)
            .long()
            .unsqueeze(0)
        ).repeat(c, 1, 1)
        feature_mask.append(mask)
    return torch.stack(feature_mask)


def _compare_explanation_per_target(
    output_explanation_per_target: tuple[Tensor, ...] | Tensor,
    expected_explanation_per_target: tuple[Tensor, ...] | Tensor,
    delta: float = 1e-5,
    visualize: bool = False,
) -> None:
    if not isinstance(output_explanation_per_target, tuple):
        output_explanation_per_target = (output_explanation_per_target,)
    if not isinstance(expected_explanation_per_target, tuple):
        expected_explanation_per_target = (expected_explanation_per_target,)

    if visualize:
        import matplotlib.pyplot as plt
        from captum.attr._utils.visualization import _normalize_attr

        for output_explanation, expected_explanation in zip(
            output_explanation_per_target, expected_explanation_per_target, strict=True
        ):
            for output, expected in zip(
                output_explanation, expected_explanation, strict=True
            ):
                output = _normalize_attr(
                    output.cpu().numpy(), "absolute_value", reduction_axis=0
                )
                expected = _normalize_attr(
                    expected.cpu().numpy(), "absolute_value", reduction_axis=0
                )
                fig, ax = plt.subplots(1, 2)
                ax[0].imshow(output)
                ax[1].imshow(expected)
                plt.show()

    for output_explanation_per_input, expected_explanation_per_input in zip(
        output_explanation_per_target, expected_explanation_per_target, strict=True
    ):
        _assert_tensor_almost_equal(
            output_explanation_per_input,
            expected_explanation_per_input,
            delta=delta,
            mode="mean",
        )


def _assert_tensor_almost_equal(
    actual, expected, delta: float = 0.0001, mode: str = "sum"
) -> None:
    assert isinstance(actual, torch.Tensor), (
        "Actual parameter given for comparison must be a tensor."
    )
    if not isinstance(expected, torch.Tensor):
        expected = torch.tensor(expected, dtype=actual.dtype)
    assert actual.shape == expected.shape, (
        f"Expected tensor with shape: {expected.shape}. Actual shape {actual.shape}."
    )
    actual = actual.cpu()
    expected = expected.cpu()

    # check if both are nan
    if torch.isnan(actual).all():
        assert torch.isnan(expected).all(), (
            f"Actual tensor is nan while expected tensor is not. Actual: {actual}, Expected: {expected}"
        )
        return

    if mode == "sum":
        assert torch.sum(torch.abs(actual - expected)).item() < delta, (
            f"Tensors are not equal with tolerance ({delta}). Actual: {actual}, Expected: {expected}"
        )
    elif mode == "mean":
        assert torch.mean(torch.abs(actual - expected)).item() < delta, (
            f"Tensors are not equal with tolerance ({delta}). Actual: {actual}, Expected: {expected}"
        )
    elif mode == "max":
        # if both tensors are empty, they are equal but there is no max
        if actual.numel() == expected.numel() == 0:
            return

        if actual.size() == torch.Size([]):
            assert torch.max(torch.abs(actual - expected)).item() < delta, (
                f"Tensors are not equal with tolerance ({delta}). Actual: {actual}, Expected: {expected}"
            )
        else:
            for index, (input, ref) in enumerate(zip(actual, expected, strict=True)):
                almost_equal = abs(input - ref) <= delta
                if hasattr(almost_equal, "__iter__"):
                    almost_equal = almost_equal.all()
                assert almost_equal, (
                    f"Values at index {index}, {input} and {ref}, differ more than by {delta}"
                )
    else:
        raise ValueError("Mode for assertion comparison must be one of `max` or `sum`.")


def _assert_all_tensors_almost_equal(tensors: list[torch.Tensor]):
    for i in range(1, len(tensors)):
        if isinstance(tensors[i], list):
            for x, y in zip(tensors[i - 1], tensors[i], strict=True):
                _assert_tensor_almost_equal(x, y)
        else:
            _assert_tensor_almost_equal(tensors[i - 1], tensors[i])


def _set_all_random_seeds(seed: int = 1234) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def _run_metric_via_ignite(
    metric, explanation_step_outputs: ExplanationStepOutputs
) -> dict[str, torch.Tensor]:
    """Helper function to run a metric via the Ignite Engine interface.

    Args:
        metric: The Ignite metric to evaluate
        explanation_state: The explanation state to process

    Returns:
        The metric output from the engine state
    """
    from ignite.engine import Engine

    def explanation_step(engine, batch) -> ExplanationStepOutputs:
        return explanation_step_outputs

    engine = Engine(explanation_step)
    engine.logger.propagate = False
    metric.attach(engine, "metric")
    state = engine.run([None], max_epochs=1)
    return state.metrics["metric"][0]
