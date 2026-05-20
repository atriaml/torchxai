import torch
from captum._utils.common import _format_output, _format_tensor_into_tuples, _is_tuple

from torchxai.data_types import TensorOrTupleOfTensorsGeneric


def _attribution_localization_impl(
    attributions: TensorOrTupleOfTensorsGeneric,
    feature_mask: TensorOrTupleOfTensorsGeneric,
    positive_attributions: bool = True,
    weighted: bool = False,
) -> TensorOrTupleOfTensorsGeneric:
    with torch.no_grad():
        is_attributions_tuple = _is_tuple(attributions)
        attributions = _format_tensor_into_tuples(attributions)
        feature_mask = _format_tensor_into_tuples(feature_mask)
        assert feature_mask[0].dtype == torch.bool, (
            "Segmentation mask must be of type bool."
        )
        assert (
            len(feature_mask) == len(attributions)
            and feature_mask[0].shape == attributions[0].shape
        ), "Segmentation mask must have the same shape as the attributions."

        if positive_attributions:
            attributions = tuple(
                torch.clamp(attribution, min=0) for attribution in attributions
            )

        bsz = attributions[0].shape[0]
        localization_scores = tuple(
            (attribution * mask).view(bsz, -1).sum(dim=1)
            / attribution.view(bsz, -1).sum(dim=1)
            for attribution, mask in zip(attributions, feature_mask, strict=False)
        )

        mask_size_ratios = tuple(
            mask.numel() / mask.contiguous().view(bsz, -1).sum(dim=1)
            for mask in feature_mask
        )

        if weighted:
            localization_scores = tuple(
                score * mask_size_ratio
                for score, mask_size_ratio in zip(
                    localization_scores, mask_size_ratios, strict=False
                )
            )

        localization_scores = _format_output(is_attributions_tuple, localization_scores)
        return localization_scores


def attribution_localization(
    attributions: tuple[torch.Tensor, ...] | list[tuple[torch.Tensor, ...]],
    feature_mask: tuple[torch.Tensor, ...] | None,
    multi_target: bool = False,
    positive_attributions: bool = True,
    weighted: bool = False,
    return_dict: bool = False,
) -> (
    TensorOrTupleOfTensorsGeneric
    | list[TensorOrTupleOfTensorsGeneric]
    | dict[str, TensorOrTupleOfTensorsGeneric]
    | dict[str, list[TensorOrTupleOfTensorsGeneric]]
):
    """Ratio of positive attribution mass inside a ground-truth region to total attribution. ↑ better.

    Measures how well the explanation concentrates on the annotated target region. A high score means
    the most-attributed features belong to the relevant object class.

    References:
        Kohlbrenner et al.: "Towards Best Practice in Explaining Neural Network Decisions with LRP."
        IJCNN (2020): 1–7.

    Args:
        attributions (tuple[Tensor, ...] or list[tuple[Tensor, ...]]): Attribution tensors, each of
            shape ``(batch_size, *input_shape)``. For multi-target mode, pass a list of tuples —
            one tuple per target.
        feature_mask (tuple[Tensor, ...] or None): Boolean segmentation masks with the same shape
            as ``attributions``, marking the ground-truth relevant region. The same mask is used
            for all targets. If ``None``, a mask of all-True values is used.
        multi_target (bool): If ``True``, ``attributions`` must be a list of tuples, one per target.
            Default: ``False``.
        positive_attributions (bool): If ``True``, only positive attribution values contribute to the
            numerator. Default: ``True``.
        weighted (bool): If ``True``, multiply the score by the inverse mask-coverage fraction,
            rewarding small, precise masks. Default: ``False``.
        return_dict (bool): If ``True``, return ``{"score": ...}`` instead of a bare tensor.
            Default: ``False``.

    Returns:
        Tensor or list[Tensor] or dict: Localization score per example, shape ``(batch_size,)``.
        Returns a list when ``multi_target=True``, and a dict when ``return_dict=True``.

    Example:
        >>> import torch
        >>> attributions = (torch.tensor([[0.1, 0.5, 0.3, 0.1]]),)
        >>> mask = (torch.tensor([[False, True, True, False]]),)
        >>> score = attribution_localization(attributions, feature_mask=mask)
    """
    if feature_mask is None:
        feature_mask = tuple(torch.ones_like(attributions[0][0], dtype=torch.bool))

    if multi_target:
        assert isinstance(attributions, list), (
            "For multi-target attributions, a list of attributions per target must be provided."
        )
        scores_per_target = [
            _attribution_localization_impl(
                attributions=attribution_per_sample,
                feature_mask=feature_mask,
                positive_attributions=positive_attributions,
                weighted=weighted,
            )
            for attribution_per_sample in attributions
        ]
        if return_dict:
            return {"score": scores_per_target}
        else:
            return scores_per_target
    else:
        assert isinstance(attributions, tuple | torch.Tensor), (
            "For single-target attributions, a tuple or list of attributions must be provided."
        )
        assert all(isinstance(t, torch.Tensor) for t in attributions), (
            "Attributions must be a tensor or a tuple of tensors."
        )
        score = _attribution_localization_impl(
            attributions=attributions,
            feature_mask=feature_mask,
            positive_attributions=positive_attributions,
            weighted=weighted,
        )
        return {"score": score} if return_dict else score
