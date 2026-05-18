import torch
from torch import Tensor

from torchxai.metrics._utils.common import (
    _construct_default_feature_mask,
    _reduce_attribution_over_features,
)


def _compute_locality_scores(
    attr: Tensor,  # [G]
    centers: Tensor,  # [G, 2]
    target_index: int,
    x_threshold: float,
    y_threshold: float,
    add_self: bool = True,
) -> Tensor:
    attr = attr.clamp(min=0)

    # normalize to sum to 1
    attr = attr / (attr.sum() + 1e-8)

    target_center = centers[target_index]
    dx = centers[:, 0] - target_center[0]
    dy = centers[:, 1] - target_center[1]

    # Mask out the target token itself
    other_mask = torch.ones(attr.shape[0], dtype=torch.bool, device=attr.device)

    if not add_self:
        other_mask[target_index] = False

    vertical_locality = attr[(dx.abs() <= x_threshold) & other_mask].sum()
    horizontal_locality = attr[(dy.abs() <= y_threshold) & other_mask].sum()

    # this is weighted rmse around the target center
    # RMSE = \sqrt{\frac{1}{\sum_{i=1}^{N} w_i} \sum_{i=1}^{N} w_i * (x_i - x_ti)^2 + (y_i - y_ti)^2}
    # where \sum_{i=1}^{N} w_i = 1 since we normalized the attributions to sum to 1, we can simplify this to:
    # RMSE = \sqrt{\sum_{i=1}^{N} w_i * (x_i - x_ti)^2 + (y_i - y_ti)^2}
    w_rmse = torch.sqrt((attr * (dx**2 + dy**2)).sum())

    return torch.stack([vertical_locality, horizontal_locality, w_rmse])


def _locality_single_sample(
    attributions_single_sample: tuple[Tensor, ...],
    feature_mask_single_sample: tuple[Tensor, ...] | None,
    bboxes_single_sample: tuple[Tensor, ...],
    target_index: int,
    x_threshold: float,
    y_threshold: float,
    use_weighted_sum: bool = False,
) -> Tensor:
    """
    Returns
    -------
    Tensor of shape [n_modalities, 3]
    """
    if not isinstance(attributions_single_sample, tuple):
        attributions_single_sample = (attributions_single_sample,)

    assert attributions_single_sample[0].shape[0] == 1

    if feature_mask_single_sample is None:
        feature_mask_single_sample = _construct_default_feature_mask(
            attributions_single_sample
        )

    reduced_attributions = _reduce_attribution_over_features(
        attributions_single_sample,
        feature_mask_single_sample,
        use_weighted_sum=use_weighted_sum,
    )  # [1, G]

    # map each bbox to its feature group id, same way we build modality_ids
    # bboxes_single_sample[mod] is [1, n_groups_mod, 4]
    all_bboxes = torch.cat(
        [bbox[0].float() for bbox in bboxes_single_sample],
        dim=0,  # [n_groups_mod, 4]
    )  # [G_total, 4]

    centers = torch.stack(
        [
            (all_bboxes[:, 0] + all_bboxes[:, 2]) / 2,
            (all_bboxes[:, 1] + all_bboxes[:, 3]) / 2,
        ],
        dim=-1,
    )  # [G_total, 2]

    return _compute_locality_scores(
        attr=reduced_attributions,
        centers=centers,
        target_index=target_index,
        x_threshold=x_threshold,
        y_threshold=y_threshold,
    )  # [3]


def attribution_locality(
    attributions: tuple[Tensor, ...] | list[tuple[Tensor, ...]],
    feature_mask: tuple[Tensor, ...] | None,
    target_indices: list[int],
    bboxes: tuple[Tensor, ...],
    x_threshold: float = 0.025,  # normalized
    y_threshold: float = 0.025,  # normalized
    use_weighted_sum: bool = False,
    return_dict: bool = False,
) -> dict | Tensor | list[Tensor]:
    """Compute attribution locality scores.

    Parameters
    ----------
    x_threshold : float
        Maximum x-distance from target center to count as local (in pixels/units).
    y_threshold : float
        Maximum y-distance from target center to count as local (in pixels/units).

    Returns
    -------
    Tensor of shape [batch_size, n_modalities, 3] where dim -1 is
    [x_locality, y_locality, spread], or list thereof (multi_target).
    If return_dict=True:
        {"x_locality": [batch, n_mod], "y_locality": [batch, n_mod], "spread": [batch, n_mod]}
    """
    with torch.no_grad():
        is_list = isinstance(attributions, list)
        assert len(attributions) == len(target_indices), (
            "Length of attributions list must match length of target_indices"
        )

        scores = []
        for target_index, attribution in zip(target_indices, attributions, strict=True):
            if not isinstance(attribution, tuple):
                attribution = (attribution,)
            if not isinstance(feature_mask, tuple) and feature_mask is not None:
                feature_mask = (feature_mask,)
            if not isinstance(bboxes, tuple):
                bboxes = (bboxes,)

            bsz = attribution[0].size(0)
            batch_scores = []

            for i in range(bsz):
                score = _locality_single_sample(
                    attributions_single_sample=tuple(
                        attr[i].unsqueeze(0) for attr in attribution
                    ),
                    feature_mask_single_sample=(
                        tuple(mask[i].unsqueeze(0) for mask in feature_mask)
                        if feature_mask is not None
                        else None
                    ),
                    bboxes_single_sample=tuple(bbox[i].unsqueeze(0) for bbox in bboxes),
                    target_index=target_index,
                    x_threshold=x_threshold,
                    y_threshold=y_threshold,
                    use_weighted_sum=use_weighted_sum,
                )
                batch_scores.append(score)

            scores.append(torch.stack(batch_scores))  # [batch_size, n_modalities, 3]

        if not is_list:
            scores = scores[0]

        if return_dict:
            if isinstance(scores, list):
                return {
                    "x_locality": [s[..., 0] for s in scores],
                    "y_locality": [s[..., 1] for s in scores],
                    "spread": [s[..., 2] for s in scores],
                }
            return {
                "x_locality": scores[..., 0],  # [batch_size, n_modalities]
                "y_locality": scores[..., 1],
                "spread": scores[..., 2],
            }

        return scores
