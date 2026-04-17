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
) -> Tensor:
    attr = attr.clamp(min=0)  # zero out negative attributions

    if attr.sum() <= 1e-8:
        return torch.zeros(3, device=attr.device)

    # normalize attributions to sum to 1 for weighted locality scores
    w = attr / attr.sum()

    target_center = centers[target_index]
    dx = centers[:, 0] - target_center[0]
    dy = centers[:, 1] - target_center[1]

    x_locality = w[dx.abs() <= x_threshold].sum()
    y_locality = w[dy.abs() <= y_threshold].sum()
    rms_distance = torch.sqrt((w * (dx**2 + dy**2)).sum())

    return torch.stack([x_locality, y_locality, rms_distance])


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
        [bbox[0].float() for bbox in bboxes_single_sample], dim=0  # [n_groups_mod, 4]
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
    bboxes: tuple[Tensor, ...],
    target_index: int,
    x_threshold: float = 50.0,
    y_threshold: float = 50.0,
    use_weighted_sum: bool = False,
    multi_target: bool = False,
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
    [x_locality, y_locality, rms_distance], or list thereof (multi_target).
    If return_dict=True:
        {"x_locality": [batch, n_mod], "y_locality": [batch, n_mod], "rms_distance": [batch, n_mod]}
    """
    is_list = isinstance(attributions, list)
    if multi_target:
        assert is_list, "attributions must be a list of tuples when multi_target=True"
    if not is_list:
        attributions = [attributions]

    scores = []
    for attribution in attributions:
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
                "rms_distance": [s[..., 2] for s in scores],
            }
        return {
            "x_locality": scores[..., 0],  # [batch_size, n_modalities]
            "y_locality": scores[..., 1],
            "rms_distance": scores[..., 2],
        }

    return scores
