"""
Modality Top-K Fraction metric.

Given attributions as a tuple of tensors (one per modality / feature group),
measures which modality contributes the most important features by counting
membership in the global top-k.

Input:
    attributions: tuple[Tensor, ...] — each [batch_size, n_features_i].
    feature_mask: optional tuple[Tensor, ...] — groups raw features into
                  semantic units (e.g. tokens → words). Same shape as attributions.
    k_fraction: float — fraction of total (reduced) features to select.
    reduce_mode: "sum" or "weighted_sum"
        - "sum": simple sum per feature group
        - "weighted_sum": mean per group × min_group_size across all modalities
                          (penalizes groups larger than the minimum)

Output:
    Tensor of shape [batch_size, n_modalities] — fraction of top-k from each modality.

Category: characterization
"""

from collections.abc import Sequence

import torch
from torch import Tensor

from torchxai.metrics._utils.common import (
    _construct_default_feature_mask,
    _reduce_attribution_over_features,
)


def _compute_topk_mass_fractions(
    all_reduced: Tensor,
    modality_ids: Tensor,
    n_modalities: int,
    k_fractions: Sequence[float] = (0.10,),
) -> Tensor:
    """Compute attribution mass fractions for each k and modality.

    Parameters
    ----------
    all_reduced : Tensor of shape [G_total]
    modality_ids : Tensor of shape [G_total]
    n_modalities : int
    k_fractions : list of float

    Returns
    -------
    Tensor of shape [n_k_fractions, n_modalities]
    """
    total_groups = all_reduced.shape[0]
    results = []

    for k_fraction in k_fractions:
        k = max(1, int(total_groups * k_fraction + 0.9999))

        topk_vals, topk_indices = all_reduced.topk(k, largest=True, sorted=False)
        topk_mods = modality_ids[topk_indices]
        total_mass = topk_vals.sum()

        fracs = torch.zeros(n_modalities, device=all_reduced.device)
        for modality_id in range(n_modalities):
            modality_mask = topk_mods == modality_id
            fracs[modality_id] = topk_vals[modality_mask].sum() / total_mass

        results.append(fracs)

    return torch.stack(results, dim=0)  # [n_k_fractions, n_modalities]


def _modality_topk_fraction_single_sample(
    attributions_single_sample: tuple[Tensor, ...],
    feature_mask_single_sample: tuple[Tensor, ...] | None,
    k_fractions: Sequence[float],
    use_weighted_sum: bool,
) -> Tensor:
    """
    Returns
    -------
    Tensor of shape [n_k_fractions, n_modalities]
    """
    n_modalities = len(attributions_single_sample)

    if feature_mask_single_sample is None:
        feature_mask_single_sample = _construct_default_feature_mask(
            attributions_single_sample
        )

    reduced_attributions = _reduce_attribution_over_features(
        attributions_single_sample,
        feature_mask_single_sample,
        use_weighted_sum=use_weighted_sum,
    )  # [1, G_total]

    modality_ids = torch.cat(
        [
            torch.full(
                (mask[0].unique().shape[0],),
                modality_id,
                dtype=torch.long,
                device=reduced_attributions.device,
            )
            for modality_id, mask in enumerate(feature_mask_single_sample)
        ]
    )  # [G_total]

    return _compute_topk_mass_fractions(
        all_reduced=reduced_attributions,
        modality_ids=modality_ids,
        n_modalities=n_modalities,
        k_fractions=k_fractions,
    )  # [n_k_fractions, n_modalities]


def _modality_topk_fraction_per_target(
    attributions: tuple[Tensor, ...],
    feature_mask: tuple[Tensor, ...] | None = None,
    k_fractions: Sequence[float] = (0.10,),
    reduce_mode: str = "sum",
) -> Tensor:
    if not isinstance(attributions, tuple):
        attributions = (attributions,)
    if not isinstance(feature_mask, tuple) and feature_mask is not None:
        feature_mask = (feature_mask,)
    use_weighted_sum = reduce_mode == "weighted_sum"
    bsz = attributions[0].shape[0]

    with torch.no_grad():
        batch_fracs = []
        for i in range(bsz):
            fracs = _modality_topk_fraction_single_sample(
                attributions_single_sample=tuple(
                    attr[i].unsqueeze(0) for attr in attributions
                ),
                feature_mask_single_sample=(
                    tuple(mask[i].unsqueeze(0) for mask in feature_mask)
                    if feature_mask is not None
                    else None
                ),
                k_fractions=k_fractions,
                use_weighted_sum=use_weighted_sum,
            )
            batch_fracs.append(fracs)

    return torch.stack(batch_fracs, dim=0)  # [batch_size, n_k_fractions, n_modalities]


def modality_topk_fraction(
    attributions: tuple[Tensor, ...] | list[tuple[Tensor, ...]],
    feature_mask: tuple[Tensor, ...] | None = None,
    modality_names: list[str] | None = None,
    k_fractions: Sequence[float] = (0.05, 0.10, 0.20),
    reduce_mode: str = "sum",
    multi_target: bool = False,
    return_dict: bool = True,
) -> dict | Tensor | list[Tensor]:
    """
    Returns
    -------
    Tensor of shape [batch_size, n_k_fractions, n_modalities], or list thereof.
    If return_dict=True, returns dict with one key per (k_fraction, modality):
        {"score_k0.05_text": ..., "score_k0.05_image": ..., "score_k0.10_text": ..., ...}
    each value of shape [batch_size].

    Examples
    --------
    >>> text_attr = torch.randn(2, 50)
    >>> img_attr = torch.randn(2, 100)
    >>> result = modality_topk_fraction(
    ...     (text_attr, img_attr),
    ...     modality_names=["text", "image"],
    ...     k_fractions=[0.05, 0.10, 0.20],
    ... )
    >>> result.keys()
    dict_keys(['score_k0.05_text', 'score_k0.05_image', 'score_k0.10_text', ...])
    """
    is_list = isinstance(attributions, list)
    if multi_target:
        assert is_list, "attributions must be a list of tuples when multi_target=True"
    if not is_list:
        attributions = [attributions]  # type: ignore

    n_modalities = len(attributions[0])
    names = (
        modality_names
        if modality_names is not None
        else [f"modality_{i}" for i in range(n_modalities)]
    )
    assert len(names) == n_modalities, (
        f"modality_names length {len(names)} must match n_modalities {n_modalities}"
    )

    scores = [
        _modality_topk_fraction_per_target(
            attrs,
            feature_mask=feature_mask,
            k_fractions=k_fractions,
            reduce_mode=reduce_mode,
        )
        for attrs in attributions
    ]  # list of [batch_size, n_k_fractions, n_modalities]

    if not is_list:
        scores = scores[0]

    if return_dict:
        if isinstance(scores, list):
            # multi_target: scores is list of [batch_size, n_k_fractions, n_modalities]
            return {
                f"score_{name}": [s[:, :, j].mean(dim=1) for s in scores]
                for j, name in enumerate(names)
            }
        else:
            # single target: scores is [batch_size, n_k_fractions, n_modalities]
            return {
                f"score_{name}": scores[:, :, j].mean(dim=1)
                for j, name in enumerate(names)
            }

    return scores
