from typing import Any

import nltk
import numpy as np
import torch
from nltk.corpus import stopwords
from torch import Tensor

from torchxai.metrics._utils.common import (
    _construct_default_feature_mask,
    _reduce_attribution_over_features,
)

nltk.download("wordnet", quiet=True)
nltk.download("stopwords", quiet=True)
STOPWORDS = frozenset(stopwords.words("english"))


def _compute_topk_words(
    attr: Tensor,  # [G]
    tokens: list[str],  # [G]
    k: int | float,  # int for top-k, float (0.0–1.0) for top-%
) -> dict:
    """Top-k words by attribution magnitude."""
    if isinstance(k, float):
        if not 0.0 < k <= 1.0:
            raise ValueError(f"k as a fraction must be in (0.0, 1.0], got {k}")
        k = max(1, int(len(tokens) * k))

    attr_clamped = attr.clamp(min=0)
    attr_norm = attr_clamped / (attr_clamped.sum() + 1e-8)
    topk_vals, topk_idx = torch.topk(attr_norm, k=min(k, len(tokens)))
    return {
        "indices": topk_idx,
        "words": [tokens[i] for i in topk_idx.tolist()],
        "scores": topk_vals,
    }


def _compute_content_stop_ratio(
    attr: Tensor,  # [G]
    tokens: list[str],  # [G]
) -> Tensor:
    """Ratio of content-word attribution to stopword attribution."""
    is_content = torch.tensor(
        [
            t.lower().strip() not in STOPWORDS and any(c.isalpha() for c in t)
            for t in tokens
        ],
        dtype=torch.bool,
        device=attr.device,
    )
    abs_attr = attr.abs()
    content_mean = (
        abs_attr[is_content].mean() if is_content.any() else torch.tensor(0.0)
    )
    stop_mean = (
        abs_attr[~is_content].mean() if (~is_content).any() else torch.tensor(0.0)
    )
    return content_mean / (stop_mean + 1e-10)


def _compute_ner_attribution(
    attr: Tensor,  # [G], already normalized
    token_labels: list[str],
    other_mask: Tensor,  # [G], bool
) -> dict[str, float]:
    attr_masked = attr * other_mask
    result: dict[str, float] = {}
    for i, label in enumerate(token_labels):
        result[label] = result.get(label, 0.0) + attr_masked[i].item()
    return result


def _compute_semantic_attribution_correlation(
    attr: Tensor,  # [G], already normalized
    embeddings: Tensor,  # [G, hidden_size], precomputed word-level
    target_index: int,
    other_mask: Tensor,  # [G], bool
) -> float:
    from scipy.stats import spearmanr
    from torch.nn import functional as F

    target_emb = embeddings[target_index]

    similarities = F.cosine_similarity(
        target_emb.unsqueeze(0),  # [1, hidden_size]
        embeddings,  # [G, hidden_size]
        dim=-1,
    )  # [G]

    mask = other_mask & ~torch.isnan(similarities)
    if mask.sum() < 3:
        return float("nan")

    corr, _ = spearmanr(similarities[mask].cpu().numpy(), attr[mask].cpu().numpy())
    return float(corr)


def _compute_label_attribution_correlation(
    attr: Tensor,  # [G], already normalized
    ner_labels: list[str],  # [G]
    target_index: int,
    other_mask: Tensor,  # [G], bool
    threshold: float = 0.75,
) -> float:
    """
    Spearman correlation between label match with target and attribution score.
    High positive = model attends to tokens with same NER type as target.
    """
    from anls import anls_score
    from scipy.stats import spearmanr

    target_label = ner_labels[target_index]

    # find anls score between the target token and all other tokens based on whether they share the same NER label
    anls_scores = [
        anls_score(
            prediction=ner_labels[i], gold_labels=[target_label], threshold=threshold
        )
        for i in range(len(ner_labels))
    ]
    anls_scores = np.array(anls_scores)
    corr, _ = spearmanr(
        anls_scores[other_mask.cpu().numpy()], attr[other_mask].cpu().numpy()
    )
    return float(corr)


def _compute_topk_target_distance(
    attr: Tensor,  # [G]
    k: int | float,  # int for top-k, float (0.0–1.0) for top-%
    target_index: int,
) -> Tensor:
    """
    Mean and max absolute distance of top-k attributed tokens from the target token.
    Returns [mean_dist, max_dist].
    """
    if isinstance(k, float):
        if not 0.0 < k <= 1.0:
            raise ValueError(f"k as a fraction must be in (0.0, 1.0], got {k}")
        k = max(1, int(attr.shape[0] * k))

    topk_idx = torch.topk(attr.abs(), k=min(k, attr.shape[0])).indices.float()
    dists = (topk_idx - target_index).abs()
    return torch.stack([dists.mean(), dists.max()])


def _compute_text_scores(
    attr: Tensor,  # [G]
    tokens: list[str],  # [G]
    k: int | float,  # int for top-k, float (0.0–1.0) for top-%
    token_embeddings: Tensor | None = None,  # [G, hidden_size]
    target_index: int | None = None,
    token_labels: list[str] | None = None,
) -> dict:
    if len(attr.shape) == 0:
        return {
            "target_word": tokens[target_index] if target_index is not None else None,
            "topk_words": [],
            "topk_scores": [],
            "topk_indices": [],
            "topk_is_stopword": [],
            "content_stop_ratio": 0.0,
            "span_mean_gap": 0.0,
            "span_n_runs": 0,
            "ner": None,
            "ner_corr": None,
            "semantic_corr": None,
        }

    attr = attr.clamp(min=0)
    attr = attr / (attr.sum() + 1e-8)

    other_mask = torch.ones(attr.shape[0], dtype=torch.bool, device=attr.device)
    if target_index is not None:
        other_mask[target_index] = False
    attr_others = attr * other_mask

    topk = _compute_topk_words(attr_others, tokens, k=k)
    content_stop_ratio = _compute_content_stop_ratio(attr_others, tokens)
    topk_dist = (
        _compute_topk_target_distance(attr_others, k=k, target_index=target_index)
        if target_index is not None
        else torch.zeros(2, device=attr.device)
    )

    return {
        "target_word": tokens[target_index] if target_index is not None else None,
        "target_label": token_labels[target_index]
        if token_labels is not None and target_index is not None
        else None,
        "topk_words": topk["words"],
        "topk_scores": topk["scores"],
        "topk_indices": topk["indices"],
        "topk_is_stopword": [
            tokens[i].lower().strip() in STOPWORDS for i in topk["indices"].tolist()
        ],
        "content_stop_ratio": content_stop_ratio,
        "topk_mean_dist": topk_dist[0],
        "topk_max_dist": topk_dist[1],
        "ner": _compute_ner_attribution(attr_others, token_labels, other_mask)
        if token_labels is not None
        else None,
        "ner_corr": _compute_label_attribution_correlation(
            attr_others, token_labels, target_index, other_mask
        )
        if token_labels is not None and target_index is not None
        else None,
        "semantic_corr": _compute_semantic_attribution_correlation(
            attr_others, token_embeddings, target_index, other_mask
        )
        if token_embeddings is not None and target_index is not None
        else None,
    }


def _text_analysis_single_sample(
    attributions_single_sample: tuple[Tensor, ...],
    feature_mask_single_sample: tuple[Tensor, ...] | None,
    tokens: list[str],
    k: int | float,  # int for top-k, float (0.0–1.0) for top-%
    token_embeddings: Tensor | None = None,
    target_index: int | None = None,
    token_labels: list[str] | None = None,
    use_weighted_sum: bool = False,
) -> dict:
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

    return _compute_text_scores(
        attr=reduced_attributions.squeeze(0),
        tokens=tokens,
        token_labels=token_labels,
        token_embeddings=token_embeddings,
        target_index=target_index,
        k=k,
    )


def attribution_text_analysis(
    attributions: tuple[Tensor, ...] | list[tuple[Tensor, ...]],
    tokens: list[list[str]],
    token_embeddings: list[Tensor] | None = None,
    target_indices: list[int] | None = None,
    token_labels: list[list[str]] | None = None,
    feature_mask: tuple[Tensor, ...] | None = None,
    k: int | float = 0.1,
    use_weighted_sum: bool = False,
) -> list[list[dict[Any, Any]]] | list[dict[Any, Any]]:
    """Analyses token-level attributions and extracts interpretability diagnostics for text inputs.

    For each sample, extracts the top-k most-attributed words, computes the content-to-stopword
    attribution ratio, measures the mean and max distance of top-k tokens from the target token,
    and optionally correlates attributions with NER labels or token embeddings.

    Args:
        attributions (tuple[Tensor, ...] or list[tuple[Tensor, ...]]): Attribution tensors, one
            per modality, each of shape ``(batch_size, n_features)``. For multi-target mode, pass
            a list of such tuples.
        tokens (list[list[str]]): Word-level token strings per sample, length matching the number
            of attribution feature groups ``G``.
        token_embeddings (list[Tensor] or None): Precomputed word embeddings per sample, each of
            shape ``(G, hidden_size)``. When provided, a Spearman correlation between attribution
            and semantic similarity to the target token is computed. Default: ``None``.
        target_indices (list[int] or None): Index of the target token per sample. Used to exclude
            the target from top-k and to compute distance metrics. Default: ``None``.
        token_labels (list[list[str]] or None): NER or other string label per token per sample.
            When provided, per-label attribution sums and a label-match correlation are computed.
            Default: ``None``.
        feature_mask (tuple[Tensor, ...] or None): Feature group masks of the same shape as
            ``attributions``, used to pool sub-word tokens into word-level groups before scoring.
            If ``None``, each feature is its own group.
        k (int or float): Number of top tokens to extract. An ``int`` selects exactly ``k`` tokens;
            a ``float`` in ``(0.0, 1.0]`` selects that fraction of all tokens. Default: ``0.1``.
        use_weighted_sum (bool): If ``True``, use weighted-sum pooling when reducing features to
            groups via ``feature_mask``. Default: ``False``.

    Returns:
        list[dict] or list[list[dict]]: Per-sample result dicts (or a list thereof for multi-target
        input). Each dict contains:

        - ``topk_words`` – list of top-k token strings.
        - ``topk_scores`` – normalised attribution scores for each top-k token.
        - ``topk_indices`` – token indices of the top-k tokens.
        - ``topk_is_stopword`` – boolean list indicating whether each top-k token is a stopword.
        - ``content_stop_ratio`` – ratio of mean content-word attribution to mean stopword attribution.
        - ``topk_mean_dist`` – mean absolute distance of top-k tokens from the target token.
        - ``topk_max_dist`` – maximum absolute distance of top-k tokens from the target token.
        - ``ner`` – per-label attribution sums (only when ``token_labels`` is provided).
        - ``ner_corr`` – Spearman correlation between label match and attribution (only when
          ``token_labels`` and ``target_indices`` are provided).
        - ``semantic_corr`` – Spearman correlation between cosine similarity to the target
          embedding and attribution (only when ``token_embeddings`` and ``target_indices``
          are provided).

    Example:
        >>> import torch
        >>> attr = (torch.tensor([[0.05, 0.4, 0.1, 0.3, 0.15]]),)
        >>> tokens_batch = [["The", "cat", "sat", "on", "mat"]]
        >>> results = attribution_text_analysis(attr, tokens=tokens_batch, k=0.4)
        >>> results[0]["topk_words"]
        ['cat', 'on']
    """
    with torch.no_grad():
        is_list = isinstance(attributions, list)
        if not is_list:
            attributions = [attributions]  # type: ignore

        target_indices_formatted = (
            [None] * len(attributions) if target_indices is None else target_indices
        )
        assert len(attributions) == len(target_indices_formatted)

        results = []
        for target_index, attribution in zip(
            target_indices_formatted, attributions, strict=True
        ):
            if not isinstance(attribution, tuple):
                attribution = (attribution,)
            if not isinstance(feature_mask, tuple) and feature_mask is not None:
                feature_mask = (feature_mask,)

            bsz = attribution[0].size(0)
            batch_results = []

            for i in range(bsz):
                result = _text_analysis_single_sample(
                    attributions_single_sample=tuple(
                        a[i].unsqueeze(0) for a in attribution
                    ),
                    feature_mask_single_sample=(
                        tuple(m[i].unsqueeze(0) for m in feature_mask)
                        if feature_mask is not None
                        else None
                    ),
                    tokens=tokens[i],
                    token_labels=token_labels[i] if token_labels is not None else None,
                    token_embeddings=token_embeddings[i]
                    if token_embeddings is not None
                    else None,
                    target_index=target_index,
                    k=k,
                    use_weighted_sum=use_weighted_sum,
                )
                batch_results.append(result)
            results.append(batch_results)

        return results if is_list else results[0]
