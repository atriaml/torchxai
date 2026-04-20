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
    k: int,
) -> dict:
    """Top-k words by attribution magnitude."""
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


def _compute_span_structure(
    attr: Tensor,  # [G]
    k: int,
) -> Tensor:
    """
    Mean gap and number of runs among top-k attributed tokens.
    Distinguishes clustered-phrase vs scattered-keyword reasoning.
    Returns [mean_gap, n_runs].
    """
    topk_idx = torch.topk(attr.abs(), k=min(k, attr.shape[0])).indices
    topk_idx, _ = topk_idx.sort()

    if len(topk_idx) > 1:
        gaps = topk_idx[1:].float() - topk_idx[:-1].float()
        mean_gap = gaps.mean()
        n_runs = 1 + (gaps > 1).sum()
    else:
        mean_gap = torch.tensor(0.0, device=attr.device)
        n_runs = torch.tensor(len(topk_idx), device=attr.device)

    return torch.stack([mean_gap, n_runs.float()])


def _compute_ner_attribution(
    attr: Tensor,  # [G], already normalized
    token_labels: list[str],
    other_mask: Tensor,  # [G], bool
) -> dict[str, float]:
    attr_masked = attr * other_mask
    result = {}
    for i, label in enumerate(token_labels):
        result[label] = result.get(label, 0.0) + attr_masked[i].item()
    entity_mass = sum(v for k, v in result.items() if k != "O")
    o_mass = result.get("O", 0.0)
    result["entity_focus_ratio"] = entity_mass / (o_mass + 1e-10)
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
            prediction=ner_labels[i].name,
            gold_labels=[target_label.name],
            threshold=0.5,
        )
        for i in range(len(ner_labels))
    ]
    anls_scores = np.array(anls_scores)
    corr, _ = spearmanr(
        anls_scores[other_mask.cpu().numpy()], attr[other_mask].cpu().numpy()
    )
    return float(corr)


def _compute_text_scores(
    attr: Tensor,  # [G]
    tokens: list[str],  # [G]
    k: int,
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
    span = _compute_span_structure(attr_others, k=k)

    return {
        "target_word": tokens[target_index] if target_index is not None else None,
        "topk_words": topk["words"],
        "topk_scores": topk["scores"],
        "topk_indices": topk["indices"],
        "topk_is_stopword": [
            tokens[i].lower().strip() in STOPWORDS for i in topk["indices"].tolist()
        ],
        "content_stop_ratio": content_stop_ratio,
        "span_mean_gap": span[0],
        "span_n_runs": span[1],
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
    k: int,
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
    k: int = 10,
    use_weighted_sum: bool = False,
) -> list[dict] | dict:
    """
    Parameters
    ----------
    tokens : list[str]
        Word-level tokens, length G (must match attribution feature groups).
    target_indices : list[int]
        Index of the target token per sample.
    k : int
        Number of top tokens to extract.

    Returns
    -------
    Per-sample dict (or list of dicts) with keys:
        topk_words, topk_scores, topk_indices,
        stopword_ratio, span_mean_gap, span_n_runs
    """
    with torch.no_grad():
        is_list = isinstance(attributions, list)
        if not is_list:
            attributions = [attributions]

        if target_indices is None:
            target_indices = [None] * len(attributions)

        assert len(attributions) == len(target_indices)

        results = []
        for target_index, attribution in zip(target_indices, attributions, strict=True):
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
