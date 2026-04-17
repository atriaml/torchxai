import nltk
import torch
from nltk.corpus import stopwords, wordnet as wn
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


def _compute_topk_semantic_coherence(topk_words: list[str]) -> float:
    """
    Mean pairwise WordNet path similarity among top-k content words.
    1.0 = identical meaning, 0.0 = no relation found.
    """
    # get first noun synset for each word, skip if not found
    synsets = []
    for word in topk_words:
        syns = wn.synsets(word, pos=wn.NOUN)
        if syns:
            synsets.append(syns[0])

    if len(synsets) < 2:
        return 0.0

    scores = []
    for i in range(len(synsets)):
        for j in range(i + 1, len(synsets)):
            sim = synsets[i].path_similarity(synsets[j])
            if sim is not None:
                scores.append(sim)

    return float(sum(scores) / len(scores)) if scores else 0.0


def _compute_text_scores(
    attr: Tensor,  # [G]
    tokens: list[str],  # [G]
    token_labels: list[str],  # [G]
    target_index: int,
    k: int,
) -> dict:
    attr = attr.clamp(min=0)
    attr = attr / (attr.sum() + 1e-8)

    other_mask = torch.ones(attr.shape[0], dtype=torch.bool, device=attr.device)
    other_mask[target_index] = False
    attr_others = attr * other_mask

    topk = _compute_topk_words(attr_others, tokens, k=k)
    content_stop_ratio = _compute_content_stop_ratio(attr_others, tokens)
    span = _compute_span_structure(attr_others, k=k)

    return {
        "target_word": tokens[target_index],
        "topk_words": topk["words"],
        "topk_scores": topk["scores"],
        "topk_indices": topk["indices"],
        "topk_is_stopword": [
            tokens[i].lower().strip() in STOPWORDS for i in topk["indices"].tolist()
        ],
        "content_stop_ratio": content_stop_ratio,
        "span_mean_gap": span[0],
        "span_n_runs": span[1],
        "ner": _compute_ner_attribution(attr_others, token_labels, other_mask),
    }


def _text_analysis_single_sample(
    attributions_single_sample: tuple[Tensor, ...],
    feature_mask_single_sample: tuple[Tensor, ...] | None,
    tokens: list[str],
    token_labels: list[str],
    target_index: int,
    k: int,
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
        target_index=target_index,
        k=k,
    )


def attribution_text_analysis(
    attributions: tuple[Tensor, ...] | list[tuple[Tensor, ...]],
    tokens: list[str],
    token_labels: list[str],
    target_indices: list[int],
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
                    token_labels=token_labels[i],
                    target_index=target_index,
                    k=k,
                    use_weighted_sum=use_weighted_sum,
                )
                batch_results.append(result)
            results.append(batch_results)

        return results if is_list else results[0]
