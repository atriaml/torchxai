import torch

from torchxai.metrics.diagnosis.attribution_text_analysis import (
    _compute_semantic_attribution_correlation,
)


def test_semantic_attribution_correlation():
    # tokens where semantic similarity to target ("dog") is known:
    # cat, wolf are similar to dog; car, table are not
    tokens = ["cat", "wolf", "car", "table", "dog"]
    target_index = 4  # "dog"

    # attribution scores should correlate: cat/wolf get high, car/table get low
    attr = torch.tensor([0.35, 0.30, 0.05, 0.05, 0.25])
    attr = attr / attr.sum()

    other_mask = torch.ones(len(tokens), dtype=torch.bool)
    other_mask[target_index] = False

    corr = _compute_semantic_attribution_correlation(
        attr=attr, tokens=tokens, target_index=target_index, other_mask=other_mask
    )

    # sanity checks
    assert not torch.isnan(torch.tensor(corr)), (
        "should have enough tokens for correlation"
    )
    assert corr > 0.0, f"expected positive correlation, got {corr:.3f}"

    # also test near-zero case: attribution inversely matches similarity
    attr_inverted = torch.tensor([0.05, 0.05, 0.35, 0.30, 0.25])
    attr_inverted = attr_inverted / attr_inverted.sum()

    print("attr", attr)
    print("tokens", tokens)
    print("target_index", target_index)
    print("other_mask", other_mask)
    print("corr", corr)
    corr_inverted = _compute_semantic_attribution_correlation(
        attr=attr_inverted,
        tokens=tokens,
        target_index=target_index,
        other_mask=other_mask,
    )
    assert corr_inverted < 0.0, (
        f"expected negative correlation, got {corr_inverted:.3f}"
    )

    # test nan when target has no wordnet synset
    tokens_no_syn = ["cat", "wolf", "car", "table", "xkdjfhg"]
    corr_nan = _compute_semantic_attribution_correlation(
        attr=attr, tokens=tokens_no_syn, target_index=4, other_mask=other_mask
    )
    print("attr", attr)
    print("tokens", tokens)
    print("target_index", target_index)
    print("other_mask", other_mask)
    print("corr", corr_nan)
    assert corr_nan != corr_nan, "expected nan for unknown target word"  # nan != nan


test_semantic_attribution_correlation()
