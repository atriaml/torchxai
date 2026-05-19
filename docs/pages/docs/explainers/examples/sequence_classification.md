---
title: Sequence Classification
---

# Sequence Classification

!!! note "Gradient-based methods require embedding-level inputs"
    BERT token IDs are discrete integers — gradients cannot flow through them.
    For gradient-based explainers (`Saliency`, `IntegratedGradients`, etc.) we pass the continuous **embedding tensor** as input. For perturbation-based explainers (`FeatureAblation`, `LIME`, etc.) we operate on embeddings as well, using a word-level `feature_mask` to score whole words.


## Setup

```python
import time

import torch
from transformers import BertForSequenceClassification, BertTokenizerFast

from torchxai.data_types import SingleTargetAcrossBatch

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2
).eval().cuda()

text = "This movie was absolutely fantastic, but the subplot felt unexpectedly overdramatized."
enc = tokenizer(text, return_tensors="pt")
input_ids      = enc["input_ids"].cuda()       # (1, seq_len)
attention_mask = enc["attention_mask"].cuda()  # (1, seq_len)

# Embeddings — used as input for all explainers
embeddings   = model.bert.embeddings(input_ids)   # (1, seq_len, 768)
baseline_emb = torch.zeros_like(embeddings)


class EmbeddingSeqCls(torch.nn.Module):
    def forward(self, emb):
        return model(inputs_embeds=emb, attention_mask=attention_mask).logits


embed_model = EmbeddingSeqCls().eval().cuda()

# 2 output classes (positive / negative)
targets = [SingleTargetAcrossBatch(index=i) for i in range(2)]


def compare(explainer_cls, model, explain_kwargs, targets, atol=1e-5, **init_kwargs):
    """Compare sequential single-target calls vs one multi-target call.
    Verifies results match and reports timing and attribution shape.
    Pass atol>1e-5 for stochastic methods (e.g. GradientShap).
    """
    explainer = explainer_cls(model, multi_target=False, **init_kwargs)
    t0 = time.perf_counter()
    attrs = [explainer.explain(**explain_kwargs, target=t) for t in targets]
    elapsed_single = time.perf_counter() - t0
    attrs_tensor = torch.stack(attrs)

    explainer_mt = explainer_cls(model, multi_target=True, **init_kwargs)
    t0 = time.perf_counter()
    attrs_mt = explainer_mt.explain(**explain_kwargs, target=targets)
    elapsed_mt = time.perf_counter() - t0
    attrs_mt_tensor = torch.stack(attrs_mt)

    assert torch.allclose(attrs_tensor, attrs_mt_tensor, atol=atol), (
        f"Results differ between single-target and multi-target, max diff: {(attrs_tensor - attrs_mt_tensor).abs().max().item():.3e}"
    )
    speedup = elapsed_single / elapsed_mt if elapsed_mt > 0 else float("inf")
    print(f"shape  : {attrs_mt_tensor.shape}")
    print(
        f"single : {elapsed_single:.3f}s  |  multi : {elapsed_mt:.3f}s  |  speedup : {speedup:.1f}x"
    )
    return attrs_mt_tensor
```

---

## Pattern A — inputs + target

No baseline or mask required. Applies to: `SaliencyExplainer`, `InputXGradientExplainer`.

```python
from torchxai.explainers import SaliencyExplainer

# Replace with InputXGradientExplainer as needed

result = compare(SaliencyExplainer, embed_model, {"inputs": embeddings}, targets)
print(f"Output shape={result.shape}, mean={result.mean().item():.3f}, std={result.std().item():.3f}, min={result.min().item():.3f}, max={result.max().item():.3f}")
```

---

## Pattern B — inputs + baseline + target

A single reference tensor (same shape as `inputs`) is required. Applies to: `IntegratedGradientsExplainer`, `InputXBaselineGradientExplainer`.

```python
from torchxai.explainers import IntegratedGradientsExplainer

# Replace with InputXBaselineGradientExplainer as needed

result = compare(IntegratedGradientsExplainer, embed_model,
        {"inputs": embeddings, "baselines": baseline_emb}, targets)
print(f"Output shape={result.shape}, mean={result.mean().item():.3f}, std={result.std().item():.3f}, min={result.min().item():.3f}, max={result.max().item():.3f}")
```

---

## Pattern C — inputs + baseline distribution + target

`baselines` is a **stacked set of reference samples** rather than a single tensor. Applies to: `GradientShapExplainer`.

!!! note
    GradientShap has internal non-determinism due to random sampling. Use `n_samples=200` and a higher `atol` for reliable comparison.

```python
from torchxai.explainers import GradientShapExplainer

baselines_dist = baseline_emb.expand(5, -1, -1)   # (5, seq_len, 768) reference distribution

result = compare(GradientShapExplainer, embed_model,
        {"inputs": embeddings, "baselines": baselines_dist}, targets, n_samples=200, atol=0.1)
print(f"Output shape={result.shape}, mean={result.mean().item():.3f}, std={result.std().item():.3f}, min={result.min().item():.3f}, max={result.max().item():.3f}")
```

---

## Pattern D — inputs + feature_mask + target

A word-level `feature_mask` groups subword tokens so the explainer scores whole words rather than individual tokens. Applies to: `FeatureAblationExplainer`, `LimeExplainer`, `KernelShapExplainer`.

```python
from torchxai.explainers import FeatureAblationExplainer

# Replace with LimeExplainer or KernelShapExplainer as needed

# Word-level feature mask: subword tokens of the same word share an ID
word_ids     = enc.word_ids(batch_index=0)   # e.g. [None, 0, 1, 1, 2, None]
seq_len      = input_ids.shape[1]

ids    = input_ids[0].tolist()
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

# Shape (1, seq_len, 1): broadcast across 768 embedding dims
feature_mask = torch.zeros(1, seq_len, 1, dtype=torch.long, device=input_ids.device)
for t_idx, w_idx in enumerate(word_ids):
    if w_idx is not None:
        feature_mask[0, t_idx, 0] = w_idx + 1  # 0 reserved for special tokens

print("Token debug view:")
print(f"{'idx':>3}  {'id':>6}  {'word_id':>7}  {'fmask':>5}  token")
print("-" * 46)
feature_mask_ids = feature_mask[0, :, 0].tolist()
for i, (tid, tok, wid, fmask) in enumerate(
    zip(ids, tokens, word_ids, feature_mask_ids, strict=True)
):
    print(f"{i:>3}  {tid:>6}  {str(wid):>7}  {fmask:>5}  {tok}")

print("\nWith word-level feature mask:")
result = compare(FeatureAblationExplainer, embed_model,
        {"inputs": embeddings, "feature_mask": feature_mask}, targets)
print(f"Output shape={result.shape}, mean={result.mean().item():.3f}, std={result.std().item():.3f}, min={result.min().item():.3f}, max={result.max().item():.3f}")
```

---

## Pattern E — inputs + sliding_window_shapes + target

`OcclusionExplainer` patches out rectangular windows of the embedding sequence. The `sliding_window_shapes` tuple `(tokens, dims)` specifies the window size.

```python
from torchxai.explainers import OcclusionExplainer

result = compare(OcclusionExplainer, embed_model,
        {"inputs": embeddings, "sliding_window_shapes": (4, 768), "strides": (2, 768)}, targets)
print(f"Output shape={result.shape}, mean={result.mean().item():.3f}, std={result.std().item():.3f}, min={result.min().item():.3f}, max={result.max().item():.3f}")
```
