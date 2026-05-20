---
title: Token Classification
---

# Token Classification

Multi-target attribution on BERT-NER explaining all token predictions in a single call.
Each section covers one **input pattern** — the minimal set of arguments that pattern of explainer requires.

We wrap the NER model so it returns one value per token — the logit of that token's predicted class.
This keeps targets simple: `SingleTargetAcrossBatch(index=token_position)` selects the logit at that position from the `(1, seq_len)` output.

!!! note "Gradient-based methods require embedding-level inputs"
    BERT token IDs are discrete integers — gradients cannot flow through them.
    We pass the continuous **embedding tensor** as input and wrap the model to accept embeddings via `inputs_embeds`.

## Setup

```python
import time

import torch
from transformers import BertForTokenClassification, BertTokenizerFast

from torchxai.data_types import SingleTargetAcrossBatch

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

ner_model = BertForTokenClassification.from_pretrained(
    "bert-base-uncased", num_labels=9   # random weights — for shape illustration only
).eval().cuda()

text           = "John works at Google"
enc            = tokenizer(text, return_tensors="pt")
input_ids      = enc["input_ids"].cuda()
attention_mask = enc["attention_mask"].cuda()
word_ids       = enc.word_ids(batch_index=0)   # e.g. [None, 0, 1, 2, 3, None]

embeddings   = ner_model.bert.embeddings(input_ids)   # (1, seq_len, 768)
baseline_emb = torch.zeros_like(embeddings)

# Get the predicted label for each token (used to build the wrapper)
with torch.no_grad():
    logits_full = ner_model(
        inputs_embeds=embeddings, attention_mask=attention_mask
    ).logits                               # (1, seq_len, num_labels)
    pred_labels = logits_full.argmax(-1)   # (1, seq_len)


class NERPredictedLogitWrapper(torch.nn.Module):
    """Returns the predicted-class logit for each token → output (1, seq_len)."""

    def forward(self, emb):
        logits = ner_model(inputs_embeds=emb, attention_mask=attention_mask).logits
        return logits.gather(-1, pred_labels.unsqueeze(-1)).squeeze(-1)


wrapped_ner = NERPredictedLogitWrapper().eval().cuda()

# One target per real (non-special) token position
targets_ner = [
    SingleTargetAcrossBatch(index=t_idx)
    for t_idx, w_idx in enumerate(word_ids)
    if w_idx is not None
]


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

## Feature mask: token → word grouping

Shape `(1, seq_len, 1)`: all 768 dims of the same token share one feature ID.
Subword tokens of the same word are assigned the same word-level ID.
Used in Pattern D.

```python
seq_len      = input_ids.shape[1]
feature_mask = torch.zeros(1, seq_len, 1, dtype=torch.long, device=input_ids.device)
for t_idx, w_idx in enumerate(word_ids):
    if w_idx is not None:
        feature_mask[0, t_idx, 0] = w_idx + 1   # 0 reserved for [CLS] / [SEP]

ids    = input_ids[0].tolist()
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

print("Token debug view:")
print(f"{'idx':>3}  {'id':>6}  {'word_id':>7}  {'fmask':>5}  token")
print("-" * 46)
feature_mask_ids = feature_mask[0, :, 0].tolist()
for i, (tid, tok, wid, fmask) in enumerate(
    zip(ids, tokens, word_ids, feature_mask_ids, strict=True)
):
    print(f"{i:>3}  {tid:>6}  {str(wid):>7}  {fmask:>5}  {tok}")
```

---

## Pattern A — inputs + target

No baseline or mask required. Applies to: `SaliencyExplainer`, `InputXGradientExplainer`.

```python
from torchxai.explainers import SaliencyExplainer

# Replace with InputXGradientExplainer as needed

result = compare(SaliencyExplainer, wrapped_ner, {"inputs": embeddings}, targets_ner)
print(f"Output shape={result.shape}, mean={result.mean().item():.3f}, std={result.std().item():.3f}, min={result.min().item():.3f}, max={result.max().item():.3f}")
```

---

## Pattern B — inputs + baseline + target

A single reference tensor (same shape as `inputs`) is required. Applies to: `IntegratedGradientsExplainer`, `InputXBaselineGradientExplainer`.

```python
from torchxai.explainers import IntegratedGradientsExplainer

# Replace with InputXBaselineGradientExplainer as needed

result = compare(IntegratedGradientsExplainer, wrapped_ner,
        {"inputs": embeddings, "baselines": baseline_emb}, targets_ner)
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

result = compare(GradientShapExplainer, wrapped_ner,
        {"inputs": embeddings, "baselines": baselines_dist}, targets_ner, n_samples=200, atol=0.1)
print(f"Output shape={result.shape}, mean={result.mean().item():.3f}, std={result.std().item():.3f}, min={result.min().item():.3f}, max={result.max().item():.3f}")
```

---

## Pattern D — inputs + feature_mask + target

A `feature_mask` groups embedding dimensions into segments so the explainer scores whole words rather than individual tokens. Applies to: `FeatureAblationExplainer`, `LimeExplainer`, `KernelShapExplainer`.

```python
from torchxai.explainers import FeatureAblationExplainer

# Replace with LimeExplainer or KernelShapExplainer as needed

result = compare(FeatureAblationExplainer, wrapped_ner,
        {"inputs": embeddings, "feature_mask": feature_mask}, targets_ner)
print(f"Output shape={result.shape}, mean={result.mean().item():.3f}, std={result.std().item():.3f}, min={result.min().item():.3f}, max={result.max().item():.3f}")
```

---

## Pattern E — inputs + sliding_window_shapes + target

`OcclusionExplainer` patches out rectangular windows of the embedding sequence. The `sliding_window_shapes` tuple `(tokens, dims)` specifies the window size.

```python
from torchxai.explainers import OcclusionExplainer

result = compare(OcclusionExplainer, wrapped_ner,
        {"inputs": embeddings, "sliding_window_shapes": (2, 768), "strides": (1, 768)}, targets_ner)
print(f"Output shape={result.shape}, mean={result.mean().item():.3f}, std={result.std().item():.3f}, min={result.min().item():.3f}, max={result.max().item():.3f}")
```
