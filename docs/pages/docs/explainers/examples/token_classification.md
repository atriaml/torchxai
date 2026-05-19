---
title: Token Classification
summary: Multi-target attribution on BERT-NER explaining all token predictions in a single call
---

# BERT — Token Classification

In NER each non-special token has its own predicted label. Running attributions for every token sequentially is expensive. With `multi_target=True` we explain all token predictions in a single call.

We wrap the NER model so it returns one value per token — the logit of that token's predicted class. This keeps targets simple: `SingleTargetAcrossBatch(index=token_position)` selects the logit at that position from the `(1, seq_len)` output.

!!! note "Gradient-based methods require embedding-level inputs"
    BERT token IDs are discrete integers — gradients cannot flow through them.
    We pass the continuous **embedding tensor** as input and wrap the model to accept embeddings via `inputs_embeds`.

!!! warning "Unsupported explainers for transformer architectures"
    `DeepLiftExplainer`, `DeepLiftShapExplainer`, and `GuidedBackpropExplainer` are not compatible with transformers out of the box. Use `IntegratedGradientsExplainer`, `GradientShapExplainer`, or `SaliencyExplainer` as alternatives.

---

## Setup

```python
import time
import torch
from transformers import BertTokenizerFast, BertForTokenClassification
from torchxai.data_types import SingleTargetAcrossBatch

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

ner_model = BertForTokenClassification.from_pretrained(
    "bert-base-uncased", num_labels=9   # random weights — for shape illustration only
).eval()

text           = "John works at Google"
enc            = tokenizer(text, return_tensors="pt")
input_ids      = enc["input_ids"]
attention_mask = enc["attention_mask"]
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

wrapped_ner = NERPredictedLogitWrapper().eval()


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

    assert torch.allclose(attrs_tensor, attrs_mt_tensor, atol=atol), \
        "Results differ between single-target and multi-target"
    speedup = elapsed_single / elapsed_mt if elapsed_mt > 0 else float("inf")
    print(f"shape  : {attrs_mt_tensor.shape}")
    print(f"single : {elapsed_single:.3f}s  |  multi : {elapsed_mt:.3f}s  |  speedup : {speedup:.1f}x")
```

---

## Feature mask: token → word grouping

```python
seq_len      = input_ids.shape[1]
# Shape (1, seq_len, 1): all 768 dims of the same token share one feature ID.
# Subword tokens of the same word are assigned the same word-level ID.
feature_mask = torch.zeros(1, seq_len, 1, dtype=torch.long)
for t_idx, w_idx in enumerate(word_ids):
    if w_idx is not None:
        feature_mask[0, t_idx, 0] = w_idx + 1   # 0 reserved for [CLS] / [SEP]
```

---

## Comparing single-target vs multi-target across all tokens

```python
from torchxai.explainers import IntegratedGradientsExplainer

# One target per real (non-special) token position
targets_ner = [
    SingleTargetAcrossBatch(index=t_idx)
    for t_idx, w_idx in enumerate(word_ids) if w_idx is not None
]

compare(IntegratedGradientsExplainer, wrapped_ner,
        dict(inputs=embeddings, baselines=baseline_emb, feature_mask=feature_mask),
        targets_ner)
```
