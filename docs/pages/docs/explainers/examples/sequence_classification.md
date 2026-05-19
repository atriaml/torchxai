---
title: Sequence Classification
summary: Multi-target attribution on BERT for sentence-level classification
---

# Sequence Classification

!!! note "Gradient-based methods require embedding-level inputs"
    BERT token IDs are discrete integers — gradients cannot flow through them.
    For gradient-based explainers (`Saliency`, `IntegratedGradients`, etc.) we pass the continuous **embedding tensor** as input. For perturbation-based explainers (`FeatureAblation`, `LIME`, etc.) we operate on embeddings as well, using a word-level `feature_mask` to score whole words.

!!! warning "Unsupported explainers for transformer architectures"
    `DeepLiftExplainer`, `DeepLiftShapExplainer`, and `GuidedBackpropExplainer` are not compatible with transformers out of the box. DeepLift requires specific activation types not present in BERT; GuidedBackprop requires ReLU activations throughout the network. Use `IntegratedGradientsExplainer`, `GradientShapExplainer`, or `SaliencyExplainer` as alternatives.

---

## Setup

```python
import time
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
from torchxai.data_types import SingleTargetAcrossBatch

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2
).eval()

text = "This movie was absolutely fantastic!"
enc = tokenizer(text, return_tensors="pt")
input_ids      = enc["input_ids"]       # (1, seq_len)
attention_mask = enc["attention_mask"]  # (1, seq_len)

# Embeddings — used as input for all explainers
embeddings   = model.bert.embeddings(input_ids)   # (1, seq_len, 768)
baseline_emb = torch.zeros_like(embeddings)

class EmbeddingSeqCls(torch.nn.Module):
    def forward(self, emb):
        return model(inputs_embeds=emb, attention_mask=attention_mask).logits

embed_model = EmbeddingSeqCls().eval()

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

    assert torch.allclose(attrs_tensor, attrs_mt_tensor, atol=atol), \
        "Results differ between single-target and multi-target"
    speedup = elapsed_single / elapsed_mt if elapsed_mt > 0 else float("inf")
    print(f"shape  : {attrs_mt_tensor.shape}")
    print(f"single : {elapsed_single:.3f}s  |  multi : {elapsed_mt:.3f}s  |  speedup : {speedup:.1f}x")
```

---

## Pattern A — inputs + target (gradient-based, no baseline)

Applies to: `SaliencyExplainer`, `InputXGradientExplainer`.

```python
from torchxai.explainers import SaliencyExplainer

# Replace with InputXGradientExplainer as needed

compare(SaliencyExplainer, embed_model, dict(inputs=embeddings), targets)
```

---

## Pattern B — inputs + baseline + target (gradient-based, with baseline)

Applies to: `IntegratedGradientsExplainer`, `InputXBaselineGradientExplainer`.

```python
from torchxai.explainers import IntegratedGradientsExplainer

# Replace with InputXBaselineGradientExplainer as needed

compare(IntegratedGradientsExplainer, embed_model,
        dict(inputs=embeddings, baselines=baseline_emb), targets)
```

---

## Pattern C — inputs + baseline distribution + target

`baselines` is a **stacked set of reference samples** rather than a single tensor.

Applies to: `GradientShapExplainer`.

```python
from torchxai.explainers import GradientShapExplainer

baselines_dist = baseline_emb.expand(5, -1, -1, -1)   # (5, seq_len, 768) reference distribution

compare(GradientShapExplainer, embed_model,
        dict(inputs=embeddings, baselines=baselines_dist), targets, atol=1e-3)
```

---

## Pattern D — inputs + feature_mask + target

A `feature_mask` groups embedding dimensions into segments so the explainer scores whole words rather than individual tokens.

Applies to: `FeatureAblationExplainer`, `LimeExplainer`, `KernelShapExplainer`.

```python
from torchxai.explainers import FeatureAblationExplainer

# Replace with LimeExplainer or KernelShapExplainer as needed

# Word-level feature mask: subword tokens of the same word share an ID
word_ids     = enc.word_ids(batch_index=0)   # e.g. [None, 0, 1, 1, 2, None]
seq_len      = input_ids.shape[1]
# Shape (1, seq_len, 1): broadcast across 768 embedding dims
feature_mask = torch.zeros(1, seq_len, 1, dtype=torch.long)
for t_idx, w_idx in enumerate(word_ids):
    if w_idx is not None:
        feature_mask[0, t_idx, 0] = w_idx + 1  # 0 reserved for special tokens

print("Without feature mask (token-level):")
compare(FeatureAblationExplainer, embed_model, dict(inputs=embeddings), targets)

print("\nWith word-level feature mask:")
compare(FeatureAblationExplainer, embed_model,
        dict(inputs=embeddings, feature_mask=feature_mask), targets)
```
