---
title: BERT
summary: Multi-target attribution examples on BERT for sequence classification and NER
---

# BERT Examples

These examples show how to apply torchxai explainers to BERT-based models. Two use cases are covered:

1. **Sequence Classification** — single label per sentence (e.g. sentiment analysis)
2. **Named Entity Recognition (NER)** — one label per token, where multi-target attribution explains all token predictions in a single call

!!! note "Gradient-based methods require embedding-level inputs"
    BERT token IDs are discrete integers — gradients cannot flow through them.
    For gradient-based explainers (`Saliency`, `IntegratedGradients`, etc.) we pass the continuous **embedding tensor** as input. For perturbation-based explainers (`FeatureAblation`, `LIME`, etc.) we can operate directly on token IDs cast to `float`.

!!! warning "Unsupported explainers for transformer architectures"
    `DeepLiftExplainer`, `DeepLiftShapExplainer`, and `GuidedBackpropExplainer` are not compatible with transformers out of the box. DeepLift requires specific activation types not present in BERT; GuidedBackprop requires ReLU activations throughout the network. Use `IntegratedGradientsExplainer`, `GradientShapExplainer`, or `SaliencyExplainer` as alternatives.

---

## Use Case 1: Sequence Classification

### Setup

```python
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torchxai.data_types import SingleTargetAcrossBatch

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2
).eval()

text = "This movie was absolutely fantastic!"
enc = tokenizer(text, return_tensors="pt")
input_ids      = enc["input_ids"]       # (1, seq_len)
attention_mask = enc["attention_mask"]  # (1, seq_len)

# Embeddings — used as input for gradient-based explainers
embeddings   = model.bert.embeddings(input_ids)   # (1, seq_len, 768)
baseline_emb = torch.zeros_like(embeddings)

class EmbeddingSeqCls(torch.nn.Module):
    """Thin wrapper: takes embeddings → returns logits (1, num_labels)."""
    def forward(self, emb):
        return model(inputs_embeds=emb, attention_mask=attention_mask).logits

embed_model = EmbeddingSeqCls().eval()
```

### Pattern A — inputs + target (gradient-based, no baseline)

Applies to: `SaliencyExplainer`, `InputXGradientExplainer`.

```python
from torchxai.explainers import SaliencyExplainer

explainer = SaliencyExplainer(embed_model, multi_target=False)
attrs = explainer.explain(
    inputs=embeddings,
    target=SingleTargetAcrossBatch(index=0),   # explain class 0 (e.g. negative)
)
print(attrs.shape)   # (1, seq_len, 768)

explainer_mt = SaliencyExplainer(embed_model, multi_target=True)
attrs_mt = explainer_mt.explain(
    inputs=embeddings,
    target=[SingleTargetAcrossBatch(index=0), SingleTargetAcrossBatch(index=1)],
)
assert torch.allclose(attrs, attrs_mt[0])
```

### Pattern B — inputs + baseline + target (gradient-based, with baseline)

Applies to: `IntegratedGradientsExplainer`, `InputXBaselineGradientExplainer`.

```python
from torchxai.explainers import IntegratedGradientsExplainer

explainer = IntegratedGradientsExplainer(embed_model, multi_target=False)
attrs = explainer.explain(
    inputs=embeddings,
    baselines=baseline_emb,
    target=SingleTargetAcrossBatch(index=0),
)
print(attrs.shape)   # (1, seq_len, 768)

explainer_mt = IntegratedGradientsExplainer(embed_model, multi_target=True)
attrs_mt = explainer_mt.explain(
    inputs=embeddings,
    baselines=baseline_emb,
    target=[SingleTargetAcrossBatch(index=0), SingleTargetAcrossBatch(index=1)],
)
assert torch.allclose(attrs, attrs_mt[0])
```

### Pattern D — feature_mask at word level (perturbation-based)

Perturbation methods work on token IDs directly (cast to `float`). A word-level `feature_mask` groups subword tokens so the explainer scores whole words rather than individual tokens.

```python
from torchxai.explainers import FeatureAblationExplainer
# Replace with LimeExplainer or KernelShapExplainer as needed

class TokenSeqCls(torch.nn.Module):
    """Takes float token IDs → returns logits (1, num_labels)."""
    def forward(self, ids):
        return model(input_ids=ids.long(), attention_mask=attention_mask).logits

token_model  = TokenSeqCls().eval()
token_inputs = input_ids.float()   # (1, seq_len)

# Build word-level feature mask: subword tokens of the same word share an ID
word_ids     = enc.word_ids(batch_index=0)   # e.g. [None, 0, 1, 1, 2, None]
seq_len      = input_ids.shape[1]
feature_mask = torch.zeros(1, seq_len, dtype=torch.long)
for t_idx, w_idx in enumerate(word_ids):
    if w_idx is not None:
        feature_mask[0, t_idx] = w_idx + 1  # 0 reserved for special tokens

# Without mask — token-level attribution
explainer    = FeatureAblationExplainer(token_model, multi_target=False)
attrs_token  = explainer.explain(
    inputs=token_inputs,
    target=SingleTargetAcrossBatch(index=0),
)
print(attrs_token.shape)   # (1, seq_len)

# With word-level mask
attrs_word = explainer.explain(
    inputs=token_inputs,
    feature_mask=feature_mask,
    target=SingleTargetAcrossBatch(index=0),
)
print(attrs_word.shape)   # (1, seq_len) — one score per word, broadcast back

# Multi-target with mask
explainer_mt = FeatureAblationExplainer(token_model, multi_target=True)
attrs_word_mt = explainer_mt.explain(
    inputs=token_inputs,
    feature_mask=feature_mask,
    target=[SingleTargetAcrossBatch(index=0), SingleTargetAcrossBatch(index=1)],
)
assert torch.allclose(attrs_word, attrs_word_mt[0])
```

---

## Use Case 2: Named Entity Recognition (NER)

In NER each non-special token has its own predicted label. Running attributions for every token sequentially is expensive. With `multi_target=True` we explain all token predictions in a single call.

We wrap the NER model so it returns one value per token — the logit of that token's predicted class. This keeps targets simple: `SingleTargetAcrossBatch(index=token_position)` selects the logit at that position from the `(1, seq_len)` output.

### Setup

```python
from transformers import BertForTokenClassification

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
```

### Feature mask: token → word grouping

```python
seq_len      = input_ids.shape[1]
# Shape (1, seq_len, 1): all 768 dims of the same token share one feature ID.
# Subword tokens of the same word are assigned the same word-level ID.
feature_mask = torch.zeros(1, seq_len, 1, dtype=torch.long)
for t_idx, w_idx in enumerate(word_ids):
    if w_idx is not None:
        feature_mask[0, t_idx, 0] = w_idx + 1   # 0 reserved for [CLS] / [SEP]
```

### Single-target: explain one token's prediction

```python
from torchxai.explainers import IntegratedGradientsExplainer

# Index of the first real (non-special) token
first_real_token = next(t for t, w in enumerate(word_ids) if w is not None)

explainer = IntegratedGradientsExplainer(wrapped_ner, multi_target=False)
attrs = explainer.explain(
    inputs=embeddings,
    baselines=baseline_emb,
    feature_mask=feature_mask,
    target=SingleTargetAcrossBatch(index=first_real_token),
)
print(attrs.shape)   # (1, num_words, 768) — one attribution vector per word
```

### Multi-target: explain all token predictions in one call

```python
# One target per real (non-special) token position
targets_mt = [
    SingleTargetAcrossBatch(index=t_idx)
    for t_idx, w_idx in enumerate(word_ids) if w_idx is not None
]

explainer_mt = IntegratedGradientsExplainer(wrapped_ner, multi_target=True)
attrs_mt = explainer_mt.explain(
    inputs=embeddings,
    baselines=baseline_emb,
    feature_mask=feature_mask,
    target=targets_mt,
)
# attrs_mt[i] → word-level attribution map for the i-th real token's predicted label
# First result must match the single-target run above
assert torch.allclose(attrs, attrs_mt[0])
print(f"Got attributions for {len(attrs_mt)} tokens in one call")
```
