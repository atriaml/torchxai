import numpy as np
import torch

DEFAULT_STOPWORDS = frozenset({
    "a","an","the","and","or","but","if","then","else","when",
    "at","by","for","with","about","into","through","before",
    "after","to","from","in","out","on","is","are","was","were",
    "this","that","it","of","we","you","he","she"
})


def is_content_word(token: str) -> bool:
    t = str(token).lower().strip()
    return t.isalpha() and t not in DEFAULT_STOPWORDS

def text_attribution_profile_grouped(
    tokens,
    attributions,   # [T] already grouped per word (or feature)
    k_ratio=0.2,
):
    """
    Minimal text attribution profiling.

    Assumptions:
        - tokens already aligned 1:1 with attributions
        - attributions already grouped (no feature reduction needed)
    """

    tokens = list(tokens)

    attr = attributions.detach().cpu().numpy() if torch.is_tensor(attributions) else np.array(attributions)

    # ---------------------------------------------------
    # normalize attribution (IMPORTANT)
    # ---------------------------------------------------
    denom = np.sum(np.abs(attr)) + 1e-10
    attr = attr / denom

    n = len(tokens)
    is_content = np.array([is_content_word(t) for t in tokens])

    # ---------------------------------------------------
    # D1: word-level table
    # ---------------------------------------------------
    word_rows = []
    for i, t in enumerate(tokens):
        word_rows.append({
            "index": i,
            "token": t,
            "is_content": bool(is_content[i]),
            "attr": float(attr[i]),
            "abs_attr": float(abs(attr[i])),
        })

    # ---------------------------------------------------
    # D2: content vs stopword signal
    # ---------------------------------------------------
    content_vals = attr[is_content]
    stop_vals = attr[~is_content]

    content_mean = float(content_vals.mean()) if len(content_vals) else 0.0
    stop_mean = float(stop_vals.mean()) if len(stop_vals) else 0.0

    content_stop_ratio = content_mean / (stop_mean + 1e-10)

    # ---------------------------------------------------
    # D3: structure of top-k attribution
    # ---------------------------------------------------
    k = max(1, int(n * k_ratio))
    topk = np.sort(np.argsort(np.abs(attr))[::-1][:k])

    if len(topk) > 1:
        gaps = np.diff(topk)
        mean_gap = float(gaps.mean())
        n_runs = int(1 + (gaps > 1).sum())
    else:
        mean_gap = 0.0
        n_runs = 1

    summary = {
        "n_tokens": n,
        "content_stop_ratio": float(content_stop_ratio),
        "topk_mean_gap": mean_gap,
        "topk_n_runs": n_runs,
    }

    return word_rows, summary

tokens = [
    "The", "movie", "was", "absolutely", "fantastic",
    "with", "great", "acting", "and", "story",
    "but", "the", "ending", "was", "rushed"
]

# already grouped per word (NO feature logic assumed)
attributions = torch.tensor([
    0.01, 0.05, 0.02, 0.20, 0.35,
    0.01, 0.15, 0.10, 0.01, 0.08,
    0.03, 0.01, 0.12, 0.02, 0.25
])

rows, stats = text_attribution_profile_grouped(tokens, attributions)

print("\n=== D1 WORD LEVEL ===")
for r in rows:
    print(r)

print("\n=== D2 / D3 SUMMARY ===")
print(stats)