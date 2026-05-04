import time
from collections import OrderedDict
from dataclasses import dataclass

import torch
import tqdm
from atria_insights.data_types._targets import BatchExplanationTarget
from datasets import load_dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer

from torchxai.data_types._target import SingleTargetPerSample
from torchxai.explainers._grad._saliency import SaliencyExplainer

# ── Data Structures ───────────────────────────────────────────────────────────


@dataclass
class NERBatch:
    input_ids: torch.Tensor  # [B, L]
    attention_mask: torch.Tensor  # [B, L]
    embeddings: torch.Tensor  # [B, L, H]
    targets: torch.Tensor  # [B]
    tokens: list[list[str]]  # decoded tokens per sample
    raw_sentences: list[str]


@dataclass
class BenchmarkResult:
    attributions: OrderedDict
    load_time_sec: float
    ingest_time_sec: float
    explain_time_sec: float
    batch_size: int


# ── Model Setup ───────────────────────────────────────────────────────────────


class BertEmbeddingWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, embeddings, attention_mask=None):
        logits = self.model(
            inputs_embeds=embeddings, attention_mask=attention_mask
        ).logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        probs = torch.gather(probs, 2, probs.argmax(dim=-1).unsqueeze(-1)).squeeze(-1)
        return probs


def load_model(model_name: str = "dslim/bert-base-NER") -> tuple:
    """Load pretrained BERT NER model, tokenizer, and wrapped version."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    model.eval()
    wrapped = BertEmbeddingWrapper(model)
    return model, wrapped, tokenizer


# ── Data Loading ──────────────────────────────────────────────────────────────


def load_conll_data(
    split: str = "validation", num_samples: int = 8, seed: int = 42
) -> list[dict]:
    """
    Load raw examples from CoNLL-2003.

    Returns a list of dicts with keys: tokens, ner_tags.
    Deliberately kept free of model/tokenizer logic so it can be
    swapped for any other NER dataset with the same interface.
    """
    dataset = load_dataset("conll2003", split=split, trust_remote_code=True)
    dataset = dataset.shuffle(seed=seed).select(range(num_samples))
    return [{"tokens": ex["tokens"], "ner_tags": ex["ner_tags"]} for ex in dataset]


# ── Ingestion (raw examples → model-ready tensors) ───────────────────────────


def ingest_batch(
    examples: list[dict],
    model: torch.nn.Module,
    tokenizer,
    max_length: int = 64,
    device: torch.device = torch.device("cpu"),
) -> NERBatch:
    """
    Convert raw CoNLL examples into a NERBatch ready for attribution.

    Steps:
      1. Tokenise sentences
      2. Run a no-grad forward pass to get predicted targets (one per sample)
      3. Extract input embeddings (gradient-ready)

    Keeping this separate from data loading means you can time tokenisation +
    embedding extraction independently from disk/network I/O.
    """
    sentences = [" ".join(ex["tokens"]) for ex in examples]

    enc = tokenizer(
        sentences,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    # Predicted label at position 1 (first real token) as scalar target per sample
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        probs = torch.gather(probs, 2, probs.argmax(dim=-1).unsqueeze(-1)).squeeze(-1)

        # randomly generate 100 targets for multi-target testing
        bs = probs.shape[0]
        targets = torch.randint(0, probs.shape[1], (100,))
        targets = [
            BatchExplanationTarget(
                value=[t.item() for _ in range(bs)],
                name=[str(t.item()) for _ in range(bs)],
            )
            for t in targets
        ]

    # Embeddings must be leaf tensors with requires_grad for Saliency
    embeddings = model.bert.embeddings(input_ids).detach().requires_grad_(True)

    decoded_tokens = [tokenizer.convert_ids_to_tokens(ids) for ids in input_ids]

    return NERBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        embeddings=embeddings,
        targets=targets,
        tokens=decoded_tokens,
        raw_sentences=sentences,
    )


def explainer_forward(
    batch: NERBatch,
    wrapped_model: torch.nn.Module,
    explainer_cls,  # SaliencyExplainer class passed in
    multi_target: bool = False,
    internal_batch_size: int = 16,
) -> tuple[torch.Tensor, ...] | list[tuple[torch.Tensor, ...]]:
    from torchxai.data_types import ExplanationTarget

    explainer = explainer_cls(
        wrapped_model, multi_target=multi_target, grad_batch_size=16
    )

    def _map_target(
        target: BatchExplanationTarget | list[BatchExplanationTarget] | None,
    ) -> ExplanationTarget | list[ExplanationTarget]:
        if target is None:
            return ExplanationTarget.from_raw_input(None)
        if isinstance(target, BatchExplanationTarget):
            return ExplanationTarget.from_raw_input(target.value)
        elif isinstance(target, list):
            return [ExplanationTarget.from_raw_input(t.value) for t in target]
        else:
            raise ValueError(
                "Target must be of type BatchExplanationTarget, list of BatchExplanationTarget, or None."
            )

    kwargs = {
        "inputs": (batch.embeddings,),
        "additional_forward_args": (batch.attention_mask,),
    }

    # map targets
    target = _map_target(batch.targets)

    print(f"Running explainer {explainer} forward with inputs:")
    if not multi_target and isinstance(target, list):
        print(
            "Running explainer forward with iterative computation for multi-target explanations."
        )
        # disable multi-target for iterative computation
        explainer.multi_target = False

        # if the input is batched we need to repeat the inputs for each target and compute explanations in a single forward pass
        # Assumption: original batch size is always 1
        per_target_explanations = []
        internal_batch_size = internal_batch_size

        print(f"internal_batch_size for iterative computation: {internal_batch_size}")
        print(f"Total number of targets: {len(target)}")

        # Calculate how many targets we can process at once
        # Since original batch size is 1, we can process internal_batch_size targets simultaneously
        num_targets_per_batch = internal_batch_size
        print("num_targets_per_batch", num_targets_per_batch)

        for batch_start in tqdm.tqdm(
            range(0, len(target), num_targets_per_batch),
            desc="Computing explanations per target batch",
        ):
            batch_end = min(batch_start + num_targets_per_batch, len(target))
            target_batch = target[batch_start:batch_end]
            final_target = SingleTargetPerSample(
                indices=[t.value[0] for t in target_batch]
            )
            print("target_batch", target_batch)
            print("final_target", final_target)
            num_targets_in_batch = len(final_target.value)

            # Repeat inputs for each target in the batch
            batched_kwargs = {}
            for key, value in kwargs.items():
                if key == "inputs" and isinstance(value, tuple):
                    # Repeat each input tensor for each target
                    batched_kwargs[key] = tuple(
                        inp.repeat_interleave(num_targets_in_batch, dim=0)
                        for inp in value
                    )
                elif key == "additional_forward_args" and isinstance(value, tuple):
                    # Repeat additional forward args
                    batched_kwargs[key] = tuple(
                        arg.repeat_interleave(num_targets_in_batch, dim=0)
                        if isinstance(arg, torch.Tensor)
                        else arg
                        for arg in value
                    )
                elif (
                    key in ["baselines", "feature_mask"]
                    and value is not None
                    and isinstance(value, tuple)
                ):
                    # Repeat baselines and feature masks
                    batched_kwargs[key] = tuple(
                        item.repeat_interleave(num_targets_in_batch, dim=0)
                        for item in value
                    )
                else:
                    # Keep other args as is
                    batched_kwargs[key] = value

            # Compute explanations for the batch
            print("Current kwargs")
            for k, v in batched_kwargs.items():
                if isinstance(v, tuple):
                    print(
                        f"  {k}: {[item.shape if isinstance(item, torch.Tensor) else item for item in v]}"
                    )
                else:
                    print(f"  {k}: {v}")
            curr_explanations = explainer.explain(**batched_kwargs, target=final_target)
            assert isinstance(curr_explanations, tuple), (
                "Explainer returned invalid type during iterative computation. "
                "Expected tuple."
            )

            # The results are organized as: [s0_t0, s0_t1, ..., s0_tT, s1_t0, s1_t1, ..., s1_tT, ...]
            # We need to reorganize them per target: each target gets [s0_ti, s1_ti, ...]
            for target_idx in range(num_targets_in_batch):
                # Extract explanations for this target across all samples
                # Every num_targets_in_batch-th element, starting from target_idx
                target_explanation = tuple(
                    exp[target_idx::num_targets_in_batch].detach().cpu()
                    for exp in curr_explanations
                )
                per_target_explanations.append(target_explanation)

        explainer.multi_target = True
        return per_target_explanations
    else:
        print(f"Running explainer forward with multi_target={explainer.multi_target}")
        # we need to map the atria_insights target to torchxai target
        if isinstance(target, list):
            explainer.multi_target = True
        explanations = explainer.explain(**kwargs, target=target)

        # validated explanations
        validated_explanations = []
        if isinstance(explanations, tuple):
            return explanations
        elif isinstance(explanations, list):
            for exp in explanations:
                if not isinstance(exp, tuple):
                    raise ValueError(
                        "Explainer returned a list but elements are not tuples."
                    )
                validated_explanations.append(exp)
            return validated_explanations
        else:
            raise ValueError(
                "Explainer returned invalid type. Expected tuple or list of tuples."
            )


# def run_attribution(
#     batch: NERBatch,
#     wrapped_model: torch.nn.Module,
#     explainer_cls,  # SaliencyExplainer class passed in
#     multi_target: bool = False,
# ) -> OrderedDict:
#     """
#     Run SaliencyExplainer on a pre-ingested NERBatch.

#     Accepts explainer_cls as a parameter so you can swap in any
#     FeatureAttributionExplainer subclass without changing this function.
#     """
#     explainer = explainer_cls(wrapped_model, multi_target=multi_target)

#     def _map_target(
#         target: BatchExplanationTarget | list[BatchExplanationTarget] | None,
#     ) -> ExplanationTarget | list[ExplanationTarget]:
#         if target is None:
#             return ExplanationTarget.from_raw_input(None)
#         if isinstance(target, BatchExplanationTarget):
#             return ExplanationTarget.from_raw_input(target.value)
#         elif isinstance(target, list):
#             return [ExplanationTarget.from_raw_input(t.value) for t in target]
#         else:
#             raise ValueError(
#                 "Target must be of type BatchExplanationTarget, list of BatchExplanationTarget, or None."
#             )

#     # map targets
#     targets = _map_target(batch.targets)

#     if isinstance(targets, list):
#         explainer.multi_target = True
#     return explainer.explain(
#         inputs=(batch.embeddings,),
#         target=targets,
#         additional_forward_args=(batch.attention_mask,),
#     )


# ── Benchmark Harness ─────────────────────────────────────────────────────────


def benchmark(
    explainer_cls,
    num_samples: int = 1,
    max_length: int = 512,
    multi_target: bool = False,
    device: torch.device = torch.device("cpu"),
    model_name: str = "dslim/bert-base-NER",
) -> BenchmarkResult:
    """
    Full pipeline benchmark: load → ingest → explain, each timed independently.
    Call this twice with different explainer_cls or settings to compare.
    """
    model, wrapped, tokenizer = load_model(model_name)
    model.to(device)
    wrapped.to(device)

    # ── Stage 1: Data loading (I/O) ──────────────────────────────────────────
    t0 = time.perf_counter()
    examples = load_conll_data(num_samples=num_samples)
    load_time = time.perf_counter() - t0

    # ── Stage 2: Ingestion (tokenise + embed) ────────────────────────────────
    t0 = time.perf_counter()
    batch = ingest_batch(
        examples, model, tokenizer, max_length=max_length, device=device
    )
    ingest_time = time.perf_counter() - t0

    # ── Stage 3: Attribution ─────────────────────────────────────────────────
    t0 = time.perf_counter()
    attributions = explainer_forward(
        batch, wrapped, explainer_cls, multi_target=multi_target
    )
    explain_time = time.perf_counter() - t0

    result = BenchmarkResult(
        attributions=attributions,
        load_time_sec=load_time,
        ingest_time_sec=ingest_time,
        explain_time_sec=explain_time,
        batch_size=num_samples,
    )

    _print_benchmark(
        result, label=f"{explainer_cls.__name__} | multi_target={multi_target}"
    )
    return result


def _print_benchmark(result: BenchmarkResult, label: str = "") -> None:
    print(f"\n{'─' * 55}")
    print(f"  {label}")
    print(f"{'─' * 55}")
    print(f"  batch size   : {result.batch_size}")
    print(f"  load time    : {result.load_time_sec:.4f}s")
    print(f"  ingest time  : {result.ingest_time_sec:.4f}s")
    print(f"  explain time : {result.explain_time_sec:.4f}s")
    print(
        f"  total        : {result.load_time_sec + result.ingest_time_sec + result.explain_time_sec:.4f}s"
    )
    print(f"{'─' * 55}\n")


# ── Usage ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Setup A — single target
    # result_a = benchmark(SaliencyExplainer, num_samples=1, multi_target=False)

    # Setup B — multi target
    result_b = benchmark(SaliencyExplainer, num_samples=1, multi_target=True)
