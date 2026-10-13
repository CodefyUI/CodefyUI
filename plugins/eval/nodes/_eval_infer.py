"""The batch loop behind Eval-BatchInference, and the two ways it predicts.

Model-agnostic on purpose: the loop takes two callables and never imports
``transformers``, and the predictors talk to their model and processor only
through the call shapes a Hugging Face model and tokenizer have
(``processor(texts, return_tensors="pt", ...)``, ``model(**encoding).logits``,
``model(input_ids=..., past_key_values=..., use_cache=True)``). A test can
therefore hand it a ten-line fake and exercise every failure rule below.

**Failure rules.** An item's record is written once and never revisited:

* an item whose batch completed is ``ok`` and stays ``ok``, whatever happens
  to the batches after it;
* a batch that raises is retried one item at a time, so one bad input marks
  ONE item ``error`` instead of its whole batch;
* a stop request is honoured between batches: every item not yet started is
  ``cancelled``, and the loop reports where it stopped so the node can mark
  its result interrupted.

**Randomness.** Sampled generation draws from a ``torch.Generator`` created
here and seeded per ITEM from ``(seed, item id)``, never from the global RNG:
an item's sample does not depend on which items ran before it, on the batch
size, or on whatever else in the process touched ``torch.manual_seed``.
"""

from __future__ import annotations

import hashlib
from typing import Any, Callable, Sequence

import torch

from ._eval_core import STATUS_CANCELLED, STATUS_ERROR, STATUS_OK

#: How many ranked candidates a classification row keeps in its ``topk`` cell.
MAX_TOPK = 5


def _error_text(exc: BaseException) -> str:
    text = str(exc).strip() or type(exc).__name__
    return f"{type(exc).__name__}: {text}"[:500]


def run_batches(
    items: Sequence[dict[str, Any]],
    *,
    batch_size: int,
    predict_batch: Callable[[list[dict[str, Any]]], list[dict[str, Any]]] | None,
    predict_one: Callable[[dict[str, Any]], dict[str, Any]],
    should_stop: Callable[[], bool],
    on_batch: Callable[[int, int, int], None] | None = None,
) -> tuple[list[dict[str, Any]], int | None]:
    """Predict every item; return one outcome per item and where it stopped.

    An outcome is ``{"status", "raw_answer", "error", "extras"}``. With
    ``predict_batch`` None every item goes through ``predict_one`` (the
    generation path, which decodes one sequence at a time).

    Returns ``(outcomes, stopped_at_batch)``; the second is None when every
    batch ran.
    """
    size = max(1, int(batch_size))
    outcomes: list[dict[str, Any] | None] = [None] * len(items)
    pending: list[int] = []
    for index, item in enumerate(items):
        if not isinstance(item.get("input"), str):
            outcomes[index] = {
                "status": STATUS_ERROR,
                "raw_answer": None,
                "error": "item has no input text",
                "extras": {},
            }
        else:
            pending.append(index)

    batches = [pending[i:i + size] for i in range(0, len(pending), size)]
    stopped_at: int | None = None
    done = 0
    for batch_index, batch in enumerate(batches):
        if should_stop():
            stopped_at = batch_index
            break
        batch_items = [items[i] for i in batch]
        results: list[dict[str, Any]] | None = None
        if predict_batch is not None:
            try:
                results = predict_batch(batch_items)
                if len(results) != len(batch_items):
                    raise RuntimeError(
                        f"predictor returned {len(results)} results for "
                        f"{len(batch_items)} items"
                    )
            except Exception:  # noqa: BLE001 - isolated per item below
                results = None
        if results is not None:
            for index, result in zip(batch, results):
                outcomes[index] = {
                    "status": STATUS_OK,
                    "raw_answer": result.get("raw_answer"),
                    "error": None,
                    "extras": dict(result.get("extras") or {}),
                }
        else:
            for index, item in zip(batch, batch_items):
                try:
                    result = predict_one(item)
                except Exception as exc:  # noqa: BLE001 - recorded on the item
                    outcomes[index] = {
                        "status": STATUS_ERROR,
                        "raw_answer": None,
                        "error": _error_text(exc),
                        "extras": {},
                    }
                else:
                    outcomes[index] = {
                        "status": STATUS_OK,
                        "raw_answer": result.get("raw_answer"),
                        "error": None,
                        "extras": dict(result.get("extras") or {}),
                    }
        done += len(batch)
        if on_batch is not None:
            on_batch(batch_index + 1, len(batches), done)

    final: list[dict[str, Any]] = []
    for outcome in outcomes:
        final.append(outcome if outcome is not None else {
            "status": STATUS_CANCELLED,
            "raw_answer": None,
            "error": "run stopped before this item was predicted",
            "extras": {},
        })
    return final, stopped_at


def _device_of(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration, TypeError):
        return torch.device("cpu")


def _logits_of(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    try:
        return output.logits
    except AttributeError:
        return output["logits"]


def id2label_of(model: Any) -> dict[int, str]:
    """``model.config.id2label`` with integer keys, or an empty dict."""
    try:
        raw = model.config.id2label
    except AttributeError:
        return {}
    out: dict[int, str] = {}
    for key, value in dict(raw or {}).items():
        try:
            out[int(key)] = str(value)
        except (TypeError, ValueError):
            continue
    return out


class ClassificationPredictor:
    """Sequence classification: argmax over the logits, named by ``id2label``."""

    def __init__(self, model: Any, processor: Any, *, max_length: int, top_k: int):
        self.model = model
        self.processor = processor
        self.max_length = int(max_length)
        self.top_k = max(1, min(int(top_k), MAX_TOPK))
        self.labels = id2label_of(model)
        self.device = _device_of(model)

    def _name(self, index: int) -> str:
        return self.labels.get(index, f"LABEL_{index}")

    def predict_batch(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        texts = [item["input"] for item in items]
        encoding = self.processor(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {
            key: value.to(self.device)
            for key, value in dict(encoding).items()
            if isinstance(value, torch.Tensor)
        }
        with torch.no_grad():
            logits = _logits_of(self.model(**inputs))
        probs = torch.softmax(logits.detach().float().cpu(), dim=-1)
        k = min(self.top_k, probs.shape[-1])
        top = probs.topk(k, dim=-1)
        out = []
        for values, indices in zip(top.values.tolist(), top.indices.tolist()):
            names = [self._name(i) for i in indices]
            out.append({
                "raw_answer": names[0],
                "extras": {"score": float(values[0]), "topk": names},
            })
        return out

    def predict_one(self, item: dict[str, Any]) -> dict[str, Any]:
        return self.predict_batch([item])[0]


def item_seed(seed: int, item_id: Any) -> int:
    """A per-item seed: the same item gets the same draw in any batch order."""
    digest = hashlib.sha256(f"{int(seed)}:{item_id}".encode("utf-8")).hexdigest()
    return int(digest[:15], 16)


class GenerationPredictor:
    """Causal-LM decoding with a KV cache and a local, per-item generator."""

    def __init__(
        self,
        model: Any,
        processor: Any,
        *,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
        seed: int,
    ):
        self.model = model
        self.processor = processor
        self.max_new_tokens = max(1, int(max_new_tokens))
        self.do_sample = bool(do_sample)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.seed = int(seed)
        self.device = _device_of(model)
        try:
            self.eos_token_id = processor.eos_token_id
        except AttributeError:
            self.eos_token_id = None

    def config(self) -> dict[str, Any]:
        """The generation configuration that actually ran."""
        return {
            "decoding": "sample" if self.do_sample else "greedy",
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "temperature": self.temperature if self.do_sample else None,
            "top_p": self.top_p if self.do_sample else None,
            "seed": self.seed,
            "seed_scope": "per item: sha256(seed:id)",
            "eos_token_id": self.eos_token_id,
        }

    def _choose(self, logits: torch.Tensor, generator: torch.Generator) -> int:
        if not self.do_sample:
            return int(logits.argmax(dim=-1).item())
        scaled = logits / max(self.temperature, 1e-6)
        probs = torch.softmax(scaled, dim=-1)
        if self.top_p < 1.0:
            sorted_probs, order = probs.sort(descending=True)
            cumulative = sorted_probs.cumsum(dim=-1)
            # Keep the smallest prefix whose mass reaches top_p; the first
            # token is always kept, so the distribution is never empty.
            keep = (cumulative - sorted_probs) < self.top_p
            keep[0] = True
            filtered = torch.zeros_like(probs)
            filtered[order[keep]] = probs[order[keep]]
            probs = filtered / filtered.sum()
        return int(torch.multinomial(probs, 1, generator=generator).item())

    def predict_one(self, item: dict[str, Any]) -> dict[str, Any]:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(item_seed(self.seed, item.get("id")))
        encoding = dict(self.processor(item["input"], return_tensors="pt"))
        input_ids = encoding["input_ids"].to(self.device)
        attention = encoding.get("attention_mask")
        if attention is None:
            attention = torch.ones_like(input_ids)
        attention = attention.to(self.device)

        produced: list[int] = []
        past = None
        step_ids = input_ids
        with torch.no_grad():
            for _ in range(self.max_new_tokens):
                output = self.model(
                    input_ids=step_ids,
                    attention_mask=attention,
                    past_key_values=past,
                    use_cache=True,
                )
                past = output.past_key_values
                logits = _logits_of(output)[0, -1, :].detach().float().cpu()
                token = self._choose(logits, generator)
                if self.eos_token_id is not None and token == self.eos_token_id:
                    break
                produced.append(token)
                step_ids = torch.tensor([[token]], device=self.device)
                attention = torch.cat(
                    [attention, torch.ones((1, 1), dtype=attention.dtype, device=self.device)],
                    dim=-1,
                )
        text = self.processor.decode(produced, skip_special_tokens=True)
        return {"raw_answer": text, "extras": {"new_tokens": len(produced)}}
