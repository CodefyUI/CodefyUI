"""TextGenerateNode — 用訓練好的 causal LM 逐 token 取樣生成文字。

`LLMChat` 呼叫外部 API；沒有節點能從「畫布上剛訓練出來的權重」生成文字。
這顆吃 MODEL + tokenizer，從 prompt 開始自迴歸取樣（temperature / top-k /
top-p，種子化可重現），是訓練成果最直觀的質性驗證：TinyStories 練完，
"Once upon a time" 應該接得出通順的小故事。
"""

from __future__ import annotations

from typing import Any

from ...core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)


class TextGenerateNode(BaseNode):
    NODE_NAME = "TextGenerate"
    CATEGORY = "LLM"
    DESCRIPTION = (
        "Autoregressively sample text from a trained causal LM: encode the "
        "prompt with the LMTokenizer handle, feed the model token by token "
        "with temperature / top-k / top-p sampling (seeded, reproducible), "
        "stop at EOS or max_new_tokens, and decode the result. The most "
        "direct qualitative check of a canvas-trained language model."
    )

    # The output depends on the model's live WEIGHTS (rule 2 of the
    # cacheable contract): a cache hit would replay text from a model the
    # training loop has since updated.
    cacheable = False

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="model", data_type=DataType.MODEL,
                           description="Trained causal LM (input_ids (B,T) -> logits (B,T,V))"),
            PortDefinition(name="tokenizer", data_type=DataType.ANY,
                           description="Tokenizer handle from LMTokenizer (encode/decode/eos_id)"),
            PortDefinition(name="prompt", data_type=DataType.STRING,
                           description="Prompt text. Optional — falls back to the `prompt` param.",
                           optional=True),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(name="text", data_type=DataType.STRING,
                           description="Prompt plus the generated completion"),
            PortDefinition(name="token_count", data_type=DataType.SCALAR,
                           description="Number of new tokens generated"),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="prompt", param_type=ParamType.STRING,
                default="Once upon a time",
                description="Prompt used when no `prompt` input is connected",
            ),
            ParamDefinition(
                name="max_new_tokens", param_type=ParamType.INT, default=200,
                min_value=1, max_value=4096,
                description="Maximum tokens to generate",
            ),
            ParamDefinition(
                name="temperature", param_type=ParamType.FLOAT, default=0.8,
                min_value=0.0, max_value=2.0,
                description="Sampling temperature (0 = greedy argmax)",
            ),
            ParamDefinition(
                name="top_k", param_type=ParamType.INT, default=50,
                min_value=0, max_value=1000, advanced=True,
                description="Keep only the k most likely tokens before sampling (0 = disabled)",
            ),
            ParamDefinition(
                name="top_p", param_type=ParamType.FLOAT, default=0.95,
                min_value=0.0, max_value=1.0, advanced=True,
                description="Nucleus sampling: keep the smallest set of tokens whose probability sums to p (1 = disabled)",
            ),
            ParamDefinition(
                name="seed", param_type=ParamType.INT, default=0,
                min_value=0, advanced=True,
                description="Sampling seed — the same seed and weights reproduce the same text",
            ),
            ParamDefinition(
                name="device", param_type=ParamType.SELECT, default="auto",
                options=["auto", "cpu", "cuda", "mps"],
                description="Device to run the forward passes on",
            ),
        ]

    def execute(
        self,
        inputs: dict[str, Any],
        params: dict[str, Any],
        progress_callback: Any | None = None,
        *,
        context: Any = None,
    ) -> dict[str, Any]:
        import torch

        from ...core.device_utils import resolve_node_device
        from ...core.loop_control import (
            ProgressThrottle,
            interrupted_result,
            stop_checker,
        )

        model = inputs.get("model")
        tokenizer = inputs.get("tokenizer")
        if model is None:
            raise ValueError("TextGenerate requires a `model` input.")
        if (
            tokenizer is None
            or not hasattr(tokenizer, "encode")
            or not hasattr(tokenizer, "decode")
            or not hasattr(tokenizer, "eos_id")
        ):
            raise ValueError(
                "TextGenerate requires a `tokenizer` input from the "
                "LMTokenizer node (an object with encode/decode/eos_id)."
            )

        prompt_input = inputs.get("prompt")
        prompt = str(prompt_input) if prompt_input is not None else str(params.get("prompt", ""))
        max_new_tokens = max(1, min(4096, int(params.get("max_new_tokens", 200))))
        temperature = max(0.0, float(params.get("temperature", 0.8)))
        top_k = max(0, int(params.get("top_k", 50)))
        top_p = float(params.get("top_p", 0.95))
        seed = int(params.get("seed", 0))
        device = resolve_node_device(params.get("device"), context)

        eos_id = int(tokenizer.eos_id)
        ids: list[int] = [int(t) for t in tokenizer.encode(prompt)]
        if not ids:
            ids = [eos_id]

        # The context window: the model says so if it can (CausalLMModel
        # exposes max_seq_len); otherwise fall back to a safe 1024.
        window = int(getattr(model, "max_seq_len", 1024) or 1024)

        model = model.to(device)
        model.eval()
        # Sampling happens on CPU in fp32 regardless of the model device, so
        # a fixed seed reproduces the same text on cpu and cuda alike.
        generator = torch.Generator(device="cpu").manual_seed(seed)
        should_stop = stop_checker(context)
        throttle = ProgressThrottle(progress_callback)

        generated = 0
        stopped_early: int | None = None
        with torch.no_grad():
            for step in range(max_new_tokens):
                if should_stop():
                    stopped_early = step
                    break
                window_ids = ids[-window:]
                input_ids = torch.tensor([window_ids], dtype=torch.int64, device=device)
                logits = model(input_ids)
                last = logits[0, -1].detach().float().cpu()

                if temperature == 0.0:
                    next_id = int(torch.argmax(last).item())
                else:
                    last = last / temperature
                    if top_k and top_k < last.shape[0]:
                        kth = torch.topk(last, top_k).values[-1]
                        last[last < kth] = float("-inf")
                    if 0.0 < top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(last, descending=True)
                        probs = torch.softmax(sorted_logits, dim=-1)
                        cumulative = torch.cumsum(probs, dim=-1)
                        # Keep the first token whose cumulative crosses p.
                        cutoff = cumulative > top_p
                        cutoff[1:] = cutoff[:-1].clone()
                        cutoff[0] = False
                        sorted_logits[cutoff] = float("-inf")
                        last = torch.full_like(last, float("-inf"))
                        last[sorted_indices] = sorted_logits
                    probs = torch.softmax(last, dim=-1)
                    next_id = int(torch.multinomial(probs, 1, generator=generator).item())

                ids.append(next_id)
                generated += 1
                throttle.emit({
                    "event": "generate",
                    "tokens": generated,
                    "max_new_tokens": max_new_tokens,
                })
                if next_id == eos_id:
                    break

        text = tokenizer.decode(ids)
        result: dict[str, Any] = {"text": text, "token_count": int(generated)}
        if stopped_early is not None:
            result.update(interrupted_result(token=stopped_early))
        return result
