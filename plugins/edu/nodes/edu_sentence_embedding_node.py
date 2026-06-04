"""EduSentenceEmbeddingNode — 把一句話壓成一條有語意的向量。

對應教材 **I1-3**。輸入一段文字，輸出一條 `(d,)` 的句子向量；意思接近的
句子，向量方向也接近（cosine 大）。這條向量丟給 CosineSimilarity 就能量
「兩句話意思有多接近」，包含跨語言（多語言模型下「我喜歡貓」與
「I like cats」會落在相近位置）。

跟玩具版的 `Edu-TokenEmbedding`（隨機表、純教 shape）不同，這顆用一個小型
**預訓練嵌入模型**算出真正帶語意的向量，所以 I1-3 的相似度實驗才會成立。
概念上是 tokenize → 每個 token 查 embedding → pooling 收成一條向量
（C1-4 講的那三步），這裡把它包成一顆節點。

效能 / 普及性：用 **model2vec** 靜態嵌入——不需要 transformers，推論是純查表
加平均（微秒級、純 CPU），模型只有十幾~一百多 MB。lazy-load 並在 process
內快取，相同文字的結果也快取。第一次使用需下載模型權重。
"""

from __future__ import annotations

from typing import Any

from app.core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)

# 模型選項 → HuggingFace repo。預設多語言（章節的中文 / 跨語言實驗需要）；
# 純英文情境可選更小的 potion-base-8M。
_MODELS: dict[str, str] = {
    "multilingual": "minishlab/potion-multilingual-128M",
    "english-small": "minishlab/potion-base-8M",
}
_DEFAULT_MODEL = "multilingual"

# process 級快取：載入過的模型、以及 (model, text) → 向量。
_MODEL_CACHE: dict[str, Any] = {}
_EMBED_CACHE: dict[tuple[str, str], Any] = {}


def _load_model(repo: str) -> Any:
    cached = _MODEL_CACHE.get(repo)
    if cached is not None:
        return cached
    try:
        from model2vec import StaticModel
    except ImportError as e:
        raise ValueError(
            "SentenceEmbedding 需要 `model2vec` 套件。"
            "請在後端環境安裝：`uv pip install model2vec`。"
        ) from e
    model = StaticModel.from_pretrained(repo)
    _MODEL_CACHE[repo] = model
    return model


class EduSentenceEmbeddingNode(BaseNode):
    NODE_NAME = "SentenceEmbedding"
    CATEGORY = "EDU"
    DESCRIPTION = (
        "把一段文字壓成一條有語意的句子向量 (d,)。意思接近的句子向量方向也接近，"
        "丟給 CosineSimilarity 就能量「兩句話意思多接近」（多語言模型下還能跨語言）。"
        "用 model2vec 靜態嵌入、跑純 CPU、結果會快取；第一次使用需下載模型權重。"
    )

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="text",
                data_type=DataType.STRING,
                description="輸入文字。沒接時退回用 `text` 參數。",
                optional=True,
            ),
        ]

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="embedding",
                data_type=DataType.TENSOR,
                description="句子向量，shape (d,)。",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="model",
                param_type=ParamType.SELECT,
                default=_DEFAULT_MODEL,
                options=list(_MODELS.keys()),
                description="嵌入模型。multilingual 支援中文與跨語言；english-small 更小更快。",
            ),
            ParamDefinition(
                name="text",
                param_type=ParamType.STRING,
                default="",
                description="要編碼的文字，僅在沒有接 `text` 輸入時使用（接了輸入就以輸入為準）。",
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

        raw = inputs.get("text")
        if raw is None:
            raw = params.get("text", "")
        if isinstance(raw, list):  # batch passthrough：取第一筆，跟 Tokenizer 一致
            raw = raw[0] if raw else ""
        text = str(raw)

        model_label = str(params.get("model", _DEFAULT_MODEL))
        repo = _MODELS.get(model_label, _MODELS[_DEFAULT_MODEL])

        key = (repo, text)
        cached = _EMBED_CACHE.get(key)
        if cached is not None:
            vec = cached
        else:
            model = _load_model(repo)
            arr = model.encode(text)  # numpy (d,) for a single string
            vec = torch.as_tensor(arr, dtype=torch.float32).reshape(-1)
            _EMBED_CACHE[key] = vec

        return {"embedding": vec}
