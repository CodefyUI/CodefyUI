---
sidebar_position: 2
title: 自訂節點
description: 把一個 Python 檔案放進 custom_nodes/ 即可新增節點行為——可熱重載，無需修改前端。
---

# 自訂節點

CodefyUI 是**後端權威**的：一個節點的連接埠、參數與類別全部來自其 Python 定義，UI 會自動渲染它。若要新增行為，把一個 `.py` 檔案放進 `backend/app/custom_nodes/`，並繼承 `BaseNode`。

## 最小範例

```python
from app.core.node_base import BaseNode, DataType, PortDefinition

class MyNode(BaseNode):
    NODE_NAME = "MyNode"
    CATEGORY = "Custom"
    DESCRIPTION = "Does something"

    @classmethod
    def define_inputs(cls):
        return [PortDefinition(name="input", data_type=DataType.TENSOR)]

    @classmethod
    def define_outputs(cls):
        return [PortDefinition(name="output", data_type=DataType.TENSOR)]

    def execute(self, inputs, params):
        return {"output": inputs["input"]}
```

## 熱重載

新增或編輯自訂節點後，無需重啟伺服器即可重新載入：

- 點擊工具列的 **Reload Nodes** 按鈕，或
- `POST /api/nodes/reload`。

節點會立即出現在面板中。你也可以使用 **Custom Node Manager** GUI 來上傳、啟用/停用與刪除自訂節點。

## 節點的剖析

| 成員 | 用途 |
|--------|---------|
| `NODE_NAME` | 在圖 JSON 中使用的唯一識別碼（例如 `"MyNode"`）。 |
| `CATEGORY` | 面板的分組與顏色。 |
| `DESCRIPTION` | 面向使用者的說明文字（支援 LaTeX）。 |
| `define_inputs()` / `define_outputs()` | 回傳 `PortDefinition` 清單——每個都有一個 `name`、一個 `data_type`，以及選用的 `description` / `optional` / `media`。 |
| `define_params()` | 回傳 `ParamDefinition` 清單——`int`、`float`、`string`、`bool`、`select`、檔案選擇器，或 `tensor_grid`，並可帶有 `default`、`options`、`min_value`/`max_value` 與 `visible_when`。 |
| `define_outputs_dynamic(params)` | 選用——依參數值變動輸出連接埠。 |
| `execute(self, inputs, params, *, context=...)` | 實際工作。回傳以輸出連接埠名稱為鍵的 dict。 |

## 資料型別

連接埠使用共用的 `DataType` 列舉：`TENSOR`、`MODEL`、`DATASET`、`DATALOADER`、`OPTIMIZER`、`LOSS_FN`、`SCALAR`、`STRING`、`IMAGE`、`LIST`、`ANY`、`TRIGGER`。型別相符才能讓一條邊有效；`TRIGGER` 型別從 [`Start`](/usage/first-graph) 節點驅動執行順序。

## 在執行結果面板顯示圖片

會產生圖片的節點，必須在輸出連接埠上以 `media=MEDIA_IMAGE` **明確宣告**。該連接埠的值就是一個 base64 編碼的 PNG 字串（不含 `data:` 前綴），執行結果面板會把它渲染成圖片：

```python
from app.core.node_base import MEDIA_IMAGE, BaseNode, DataType, PortDefinition

@classmethod
def define_outputs(cls):
    return [
        PortDefinition(
            name="image",
            data_type=DataType.STRING,
            media=MEDIA_IMAGE,
        ),
    ]
```

宣告是**必要的**——系統不會去檢查你的值長什麼樣子來猜它是不是圖片。沒有宣告的連接埠，不論多像圖片都仍然是一般資料。正是這一點，讓長文字輸出（LLM 的回答、token 傾印）不會被當成圖片、渲染成一張壞掉的圖。

## 在執行結果面板畫圖表

`media=MEDIA_CHART` 是同一套機制，用來畫圖表。連接埠的值是一份 JSON **圖表規格**（一個普通的 dict），由編輯器用自己的 SVG 元件繪製，因此圖表會套用主題、可以滑鼠停留查看數值，而且放大也不會糊掉——這些都是固定尺寸的 PNG 做不到的：

```python
from app.core.node_base import MEDIA_CHART, BaseNode, DataType, PortDefinition

@classmethod
def define_outputs(cls):
    return [
        PortDefinition(name="chart", data_type=DataType.ANY, media=MEDIA_CHART),
    ]

def execute(self, inputs, params, progress_callback=None, *, context=None):
    return {"chart": {
        "kind": "bar",                       # bar | line | scatter | heatmap
        "title": "各品種的平均花瓣長度",
        "bars": [{"label": "setosa", "value": 1.462}],
    }}
```

規格裡的每個數字都必須是有限的、原生的 Python `float` 或 `int`：執行事件是以 `allow_nan=False` 序列化的，而 `numpy.float32` 根本無法被 JSON 序列化。規格也要小——`node_status` 訊息超過 128 KB 就會整份被省略標記取代。各種 `kind` 的完整欄位參考，請見 [stats 外掛包的 README](https://github.com/CodefyUI/CodefyUI/blob/main/plugins/stats/README.md)，那就是本機制的參考實作。

### 自訂你自己的媒體種類

這兩種 media 都沒有被特別寫死。解析器只認連接埠宣告的那個字串，而且只要連接埠的值是非空的 dict 就會原封不動送出。因此一個外掛包宣告 `media="waveform"`，瀏覽器端就已經會收到 `{"output_kind": "waveform", ...}`；只有*繪製*它才需要改前端。遇到不認識的種類時，編輯器會直接忽略，而不是壞掉。

:::tip
需要封裝既有節點而不是撰寫新行為嗎？使用**[預設模組](./presets)**。想以可安裝的套件與他人分享節點嗎？建立一個**[外掛包](./plugins)**。
:::
