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

:::tip
需要封裝既有節點而不是撰寫新行為嗎？使用**[預設模組](./presets)**。想以可安裝的套件與他人分享節點嗎？建立一個**[外掛包](./plugins)**。
:::
