---
sidebar_position: 2
title: 自訂節點
description: 把一個 Python 檔案放進 custom_nodes/ 即可新增節點行為——可熱重載，無需修改前端。
---

# 自訂節點

CodefyUI 是**後端權威**的：一個節點的連接埠、參數與類別全部來自其 Python 定義，UI 會自動渲染它。若要新增行為，把一個 `.py` 檔案放進 `backend/app/custom_nodes/`，並繼承 `BaseNode`。

:::tip 只有幾行程式碼？先試試畫布
如果只需要簡短的轉換，或對圖表已產生的結果計算統計資料，[PythonScript 節點](./python-script-node.md)可直接執行你在畫布上輸入的 Python，不需建立檔案或重新啟動。當程式碼不再適合放在單一節點中，或需要存取檔案、網路或允許清單以外的相依套件時，再建立自訂節點。
:::

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

- 點擊工具列的**重新載入節點**按鈕，或
- `POST /api/nodes/reload`。

節點會立即出現在面板中。若要上傳檔案而不是將檔案複製至目錄，請使用[自訂節點管理](#uploading-through-the-custom-node-manager)；每次操作後都會重新載入節點定義。

## 透過自訂節點管理上傳 {/* #uploading-through-the-custom-node-manager */}

請使用側邊欄**自訂與外掛**分頁之**自訂節點**區段中的**管理...**按鈕開啟管理視窗。它會列出 `custom_nodes/` 中的每個檔案、該檔案定義的節點名稱，以及三種操作：

- **上傳 .py** 會將一個檔案送至 `POST /api/custom-nodes/upload`。檔案必須使用 `.py` 副檔名，而且不得超過 `CODEFYUI_MAX_UPLOAD_SIZE`（500 MB）。伺服器會以外掛 AST 閘門的[第 0 級](/advanced/plugins#安全性三個層級)掃描檔案。自訂節點無法宣告能力，因此 `requests` 或 `os` 等 imports 會產生 `400` response 與閘門訊息。需要第 0 級以外 imports 的節點，應放入具有 `[security]` 區段的[外掛包](./plugins)。直接複製到 `backend/app/custom_nodes/` 的檔案，會在下次重新載入時載入，不會進行這項掃描。
- **啟用／停用**會在 `name.py` 與 `name.py.disabled` 之間重新命名檔案；停用的檔案仍留在磁碟上，但探索時會跳過。
- **刪除**會移除檔案（名稱以 `__` 開頭者受保護）。

每次操作後，伺服器都會重新探索自訂節點、外掛包與預設模組。request 完成時，節點面板會反映結果，不需另外重新載入。

## 節點的剖析

| 成員 | 用途 |
|--------|---------|
| `NODE_NAME` | 在圖 JSON 中使用的唯一識別碼（例如 `"MyNode"`）。 |
| `CATEGORY` | 面板的分組與顏色。 |
| `DESCRIPTION` | 面向使用者的說明文字（支援 LaTeX）。 |
| `define_inputs()` / `define_outputs()` | 回傳 `PortDefinition` 清單——每個都有一個 `name`、一個 `data_type`，以及選用的 `description` / `optional` / `media`。 |
| `define_params()` | 回傳 `ParamDefinition` 清單——`int`、`float`、`string`、`bool`、`select`、檔案選擇器（`model_file`、`image_file`、`data_file`）、`tensor_grid`、`code`（具語法標示的多行編輯器；仍是普通的 string 參數），或 `secret`，並可帶有 `default`、`options`、`min_value`/`max_value` 與 `visible_when`。`secret` 參數（例如 API key）在編輯器裡會被遮罩，而且它的值**永遠不會被保存**——存檔、匯出與發佈時都會被清空，所以要把它提供給已發佈的應用程式，請改用環境變數。 |
| `define_outputs_dynamic(params)` / `define_inputs_dynamic(params)` | 選用。依參數值變更輸出或輸入連接埠，例如 `Split` 的 `chunks` 或 `PythonScript` 的 `input_ports`。靜態方法必須描述預設參數，因為節點面板會使用這些定義；驗證、渲染與預設模組匯出會使用動態定義。 |
| `execute(self, inputs, params, progress_callback=None, *, context=None)` | 執行節點，並回傳以輸出連接埠名稱為鍵的 dict。執行引擎只會在函式簽名宣告時傳入各個選用關鍵字參數。`progress_callback` 會為每個進度事件接收一個 dict；例如，訓練迴圈會送出 `{"event": "epoch", ...}`。`context` 會提供該次執行的裝置、seed 與 determinism 旗標。 |
| `REQUIRES_PACK` | 選用的類別屬性，用來識別執行時需要的[選用套件包](/usage/optional-packs)（預設為 `None`）。`/api/nodes` 會以 `requires_pack` 提供此值，讓節點面板顯示套件包徽章，並讓編輯器在執行失敗前提供安裝操作。 |
| `cacheable` / `align_inputs` / `cache_fingerprint(params)` | 選用的快取與裝置控制。當節點具有可訓練狀態、回傳即時物件參照，或產生未反映在回傳值中的副作用時，請設定 `cacheable = False`。將輸入直接傳給 numpy、sklearn 或 PIL 時，請設定 `align_inputs = False`；否則執行引擎會將輸入張量移至該次執行的裝置，而 `Tensor.numpy()` 對 CPU 以外的張量會失敗。覆寫 `cache_fingerprint`，可將參數所參照的外部狀態（例如檔案修改時間）加入快取鍵。 |

## 資料型別

連接埠使用共用的 `DataType` 列舉：`TENSOR`、`MODEL`、`DATASET`、`DATALOADER`、`OPTIMIZER`、`LOSS_FN`、`SCALAR`、`STRING`、`IMAGE`、`LIST`、`TRANSFORM`、`ANY`、`TRIGGER`。型別相符才能讓一條邊有效；`TRIGGER` 型別從 [`Start`](/usage/first-graph) 節點驅動執行順序。

## 在結果面板顯示圖片 {/* #在執行結果面板顯示圖片 */}

會產生圖片的節點，必須在輸出連接埠上以 `media=MEDIA_IMAGE` **明確宣告**。該連接埠的值就是一個 base64 編碼的 PNG 字串（不含 `data:` 前綴），結果面板會把它渲染成圖片：

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

系統不會根據值的內容判斷它是否為圖片，因此媒體宣告是**必要的**。未宣告媒體類型的連接埠一律視為一般資料，即使內容看似圖片也是如此。這可避免將長文字輸出（LLM 的回答、token 傾印）誤當成圖片而渲染失敗。

## 在結果面板畫圖表 {/* #在執行結果面板畫圖表 */}

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

規格裡的每個數字都必須是有限的原生 Python `float` 或 `int`：執行事件會以 `allow_nan=False` 序列化，而 `numpy.float32` 無法由 JSON 序列化。規格也必須保持精簡；超過 `CODEFYUI_RUN_EVENT_PAYLOAD_CAP_BYTES`（預設 128 KB）的 `node_status` 訊息會整份由省略標記取代。各種圖表種類的完整欄位參考，請見 [stats 外掛包的 README](https://github.com/CodefyUI/CodefyUI/blob/main/plugins/stats/README.md)；這也是本機制的參考實作。

## 產生可播放的影片 {/* #emitting-a-playable-video */}

`media=MEDIA_VIDEO` 是第三種內建媒體種類，並以**參照**方式運作：影片片段無法直接放入事件串流（單一 `node_status` 事件的上限是 128 KB），所以檔案會放在 `settings.MEDIA_DIR` 底下，連接埠的值則是一個指向該檔案的小型 dict。`/api/media/<path>` 會以正確的 `Content-Type` 內嵌提供檔案，讓編輯器的 `<video>`/`<img>` 元素可以直接播放：

```python
{"path": "rollouts/run1.mp4",          # POSIX 風格，相對於 MEDIA_DIR
 "url": "/api/media/rollouts/run1.mp4",
 "format": "mp4",                       # mp4 | gif | webm
 "fps": 10.0, "frames": 240, "width": 96, "height": 96, "bytes": 81234}
```

請勿自行建立這些參照。請透過 `VideoWrite` 節點寫入影格，或在你的節點中呼叫 `core.video_io`。它負責編碼（mp4 使用 PATH 中的 `ffmpeg` 執行檔；gif 使用 Pillow，完全不需要額外相依套件）、限制路徑範圍，以及建立參照格式。絕對路徑的 `path` 會在傳輸層被拒絕。

### 自訂你自己的媒體種類

系統不會對任何媒體種類加入特殊判斷。解析器以連接埠宣告的字串為鍵；只要連接埠的值是非空 dict，就會原封不動送出。因此，外掛包宣告 `media="waveform"` 後，瀏覽器便會收到 `{"output_kind": "waveform", ...}`；只有*繪製*該資料時才需要修改前端。編輯器遇到不認識的種類時會忽略，不會發生錯誤。

:::tip
需要封裝既有節點而不是撰寫新行為嗎？使用**[預設模組](./presets)**。想以可安裝的套件與他人分享節點嗎？建立一個**[外掛包](./plugins)**。
:::
