---
sidebar_position: 7
title: PythonScript 節點
description: 直接在畫布上寫 Python——函式介面、Tier-0 匯入政策、誠實的安全模型，以及統計常用範例。
---

# PythonScript 節點

CodefyUI 裡其他每個節點，都是某個人寫好、安裝進來的 Python 檔案。**PythonScript** 則是內容由你在瀏覽器裡直接輸入的節點。它存在的理由很簡單：統計與研究的需求永遠跑在節點庫前面，而為了四行 numpy 去寫一個[自訂節點](./custom-nodes.md)或發一個[外掛套件](./plugins.md)，手續實在太多。

從節點面板拖出它（分類 **Utility**），在節點上雙擊打開 **程式碼** 分頁，然後寫：

```python
def run(inputs, params):
    x = inputs["in1"]
    return {"out1": x.mean()}
```

## 函式介面

```python
def run(inputs: dict, params: dict) -> dict
```

| | |
|---|---|
| `inputs` | **有連線**的每個輸入連接埠各一個鍵：`in1`、`in2`……。沒接線的連接埠不會出現在字典裡，所以在你自己的設計中屬於選用的連接埠請用 `inputs.get("in2")`。 |
| `params` | 這個節點參數的複本（也包含 `code` 本身）。改動它不會有任何效果——節點交給你的是複本。 |
| 回傳 | 以輸出連接埠為鍵的字典：`{"out1": ..., "out2": ...}`。**直接回傳單一值時會當作 `out1`**，所以只有一個輸出時 `return x` 就夠了。 |
| 多餘的鍵 | 回傳了不對應任何連接埠的鍵會被捨棄，並在執行紀錄中說明，而不是默默吞掉。 |

`run` 必須定義在模組層級、名稱完全一致。少了它時編輯器會立刻提醒，不會等到執行失敗才說。

`async def run` 會**在檢查階段就被拒絕**：節點是在沒有事件迴圈的工作執行緒上呼叫 `run`，不會有人 await 它，輸出連接埠只會拿到一個 coroutine 物件。

除了允許的函式庫之外，命名空間裡還有兩個名稱：

* `device`——本次執行解析出來的運算裝置，讓 `torch.zeros(3, device=device)` 落在流程其他部分所在的裝置上。
* `should_stop()`——本次執行的協作式停止旗標。腳本無法從外部被中斷（見下方的安全模型），所以想要能被停止的長迴圈必須自己詢問：

  ```python
  def run(inputs, params):
      total = 0.0
      for row in inputs["in1"]:
          if should_stop():
              break
          total += float(row.sum())
      return total
  ```

## 連接埠

`input_ports` 與 `output_ports`（各 1–8）決定節點有幾個連接點。**程式碼** 分頁的連接埠區塊可以同時設定數量與每個連接埠的 `DataType`；同樣的值以逗號分隔字串存在 `input_types` / `output_types` 參數裡，所以流程 JSON 仍然易讀。

* 輸入連接埠一律是**選用**的：需要哪些由腳本自己決定，所以宣告四個連接埠只接兩條線仍是合法的流程。
* 輸出連接埠預設是 `ANY`，可以接到任何地方。改成實際型別（`TENSOR`、`SCALAR`、`STRING`……）之後，流程驗證器就會幫你檢查接線——腳本寫好之後很值得補上。
* 調降連接埠數量時，接在已消失連接埠上的連線會被**移除**，受影響的下游節點也會被標記為需要重跑。

## 快取

`code` 就是一個普通參數，所以執行快取跟其他參數一樣把它算進金鑰：改了腳本就會重跑這個節點與其下游；同樣的腳本、同樣的輸入跑第二次則直接命中快取。

## 輸出與錯誤

腳本裡的 `print()` 會被擷取（每次執行上限 64 KB），以節點的紀錄行出現在**執行紀錄**中。這不是全域的 stdout 轉向——直接寫入 `sys.stdout` 的函式庫仍會輸出到伺服器主控台——因為節點共用執行緒池，劫持整個行程的 stdout 會把其他節點同時印出的內容一起吃掉。

發生例外時，訊息會標出**你腳本裡的行號**：

```
PythonScript failed at line 4: ZeroDivisionError: division by zero
```

標出的是你程式碼裡最深的那一層：在 `statistics.mean([])` 裡爆掉會指向呼叫它的那一行，在你自己寫的輔助函式裡爆掉則指向那個函式。腳本在失敗前印出的內容也會附在訊息後面。流程的錯誤處理模式（fail-fast／continue／retry）與其他節點完全一致。

## Tier-0 政策

程式碼在**每次編輯**時就會檢查，而且是在編譯之前——用的是把關外掛套件的同一套 AST 檢查器（`backend/app/core/plugin_validator.py`），只是切換成允許清單模式。被拒絕時你會立刻在編輯器下方看到紅色訊息並標出該行，而不是在訓練流程跑了十分鐘後才失敗。

**可匯入的模組**（`backend/app/core/script_policy.py` 的 `TIER0_MODULES`）：

```
collections   itertools   json   math   numpy   re   statistics   torch
```

這些模組也已經以同名**預先綁定**在命名空間裡，所以不寫 import 也能直接用 `math.floor(x)`；喜歡別名的話，`import numpy as np` 也可以。

**會被拒絕，並附上替代路徑說明：**

* 其他任何匯入——`os`、`sys`、`pathlib`、`subprocess`、`socket`、`urllib`、`requests`，以及 `pandas`、`sklearn` 這類並不危險、只是不在清單上的模組。相對匯入同樣被拒絕。
* `exec`、`eval`、`compile`、`__import__`、`open`、`input`、`globals`、`locals`、`vars`、`dir`、`breakpoint`、`exit`。
* 直接使用模組機制的名稱——`__loader__`、`__spec__`、`__builtins__`、`__package__`——以及雙底線屬性存取（`__class__`、`__globals__`、`__subclasses__`、`__code__`……）。這些是通用的逃逸手法：`__loader__.load_module('nt')` 不用任何 import 就能拿到真正的 `os` 模組。
* 允許的函式庫**內部**通往檔案系統、網路、編譯器或其他行程的那些門：`torch.hub`（會下載並執行遠端的 `hubconf.py`）、`torch.utils.cpp_extension`（編譯並執行 C++）、`torch.distributed`、`torch.multiprocessing`、`numpy.savetxt` / `loadtxt` / `fromfile` / `tofile` / `save` / `memmap`、`numpy.ctypeslib`，以及 `script_policy.py` 裡 `TIER0_DENIED_ATTRS` 的其餘項目。用 import 把它們的名稱帶進來、或用字面值 `getattr` 取得，同樣會被拒絕。
* 沒有明確寫 `weights_only=True` 的 `torch.load(...)` / `numpy.load(...)`，以及任何 `load(allow_pickle=True)`：它們會執行來源檔案裡的程式碼。接收端會透過你的 import 別名解析，所以 `import torch as t; t.load(...)` 一樣會被擋下。

`json.load` 與 `json.loads` **不**受最後這條規則影響——`json` 本來就是 Tier-0 模組，解析 JSON 正是它的用途。

上述規則中有兩條在執行期還有第二道鎖：命名空間的 builtins 以明確的允許清單建立（所以 `open`、`eval` 這些不只是不能改，而是根本不存在），而 `__import__` 被換成會重新檢查模組清單的版本。雙底線與函式庫門戶這兩條規則**只在 AST 階段**檢查——程式碼開始執行後就沒有人再檢查一次，這正是這道關卡必須在編譯之前跑的原因。

### 需要清單以外的東西？

那正是另外兩條路存在的理由，而且它們更適合：

* [自訂節點](./custom-nodes.md)——放在你自己專案裡的檔案，你寫的、你能檢查。
* [外掛套件](./plugins.md)——可安裝、有版本，還能在 manifest 裡宣告額外模組，由使用者以 `--trust-author` 接受。

## 安全模型——這是防護欄，不是沙箱

在把 CodefyUI 開到網路介面上之前，請先讀這一節。

這道關卡擋的是**容易的**逃逸手法。它**不是**沙箱，也沒有打算變成沙箱：

* **它限制的是腳本能碰到哪些函式庫，而不是那些函式庫能做什麼。** numpy 與 torch 本身就大到內建了檔案 IO、下載功能與一個 C++ 編譯器。上面那份拒絕清單關掉的是我們知道的門；它是架在兩套龐大 API 之上的黑名單，只能視為提高逃逸的成本，絕不是「檔案與網路碰不到」的保證。
* 腳本在 **CodefyUI 伺服器行程內**執行，使用你的使用者權限。沒有任何容器隔離。
* 已經能在你畫布上打字的攻擊者，只要夠有決心，多半仍找得到出口。這道關卡提高的是隨手執行程式碼的成本，並不能讓這個面向對有備而來的對手變得安全。
* 沒有任何 CPU 或記憶體限制，而且失控的腳本不只影響自己的節點。節點是在直譯器的**預設執行緒池**上執行，所以某個腳本裡的 `while True:` 會餓死這個行程裡**所有**節點的執行，直到伺服器重啟。「停止」是協作式的、只在節點**之間**檢查，無法中斷節點內部的迴圈——長迴圈必須自己呼叫 `should_stop()`。
* **模組物件是與主行程共用的。** 允許清單裡的模組就是本尊，不是複本，所以腳本裡的 `torch.zeros = 別的東西` 會影響其他所有節點、所有流程，直到伺服器重啟為止。run 與 run 之間不會還原任何東西。
* `code` 參數跟其他參數一樣會存進流程 JSON。**打開來路不明的流程並按下執行，等於執行對方的 Python。** 政策檢查是這中間唯一的一道防線，這正是它在編譯之前就跑、而不是等到匯入時才跑的原因。

真正的界線是**誰能碰到這個編輯器**。CodefyUI 預設只綁定本機；除非你信任該網路上的所有人，否則請維持原狀。

## 匯出

「匯出為 Python」會把腳本以「一行原始碼一個字串常量」的形式寫進產生的檔案，並附上來源註解：

```python
def n02_pythonscript(ctx):
    "PythonScript - node 'py1'."
    params = {
        # ---- 'code' of node 'py1': the script this node runs, verbatim ----
        'code': (
            'def run(inputs, params):\n'
            '    x = inputs["in1"]\n'
            '    return {"out1": x.mean(dim=0)}\n'
        ),
        'input_ports': 1,
        'output_ports': 1,
    }
    return _call('PythonScript', 'py1', params, ctx, inputs={'in1': in1})
```

它易讀、可改——改一行，匯出的流程就照著改動跑——而且仍然只是字串常量，所以你程式碼裡的 `'''` 永遠不可能變成程式本體。

## 統計常用範例

### 每個通道的平均與標準差

一個 TENSOR 進、兩個 TENSOR 出（`output_ports: 2`、`output_types: TENSOR,TENSOR`）：

```python
def run(inputs, params):
    x = inputs["in1"]                     # (N, C, H, W)
    flat = x.reshape(x.shape[0], x.shape[1], -1)
    return {
        "out1": flat.mean(dim=(0, 2)),
        "out2": flat.std(dim=(0, 2)),
    }
```

### 標籤批次的類別分佈

一個 TENSOR 進、一個 STRING 出：

```python
import collections

def run(inputs, params):
    labels = inputs["in1"].flatten().tolist()
    counts = collections.Counter(labels)
    total = sum(counts.values())
    lines = [
        f"class {int(k)}: {v} ({100 * v / total:.1f}%)"
        for k, v in sorted(counts.items())
    ]
    print("\n".join(lines))          # 同時會進到執行紀錄
    return {"out1": "\n".join(lines)}
```

### 穩健統計摘要（中位數、四分位距、離群值數量）

```python
import statistics

def run(inputs, params):
    values = sorted(float(v) for v in inputs["in1"].flatten().tolist())
    q1, _, q3 = statistics.quantiles(values, n=4)
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return {
        "out1": {
            "median": statistics.median(values),
            "iqr": iqr,
            "outliers": sum(1 for v in values if v < low or v > high),
        }
    }
```

### 比較兩個張量

兩個 TENSOR 進（`input_ports: 2`）、一個 SCALAR 出：

```python
def run(inputs, params):
    a, b = inputs["in1"], inputs["in2"]
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    return float((a - b).abs().max())     # 單一值 -> out1
```

### 在本次執行的裝置上建立張量

`device` 是本次執行解析出來的運算裝置，讓建立張量的腳本跟流程其他部分待在同一個裝置上：

```python
def run(inputs, params):
    return torch.zeros(4, 4, device=device)
```

## 限制

| | |
|---|---|
| 腳本長度 | 100,000 個字元。超過這個長度請改寫成自訂節點。 |
| 連接埠 | 每邊 1–8 個。 |
| 擷取的輸出 | 每次執行 64,000 個字元，超過後截斷並附註。 |
| 非同步 | `run` 必須是一般的 `def`；`async def run` 會在檢查階段被拒絕，`asyncio` 也無法匯入。 |
| 狀態 | 每次執行都會拿到全新的命名空間，所以你自己的模組層級變數不會延續。但匯入進來的**模組**是主行程的那一份——見安全模型一節。真正需要保留的狀態請交給流程本身。 |
