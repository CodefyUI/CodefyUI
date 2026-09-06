---
sidebar_position: 7.5
title: 把 graph 當成函式呼叫
description: 透過 HTTP 以具名函式呼叫任何已儲存的 graph，不需操作圖形介面 — 傳入宣告的輸入，取得宣告的輸出。
---

# 把 graph 當成函式呼叫

任何能在畫布上執行的 graph，也都能透過 HTTP 當成一個具名函式來呼叫：你用兩種節點宣告它的輸入與輸出、存檔，然後 `POST /api/graph/run/{name}` 就會執行它並回傳 JSON。這個端點永遠執行**最新儲存的檔案**，並使用每次重新啟動都會輪換的工作階段權杖（session token）。若需要穩定介面，請把它[發佈](./publish)成有版本、由長效 API key 保護的應用程式。

**何時使用這個端點，何時使用 [CLI 圖形執行器](./cli-runner)：** CLI 執行器會在全新的 Python 行程中執行 `graph.json`，不需要伺服器，適合批次工作與 CI。執行 API 則會呼叫*執行中* CodefyUI 伺服器上的*已儲存* graph，適合腳本、筆記本與自動化流程：這些場景需要的是具型別輸入與輸出的函式呼叫，而不是啟動一個行程。

## 1. 在畫布上宣告合約 {/* #1-declare-the-contract-on-the-canvas */}

面板 **IO** 群組裡的兩種節點定義了 graph 的函式簽章：

- **GraphInput** —— 每個輸入一個。參數：`name`（須為合法識別字：`^[a-zA-Z_][a-zA-Z0-9_]{0,63}$`）、`type`（`string` / `number` / `integer` / `boolean` / `json` / `image`）、`required`、`default`、`description`。
- **GraphOutput** —— 每個輸出一個。參數：`name`、`description`。把你想回傳的值接進它的 `value` 連接埠。

**把 Start 接到每一個 GraphInput。** 執行端點要求每一個 GraphInput 都必須帶有一條 trigger 邊，否則會在一開始就拒絕這張 graph（409 `untriggered_input`）。這是合約規則，不是引擎限制。未連接 trigger、但將資料傳給執行中節點的資料根節點[仍然會執行](./running-graphs#沒有-trigger-的節點仍然可能執行)。這項規則可確保宣告的輸入不受引擎可達性判定影響。

```text
[Start] --trigger--> [GraphInput name="message"] --value--> [Print] --value--> [GraphOutput name="echo"]
```

`default` 參數永遠是畫布上的測試值，所以同一張 graph 從**執行**按鈕執行時，行為仍不變。對 API 呼叫來說，`default` 只有在 `required` 關閉時才會套用。`default` 是一個字串欄位，依 `type` 解析（`2.5`、`true`、`{"k": 1}`）—— 這種字串解析是 API 的嚴格型別檢查唯一不適用的地方。

## 2. 替外部腳本取得權杖 {/* #2-getting-the-token-for-external-scripts */}

會改變狀態的請求需要 `X-CodefyUI-Token` 標頭。取得它有兩種方式：

- 讀取權杖檔。由 `cdui start` 或 `cdui dev` 啟動的伺服器，會把使用者資料放在 `<install dir>/.codefyui_dev/` 底下，而預設的安裝目錄是 `~/CodefyUI`（Windows 上是 `$HOME\CodefyUI`）—— 所以權杖就在 `~/CodefyUI/.codefyui_dev/session.token`。同一個目錄也包含外掛的 lockfile（`plugins/installed.json`）、素材快取（`cache/`）、ChatGPT 登入（`llm/codex_auth.json`）與重新啟動安裝的檔案（`packs/`）。如果伺服器啟動前已設定 `CODEFYUI_USER_DATA_DIR`，則改用該目錄。platformdirs 的位置 —— `%LOCALAPPDATA%\codefyui`、`~/.local/share/codefyui`、`~/Library/Application Support/codefyui` —— 只適用於未設定 `CODEFYUI_USER_DATA_DIR` 且手動啟動的 `uvicorn app.main:app`。
- 或者用 `GET /api/auth/bootstrap`，它會對 `Host` 標頭在白名單內的任何請求回傳 `{"token": "..."}`。這不限於伺服器所在的電腦：綁定區域網路時，透過白名單位址連線的遠端用戶端也能取得權杖。

權杖在每次伺服器重啟時都會輪換 —— 腳本應該每次重新讀檔，而不是快取權杖值。

## 3. 檢視合約 {/* #3-inspect-the-contract */}

```text
GET /api/graph/contract/{name}     (no auth required, like /api/graph/load)
```

```json
{
  "graph": "my-graph",
  "inputs":  [{"name": "prompt", "type": "string", "required": true, "default": null, "description": ""}],
  "outputs": [{"name": "answer", "type": "SCALAR", "description": ""}],
  "problems": []
}
```

- 必填輸入的 `default` 是 `null` —— API 不會套用必填輸入的預設值，因此合約也不會列出該值。選填輸入會顯示解析後的預設值。
- 輸出的 `type` 是從連接至 GraphOutput 的連接埠推導出來的（無法解析時為 `ANY`）。
- `problems` 列出合約問題（名稱不合法、名稱重複、選填預設值無法解析、選填的 image 輸入）。這些問題在此不會造成請求失敗，方便你檢視尚未完成的 graph，但它們會讓 `/run` 以 409 被拒絕。

## 4. 執行 graph {/* #4-run-the-graph */}

```text
POST /api/graph/run/{name}         (auth: X-CodefyUI-Token header)
Content-Type: application/json
```

request body 是**選填的**（沒有 body 等同 `{}`），而且每個欄位都是選填：

```json
{
  "inputs": {"prompt": "hello"},   // default {}
  "timeout_s": 300,                // default 300, min 1, max 3600
  "device": "cuda",                // "cpu" / "cuda" / "mps"; falls back to CPU when unavailable
  "record_outputs": false          // default false; see gotchas before enabling
}
```

### 回應封裝格式 {/* #the-response-envelope */}

每一個 `/run` 的回應 —— 不論成功或失敗 —— 都是這同一種形狀，而且**所有**的鍵永遠都在（不適用時為 `null`）：

```json
{
  "status": "ok",                 // "ok" | "error"
  "run_id": "9f2c...",            // assigned at request entry; NEVER null
  "graph": "my-graph",
  "app": null,                    // published-app slug; always null on this editor route
  "version": null,                // published version; always null on this editor route
  "device": "cuda",               // what you actually got; null on early rejections
  "outputs": {"answer": 0.93},    // null unless status == "ok"
  "error": null,                  // null on success, else {"code", "message", "node_id", "details"}
  "timing": {"total_s": 1.234}    // null when execution was never attempted
}
```

`app` 與 `version` 標明的是 [`POST /api/apps/{slug}/invoke`](./publish) 上的已發佈應用程式 —— invoke 路由會填入這兩個欄位（slug 在每一種結果中都有值；version 則在解析完成後才有值），而這條編輯器路由的兩個欄位永遠都是 `null`。

HTTP 狀態碼與 `status`／`error.code` 相對應，因此 `raise_for_status()` 可以正常運作。

向前相容：**客戶端必須忽略封裝格式裡不認識的欄位。** **`error.code` 是一個開放的列舉 —— 不認識的代碼請當成一般錯誤處理。** 未來的非同步模式會回傳 202 `{"status": "queued", "run_id": ..., "job": {...}}` —— 依 `status` 分支的同步客戶端不需修改即可繼續運作。

### 錯誤分類 {/* #error-taxonomy */}

分流原則：**404 = 名稱錯誤；409 = 修正 graph；413 = 縮小 payload；422 = 修正 payload；500 = 執行失敗（或無法讀取 graph 檔案）。**

| `error.code` | HTTP | 觸發條件 |
| --- | --- | --- |
| `graph_not_found` | 404 | 沒有名稱完全相符的 graph 檔案（嚴格比對名稱 —— `my.graph` 永遠不會被當成 `my_graph` 的別名） |
| `graph_unreadable` | 500 | graph 檔案存在，但 JSON 已損毀 |
| `invalid_contract` | 409 | 合約的 `problems[]` 不是空的（`details` 會列出它們） |
| `no_entry_points` | 409 | graph 裡完全沒有 Start trigger |
| `untriggered_input` | 409 | 某個 GraphInput 沒有進來的 trigger 邊（`details`：輸入名稱） |
| `unreachable_output` | 409 | 某個 GraphOutput 從任何 Start 都到不了（`details`：輸出名稱） |
| `invalid_graph` | 409 | graph 驗證失敗，不論是靜態驗證或執行期（`details`：錯誤） |
| `invalid_input` | 422 | 不認識的輸入名稱（區分大小寫）、缺少必填輸入、型別不符、body 格式錯誤 —— 全部彙整在 `details` |
| `payload_too_large` | 413 | body 超過 `MAX_RUN_BODY_BYTES`（預設 64 MB，`CODEFYUI_MAX_RUN_BODY_BYTES`） |
| `execution_error` | 500 | 某個節點拋出例外；`node_id` 指出是哪一個 |
| `timeout` | 500 | `timeout_s` 到期；`timing.total_s` = 已經過的時間 |
| `output_not_produced` | 500 | 宣告的輸出不在引擎的結果裡（安全網） |
| `output_too_large` / `unserializable_output` | 500 | 見下方的輸出序列化 |

## 5. 輸入型別 {/* #5-input-types */}

| `type` | 該送什麼（JSON） | 會被拒絕 |
| --- | --- | --- |
| `string` | 一個 JSON 字串 | 數字、布林值、null（不會隱含地做 `str()`） |
| `number` | int 或 float | 字串（`"3"`）、布林值、null |
| `integer` | int，或小數部分為零的 float（`3.0` -> `3`） | `3.5`、字串、布林值 |
| `boolean` | `true` / `false` | `0`/`1`、`"true"` |
| `json` | 任何 JSON 值 | 沒有 |
| `image` | 一個 base64 字串（可以加上 `data:image/...;base64,` 前綴） | 非字串、無法解碼的資料 |

型別檢查刻意採用嚴格規則：JSON 本來就帶有型別，因此輸入錯誤會直接回報，不會自動轉型。唯一放寬的是 `integer` 接受整數值的 float（JS 客戶端無法控制 `3` 是否序列化成 `3.0`）。

`image` 輸入送達 graph 時，是一個值域在 `[0, 1]` 的 `(C, H, W)` float32 tensor，格式與 `ImageReader` 的輸出相同。image 輸入必須是 `required`（沒有合理的 base64 預設值）；節點的 `default` 是伺服器本機檔案路徑，只用於畫布執行。

```python
import base64
from pathlib import Path

img_b64 = base64.b64encode(Path("cat.png").read_bytes()).decode("ascii")
# ... json={"inputs": {"photo": img_b64}}
```

## 6. 輸出序列化 {/* #6-output-serialization */}

| GraphOutput 上的值 | JSON 形式 |
| --- | --- |
| `None`、bool、int、float、str | 原樣（base64 字串形式的圖表以純字串通過） |
| dict / list / tuple | 遞迴序列化（tuple 會變成 list） |
| numpy 純量 | 一般數字 |
| tensor / ndarray | `{"__type__": "tensor", "shape": [...], "dtype": "torch.float32", "values": [...]}` —— 上限 **65,536 個元素**；0 維 tensor 保留 `"shape": []` |
| PIL 圖片 | `{"__type__": "image", "format": "png", "base64": "..."}` |
| `torch.nn.Module` | 錯誤 —— 請在 graph 裡用 ModelSaver 節點把它存檔，改回傳路徑字串 |
| 其他值 | 錯誤 `unserializable_output`，並指出型別 |

## 7. 範例 {/* #7-examples */}

先在畫布上把 graph 存檔（執行 API 是依名稱執行*已儲存*的 graph）。Python `requests`：

```python
from pathlib import Path

import requests

# `cdui start` / `cdui dev` keep the token under the install dir (section 2):
#   ~/CodefyUI/.codefyui_dev/session.token   (Windows: $HOME\CodefyUI\.codefyui_dev\session.token)
token = (Path.home() / "CodefyUI" / ".codefyui_dev" / "session.token").read_text().strip()

resp = requests.post(
    "http://127.0.0.1:8000/api/graph/run/Api-Function",
    headers={"X-CodefyUI-Token": token},
    json={"inputs": {"message": "hello from Python"}},
    timeout=310,  # slightly above the server-side default timeout_s=300
)
resp.raise_for_status()
envelope = resp.json()
print(envelope["outputs"]["echo"])
```

Windows 上使用 curl 時，請用 `curl.exe`（PowerShell 會把 `curl` 當成 `Invoke-WebRequest` 的別名），而且**一律**用 `--data "@payload.json"` 以檔案傳入 request body。行內 JSON 會受到 cmd 的 8191 字元上限與 PowerShell 引號規則限制，行內 base64 圖片則完全無法傳入：

```powershell
# payload.json: {"inputs": {"message": "hello from curl"}}
$token = Get-Content "$HOME\CodefyUI\.codefyui_dev\session.token"
curl.exe -s -X POST "http://127.0.0.1:8000/api/graph/run/Api-Function" `
  -H "X-CodefyUI-Token: $token" -H "Content-Type: application/json" `
  --data "@payload.json"
```

同一個呼叫的 bash 版本：

```bash
curl -s -X POST "http://127.0.0.1:8000/api/graph/run/Api-Function" \
  -H "X-CodefyUI-Token: $(cat ~/CodefyUI/.codefyui_dev/session.token)" \
  -H "Content-Type: application/json" \
  --data "@payload.json"
```

為不熟悉的 graph 撰寫腳本時，請先檢視其合約：

```powershell
curl.exe -s "http://127.0.0.1:8000/api/graph/contract/Api-Function"
```

上述呼叫所需的 graph 位於 `examples/Usage_Example/Api-Function/`。從範例集開啟並儲存後，上面的指令不需修改即可執行。

## 8. 限制與注意事項 {/* #8-limits-and-gotchas */}

- 403（權杖缺失／無效）與 421（Host 防護）**不使用**上述回應封裝格式，因為它們在進入路由前就會觸發。413 在這條路由與 `/invoke` 上仍使用該封裝格式。
- `MAX_RUN_BODY_BYTES` 上限不只適用於 `/run`，而是適用於**每個**端點。只有四條上傳路由例外，改用 `MAX_UPLOAD_SIZE`（預設 500 MB，`CODEFYUI_MAX_UPLOAD_SIZE`）。因此，以 `POST /api/graph/save` 儲存 70 MB graph 時也會收到 413。系統會在位元組抵達時累計大小，而不是依賴 `Content-Length`，所以未宣告長度的 chunked request 也會接受相同的大小檢查。
- 畫布不是透過 HTTP 傳送 graph，而是使用 `WS /ws/execution`，因此另受 `WS_MAX_MESSAGE_BYTES`（`CODEFYUI_WS_MAX_MESSAGE_BYTES`）限制。它預設採用 `MAX_RUN_BODY_BYTES` 的值，所以預設的 64 MB graph 上限同時涵蓋兩種傳輸；調高 HTTP 上限也會調高這個上限。WebSocket 沒有 request body，因此不會回傳 413：傳輸層會在訊息片段組裝期間拒絕該訊息，連線並以 WebSocket 代碼 **1009**（`message too big`）關閉。編輯器會把該代碼轉成「graph 太大」訊息後重新連線；非瀏覽器客戶端會在 close frame 中收到 1009 與原因。`cdui start` 與 `cdui dev` 會透過 `--ws-max-size` 把這個值傳給 uvicorn。手動啟動 uvicorn 時會使用其預設值 16 MB，比 HTTP 上限更嚴格，因此需要自行傳入該參數。
- 這台伺服器從不送出 504；504 一定來自中間的某個代理。
- `record_outputs=true` 會讓區域網路上任何知道 `run_id` 的使用者都可讀取輸入與結果（GET 輸出端點不需要驗證；傳輸是純 HTTP）。已發佈應用程式的執行紀錄儲存在 SQLite 並受 key 保護；檢視器儲存區只供編輯器使用，invoke 不會寫入。若要讓 graph 的執行內容固定在特定版本並受 key 保護，請將它[發佈](./publish)。
- 不要把機密寫入 `default` 值 —— `GET /contract` 與 `/load` 都不需要驗證。
- `device: "auto"`（或無法使用的裝置）會自動解析為 CPU，且不會回報錯誤；封裝格式中的 `device` 欄位會顯示實際使用的裝置。
- 單一 tensor 輸出超過 65,536 個元素時，整次呼叫會失敗。請移除該 GraphOutput，或改用 `record_outputs` 與可切片的輸出 API（`GET /api/execution/outputs/{run_id}/{node_id}/{port}?slice=...`）；輸出篩選器尚未實作。
- 並行執行共用行程的預設執行緒池（每次執行的平行上限為 4，並非全域上限），因此高負載執行會競爭 CPU／GPU 資源。
- 伺服器中只要有任何**設定亂數種子的**執行正在進行，其他呼叫就會等待；設定亂數種子的執行也會等待既有呼叫完成。一般呼叫仍可彼此重疊。請見[可重現的執行](./running-graphs#可重現的執行亂數種子)—— 如果同一台伺服器也用於設定亂數種子的訓練，請將等待時間計入 `timeout_s`。
- 逾時並取消後，不會再啟動新節點；正在執行的節點會在背景完成（節點可以輪詢 `context.cancelled`，以提早停止）。
- 客戶端斷線不會停止執行；只有逾時會停止。斷線後的執行結果不會保留，除非 `record_outputs=true`。

## 9. 路線圖 {/* #9-roadmap */}

- **第二階段（已推出）：[發佈](./publish)** —— 以 `POST /api/apps/{slug}/invoke` 提供版本化應用程式，並包含長效 API key、受 key 保護的 SQLite 執行紀錄、每張圖片的像素預算、每個應用程式各自的 OpenAPI 文件，以及用於區域網路服務的 `cdui start --host/--port`。
- `cdui call <graph> --input k=v` / `cdui publish` / `cdui keys` —— 這些 API 的 CLI 包裝（接續的開發體驗項目）。
- 非同步工作模式（202 + `job.status_url`），沿用同一個封裝格式。
