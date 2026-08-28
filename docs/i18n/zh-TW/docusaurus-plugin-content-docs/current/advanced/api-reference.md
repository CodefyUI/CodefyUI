---
sidebar_position: 5
title: API 參考
description: CodefyUI 後端的 REST 與 WebSocket 端點——節點、預設模組、圖、外掛、LLM 代理、模型、影像與執行輸出。
---

# API 參考

後端提供一組 REST API，加上一個用於執行的 WebSocket。所有端點都位於與應用程式相同的來源下（預設為 `http://localhost:8000`）。

| 端點 | 方法 | 說明 |
|----------|--------|-------------|
| `/api/health` | GET | 健康探測——回傳 `version`、`boot_id`（回應的是哪一個行程，客戶端靠它分辨伺服器真的重啟過，還是根本沒停過）、`nodes_loaded`、`presets_loaded`，以及 `caches`（各記憶體內儲存區目前佔用的位元組與其上限，參閱[訓練記憶體](./training-memory)）。 |
| `/api/nodes` | GET | 列出所有節點定義。 |
| `/api/nodes/{node_name}` | GET | 取得單一節點定義。 |
| `/api/nodes/reload` | POST | 熱重載所有內建與自訂節點。 |
| `/api/presets` | GET | 列出預設模組定義。 |
| `/api/presets/{name}` | GET | 取得單一預設模組定義。 |
| `/api/presets/create` | POST | 從選取的節點建立新預設模組。 |
| `/api/graph/validate` | POST | 驗證一張圖。 |
| `/api/graph/save` | POST | 儲存一張圖。 |
| `/api/graph/load/{name}` | GET | 載入一張已儲存的圖。 |
| `/api/graph/list` | GET | 列出已儲存的圖。 |
| `/api/graph/export` | POST | 匯出內嵌圖表的單檔無伺服器 Python 執行器；需要相容的 CodefyUI 後端環境，但不需啟動伺服器。 |
| `/api/examples/list` | GET | 列出範例圖。 |
| `/api/examples/load` | GET | 載入一張範例圖。 |
| `/api/custom-nodes` | GET | 列出自訂節點。 |
| `/api/custom-nodes/upload` | POST | 上傳一個自訂節點。 |
| `/api/custom-nodes/toggle` | POST | 啟用/停用一個自訂節點。 |
| `/api/custom-nodes/{filename}` | DELETE | 刪除一個自訂節點。 |
| `/api/plugins` | GET | 列出已安裝的外掛包。 |
| `/api/plugins/{id}` | GET | 取得某外掛的資訊清單 (manifest) 與 README。 |
| `/api/plugins/reload` | POST | 熱重載所有節點與預設模組來源。 |
| `/api/packs` | GET | 列出每一個選用套件包，包含已安裝的部分、下載要花多少空間，以及這台機器能不能裝。 |
| `/api/packs/{id}/install` | POST | 啟動一個安裝工作——回傳 `202` 與 `job_id`。同一時間只會有一個工作在跑。 |
| `/api/packs/jobs/{job_id}/cancel` | POST | 要求執行中的工作停下來。下載會在檔案途中就中斷，不是等這個檔下完——這也會一併中斷執行中的圖在那一刻正在進行的任何 Hugging Face 下載（例如資料集或斷詞器），因為兩者共用同一個傳輸層。 |
| `/api/packs/jobs/{job_id}/events` | GET | 取得某個工作在 `?cursor=` 之後的記錄與進度事件；加上 `?wait=` 最多可長輪詢 60 秒，面板就不用靠不斷重試來跟蹤。 |
| `/api/packs/{id}/items/{item_id}` | DELETE | 刪除一個已下載的模型並釋放空間。套件包的 Python 套件無法從執行中的伺服器移除——請見 `cdui packs remove`。 |
| `/api/llm/chat` | POST | 從設定的供應商串流統一格式的 SSE 對話回應（OpenAI / OpenRouter / Anthropic / OpenAI-Codex / 自訂 OpenAI 相容端點）。 |
| `/api/llm/models` | POST | 列出某供應商可用的模型。 |
| `/api/llm/codex/login` | POST | 啟動 OpenAI-Codex（ChatGPT 帳號）OAuth 登入流程。 |
| `/api/llm/codex/status` | GET | 回報 OpenAI-Codex OAuth 登入狀態。 |
| `/api/llm/codex/logout` | POST | 清除已儲存的 OpenAI-Codex OAuth token。 |
| `/api/models` | GET | 列出已上傳的模型檔案。 |
| `/api/models/upload` | POST | 上傳一個模型權重檔。 |
| `/api/models/download/{filename}` | GET | 下載一個模型權重檔（支援巢狀路徑）。 |
| `/api/models/{filename}` | DELETE | 刪除一個模型檔案。 |
| `/api/images` | GET | 列出已上傳的影像檔案。 |
| `/api/images/upload` | POST | 上傳一個影像檔案。 |
| `/api/images/download/{filename}` | GET | 下載一個影像檔案。 |
| `/api/images/{filename}` | DELETE | 刪除一個影像檔案。 |
| `/api/execution/outputs/{run_id}` | GET | 列出某次執行所捕獲的連接埠。 |
| `/api/execution/outputs/{run_id}` | DELETE | 清除某次捕獲的執行。 |
| `/api/execution/outputs/{run_id}/{node_id}/{port}` | GET | 取得一個已捕獲的張量（支援 `?slice=0,:,:`）。 |
| `/api/execution/outputs/{run_id}/{node_id}/__steps_index` | GET | 某節點的步驟追蹤 metadata（檢視器 → Steps 分頁）。 |
| `/api/execution/outputs/{run_id}/{node_id}/__grad_index` | GET | 已捕獲的梯度 metadata（檢視器 → Backward 分頁）。 |
| `/api/execution/state/reset` | POST | 重設已保存的層權重（單節點或整張圖）。 |
| `/api/execution/state/list` | GET | 列出有多少模組被保存（診斷用）。 |
| `/ws/execution` | WebSocket | run 的訂閱視圖：`execute` 啟動一個 run，`attach` 從指定 cursor 重播事件記錄再接著即時追蹤，`detach` 取消訂閱，`cancel` 停止 run。關閉連線不會取消 run。 |

:::note WebSocket 驗證
執行 WebSocket 以查詢參數的形式取得其工作階段 token，因為瀏覽器無法在 WebSocket 交握時設定自訂標頭。前端會為你處理這件事。
:::

:::note 安裝套件包只能從本機操作
所有會造成變更的 `/api/packs` 端點，在伺服器不是綁定在回送（loopback）位址時一律拒絕：啟動安裝等於對「正在服務這個請求的那一個直譯器」執行套件管理程式，而「任何連得上這個埠的人」不是適合做這件事的對象。刻意對區網提供服務的教室或公司環境，可用 `CODEFYUI_ALLOW_REMOTE_PACK_INSTALL=1` 重新開放。不論開不開，能被要求的東西都只限於型錄中列出的項目——請求內容裡的 pip 安裝字串、repo id 或網址永遠不會進到子行程。
:::

:::note 節點清單裡的選用套件包
`/api/nodes` 會為每個節點附上 `requires_pack`（該節點執行前需要的套件包 id，沒有則為 `null`），並為每個 SELECT 參數附上 `option_packs`（選項值 → 套件包 id，用於某個選項需要特定下載的情況）。這兩項是讓編輯器能把尚未安裝的選項變灰並提供安裝；真正的把關不論如何都在後端執行時做。
:::
