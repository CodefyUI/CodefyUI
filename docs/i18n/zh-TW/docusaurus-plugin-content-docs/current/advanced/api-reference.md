---
sidebar_position: 5
title: API 參考
description: CodefyUI 後端的 REST 與 WebSocket 端點——節點、預設模組、圖、外掛、模型、影像與執行輸出。
---

# API 參考

後端提供一組 REST API，加上一個用於執行的 WebSocket。所有端點都位於與應用程式相同的來源下（預設為 `http://localhost:8000`）。

| 端點 | 方法 | 說明 |
|----------|--------|-------------|
| `/api/health` | GET | 健康探測——回傳 `nodes_loaded`、`presets_loaded`。 |
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
| `/api/graph/export` | POST | 將一張圖匯出為 Python 腳本。 |
| `/api/examples/list` | GET | 列出範例圖。 |
| `/api/examples/load` | GET | 載入一張範例圖。 |
| `/api/custom-nodes` | GET | 列出自訂節點。 |
| `/api/custom-nodes/upload` | POST | 上傳一個自訂節點。 |
| `/api/custom-nodes/toggle` | POST | 啟用/停用一個自訂節點。 |
| `/api/custom-nodes/{filename}` | DELETE | 刪除一個自訂節點。 |
| `/api/plugins` | GET | 列出已安裝的外掛包。 |
| `/api/plugins/{id}` | GET | 取得某外掛的資訊清單 (manifest) 與 README。 |
| `/api/plugins/reload` | POST | 熱重載所有節點與預設模組來源。 |
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
| `/ws/execution` | WebSocket | 即時圖執行（接受 `run_id`、`record_outputs`）。 |

:::note WebSocket 驗證
執行 WebSocket 以查詢參數的形式取得其工作階段 token，因為瀏覽器無法在 WebSocket 交握時設定自訂標頭。前端會為你處理這件事。
:::
