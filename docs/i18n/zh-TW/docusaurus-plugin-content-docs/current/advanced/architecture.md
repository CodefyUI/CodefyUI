---
sidebar_position: 6
title: 架構
description: CodefyUI 如何組合而成——後端權威的節點定義、WebSocket 執行、拓撲排程、前端的 store 與執行事件，以及註冊表與外掛系統。
---

# 架構

```
frontend/   React 19 · TypeScript · React Flow 12 · Zustand 5 · Vite 6
backend/    Python 3.10+ · FastAPI · PyTorch
```

單一 uvicorn 行程同時提供 REST API、執行 WebSocket，以及預先建置好的 React 應用程式。

## 核心原則 {/* #core-principles */}

| 原則 | 說明 |
|-----------|--------|
| **後端權威** | `GET /api/nodes` 回傳每一個節點定義。新增一個後端節點即可讓它自動出現在 UI 中——無需修改前端。 |
| **預設節點渲染元件** | 一個元件（`BaseNode`）依後端定義渲染任何節點，因此新增後端節點不需要任何前端程式碼。前端允許清單（`VIZ_NODE_TYPES`）讓九個教學節點改用建立在同一張卡片上的進階渲染元件；外掛節點可以透過[外掛前端 API](./plugin-frontend-extensions)（`api.nodes.registerRenderer`）自行繪製卡片本體；預設組合、子圖實例、`Start` 與註記則各有自己的元件。 |
| **WebSocket 執行** | `ws://host/ws/execution` 是伺服器所擁有之 run 的*視圖*：即時串流每個節點的狀態，也能從任意 cursor 重播該 run 已儲存的事件紀錄，讓重新連線的分頁完整接上。run 本身由 run service 擁有，而不是 socket。 |
| **拓撲排序執行** | 使用 Kahn 演算法進行 DAG 排序 + 循環偵測，並對獨立節點進行平行執行。 |

## 執行流程 {/* #execution-flow */}

1. **送出**——畫布、`cdui run`、`POST /api/runs` 或 sweep 會將圖送至 run service。該服務會將圖持久化並排程。queued 通道的 run 會進入各裝置的 FIFO；interactive 畫布 run 則會跳過佇列。詳見[執行佇列](/usage/run-queue)。
2. **展開**——在任何東西執行之前：丟棄註記；內聯展開子圖實例（最多 10 層；包含自己的方塊會被拒絕），其內部節點以 `<instance>/<node>` 的 id 執行；把預設組合節點攤平成其內部節點（最多 10 層）；移除被略過的節點，並把它的每個輸出改由相符的輸入轉送。請參閱[子圖](./subgraphs)。
3. **驗證**——檢查 DAG、連接埠／型別安全，以及是否有必要的 [`Start`](/usage/first-graph) 節點。節點只要能沿著 trigger 邊抵達，或透過 data 邊餵給一個能直接或間接如此抵達的節點，就會執行；data 邊接的是必要還是選用連接埠都沒有差別。像 `Dataset` 或轉換鏈開頭這類自身沒有 trigger 的根節點，只要其輸出會被執行中的節點使用，就會保留而不被剪掉（core#201）。
4. **拓撲排序**——使用 Kahn 演算法並進行循環偵測。
5. **平行執行**——獨立節點並行執行，一次最多 `CODEFYUI_MAX_PARALLEL_NODES` 個。設定了亂數種子的 run 一次只執行一個節點，因為每個節點的種子設定會改動整個行程共用的亂數產生器。
6. **快取／髒節點追蹤**——具決定性的節點輸出會依 WebSocket 連線個別快取（預設 256 筆與 1 GB），快取鍵包含節點類型、params、每條輸入邊的一份參照（上游鍵加上兩端的連接埠名稱）、解析後的裝置，以及讀檔節點的內容指紋；變更一個節點會把它與其下游標記為 dirty，因此只有受影響的子圖會重新執行。非決定性節點（或 `cacheable = False`）一律執行。
7. **裝置解析**——沒有指定裝置時為 `cpu`，`auto` 代表最佳的加速器（依序為 `cuda`、`mps`、`cpu`）。無法使用的 `cuda` 或 `mps` 會退回 CPU 並發出警告；在有 CUDA 的機器上，超出範圍或格式錯誤的 `cuda:N` 會改用目前的 CUDA 裝置。節點自己的 `device` 參數不是 `auto` 時，會覆寫 run 的裝置。請參閱[裝置後端](./device-backends)。

## 狀態、輸出與梯度 {/* #state-outputs-and-gradients */}

- **Run service**——`RunService` 會獨立於 WebSocket 連線管理每個 run。它會將每個引擎事件附加至持久事件紀錄、批次寫入純量指標，並透過執行環境協作式取消。啟動時，它會將上一個行程留下的 `queued` 或 `running` 資料列標記為 `interrupted`。run、事件、指標與產出會儲存在 SQLite（`exec_runs`、`exec_run_events`、`exec_run_metrics`、`exec_run_artifacts`），讓分頁可以重新連線，終端機也可以監控由其他用戶端啟動的 run。
- **執行環境**攜帶每次執行的選項：裝置、亂數種子與決定性旗標、「顯示內部步驟」模式、權重持久化、反向傳播與梯度設定，以及協作式停止旗標。
- **有狀態模組**——一個 mixin 透過以（graph id、node id、structure hash）為鍵的鍵值儲存區，在多次執行之間保留 `nn.Module` 權重，因此開啟*在多次執行間保留權重*時，模型能在多次**執行**點擊之間持續學習。
- **Run 輸出儲存區**——全伺服器共用的記憶體儲存區，會保留[教學檢視器](/usage/teaching-inspector)使用的擷取輸出，並透過 REST 依需求提供。預設最多追蹤 20 次 run 與 2 GiB；超過任一限制時，會逐出完整且最舊的 run。
- **反向傳播**——開啟*擷取梯度*時，引擎會掛上 hook、呼叫 `.backward()`，並把每層的梯度與輸出一起儲存。
- **步驟追蹤**——在「顯示內部步驟」模式下，受插樁的節點會發出 `__steps__` 追蹤，記錄供檢視器的**步驟**分頁使用。

## 節點註冊表與可擴充性 {/* #node-registry--extensibility */}

- **註冊表**透過走訪節點套件來探索 `BaseNode` 子類別。內建節點使用裸名稱（`Conv2d`）；外掛節點則加上命名空間（`foundations:Edu-KNN`），以避免衝突並讓圖能自我說明。
- **[自訂節點](./custom-nodes)**——把一個 `.py` 檔案放進 `custom_nodes/`，即可熱重載。
- **[外掛包](./plugins)**——加上命名空間的節點包，也可以附帶預設組合、範例與前端；請參閱下方的[外掛](#plugins)。
- **[預設組合](./presets)**——可重用的子圖，於執行時展開。

## 前端資料流 {/* #frontend-data-flow */}

編輯器把狀態放在 `frontend/src/store/` 底下的 Zustand store，並以兩種方式和伺服器溝通：大多由 `frontend/src/api/` 裡的模組發出的 REST 呼叫（`rest.ts`，以及版本控制用的 `git.ts` 與擷取輸出用的 `executionOutputs.ts`），以及每個畫布分頁各一條執行 WebSocket。會變更資料的 REST 呼叫都經過 `api/_auth.ts` 的 `apiFetch`，它會加上 `GET /api/auth/bootstrap` 發給的 session token；socket 則把同一個 token 放在網址裡（[驗證](./api-reference#authentication)）。

| Store | 內容 | 資料來源 |
|-------|------|----------|
| `nodeDefStore` | 節點與預設組合的定義 | `GET /api/nodes` 與 `GET /api/presets`：啟動時載入，之後每當型錄改變就再載入一次，包括工具列的**重新載入節點**、自訂節點管理、**匯出為子圖**，以及在外掛中心變更外掛 |
| `tabStore` | 每個開啟的分頁：它的圖、復原歷史、dirty 節點、執行狀態、執行紀錄、輸出摘要、`lastRunId` 與 socket | 啟動時從 IndexedDB 還原，之後是畫布與各面板上的編輯，以及執行事件 |
| `runStore` | 執行任務面板：run 清單與選取的 run | `GET /api/runs` 與 `GET /api/runs/{id}/events` |

其他 store 各自負責編輯器的一部分，例如 `pluginStore` 對應外掛中心，`packStore` 對應套件中心，`gitStore` 對應版本控制，`uiStore` 則保存版面與編輯器設定，包括在設定中選擇的裝置。

### 從編輯到執行 {/* #from-an-edit-to-a-run */}

1. **編輯**——從側邊欄拖放節點時，會依 `nodeDefStore` 裡的定義建立節點，並把定義與預設參數複製進節點的資料；節點卡片與設定面板都依這份資料繪製。`FlowCanvas` 以 React Flow 繪製目前的分頁，並把每一次移動、刪除與新連線透過 `tabStore` 的 `onNodesChange`、`onEdgesChange` 與 `onConnect` 寫回；設定面板則透過 `updateNodeParams` 寫入參數。修改參數會把該節點標記為 dirty，新連線則會標記它的目標節點。
2. **自動儲存**——`tabStore` 會在最後一次變更的 250 ms 後，把變更存進 IndexedDB，每個分頁一筆資料（[自動儲存](/usage/tabs-persistence#automatic-saving)）。這份資料只存在瀏覽器裡；把圖存到伺服器是 `POST /api/graph/save`。
3. **執行**——工具列的**執行**會呼叫 `useGraphExecution`。它先檢查畫面上的畫布有沒有進入點，再把分頁的整張圖序列化（略過註記），交給 `POST /api/graph/validate` 檢查；任一步被拒絕時都會顯示 toast，不送出任何東西。接著它會清空該分頁的紀錄、節點狀態與輸出摘要，把 dirty 節點及其所有下游節點整理成 `changed_nodes`，然後在分頁的 socket 上送出 `execute` 訊息，附上圖、裝置（圖自己指定的裝置，沒有時用設定中選擇的裝置），以及分頁的執行設定，例如亂數種子與是否錄製節點輸出。
4. **送出**——`ws_execution.py` 會把圖交給 run service 的 interactive 通道，連同這個 socket 的執行快取與 `changed_nodes`，並讓 socket 接上新的 run。從這裡開始，run 由伺服器負責；請參閱[執行流程](#execution-flow)。

### 執行事件如何回到畫布 {/* #from-run-events-back-to-the-canvas */}

1. **轉送**——run service 會把每一個引擎事件附加到該 run 的事件紀錄，再交給接在該 run 上的 socket，由 socket 以 `{type, ...payload, run_id, cursor}` 的形式送出。[WebSocket 協定](./api-reference#websocket-protocol)列出了所有事件類型。
2. **每個分頁**——`useGraphExecution` 會監聽每個分頁的 socket，因此在背景的分頁也會持續收到自己那個 run 的事件。
   - `execution_start` 會把分頁切換成執行中，並把 run 的 id 記為 `lastRunId`。
   - `node_status` 訊息會交給 `nodeUpdateQueue`，由它每個動畫影格一次套用到 `tabStore`；因此即使 run 每秒回報很多次，節點卡片每次繪製最多也只重新渲染一次。同一批訊息中的文字、圖片、影片與圖表會各自成為執行紀錄中的一行，張量摘要則是執行後點擊連線時顯示的內容。
   - `execution_complete`、`execution_error` 與 `execution_stopped` 會設定分頁的最終狀態。
3. **重新連線**——每則訊息的 `run_id` 與 `cursor` 都會記在分頁上。連線中斷後，socket 會重新連線，並從最後的 cursor 重新接上 run。重新載入頁面後，原本在執行的分頁會以 `GET /api/runs/{id}` 詢問它的 run 是否仍在進行，若是就從頭重新接上。關閉分頁只會斷開連線，run 會繼續執行；工具列的**停止**會帶著 run 的 id 送出 `cancel`。

### 其他讀取 run 的地方 {/* #other-readers-of-a-run */}

- [檢視器](/usage/teaching-inspector#the-inspector-panel)會依分頁的 `lastRunId`，從 `/api/execution/outputs/` 取得節點擷取的輸出。
- [執行任務面板](/usage/run-queue#runs-panel)不使用 socket：`runStore` 會輪詢 `GET /api/runs`，並對選取的 run 長輪詢 `GET /api/runs/{id}/events`。它的**觀看**按鈕會讓目前分頁的 socket 接上該 run，並把它重播到分頁的執行紀錄。
- 外掛透過 [`api.events`](./plugin-frontend-extensions#apievents--live-run-events) 收到這些事件，名稱已換成一套穩定的詞彙。

## 外掛 {/* #plugins */}

外掛包是一個目錄，裡面有一份 `cdui.plugin.toml` manifest，內容則放在名稱固定的子目錄：`nodes/`、`presets/`、`examples/`、`assets/`，以及有自己 UI 的外掛包才會有的 `frontend/`。參考文件是[外掛包](./plugins)與[外掛前端擴充](./plugin-frontend-extensions)；本節說明一個外掛包在程式碼中經過的路徑。

1. **安裝**——外掛中心與 `cdui plugin install` 執行同一套安裝流程，位於 `backend/app/core/plugins/`。對來自 GitHub 的外掛包，它會在一個固定的 commit 讀取 manifest、下載並解開該 commit，用 AST 閘門掃描所有 Python 能匯入的檔案，接著安裝外掛包的 Python 相依套件、複製檔案，並記錄到 lockfile `installed.json`。`plugins/` 底下的內建包，以及用 `cdui plugin link` 連結的資料夾，都不會被掃描。請參閱[安裝如何進行](./plugins#how-an-install-runs)與[安全性](./plugins#security--three-tiers)。
2. **載入**——啟動時與每次重載時，`plugin_loader.py` 會讀取 lockfile，把每個已啟用的外掛包公開為 Python 套件 `cdui_plugins.<id>`（id 轉為 snake_case）。註冊表以走訪內建節點的同一種方式走訪外掛包的 `nodes/`，並把每個節點註冊為 `<id>:<NODE_NAME>`；外掛包的預設組合也以相同方式載入。這一步不會再掃描：閘門只在安裝時執行。
3. **編輯器中的節點**——外掛節點與其他節點一起由 `GET /api/nodes` 回傳，因此列出、設定與執行它們都不需要外掛專屬的程式碼。在畫布上，加上命名空間但沒有內建卡片（`VIZ_NODE_TYPES`，請參閱[核心原則](#core-principles)）的節點由 `PluginNodeBridge` 繪製，它會使用外掛為該類型註冊的卡片本體，沒有註冊時則使用預設的卡片本體。
4. **前端**——`GET /api/plugins` 會列出每個已啟用、且宣告了前端的外掛包的 `frontend_entry`，也就是由 `/plugins/<id>/frontend/` 提供的模組。`PluginHost` 會先等待節點型錄載入，再匯入每個模組，並以 `plugins/api.ts` 建立的物件呼叫它的預設匯出；這個物件提供面板、工具列按鈕、節點 renderer、經由 `tabStore` 的圖讀寫、執行事件與執行歷史。公開的型別合約是 `plugins/contract.ts`：宿主與它不一致時 `tsc -b` 會失敗，而 `scripts/sync_plugin_sdk.py` 會把它複製到 `cdui plugin new` 的範本裡（這份複本過期時會有後端測試失敗）。在外掛中心安裝、更新、移除、啟用或停用外掛之後，編輯器會重新載入節點型錄，並重新啟用所有外掛前端（[activate 合約](./plugin-frontend-extensions#the-activate-contract)）。

## 進入點 {/* #entry-points */}

| 區域 | 檔案 |
|------|------|
| FastAPI 應用程式、lifespan、路由 | `backend/app/main.py` |
| BaseNode ABC | `backend/app/core/node_base.py` |
| 節點註冊表 + 命名空間 | `backend/app/core/node_registry.py` |
| 圖驗證 + 執行 | `backend/app/core/graph_engine.py` |
| Run service（排程、事件紀錄、取消、復原） | `backend/app/core/run_service.py` |
| Run 儲存區（SQLite 資料列、保留政策） | `backend/app/core/run_store.py` |
| Run REST 路由 | `backend/app/api/routes_runs.py` |
| 參數掃描 | `backend/app/api/routes_sweeps.py`, `backend/app/core/sweep_compiler.py` |
| WebSocket 處理器 | `backend/app/api/ws_execution.py` |
| 外掛探索 | `backend/app/core/plugin_loader.py` |
| 外掛 AST 閘門 | `backend/app/core/plugins/gate.py`, `backend/app/core/plugin_validator.py` |
| 外掛安裝服務（外掛中心與 `cdui plugin`） | `backend/app/core/plugins/`, `backend/app/api/routes_plugins.py` |
| 套件中心（選用套件包） | `backend/app/core/packs/`, `backend/app/api/routes_packs.py` |
| 安裝 job 執行器（套件包與外掛同時只執行一個 job） | `backend/app/core/jobs.py` |
| 版本控制（git） | `backend/app/core/git/`, `backend/app/api/routes_git.py` |
| 已發佈的應用程式與 API key | `backend/app/api/routes_apps.py`, `backend/app/api/routes_keys.py`, `backend/app/core/api_keys.py` |
| SQLite 資料庫（應用程式、key、run） | `backend/app/core/db.py` |
| 把 graph 當成函式呼叫 | `backend/app/api/routes_graph_run.py`, `backend/app/core/api_contract.py` |
| LLM provider 代理 | `backend/app/core/llm_proxy/`, `backend/app/api/routes_llm.py` |
| 專案目錄 | `backend/app/core/project.py` |
| Host 允許清單、session token、body 上限 | `backend/app/core/auth.py`, `backend/app/core/body_limit.py` |
| Python 匯出 | `backend/app/core/codegen.py` |
| CLI 圖形執行器 | `backend/run_graph.py` |
| 前端根元件 | `frontend/src/App.tsx` |
| 分頁與圖的狀態（復原、dirty 節點、自動儲存） | `frontend/src/store/tabStore.ts`, `frontend/src/store/tabPersistence.ts` |
| 畫布 | `frontend/src/components/Canvas/FlowCanvas.tsx` |
| REST 客戶端與 session token | `frontend/src/api/rest.ts`, `frontend/src/api/_auth.ts` |
| WebSocket 客戶端 | `frontend/src/api/ws.ts` |
| 執行、停止與畫布上的執行事件 | `frontend/src/hooks/useGraphExecution.ts`, `frontend/src/store/nodeUpdateQueue.ts` |
| 外掛前端宿主與合約 | `frontend/src/plugins/PluginHost.tsx`, `frontend/src/plugins/api.ts`, `frontend/src/plugins/contract.ts` |

:::tip 貢獻
後端權威的設計意味著，大多數「新增一項功能」只需要新增一個 Python 節點。請先參閱[自訂節點](./custom-nodes)，接著再進階到[外掛包](./plugins)來分享它。
:::
