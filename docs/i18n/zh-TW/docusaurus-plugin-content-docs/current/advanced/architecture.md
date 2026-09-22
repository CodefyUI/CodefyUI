---
sidebar_position: 6
title: 架構
description: CodefyUI 如何組合而成——後端權威的節點定義、WebSocket 執行、拓撲排程，以及註冊表與外掛系統。
---

# 架構

```
frontend/   React 19 · TypeScript · React Flow 12 · Zustand 5 · Vite 6
backend/    Python 3.10+ · FastAPI · PyTorch
```

單一 uvicorn 行程同時提供 REST API、執行 WebSocket，以及預先建置好的 React 應用程式。

## 核心原則

| 原則 | 說明 |
|-----------|--------|
| **後端權威** | `GET /api/nodes` 回傳每一個節點定義。新增一個後端節點即可讓它自動出現在 UI 中——無需修改前端。 |
| **預設節點渲染元件** | 一個元件（`BaseNode`）依後端定義渲染任何節點，因此新增後端節點不需要任何前端程式碼。前端允許清單（`VIZ_NODE_TYPES`）讓九個教學節點改用建立在同一張卡片上的進階渲染元件；外掛節點可以透過[外掛前端 API](./plugin-frontend-extensions)（`api.nodes.registerRenderer`）自行繪製卡片本體；預設組合、子圖實例、`Start` 與註記則各有自己的元件。 |
| **WebSocket 執行** | `ws://host/ws/execution` 是伺服器所擁有之 run 的*視圖*：即時串流每個節點的狀態，也能從任意 cursor 重播該 run 已儲存的事件紀錄，讓重新連線的分頁完整接上。run 本身由 run service 擁有，而不是 socket。 |
| **拓撲排序執行** | 使用 Kahn 演算法進行 DAG 排序 + 循環偵測，並對獨立節點進行平行執行。 |

## 執行流程

1. **送出**——畫布、`cdui run`、`POST /api/runs` 或 sweep 會將圖送至 run service。該服務會將圖持久化並排程。queued 通道的 run 會進入各裝置的 FIFO；interactive 畫布 run 則會跳過佇列。詳見[執行佇列](/usage/run-queue)。
2. **展開**——在任何東西執行之前：丟棄註記；內聯展開子圖實例（最多 10 層；包含自己的方塊會被拒絕），其內部節點以 `<instance>/<node>` 的 id 執行；把預設組合節點攤平成其內部節點（最多 10 層）；移除被略過的節點，並把它的每個輸出改由相符的輸入轉送。請參閱[子圖](./subgraphs)。
3. **驗證**——檢查 DAG、連接埠／型別安全，以及是否有必要的 [`Start`](/usage/first-graph) 節點。節點只要能沿著 trigger 邊抵達，或透過 data 邊餵給一個能直接或間接如此抵達的節點，就會執行；data 邊接的是必要還是選用連接埠都沒有差別。像 `Dataset` 或轉換鏈開頭這類自身沒有 trigger 的根節點，只要其輸出會被執行中的節點使用，就會保留而不被剪掉（core#201）。
4. **拓撲排序**——使用 Kahn 演算法並進行循環偵測。
5. **平行執行**——獨立節點並行執行，一次最多 `CODEFYUI_MAX_PARALLEL_NODES` 個。設定了亂數種子的 run 一次只執行一個節點，因為每個節點的種子設定會改動整個行程共用的亂數產生器。
6. **快取／髒節點追蹤**——具決定性的節點輸出會依 WebSocket 連線個別快取（預設 256 筆與 1 GB），快取鍵包含節點類型、params、每條輸入邊的一份參照（上游鍵加上兩端的連接埠名稱）、解析後的裝置，以及讀檔節點的內容指紋；變更一個節點會把它與其下游標記為 dirty，因此只有受影響的子圖會重新執行。非決定性節點（或 `cacheable = False`）一律執行。
7. **裝置解析**——沒有指定裝置時為 `cpu`，`auto` 代表最佳的加速器（依序為 `cuda`、`mps`、`cpu`）。無法使用的 `cuda` 或 `mps` 會退回 CPU 並發出警告；在有 CUDA 的機器上，超出範圍或格式錯誤的 `cuda:N` 會改用目前的 CUDA 裝置。節點自己的 `device` 參數不是 `auto` 時，會覆寫 run 的裝置。請參閱[裝置後端](./device-backends)。

## 狀態、輸出與梯度

- **Run service**——`RunService` 會獨立於 WebSocket 連線管理每個 run。它會將每個引擎事件附加至持久事件紀錄、批次寫入純量指標，並透過執行環境協作式取消。啟動時，它會將上一個行程留下的 `queued` 或 `running` 資料列標記為 `interrupted`。run、事件、指標與產出會儲存在 SQLite（`exec_runs`、`exec_run_events`、`exec_run_metrics`、`exec_run_artifacts`），讓分頁可以重新連線，終端機也可以監控由其他用戶端啟動的 run。
- **執行環境**攜帶每次執行的選項：裝置、亂數種子與決定性旗標、「顯示內部步驟」模式、權重持久化、反向傳播與梯度設定，以及協作式停止旗標。
- **有狀態模組**——一個 mixin 透過以（graph id、node id、structure hash）為鍵的鍵值儲存區，在多次執行之間保留 `nn.Module` 權重，因此開啟*在多次執行間保留權重*時，模型能在多次**執行**點擊之間持續學習。
- **Run 輸出儲存區**——全伺服器共用的記憶體儲存區，會保留[教學檢視器](/usage/teaching-inspector)使用的擷取輸出，並透過 REST 依需求提供。預設最多追蹤 20 次 run 與 2 GiB；超過任一限制時，會逐出完整且最舊的 run。
- **反向傳播**——開啟*擷取梯度*時，引擎會掛上 hook、呼叫 `.backward()`，並把每層的梯度與輸出一起儲存。
- **步驟追蹤**——在「顯示內部步驟」模式下，受插樁的節點會發出 `__steps__` 追蹤，記錄供檢視器的**步驟**分頁使用。

## 節點註冊表與可擴充性

- **註冊表**透過走訪節點套件來探索 `BaseNode` 子類別。內建節點使用裸名稱（`Conv2d`）；外掛節點則加上命名空間（`foundations:Edu-KNN`），以避免衝突並讓圖能自我說明。
- **[自訂節點](./custom-nodes)**——把一個 `.py` 檔案放進 `custom_nodes/`，即可熱重載。
- **[外掛包](./plugins)**——從編輯器的**外掛中心**或以 `cdui plugin install` 安裝（兩者共用 `backend/app/core/plugins/` 這一套安裝實作）、藉由 lockfile 探索，並在載入第三方程式碼之前經過 **AST 驗證**。
- **[預設組合](./presets)**——可重用的子圖，於執行時展開。

## 進入點

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
| WebSocket 客戶端 | `frontend/src/api/ws.ts` |

:::tip 貢獻
後端權威的設計意味著，大多數「新增一項功能」只需要新增一個 Python 節點。請先參閱[自訂節點](./custom-nodes)，接著再進階到[外掛包](./plugins)來分享它。
:::
