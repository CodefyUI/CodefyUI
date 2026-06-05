---
sidebar_position: 6
title: 架構
description: CodefyUI 如何組合而成——後端權威的節點定義、WebSocket 執行、拓撲排程，以及註冊表/外掛系統。
---

# 架構

```
frontend/   React 19 · TypeScript · React Flow 12 · Zustand 5 · Vite 6
backend/    Python 3.10+ · FastAPI · PyTorch
```

單一 uvicorn 程序同時提供 REST API、執行 WebSocket，以及預先建置好的 React 應用程式。

## 核心原則

| 原則 | 說明 |
|-----------|--------|
| **後端權威** | `GET /api/nodes` 回傳每一個節點定義。新增一個後端節點即可讓它自動出現在 UI 中——無需修改前端。 |
| **單一 BaseNode 元件** | 一個 React 元件渲染所有節點類型，由後端定義參數化。 |
| **WebSocket 執行** | `ws://host/ws/execution` 串流每個節點的狀態；REST 處理圖的 CRUD 與輸出抓取。 |
| **拓撲排序執行** | 使用 Kahn 演算法進行 DAG 排序 + 循環偵測，並對獨立節點進行平行執行。 |

## 執行流程

1. **預設模組展開**——預設模組節點會在任何東西執行之前被攤平成其內部節點。
2. **驗證**——DAG 檢查、連接埠/型別安全，以及一個必要的 [`Start`](/usage/first-graph) 節點。只有能透過 trigger 邊抵達的節點才會執行。
3. **拓撲排序**——使用 Kahn 演算法並進行循環偵測。
4. **平行執行**——獨立節點並行執行。
5. **快取 / 髒節點追蹤**——具決定性的節點輸出會以節點類型、參數與上游輸出為鍵進行快取；變更一個節點會把它與其下游標記為髒，因此只有受影響的子圖會重新執行。非決定性節點（或 `cacheable = False`）總是會執行。
6. **裝置解析**——被請求的裝置會與可用的裝置比對，若不存在則退回 CPU 並發出警告。請參閱 [裝置後端](./device-backends)。

## 狀態、輸出與梯度

- **執行環境**攜帶每次執行的選項：裝置、詳細模式、權重保存與梯度目標。
- **有狀態模組**——一個 mixin 透過以（圖 id、節點 id、結構雜湊）為鍵的鍵值儲存，在多次執行之間保存 `nn.Module` 權重，因此在開啟*保留權重 (Persist weights)* 時，模型能在多次 **Run** 點擊之間持續學習。
- **執行輸出儲存**——一個 LRU 快取（最近約 20 次執行）保存供 [Teaching Inspector](/usage/teaching-inspector) 使用的已捕獲輸出，並透過 REST 依需求抓取。
- **反向傳播**——當*捕獲梯度 (Capture gradients)* 開啟時，引擎會掛上 hook、呼叫 `.backward()`，並把每層的梯度與輸出一起儲存。
- **步驟追蹤**——在詳細模式下，已植入儀器的節點會發出一個 `__steps__` 追蹤，記錄供檢視器的 **Steps** 分頁使用。

## 節點註冊表與可擴充性

- **註冊表**透過走訪節點套件來探索 `BaseNode` 子類別。內建節點使用裸名稱（`Conv2d`）；外掛節點則加上命名空間（`foundations:Edu-KNN`），以避免衝突並讓圖能自我說明。
- **[自訂節點](./custom-nodes)**——把一個 `.py` 檔案放進 `custom_nodes/` 並熱重載。
- **[外掛包](./plugins)**——透過 CLI 安裝、藉由 lockfile 探索，並在載入第三方程式碼之前經過 **AST 驗證**。
- **[預設模組](./presets)**——可重用的子圖，於執行時展開。

## 進入點

| 區域 | 檔案 |
|------|------|
| FastAPI 應用程式、lifespan、路由 | `backend/app/main.py` |
| BaseNode ABC | `backend/app/core/node_base.py` |
| 節點註冊表 + 命名空間 | `backend/app/core/node_registry.py` |
| 圖驗證 + 執行 | `backend/app/core/graph_engine.py` |
| WebSocket 處理器 | `backend/app/api/ws_execution.py` |
| 外掛探索 + AST 閘門 | `backend/app/core/plugin_loader.py` |
| CLI 圖執行器 | `backend/run_graph.py` |
| 前端根 | `frontend/src/App.tsx` |
| WebSocket 用戶端 | `frontend/src/api/ws.ts` |

:::tip 貢獻
後端權威的設計意味著大多數「新增一項功能」的工作都是單一個 Python 節點。請參閱 [自訂節點](./custom-nodes) 來上手，接著進階到 [外掛包](./plugins) 來分享它。
:::
