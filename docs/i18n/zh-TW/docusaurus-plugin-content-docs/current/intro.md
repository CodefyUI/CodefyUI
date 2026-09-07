---
sidebar_position: 1
slug: /
title: 簡介
description: 視覺化、節點式的深度學習管線建構工具。在瀏覽器中設計 CNN、RNN、Transformer 與 RL 架構並即時執行。
---

# CodefyUI

**視覺化、節點式的深度學習管線建構工具。** 透過拖曳節點到畫布、連接成 DAG 並執行管線，直接在瀏覽器中設計 CNN、RNN、Transformer 與 RL 架構。

![CodefyUI 截圖](/img/ui-screenshot-zh-TW.png)

## 你可以做什麼

- **以視覺方式建構模型。** 拖放節點、以型別安全的連線接上連接埠，並即時取得驗證結果。CodefyUI 內建 **152 個節點**，分為 16 個類別，包括 CNN、RNN、Transformer、RL、資料、訓練、LLM、Diffusion 與 Classical。
- **檢視張量。** **教學檢視器**會錄製每個節點的輸出。你可以逐格比較輸入與輸出、擷取梯度，並使用段落只比較子圖起點的輸入與終點的輸出。
- **監看執行。** WebSocket 串流會在執行期間回報每個節點的進度、即時訓練 loss 圖表與 `Print` 輸出。**執行任務**面板會追蹤排隊中、執行中與已完成的執行。請參閱[執行佇列](/usage/run-queue)。
- **擴充節點系統。** 將選取的節點收合成可重用的[子圖](/advanced/subgraphs)、將 graph 儲存為可重用的預設組合，或從 `.py` 檔加入自訂節點。你可以從套件中心、[外掛中心](/advanced/plugins#plugin-center)或 CLI 安裝選用套件與外掛包。
- **選擇裝置後端。** 可使用 CPU、NVIDIA CUDA、Apple Silicon MPS 或 AMD ROCm 執行，並在安裝時及每次執行時選擇後端。

## 快速開始

只需安裝執行應用程式所需的工具（`git`、`uv` 與 Python）——**一般使用者不需要 Node.js**：

```bash
# macOS / Linux
curl -fsSL https://raw.githubusercontent.com/CodefyUI/CodefyUI/main/install.sh | bash
```

```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://raw.githubusercontent.com/CodefyUI/CodefyUI/main/install.ps1 | iex"
```

接著開啟新的終端機並執行：

```bash
cdui start
```

開啟 [http://localhost:8000](http://localhost:8000)——單一 uvicorn 行程會同時提供 API 與預先建置的 React 應用程式。

→ 完整說明請見 **[安裝](/getting-started/installation)**。

## 接下來

| 你想要… | 從這裡開始 |
|---------|-----------|
| 安裝並啟動應用程式 | [開始使用 → 安裝](/getting-started/installation) |
| 選對 GPU / CUDA / MPS 版本 | [GPU 與裝置設定](/getting-started/gpu-device) |
| 建立並執行你的第一個圖 | [使用方式 → 你的第一個圖](/usage/first-graph) |
| 邊學邊檢視張量與梯度 | [教學檢視器](/usage/teaching-inspector) |
| 瀏覽所有內建節點 | [節點參考](/usage/node-reference) |
| 撰寫自訂節點或外掛 | [進階 → 自訂節點](/advanced/custom-nodes) · [外掛](/advanced/plugins) |
| 了解執行機制 | [架構](/advanced/architecture) |

## 架構一覽

```
frontend/   React 19 · TypeScript · React Flow 12 · Zustand 5 · Vite 6
backend/    Python 3.10+ · FastAPI · PyTorch
```

CodefyUI **以後端為準**：`GET /api/nodes` 回傳所有節點定義，並由單一 React 元件根據這些定義渲染所有節點類型。在後端新增節點後，該節點會自動出現在 UI 中。完整說明請見[架構](/advanced/architecture)。

## 授權

CodefyUI 採用雙軌授權模式：

- **開源路徑** — [AGPL-3.0-only](https://github.com/CodefyUI/CodefyUI/blob/main/LICENSE)，適用於個人開發者、小型團隊、教育、研究、社群使用，**以及任何能遵守 AGPL-3.0 的其他使用情境**。
- **商業路徑** — 若專有、閉源、SaaS、OEM 或企業用途需要 AGPL-3.0 以外的條款，請[聯絡維護者](https://github.com/CodefyUI/CodefyUI/issues)。

**未經修改直接執行 CodefyUI——包括部署於公司內部伺服器——是 AGPL-3.0 明文允許的，不需要購買授權。** 第 13 條的「提供原始碼」義務，前提是你*修改了*本程式。[授權常見問題](/licensing)會說明其實務意義，包括自訂節點與外掛的處理方式，以及商業授權的實際涵蓋範圍。

Copyright (C) 2026 CodefyUI 及 CodefyUI 貢獻者。貢獻以 Developer Certificate of Origin 1.1 為準——請見 [CONTRIBUTING.md](https://github.com/CodefyUI/CodefyUI/blob/main/CONTRIBUTING.md)。
