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

- **視覺化建構模型** — 拖放節點、用型別安全的連線連接連接埠、即時驗證。**94 個內建節點**，涵蓋 15 大類別（CNN、RNN、Transformer、RL、資料、訓練、LLM、Diffusion、傳統機器學習等）。
- **看著張量流動** — **教學檢視器** 會記錄每個節點的輸出，讓你逐格檢視輸入→輸出的差異、捕獲梯度，並用段落比較只看一段子圖的頭部輸入與尾部輸出。
- **即時執行** — WebSocket 串流回報每個節點的進度、即時訓練 loss 圖表，以及執行時的 `Print` 輸出。
- **可擴充** — 把子圖存成可重用的 **預設模組**、放入 **自訂節點**（`.py` 檔），或安裝教學節點的 **外掛包**。
- **支援各種後端** — CPU、NVIDIA CUDA、Apple Silicon（MPS）或 AMD ROCm，可在安裝時與每次執行時選擇。

## 快速開始

只安裝執行所需的東西（`git`、`uv` 與 Python）——**一般使用者不需要 Node.js**：

```bash
# macOS / Linux
curl -fsSL https://raw.githubusercontent.com/treeleaves30760/CodefyUI/main/install.sh | bash
```

```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://raw.githubusercontent.com/treeleaves30760/CodefyUI/main/install.ps1 | iex"
```

接著重新開啟一個 terminal 並執行：

```bash
cdui start
```

開啟 [http://localhost:8000](http://localhost:8000)——單一 uvicorn 同時處理 API 與預編好的 React 前端。

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

CodefyUI 採 **後端權威** 設計：`GET /api/nodes` 回傳所有節點定義，並由單一 React 元件依這些定義渲染所有節點類型。在後端新增節點後，UI 會自動出現——完整說明請見 [架構](/advanced/architecture)。

## 授權

CodefyUI 採用雙軌授權模式：

- **開源路徑** — [AGPL-3.0-only](https://github.com/treeleaves30760/CodefyUI/blob/main/LICENSE)，適用於個人開發者、小型團隊、教育、研究與社群使用。
- **商業路徑** — 若需要閉源、SaaS、OEM 或企業部署等不適合 AGPL-3.0 的條款，請[聯絡維護者](https://github.com/treeleaves30760/CodefyUI/issues)。
