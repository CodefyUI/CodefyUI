---
sidebar_position: 4
title: 裝置後端
description: CodefyUI 如何在 CPU、CUDA、MPS 與 ROCm 之間選擇與退回——以及實驗性的原生 MLX 推論 spike。
---

# 裝置後端

CodefyUI 執行於 PyTorch 之上，因此繼承了 PyTorch 的裝置後端：**CPU**、**NVIDIA CUDA**、**Apple Silicon（MPS）**與 **AMD ROCm**（Linux）。關於安裝正確的 wheel，請參閱 **[GPU 與裝置設定](/getting-started/gpu-device)**；本頁說明裝置選擇在執行時的行為。

## 全域裝置選擇

**預設是 CPU，而且不會有任何機制替你切走。** 除非你在設定裡自行選擇加速器，否則執行一律使用 CPU；下拉選單會列出 PyTorch 實際能看到的每個裝置（透過 `device_utils.get_available_devices()`）。被請求的裝置會與可用的裝置比對，若不存在則**退回 CPU 並發出警告**。設定一次即可，不必逐節點設定。

### 裝置對齊由引擎保證

你不需要去推敲某個張量到底在哪個裝置上。節點執行前，`graph_engine.invoke_node` 會把它輸入裡的每個張量搬到該節點要跑的裝置——節點自己宣告了 `device` 參數就用它的，否則用這次執行的全域裝置。由於所有進入節點的路徑都經過那一個函式，這個保證同時涵蓋內建節點、外掛節點與你自己的[自訂節點](./custom-nodes)。

這件事之所以重要，是因為裝置不一致在只有 CPU 的機器上根本不可能發生，所以在多數開發過程中完全隱形，只會在別人的 GPU 機器上爆出來。在對齊移進引擎之前，已經有兩張出貨的圖是這樣死的——`Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor) should be the same`。

對齊**刻意不碰**的東西：

- **模組。** `nn.Module.to()` 是就地修改，把一個從別的節點傳過來的模型搬走，等於在擁有它的節點腳下換掉權重。需要把收到的模型放到自己裝置上的節點，要自己明確呼叫 `to_device`。
- **Dataset、DataLoader、環境等非張量值。** 它們原樣通過，所以 dataset 維持惰性，`TrainingLoop` 仍然是一次一個 batch 串流到 GPU，而不是整份常駐 VRAM。

## 在多張卡之中指定其中一張

在有一張以上 CUDA 裝置的機器上，每個下拉選單也會逐張列出——`cuda:0`、`cuda:1` 等等——並與單純的 `cuda` 並列，後者代表「torch 目前指向的那一張」。只有一張 GPU 的機器只會顯示 `cuda`，因為在那裡兩者指的是同一塊硬體。

指定了這台機器上不存在的編號時，會退回**目前的 CUDA 裝置**而不是 CPU：一張在工作站上釘在 `cuda:2` 的圖，在筆電上打開時仍然應該用 GPU 訓練。每張卡也各自有自己的執行佇列。完整說明（包含刻意排除在外的分散式訓練）請參閱 **[訓練記憶體](./training-memory)**。

## float64 + MPS 的限制

MPS 是 **float32 原生**的，會拒絕 float64 張量。CodefyUI 在 `device_utils.to_device` 中將其正規化，但如果你撰寫一個直接建立張量的[自訂節點](./custom-nodes)，請在 Apple GPU 上將它們維持為 float32，以避免執行時錯誤。

## ROCm 呈現為 CUDA

在 AMD + Linux 上搭配 ROCm 版本的 PyTorch 時，`torch.cuda.is_available()` 會回傳 `True`，因為 ROCm 暴露了一個與 CUDA 相容的介面。該裝置在下拉選單中會顯示為 `cuda`；這是預期的行為。

## 實驗性：原生 MLX（spike）

有一個**概念驗證 (proof-of-concept)**，把一個小型 MLP 的*前向推論*從 PyTorch 移植到 Apple 的 [MLX](https://github.com/ml-explore/mlx) 框架，產生數值上完全相同的結果（最大絕對差約 1.9e-7）。重點如下：

- **真正圖引擎中的 Apple 加速是 PyTorch MPS**，它已接好並完成端到端驗證。MLX **並非**已交付的執行後端。
- MLX 是一個*獨立的陣列框架*，並非 PyTorch 後端——並沒有 `torch.device("mlx")`——所以它無法成為全域裝置選擇器（驅動 `torch`）中的一個值。
- 此 spike 僅供**推論**且為 **float32**，可臨時執行：

  ```bash
  uv pip install mlx        # Apple Silicon only
  python scripts/mlx_spike.py
  ```

- `mlx` **並非**已納入的相依套件；主應用程式從不匯入它。只透過 `device_utils.mlx_available()`（偵測）與 spike 腳本來呈現它。

**建議：**將 **MPS** 維持為所有執行（訓練 + 推論）的 Apple 預設；把 MLX 當作選用的推論加速器，只在推論密集的教學示範上有可量測的效益時才回頭考慮。
