---
sidebar_position: 9
title: 範例集
description: 預建的範例工作流程 — 模型架構、端到端訓練，以及可載入並執行的 LLM 範例。
---

# 範例集

CodefyUI 在 `examples/` 底下隨附一整套可直接執行的範例圖。你可以從應用程式的 **Examples** 選單載入它們，或用 [CLI 圖形執行器](./cli-runner) 在無介面（headless）下執行。

| 類別 | 範例 |
|----------|----------|
| **模型架構** | ResNet、ConvNeXt、EfficientNet、UNet、ViT、SwinTransformer、BERT、GPT、LLaMA、DiT、LSTM TimeSeries、BiGRU SpeechRecognition、Seq2Seq Attention、DQN Atari、PPO Robotics |
| **使用範例** | CNN-MNIST 訓練、CNN-MNIST 推論、GPT-Mini 訓練、ResNet-CIFAR10 訓練、**ResNet-18 / CIFAR-10 baseline**（實測 95.48%，見[重現標準結果](./reproducing-baselines)）|
| **LLM** | Word Embedding Analogy（用離線的 `demo-16d` backend 計算 `king − man + woman ≈ queen`）、**Sentence Similarity (zh-TW)**（用真正的句子編碼器比較八句繁體中文）、**Train a Causal LM on TinyStories**（從零開始預訓練一個 203,668,480 參數的 GPT 式 decoder）|

這個 repository 在磁碟上也依主題把範例分組：`Classical/`、`Diffusion/`、`LLM/`、`Model_Architecture/`、`RL/`、`RNN/`、`Transformer/`、`Usage_Example/` 與 `Others/`。

所有列出的範例都可以離線直接執行，只有兩個例外：

- **Train a Causal LM on TinyStories** 第一次執行時會從 Hugging Face Hub 下載 TinyStories 語料與 gpt2 的 BPE ranks，並且需要一張放得下 203,668,480 參數模型的 GPU。它的說明卡開頭就會先講這兩個前提條件；完整的訓練配方、token 預算與記憶體開關則放在圖旁邊的 `README.md`（`examples/LLM/TrainCausalLM-TinyStories/`）。兩份下載都會被快取，之後再跑就同樣是離線的。
- **Sentence Similarity (zh-TW)** 需要 `sentence-embeddings` 套件包，那是一次性的安裝，可以在套件中心（工具列 > 設定 > 選用套件包）裝，或執行 `cdui packs install sentence-embeddings` —— 執行圖形時永遠不會幫你下載它。裝好之後，這個範例就能離線在 CPU 上跑，幾秒鐘就結束。見[選用套件包](./optional-packs)。

## 載入範例

- **在應用程式中** — 開啟 **Examples** 選單並選一個圖；它會載入到新分頁，準備好可以 **執行**。
- **從 CLI** — 把 `run_graph.py` 指向該圖的 JSON：

  ```bash
  cd backend
  python run_graph.py ../examples/Usage_Example/CNN-MNIST/TrainCNN-MNIST/graph.json
  ```

## 適合的第一次執行

載入 **Train CNN on MNIST**，然後：

1. 在 設定 popover 中開啟 **記錄輸出** 與 **跨 run 保留權重**。
2. 點擊 **執行**，並在 **訓練** 分頁觀看即時 loss 圖表。
3. 點一個 `Conv2d` 節點，在 **[教學檢視器](./teaching-inspector)** 中檢視它的 kernel 與 activation。
4. 再執行一次 — 因為權重已保留，模型會跨次執行持續學習。
