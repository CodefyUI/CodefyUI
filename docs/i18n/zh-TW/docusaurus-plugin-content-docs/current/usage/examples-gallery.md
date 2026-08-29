---
sidebar_position: 9
title: 範例集
description: 預建的範例工作流程 — 模型架構、端到端訓練，以及可載入並執行的 LLM 範例。
---

# 範例集

CodefyUI 在 `examples/` 底下隨附一整套可直接執行的範例圖。只要目前分頁的畫布是空的，範例集就會直接出現在畫布上 —— 挑一張卡片，圖就會載入該分頁，隨時可以 **執行**。你也可以用 [CLI 圖形執行器](./cli-runner) 在無介面（headless）下執行任何一個範例。

範例集依固定順序分成幾個區塊：

| 區塊 | 內容 |
|---------|----------|
| **Quick Start** | 三個釘選的起手式：**Train CNN on MNIST**、**Inference CNN on MNIST**，以及 **Api-Function**（graph-as-a-function 示範）。|
| **Advanced Examples** | 其他每一個可執行的內建範例 —— LLM（用離線 `demo-16d` backend 的 Word Embedding Analogy、用真正句子編碼器的 **Sentence Similarity (zh-TW)**、**Train a Causal LM on TinyStories**，以及兩個檢索範例 **RAG, fully local** 與 **RAG with a chat API**）、Diffusion（Forward Process、Toy Sampling、Mini U-Net）、傳統機器學習（Iris KNN、表格資料流程）、Transformer（MoE routing）、RNN、RL（RLHF reward + KL）、VLA（**Train a VLA on PushWorld**，需要 GPU 與大約一小時），以及其餘的訓練範例（GPT-Mini、ResNet-CIFAR10，還有實測過的 **ResNet-18 / CIFAR-10 baseline** —— 見[重現標準結果](./reproducing-baselines)）。|
| **Plugin Examples** | 由已安裝的[外掛](/advanced/plugins)提供的範例（以及任何無法辨識的分類）。只有存在時才會顯示。|
| **Model Architectures** | 15 個經典架構導覽，永遠排在最後：ResNet、ConvNeXt、EfficientNet、UNet、ViT、SwinTransformer、BERT、GPT、LLaMA、DiT、LSTM TimeSeries、BiGRU SpeechRecognition、Seq2Seq Attention、DQN Atari、PPO Robotics。|

在磁碟上，範例依主題資料夾分組：`Classical/`、`Diffusion/`、`LLM/`、`Model_Architecture/`、`RL/`、`RNN/`、`Transformer/`、`Usage_Example/` 與 `VLA/`。

所有列出的範例都可以離線直接執行，只有四個例外：

- **Train a Causal LM on TinyStories** 第一次執行時會從 Hugging Face Hub 下載 TinyStories 語料與 gpt2 的 BPE ranks，並且需要一張放得下 203,668,480 參數模型的 GPU。它的說明卡開頭就會先講這兩個前提條件；完整的訓練配方、token 預算與記憶體開關則放在圖旁邊的 `README.md`（`examples/LLM/TrainCausalLM-TinyStories/`）。兩份下載都會被快取，之後再跑就同樣是離線的。
- **Sentence Similarity (zh-TW)** 需要 `sentence-embeddings` 套件包，那是一次性的安裝，可以在套件中心（工具列 > 設定 > 選用套件包）裝，或執行 `cdui packs install sentence-embeddings` —— 執行圖形時永遠不會幫你下載它。裝好之後，這個範例就能離線在 CPU 上跑，幾秒鐘就結束。見[選用套件包](./optional-packs)。
- **RAG, fully local** 要下載的是兩樣東西，不是一樣：`rag` 套件包裡的 `qwen2.5-0.5b-instruct`，以及 `sentence-embeddings` 裡的 `multilingual-e5-small` —— 合計約 1.5 GB。裝 `rag` 只會帶進那個套件包的 Python 套件，不會帶進任何編碼器，所以第二個項目要自己另外勾。兩個都到位之後，就沒有任何東西會離開這台機器：讀文件、搜尋、生成全都在本機完成，CPU 上大約每秒幾個 token，所以答案大概要等幾秒到幾十秒 —— 這是依模型大小估的，不是量出來的，有 GPU 會快很多。
- **RAG with a chat API** 是同一條檢索鏈，只把最後一棒換成 `LLMChat`，所以它只需要 `multilingual-e5-small` —— 外加一個可以送提示詞過去的地方。預設是本機的 [Ollama](https://ollama.com)（先執行 `ollama pull qwen2.5:0.5b`），這樣一切仍然留在這台機器上；把 `provider` 換成雲端模型的話，檢索到的內容就會送給第三方，而且需要環境變數裡有金鑰。

兩個 RL 架構圖（**DQN Atari**、**PPO Robotics**）是用合成的觀測張量（`TensorCreate`、`randn`）餵給網路，而不是接真的 gym 環境，所以不需要安裝 `ale-py`/`mujoco` —— 想用真環境驅動它們，把 `EnvWrapper` 節點換進去即可。

## 載入範例

- **在應用程式中** — 開一個新的（空的）分頁，範例集就會出現在畫布上。挑一張卡片，圖會載入該分頁，隨時可以 **執行**。
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

訓練也會存下 `model_weights.pt`（在 `backend/data/models/` 底下）。之後載入 **Inference CNN on MNIST** — 它會用你剛訓練好的權重，去分類 `test_digit.png`，那是一張放在 `backend/data/images/` 底下的真實 MNIST 數字。
