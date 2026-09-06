---
sidebar_position: 9
title: 範例集
description: 預建的範例工作流程 — 模型架構、端到端訓練，以及可載入並執行的 LLM 範例。
---

# 範例集

CodefyUI 在 `examples/` 底下隨附一整套可直接執行的範例 graph。只要目前分頁的畫布是空的，範例集就會直接出現在畫布上；選取卡片後，graph 會載入該分頁並可立即**執行**。其他開啟方式見[畫布不是空白時](#when-the-canvas-is-not-empty)。你也可以用 [CLI 圖形執行器](./cli-runner)在無介面（headless）下執行任何範例。

範例集依固定順序分成幾個區塊：

| 區塊 | 內容 |
|---------|----------|
| **快速上手** | 三個置頂的入門範例：**Train CNN on MNIST**、**Inference CNN on MNIST**，以及 **Api-Function**（graph-as-a-function 示範）。 |
| **進階範例** | 其他每一個可執行的內建範例 — LLM（用離線 `demo-16d` backend 的 Word Embedding Analogy、用真正句子編碼器的 **Sentence Similarity (zh-TW)**、**Train a Causal LM on TinyStories**，以及兩個檢索範例 **RAG, fully local** 與 **RAG with a chat API**）、Diffusion（Forward Process、Toy Sampling、Mini U-Net）、Classical ML（Iris KNN、表格資料管線）、Transformer（MoE routing）、RNN、RL（RLHF reward + KL）、VLA（**Train a VLA on PushWorld** — 需要 CUDA GPU 與大約一小時；操作方式在它的 [README](https://github.com/CodefyUI/CodefyUI/blob/main/examples/VLA/TrainVLA-PushWorld/README.md) 中），以及其餘的訓練範例（GPT-Mini、ResNet-CIFAR10，還有實測過的 **ResNet-18 / CIFAR-10 baseline** — 見[重現標準結果](./reproducing-baselines)）。 |
| **外掛範例** | 由已安裝的[外掛](/advanced/plugins)提供的範例（以及任何無法辨識的分類）。只有存在時才會顯示。 |
| **模型架構範例** | 15 個經典架構導覽，永遠排在最後：ResNet、ConvNeXt、EfficientNet、UNet、ViT、SwinTransformer、BERT、GPT、LLaMA、DiT、LSTM TimeSeries、BiGRU SpeechRecognition、Seq2Seq Attention、DQN Atari、PPO Robotics。 |

在磁碟上，範例依主題資料夾分組：`Classical/`、`Diffusion/`、`LLM/`、`Model_Architecture/`、`RL/`、`RNN/`、`Transformer/`、`Usage_Example/` 與 `VLA/`。

所有列出的範例都可以離線直接執行，只有四個例外：

- **Train a Causal LM on TinyStories** 第一次執行時會從 Hugging Face Hub 下載 TinyStories 語料與 gpt2 的 BPE ranks，並且需要一張有足夠空間容納 203,668,480 參數模型的 GPU。它的說明卡開頭會列出這兩項需求；完整步驟、token 預算與記憶體調整選項則放在 graph 旁邊的 `README.md`（`examples/LLM/TrainCausalLM-TinyStories/`）。兩份下載都會被快取，之後再次執行也能離線完成。
- **Sentence Similarity (zh-TW)** 需要 `sentence-embeddings` 套件包。這是一次性的安裝，可以在套件中心（工具列 > 設定 > 選用套件）安裝，或執行 `cdui packs install sentence-embeddings`；graph 執行時不會自動下載該套件包。安裝後，這個範例就能離線在 CPU 上執行，幾秒鐘就結束。見[選用套件包](./optional-packs)。
- **RAG, fully local** 需要下載兩個項目：`rag` 套件包裡的 `qwen2.5-0.5b-instruct`，以及 `sentence-embeddings` 裡的 `multilingual-e5-small`，合計約 1.5 GB。安裝 `rag` 只會加入該套件包的 Python 套件，不含編碼器，所以還需要另外選取第二個項目。兩個項目都安裝後，文件、搜尋與生成都在本機處理，不會把資料傳送到外部。CPU 上大約每秒生成幾個 token，所以答案可能需要幾秒到幾十秒；這是依模型大小估算，不是實測值，使用 GPU 會快得多。
- **RAG with a chat API** 使用同一條檢索鏈，但最後一個節點是 `LLMChat`，所以它只需要 `multilingual-e5-small`，以及可接收 prompt 的服務。預設使用本機的 [Ollama](https://ollama.com)（先執行 `ollama pull qwen2.5:0.5b`），資料仍不會離開本機；把 `provider` 換成 hosted model 後，檢索到的內容會送給第三方，而且需要在環境變數中設定 key。

兩個 RL 架構 graph（**DQN Atari**、**PPO Robotics**）會將合成的觀測張量（`TensorCreate`、`randn`）傳入網路，而不是連接真正的 gym 環境，所以不需要安裝 `ale-py`/`mujoco` — 若要用真實環境驅動它們，換成 `EnvWrapper` 節點即可。

## 載入範例

- **在應用程式中** — 開一個新的空白分頁，範例集就會出現在畫布上。選取卡片後，graph 會載入該分頁並可立即**執行**。
- **從 CLI** — 把 `run_graph.py` 指向該 graph 的 JSON：

  ```bash
  cd backend
  python run_graph.py ../examples/Usage_Example/CNN-MNIST/TrainCNN-MNIST/graph.json
  ```

### 畫布不是空白時 {/* #when-the-canvas-is-not-empty */}

**範例圖庫**可從工具列的**範例圖庫**按鈕、空白畫布 overlay 上的**瀏覽全部範例**，以及側邊欄的**範例**分頁開啟。圖庫依類別將範例分組；選取範例後，會顯示說明、節點與連線數量，以及它是內建範例還是來自外掛。每個範例提供兩個動作：

- **在新分頁開啟**會保留目前的 graph，不做任何變動。
- **插入目前畫布**會把範例加入正在編輯的 graph：系統會為插入的節點產生新的 id，並將它們放在目前 graph 的下方，因此不會覆蓋任何內容，而且只要復原一次就能移除。

在側邊欄的**範例**分頁中，點擊範例即可插入，也可以把它拖到畫布上想放的位置。

## 適合的第一次執行

載入 **Train CNN on MNIST**，然後：

1. **錄製節點輸出**與**在多次執行間保留權重**預設都已開啟 — 可在「設定」popover 中的**錄製與檢視**及**訓練行為**確認。
2. 點擊**執行**，並在**訓練**分頁觀看即時 loss 圖表。
3. 點擊一個 `Conv2d` 節點，在 **[教學檢視器](./teaching-inspector)**中檢視它的 kernel 與 activation。
4. 再執行一次 — 因為權重已保留，模型會跨次執行持續學習。

訓練也會存下 `model_weights.pt`（在 `backend/data/models/` 底下）。之後載入 **Inference CNN on MNIST**；它會使用剛才訓練出的權重分類 `test_digit.png`。這是放在 `backend/data/images/` 底下的真實 MNIST 數字影像。
