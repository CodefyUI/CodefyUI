---
sidebar_position: 9
title: 範例集
description: 預建的範例工作流程 — 模型架構、端到端訓練，以及可載入並執行的 LLM 範例。
---

# 範例集

CodefyUI 在 `examples/` 底下隨附一整套可直接執行的範例 graph。只要目前分頁的畫布是空的，範例集就會直接出現在畫布上；選取卡片後，graph 會載入該分頁並可立即**執行**。其他開啟方式見[畫布不是空白時](#when-the-canvas-is-not-empty)。你也可以用 [CLI 圖形執行器](./cli-runner)在無介面（headless）下執行任何範例。

範例集依「這個範例是做什麼用的」分組，順序如下：

| 區塊 | 內容 |
|---------|----------|
| **快速開始** | 最先跑的三個：**Train CNN on MNIST**、**Inference CNN on MNIST**，以及 **Call a graph as an API**（graph-as-a-function 示範）。 |
| **訓練** | 從頭訓練出一個模型的圖：**Train a CNN on a HuggingFace dataset**、**Train ResNet on CIFAR10**、**Train a Transformer classifier on MNIST**、實測過的 **ResNet-18 / CIFAR-10 baseline**（見[重現標準結果](./reproducing-baselines)）、**Train a Causal LM on TinyStories**，以及 **Train a VLA on PushWorld** — 它需要 CUDA GPU 與大約一小時，操作方式在它的 [README](https://github.com/CodefyUI/CodefyUI/blob/main/examples/VLA/TrainVLA-PushWorld/README.md) 中。 |
| **LLM 與 RAG** | **Word Embedding Analogy**、**Sentence Similarity (zh-TW)**，以及兩個檢索範例 **RAG, fully local** 與 **RAG with a chat API**。 |
| **觀念** | 一張圖講一個觀念，小到可以從頭看到尾：兩個 Iris 管線、**RNN unrolled**、**Mixture of Experts**、三個 diffusion 範例（**Forward Diffusion**、**Toy Sampling**、**Mini U-Net**），以及 **RLHF building blocks: reward + KL**。 |
| **模型架構** | 15 個經典架構導覽，再依模型家族分成 CNN、RNN、Transformer、Diffusion、RL 五組。 |
| **擴充套件包** | 由已安裝的[外掛](/advanced/plugins)提供的範例，每個套件包一個小標題。只有存在時才會顯示。 |
| **其他** | 沒有宣告區塊、或宣告了這份清單不認識的區塊的內建範例。 |

列出範例的三個地方都用這一套分組：空白畫布上的 overlay、**範例圖庫**，以及側邊欄的**範例**分頁。

在磁碟上，範例依主題資料夾分組：`Classical/`、`Diffusion/`、`LLM/`、`Model_Architecture/`、`RL/`、`RNN/`、`Transformer/`、`Usage_Example/` 與 `VLA/`。資料夾不等於區塊 — 見[新增範例](#adding-an-example)。

所有列出的範例都可以離線直接執行，以下是例外。每一個都會在自己的卡片上用卡片放得下的字說明 —「需要下載」、「需套件包」、「需 GPU」：

- 四個資料集訓練範例會在第一次執行時下載資料，之後就能離線執行：**Train CNN on MNIST** 與 **Train a Transformer classifier on MNIST** 下載 MNIST，**Train ResNet on CIFAR10** 與 **ResNet-18 / CIFAR-10 baseline** 下載 CIFAR-10（約 170 MB）。兩份都放在 `backend/data/` 底下。
- **Train a CNN on a HuggingFace dataset** 第一次執行時會從 Hugging Face Hub 下載 `AI-Lab-Makerere/beans`（1,295 張照片，約 170 MB）到 Hugging Face 的快取目錄。把圖裡三個 `HuggingFaceDataset` 節點指到別的 repo，下載的就會換成那一個。
- **Train a Causal LM on TinyStories** 第一次執行時會從 Hugging Face Hub 下載 TinyStories 語料與 gpt2 的 BPE ranks，並且需要一張有足夠空間容納 203,668,480 參數模型的 GPU。它的說明卡開頭會列出這兩項需求；完整步驟、token 預算與記憶體調整選項則放在 graph 旁邊的 `README.md`（`examples/LLM/TrainCausalLM-TinyStories/`）。兩份下載都會被快取，之後再次執行也能離線完成。
- **Sentence Similarity (zh-TW)** 需要 `sentence-embeddings` 套件包。這是一次性的安裝，可以在套件中心（工具列 > 設定 > 選用套件與外掛）安裝，或執行 `cdui packs install sentence-embeddings`；graph 執行時不會自動下載該套件包。安裝後，這個範例就能離線在 CPU 上執行，幾秒鐘就結束。見[選用套件包](./optional-packs)。
- **RAG, fully local** 需要下載兩個項目：`rag` 套件包裡的 `qwen2.5-0.5b-instruct`，以及 `sentence-embeddings` 裡的 `multilingual-e5-small`，合計約 1.5 GB。安裝 `rag` 只會加入該套件包的 Python 套件，不含編碼器，所以還需要另外選取第二個項目。兩個項目都安裝後，文件、搜尋與生成都在本機處理，不會把資料傳送到外部。CPU 上大約每秒生成幾個 token，所以答案可能需要幾秒到幾十秒；這是依模型大小估算，不是實測值，使用 GPU 會快得多。
- **RAG with a chat API** 使用同一條檢索鏈，但最後一個節點是 `LLMChat`，所以它只需要 `multilingual-e5-small`，以及可接收 prompt 的服務。預設使用本機的 [Ollama](https://ollama.com)（先執行 `ollama pull qwen2.5:0.5b`），資料仍不會離開本機；把 `provider` 換成 hosted model 後，檢索到的內容會送給第三方，而且需要在環境變數中設定 key。
- **擴充套件包**區塊裡有六個範例會在第一次執行時下載東西，卡片上也會寫「需下載」：`foundations` 與 `deep` 套件包各有一個 MNIST 訓練範例，會把 MNIST 下載到 `backend/data/`；`deep` 套件包裡處理真實文字的四個範例 **Self-Attention 101**、**Multi-Head Causal Attention**、**C4-2 Self-attention over a tokenised sentence** 與 **C4-4 LLM inference** 則會把 `cl100k_base` 的 BPE 表下載到快取目錄，之後每次執行都直接讀本機的快取。這四個套件包的其他範例從第一次點下去就能離線執行。

兩個 RL 架構 graph（**DQN Atari**、**PPO Robotics**）會將合成的觀測張量（`TensorCreate`、`randn`）傳入網路，而不是連接真正的 gym 環境，所以不需要安裝 `ale-py`/`mujoco` — 若要用真實環境驅動它們，換成 `EnvWrapper` 節點即可。

## 載入範例

- **在應用程式中** — 開一個新的空白分頁，範例集就會出現在畫布上。選取卡片後，graph 會載入該分頁並可立即**執行**。
- **從 CLI** — 把 `run_graph.py` 指向該 graph 的 JSON：

  ```bash
  cd backend
  python run_graph.py ../examples/Usage_Example/CNN-MNIST/TrainCNN-MNIST/graph.json
  ```

### 畫布不是空白時 {/* #when-the-canvas-is-not-empty */}

**範例圖庫**可從工具列的**範例圖庫**按鈕、空白畫布 overlay 上的**瀏覽全部範例**，以及側邊欄的**範例**分頁開啟。圖庫依上面那幾個區塊分組；選取範例後，會顯示說明、節點與連線數量，以及它是內建範例還是來自外掛。每個範例提供兩個動作：

- **在新分頁開啟**會保留目前的 graph，不做任何變動。
- **插入目前畫布**會把範例加入正在編輯的 graph：系統會為插入的節點產生新的 id，並將它們放在目前 graph 的下方，因此不會覆蓋任何內容，而且只要復原一次就能移除。

在側邊欄的**範例**分頁中，點擊範例即可插入，也可以把它拖到畫布上想放的位置。

### 語言

編輯器切到繁體中文後，每個內建範例的說明都是中文。**名稱兩種語言都保持英文**：範例的名稱就是你在這份文件、在磁碟上的 `examples/`、以及 `run_graph.py` 參數裡再次找到它的依據。搜尋兩種語言都吃，`attention` 與「注意力」會找到同一張圖。

還沒翻譯的範例（例如第三方外掛帶來的）會保留英文說明，不會變成空白。翻譯放在 `frontend/src/i18n/exampleLocales/`，以範例的路徑為 key。

## 新增範例 {/* #adding-an-example */}

一個範例就是一個放著 `graph.json` 的資料夾 — 內建範例放在 `examples/` 底下，外掛提供的則放在 `<套件包>/examples/` 底下。這個檔案裡有三件事決定它在範例集裡長什麼樣子。

**它會出現在哪一區。** 最上層可以放一個選用的 `gallery` 區塊：

```json
{
  "name": "UNet for Image Segmentation",
  "description": "...",
  "gallery": { "section": "architectures", "family": "CNN", "order": 2 },
  "nodes": [],
  "edges": []
}
```

`section` 只能是 `quickstart`、`training`、`llm`、`concepts` 與 `architectures` 其中之一。`family` 是**模型架構**區塊裡的小標題：上面那五個家族照列出的順序排在前面，其他字串接在後面依字母排序。只要範例有寫 `family`，卡片上的標籤也會改成顯示它，而不是分類名稱；如果卡片上方的小標題已經是同一個字，卡片就不再放標籤。`order` 在同一區塊（模型架構則是同一個家族）內由小到大排序，沒有寫 `order` 的範例排在有寫的後面。清單讀不懂的欄位只有那一個欄位會變成空值，而不是讓整個範例集掛掉：不認識的 `section` 會讓內建範例落到**其他**區塊，不是整數的 `order` 則只是讓它排在有寫 `order` 的範例後面。外掛提供的範例不管宣告什麼，都會列在它自己的套件包底下。

範例所在的資料夾不等於它的區塊。資料夾是路徑，而這份文件、翻譯表與 `run_graph.py` 的參數指的都是路徑，所以重新分組範例集時，磁碟上的東西不會動。

**卡片上會看到什麼。** `description` 是一行、最多 56 欄寬 — 卡片、側邊欄那一列與詳細資訊面板都能完整顯示而不會被截斷。用欄寬而不是字數計算：中文說明會夾雜英文節點名稱，而一個漢字佔兩欄。英文的部分由 `backend/tests/test_example_descriptions.py` 把關，中文的部分由 `frontend/src/i18n/exampleLocales/zh-TW.test.ts` 把關。

**比較長的說明放哪裡。** 再長的內容就放到畫布上的[註記](./canvas-basics#notes)，貼在它所說明的節點旁邊。註記是 type 為 `note` 的節點：驗證會跳過它，執行不會走到它，卡片上的節點數量也不會把它算進去。每則註記在同一個註記裡寫兩次 — 先一段英文，空一行，再用繁體中文寫同一件事 — 這樣一則註記就兩種語言都能讀，也不必再多維護一份翻譯表。

## 適合的第一次執行

載入 **Train CNN on MNIST**，然後：

1. 先讀 `Start` 節點左邊那則註記 — 它會說這張圖在做什麼，以及跑完要看哪裡。
2. **錄製節點輸出**與**在多次執行間保留權重**預設都已開啟 — 可在「設定」popover 中的**錄製與檢視**及**訓練行為**確認。
3. 點擊**執行**，並在**訓練**分頁觀看即時 loss 圖表。第一次執行會下載 MNIST；5 個 epoch 在 CPU 上大約一兩分鐘。
4. 點擊一個 `Conv2d` 節點，在 **[教學檢視器](./teaching-inspector)**中檢視它的 kernel 與 activation。
5. 跑完之後看 **Test accuracy** 那個 `Print`，大約 0.99，算在訓練迴圈沒看過的 10,000 張測試影像上；再把 16 個預測數字和旁邊 `Visualize` 排出來的 16 張影像對照。
6. 再執行一次 — 因為權重已保留，模型會跨次執行持續學習。

訓練也會存下 `model_weights.pt`（在 `backend/data/models/` 底下）。之後載入 **Inference CNN on MNIST**；它會使用剛才訓練出的權重分類 `test_digit.png`（放在 `backend/data/images/` 底下的真實 MNIST 數字影像），並印出它讀到的數字。
