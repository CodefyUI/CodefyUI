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
| **快速開始** | 最先跑的三個：**Train CNN on MNIST**、**Inference CNN on MNIST**，以及 **Call a graph over HTTP**（graph-as-a-function 示範）。 |
| **訓練** | 從頭訓練出一個模型的圖：**Train a CNN on beans**（資料集來自 Hugging Face Hub）、**Train ResNet on CIFAR10**、**Train a Transformer on MNIST**、實測過的 **ResNet-18 CIFAR-10 baseline**（見[重現標準結果](./reproducing-baselines)）、**Pretrain an LM on TinyStories**，以及 **Train a VLA on PushWorld** — 它需要 CUDA GPU 與大約一小時，操作方式在它的 [README](https://github.com/CodefyUI/CodefyUI/blob/main/examples/VLA/TrainVLA-PushWorld/README.md) 中。 |
| **LLM 與 RAG** | **Word embedding analogy**、**Sentence similarity in zh-TW**，以及兩個檢索範例 **Fully local RAG** 與 **RAG with a chat API**。 |
| **觀念** | 一張圖講一個觀念，小到可以從頭看到尾：兩個 Iris 管線、**RNN unrolled over three steps**、**Mixture of Experts top-k routing**、三個 diffusion 範例（**Forward diffusion on a digit**、**Toy reverse diffusion sampling**、**Mini U-Net node**），以及 **RLHF reward and KL terms**。 |
| **模型架構** | 15 個經典架構導覽，再依模型家族分成 CNN、RNN、Transformer、Diffusion、RL 五組。 |
| **擴充套件包** | 由已安裝的[外掛](/advanced/plugins)提供的範例，每個套件包一個小標題。只有存在時才會顯示。 |
| **其他** | 沒有宣告區塊、或宣告了這份清單不認識的區塊的內建範例。 |

列出範例的四個地方都用這一套分組：空白畫布上的 overlay、沒有開啟分頁時顯示的[歡迎畫面](./tabs-persistence#the-welcome-screen)、**範例圖庫**，以及側邊欄的**範例**分頁。

在磁碟上，範例依主題資料夾分組：`Classical/`、`Diffusion/`、`LLM/`、`Model_Architecture/`、`RL/`、`RNN/`、`Transformer/`、`Usage_Example/` 與 `VLA/`。資料夾不等於區塊 — 見[新增範例](#adding-an-example)。

所有列出的範例都可以離線直接執行，以下是例外。卡片不會寫這些：卡片只有一行，說的是這張圖在做什麼。範例執行前需要準備什麼，寫在它畫布上的[註記](./canvas-basics#notes)裡，貼在相關的節點旁邊 — 那裡才有空間說下載多大、套件包從哪裡來 — 也寫在下面這份清單裡：

- 兩個 CIFAR-10 訓練範例 **Train ResNet on CIFAR10** 與 **ResNet-18 CIFAR-10 baseline** 會在第一次執行時把 CIFAR-10（約 170 MB）下載到 `backend/data/`，之後就能離線執行。MNIST 隨 CodefyUI 一起提供，放在 `backend/data/MNIST/raw/`，所以 **Train CNN on MNIST** 與 **Train a Transformer on MNIST** 從第一次點下去就能離線執行。在[專案目錄](./project-directories)中，相對路徑的 `data_dir` 會指向專案的 `assets/data/`，所以在那裡這四個訓練範例第一次執行時都會下載各自的資料集。
- **Inference CNN on MNIST** 會載入 `model_weights.pt`，這個檔案由 **Train CNN on MNIST** 寫到 `backend/data/models/`。CodefyUI 不附任何訓練好的權重，所以在剛安裝好的電腦上，它會停在 `ModelLoader`，直到訓練範例跑過一次為止。
- **Train a CNN on beans** 第一次執行時會從 Hugging Face Hub 下載 `AI-Lab-Makerere/beans`（1,295 張照片，約 170 MB）到 Hugging Face 的快取目錄。把圖裡三個 `HuggingFaceDataset` 節點指到別的 repo，下載的就會換成那一個。
- **Pretrain an LM on TinyStories** 第一次執行時會從 Hugging Face Hub 下載 TinyStories 語料與 gpt2 的 BPE ranks，並且需要一張有足夠空間容納 203,668,480 參數模型的 GPU。它的總覽註記會用英文與中文各寫一次這兩項需求；完整步驟、token 預算與記憶體調整選項則放在 graph 旁邊的 `README.md`（`examples/LLM/TrainCausalLM-TinyStories/`）。兩份下載都會被快取，之後再次執行也能離線完成。
- **Sentence similarity in zh-TW** 需要 `sentence-embeddings` 套件包。這是一次性的安裝，可以在套件中心（工具列 > 設定 > 選用套件與外掛）安裝，或執行 `cdui packs install sentence-embeddings`；graph 執行時不會自動下載該套件包。安裝後，這個範例就能離線在 CPU 上執行，幾秒鐘就結束。見[選用套件包](./optional-packs)。
- **Fully local RAG** 需要下載兩個項目：`rag` 套件包裡的 `qwen2.5-0.5b-instruct`，以及 `sentence-embeddings` 裡的 `multilingual-e5-small`，合計約 1.5 GB。安裝 `rag` 只會加入該套件包的 Python 套件，不含編碼器，所以還需要另外選取第二個項目。兩個項目都安裝後，文件、搜尋與生成都在本機處理，不會把資料傳送到外部。CPU 上大約每秒生成幾個 token，所以答案可能需要幾秒到幾十秒；這是依模型大小估算，不是實測值，使用 GPU 會快得多。
- **RAG with a chat API** 使用同一條檢索鏈，但最後一個節點是 `LLMChat`，所以它只需要 `multilingual-e5-small`，以及可接收 prompt 的服務。預設使用本機的 [Ollama](https://ollama.com)（先執行 `ollama pull qwen2.5:0.5b`），資料仍不會離開本機；把 `provider` 換成 hosted model 後，檢索到的內容會送給第三方，而且需要在環境變數中設定 key。
- **擴充套件包**區塊裡有四個範例會在第一次執行時下載東西，每一個都會在自己畫布上的註記裡寫明：`deep` 套件包裡處理真實文字的範例 **Self-Attention 101**、**Multi-Head Causal Attention**、**C4-2 Self-attention on real text** 與 **C4-4 LLM inference pipeline** 會把 `cl100k_base` 的 BPE 表下載到快取目錄，之後每次執行都直接讀本機的快取。`foundations` 與 `deep` 套件包各有一個 MNIST 訓練範例，它們讀的是 CodefyUI 隨附的 MNIST，和內建範例一樣，只有在專案目錄中才會下載。這四個套件包的其他範例從第一次點下去就能離線執行。

兩個 RL 架構 graph（**DQN on Atari pixels**、**PPO Robotics Controller**）會將合成的觀測張量（`TensorCreate`、`randn`）傳入網路，而不是連接真正的 gym 環境，所以不需要安裝 `ale-py`/`mujoco` — 若要用真實環境驅動它們，換成 `EnvWrapper` 節點即可。

## 載入範例

- **在應用程式中** — 開一個新的空白分頁，範例集就會出現在畫布上。選取卡片後，graph 會載入該分頁並可立即**執行**。沒有開啟任何分頁時，在歡迎畫面上選取卡片，它會在新分頁中開啟。
- **從 CLI** — 把 `run_graph.py` 指向該 graph 的 JSON：

  ```bash
  cd backend
  python run_graph.py ../examples/Usage_Example/CNN-MNIST/TrainCNN-MNIST/graph.json
  ```

### 畫布不是空白時 {/* #when-the-canvas-is-not-empty */}

**範例圖庫**可從工具列的**範例圖庫**按鈕、空白畫布 overlay 與歡迎畫面上的**瀏覽全部範例**，以及側邊欄**範例**分頁底部的**瀏覽全部範例**按鈕開啟。圖庫依上面那幾個區塊分組；選取範例後，會顯示說明、節點與連線數量，以及它是內建範例還是來自外掛。每個範例提供兩個動作：

- **在新分頁開啟**會保留目前的 graph，不做任何變動。雙擊卡片的效果相同。
- **插入目前畫布**會把範例加入正在編輯的 graph：系統會為插入的節點產生新的 id，並將它們放在目前 graph 的下方，因此不會覆蓋任何內容，而且只要復原一次就能移除。由較新版 CodefyUI 寫出的範例不會被插入（「沒有插入任何內容：…」）；請改用在新分頁開啟，它會以唯讀方式開啟。

沒有開啟任何分頁時，圖庫只提供**在新分頁開啟**。

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

**卡片上會看到什麼。** `name` 是標籤，不是句子：最多五個英文單字、34 欄寬，而且不能和其他範例重複 — 卡片、側邊欄那一列、空白畫布 overlay 與歡迎畫面都只印這一個名稱，旁邊沒有別的東西能用來分辨兩個範例。`description` 是名稱底下那一行，最多 40 欄寬，也就是二十個漢字，寫的是這張圖在做什麼、在解釋什麼。它不寫執行前的需求，也不寫成績：下載、GPU、套件包、API 金鑰、必須先跑過的另一個範例、跑完得到的準確率，這些全部改寫在畫布上的註記裡，貼在相關的節點旁邊；其中會決定範例跑不跑得起來的，也會列在本頁開頭的例外清單裡。這兩個上限就是卡片放得下的量：卡片寬 `13rem`，裡面的字級跟著根字級一起放大，所以螢幕愈寬只會讓標題變大、卡片不會跟著變寬，超出的部分就被截掉。用欄寬而不是字數計算：中文說明會夾雜英文節點名稱，而一個漢字佔兩欄。英文的兩個上限由 `backend/tests/test_example_descriptions.py` 把關，「不寫需求」這一條則由 `backend/tests/test_builtin_examples.py` 對兩個來源目錄裡的每個範例把關；中文的上限與這一條都由 `frontend/src/i18n/exampleLocales/zh-TW.test.ts` 把關 — 名稱不翻譯。

**比較長的說明放哪裡。** 這兩行放不下的內容就放到畫布上的[註記](./canvas-basics#notes)，貼在它所說明的節點旁邊。註記是 type 為 `note` 的節點：驗證會跳過它，執行不會走到它，卡片上的節點數量也不會把它算進去。每則註記在同一個註記裡寫兩次 — 先一段英文，空一行，再用繁體中文寫同一件事 — 這樣一則註記就兩種語言都能讀，也不必再多維護一份翻譯表。

## 適合的第一次執行

載入 **Train CNN on MNIST**，然後：

1. 先讀 `Start` 節點左邊那則註記 — 它會說這張圖在做什麼，以及跑完要看哪裡。
2. **錄製節點輸出**與**在多次執行間保留權重**預設都已開啟 — 可在「設定」popover 中的**錄製與檢視**及**訓練行為**確認。
3. 點擊**執行**，並在**訓練**分頁觀看即時 loss 圖表。5 個 epoch 在 CPU 上大約一兩分鐘。
4. 點擊 `Inference` 節點，在 **[教學檢視器](./teaching-inspector)**中查看送進去的內容（16 張測試影像的批次）與輸出的結果（每張影像 10 個 logits）。兩個 `Conv2d` 層在 `SequentialModel` 節點裡面；雙擊它即可在「模型架構」編輯器中看到。
5. 跑完之後看 **Test accuracy** 那個 `Print`，大約 0.99，算在訓練迴圈沒看過的 10,000 張測試影像上；再把 16 個預測數字和旁邊 `Visualize` 排出來的 16 張影像對照。
6. 再執行一次 — 因為權重已保留，模型會跨次執行持續學習。

訓練也會存下 `model_weights.pt`（在 `backend/data/models/` 底下）。之後載入 **Inference CNN on MNIST**；它會使用剛才訓練出的權重分類 `test_digit.png`（放在 `backend/data/images/` 底下的真實 MNIST 數字影像），並印出它讀到的數字。
