---
sidebar_position: 9
title: 範例集
description: 預建的範例工作流程 — 快速上手、訓練與評估的做法、LLM 範例、模型架構與論文重現，都可以載入並執行。
---

# 範例集

CodefyUI 在 `examples/` 底下隨附一整套可直接執行的範例 graph。只要目前分頁的畫布是空的，範例集就會直接出現在畫布上（打開的子圖是空的時，只會顯示一段簡短的提示）；選取卡片後，graph 會開啟並可立即**執行**：分頁是空的、也沒有可復原的步驟時放進該分頁，否則開在新分頁，所以不會取代任何 graph。其他開啟方式見[畫布不是空白時](#when-the-canvas-is-not-empty)。你也可以用 [CLI 圖形執行器](./cli-runner)在無介面（headless）下執行任何範例。

範例集依**主標籤**分組，也就是「這個範例是做什麼用的」；實務類別排在前面，論文與課程放在後面：

| 主標籤 | 內容 |
|----------|----------|
| **快速開始** | 最先跑的四個：**Train CNN on MNIST**、**Inference CNN on MNIST**、**Classify Iris with sklearn KNN**，以及 **Call a graph over HTTP**（graph-as-a-function 示範）。新分頁上的範例集預設停在這一類。 |
| **訓練** | 從頭訓練出一個模型的圖：**Train a CNN on beans**（資料集來自 Hugging Face Hub）、**Train ResNet on CIFAR10**、實測過的 **ResNet-18 CIFAR-10 baseline**（見[重現標準結果](./reproducing-baselines)）、**Train a Transformer on MNIST**、**Train a VLA on PushWorld** — 它需要 CUDA GPU 與大約一小時，操作方式在它的 [README](https://github.com/CodefyUI/CodefyUI/blob/main/examples/VLA/TrainVLA-PushWorld/README.md) 中 — 以及三個用產生的資料、在 CPU 上一分鐘內訓練完的範例：**Segment shapes pixel by pixel**、**Train a DDPM on blobs** 與 **LSTM remembers the first token**。另外四個示範訓練迴圈周邊的做法：**Early stopping, keep the best** 依驗證 loss 提早停止，並用 `CheckpointSaver` 存下最佳 epoch 的權重；**Resume from a checkpoint** 用 `CheckpointSaver` 存下一次訓練，再用 `CheckpointLoader` 接著訓練；**Augment, then train** 把轉換鏈接到 Dataset 的 `train_transform`；**Transfer with a frozen backbone** 用 `FreezeParameters` 凍結預訓練好的骨幹網路，只訓練新的輸出層，再和從頭訓練的結果比較。 |
| **LLM 與 RAG** | **Pretrain an LM on TinyStories**；**Fine-tune a tiny char LM**，先用短篇故事預訓練一個字元層級的 Transformer，再用你提供的文字微調，並在兩份文字上量 perplexity；兩條檢索鏈 **Fully local RAG** 與 **RAG with a chat API**；**Measure retrieval hit rate**，用詞袋檢索器在隨附的筆記上量 hit@1 與 hit@3，不需安裝任何套件包；**Data mix ratios**，依權重混合兩個隨附的文字檔；**Sentence similarity in zh-TW**；以及 **Word embedding analogy**。 |
| **資料** | 訓練之前先看資料：**Inspect a dataset first**（隨附手寫數字資料的筆數、樣本與像素統計）、**Split and check for leaks**（分層切出訓練／驗證／測試資料，再檢查評估資料裡有沒有和訓練資料重複的列）、**Tabular Iris feature pipeline**，以及 **Image batch IO**，它用 `ImageWriter` 寫出一個批次，再用 `ImageBatchReader` 把資料夾讀回來。 |
| **評估** | 衡量模型，而不是訓練模型：**Five seeds, mean and spread**（同一個訓練子圖放五次，回報平均與標準差）、**Evaluate a saved checkpoint**（混淆矩陣與各數字的正確率）、**Compare two checkpoints**（同一次訓練的兩個 epoch，用同一份測試資料比較，並列出兩者判讀不同的影像）、**Project embeddings to 2D**（用 PCA 投影隱藏層）、**K-fold cross-validation**（`KFoldSplit`），以及 **Six classifiers on two moons**。 |
| **實驗** | 決定要不要相信某個設定之前，先做的小型研究：**Find a learning rate**（先做 LR range test，再用它選出的學習率訓練）、**Learning rate x batch size**（用兩個子圖、各放兩次組成 2x2 網格，並排出名次），以及 **Ablation: input scaling**（同一個模型，有無標準化輸入的對照）。 |
| **部署** | 把訓練好的模型變成服務：**Train and export a classifier** 用 iris 訓練並存成 `iris-classifier.safetensors`，**Serve a classifier over HTTP** 載入它，經由 `GraphInput` 與 `GraphOutput` 回答，因此可以[透過 HTTP 當成函式呼叫](./graph-as-a-function)。 |
| **模型架構** | 15 個經典架構導覽，再依模型家族分成 CNN、RNN、Transformer、Diffusion、RL 五組。 |
| **概念** | 一張圖講一個觀念，小到可以從頭看到尾：**Tensor ops tour**（調整張量形狀的節點）、**Reduce and Switch**（`Reduce` 與 `Switch` 這兩個資料流節點）、**One backward pass**（可以用手算驗證的梯度，需開啟「擷取梯度」）、**RNN unrolled over three steps**、**Conv1d classifies sequences**（放在 RNN 範例旁邊的時間卷積）、**Causal attention from scratch**、**Mixture of Experts top-k routing**、三個 diffusion 範例（**Forward diffusion on a digit**、**Toy reverse diffusion sampling**、**Mini U-Net node**），以及強化學習的觀念：**Discounted returns**（gamma 如何把獎勵變成回報）、**Reward model from preferences**（RLHF 第一步）、**RLHF reward and KL terms**、**Reward hacking via a shortcut** 與 **One GRPO step on rollouts**。 |
| **論文重現** | 一張圖重現一篇論文的方法，用的節點和其他範例相同，在 CPU 上幾秒內就能從頭訓練完。總覽註記會寫明論文、配方，以及這張圖預期達到的數字：**Dropout narrows the overfit gap**、**BatchNorm tames a high LR**、**Skip connections train deeper**、**Adam vs SGD vs momentum**、**Label smoothing lowers confidence**、**Cosine warm restarts (SGDR)**、**Sigmoid gradients vanish in depth**，以及極小 batch 下的正規化：**GroupNorm holds at batch 2**。序列與回歸：**Gradient clipping rescues an RNN**、**Autoencoder squeezes digits to 2-D**、**LSTM forecasts a noisy wave**、**Transformer learns to sort**、**Weight decay smooths the fit** 與 **Early stopping on validation loss**。強化學習的四個範例用 **RLTrainingLoop** 節點訓練：**REINFORCE balances CartPole**、**DQN balances CartPole**、**PPO balances CartPole** 與 **Q-learning crosses a trap grid**。 |
| **課程** | CodefyUI 隨附的課程套件包 — `foundations`、`deep`、`rl`、`stats` 與 `eval` — 提供的範例，每個套件包一個小標題。只有已啟用的套件包帶有範例時才會顯示；見[外掛](/advanced/plugins)。 |
| **插件範例** | 已安裝的[外掛](/advanced/plugins)提供、但沒有說明自己屬於哪一類的範例，每個套件包一個小標題。只有存在時才會顯示。 |
| **其他** | 沒有宣告主標籤、或宣告了這份清單不認識的主標籤的內建範例。測試會確保隨附的範例不會落到這裡。 |

其中七個實務範例 — **Inspect a dataset first**、**Early stopping, keep the best**、**Transfer with a frozen backbone**，以及**評估**底下四個用手寫數字的範例 — 用的是 `backend/data/samples/digits.csv`，也就是 UCI 的 1,797 張 8x8 手寫數字（CC BY 4.0），隨 CodefyUI 一起提供。每一個都能離線在 CPU 上幾秒內跑完，並固定了亂數種子。

外掛的範例只要宣告了主標籤，就會和內建範例一起排在那個主標籤底下；只有**課程**與**插件範例**會為每個套件包加一個小標題。

列出範例的四個地方都用這一套分組：空白畫布上的 overlay、沒有開啟分頁時顯示的[歡迎畫面](./tabs-persistence#the-welcome-screen)、**範例圖庫**，以及側邊欄的**範例**分頁。

在磁碟上，範例依主題資料夾分組：`Classical/`、`Data/`、`Deploy/`、`Diffusion/`、`Evaluation/`、`LLM/`、`Model_Architecture/`、`Others/`、`Papers/`、`Research/`、`RL/`、`RNN/`、`Training/`、`Transformer/`、`Usage_Example/` 與 `VLA/`。資料夾不等於主標籤 — 見[新增範例](#adding-an-example)。

所有列出的範例都可以離線直接執行，以下是例外。卡片不會寫這些：卡片只有一行，說的是這張圖在做什麼。範例執行前需要準備什麼，寫在它畫布上的[註記](./canvas-basics#notes)裡，貼在相關的節點旁邊 — 那裡才有空間說下載多大、套件包從哪裡來 — 也寫在下面這份清單裡：

- 兩個 CIFAR-10 訓練範例 **Train ResNet on CIFAR10** 與 **ResNet-18 CIFAR-10 baseline** 會在第一次執行時把 CIFAR-10（約 170 MB）下載到 `backend/data/`，之後就能離線執行。MNIST 隨 CodefyUI 一起提供，放在 `backend/data/MNIST/raw/`，所以 **Train CNN on MNIST** 與 **Train a Transformer on MNIST** 從第一次點下去就能離線執行。在[專案目錄](./project-directories)中，相對路徑的 `data_dir` 會指向專案的 `assets/data/`，所以在那裡這四個訓練範例第一次執行時都會下載各自的資料集。**Augment, then train** 讀的是同一份 MNIST，在專案目錄中第一次執行時同樣會下載 MNIST。
- **Inference CNN on MNIST** 會載入 `model_weights.pt`，這個檔案由 **Train CNN on MNIST** 寫到 `backend/data/models/`。CodefyUI 不附任何訓練好的權重，所以在剛安裝好的電腦上，它會停在 `ModelLoader`，直到訓練範例跑過一次為止。
- **Serve a classifier over HTTP** 會載入 `iris-classifier.safetensors`，這個檔案由 **Train and export a classifier** 寫到模型目錄（預設是 `backend/data/models/`）。請先把訓練範例跑過一次；沒有這個檔案時，`ModelLoader` 會停下並顯示「Weights file not found: &lt;path&gt;」。
- **Train a CNN on beans** 第一次執行時會從 Hugging Face Hub 下載 `AI-Lab-Makerere/beans`（1,295 張照片，約 170 MB）到 Hugging Face 的快取目錄。把圖裡三個 `HuggingFaceDataset` 節點指到別的 repo，下載的就會換成那一個。
- **Pretrain an LM on TinyStories** 第一次執行時會從 Hugging Face Hub 下載 TinyStories 語料與 gpt2 的 BPE ranks，並且需要一張有足夠空間容納 203,668,480 參數模型的 GPU。它的總覽註記會用英文與中文各寫一次這兩項需求；完整步驟、token 預算與記憶體調整選項則放在 graph 旁邊的 `README.md`（`examples/LLM/TrainCausalLM-TinyStories/`）。兩份下載都會被快取，之後再次執行也能離線完成。
- **Sentence similarity in zh-TW** 需要 `sentence-embeddings` 套件包。這是一次性的安裝，可以在套件中心（工具列 > 設定 > 選用套件與外掛）安裝，或執行 `cdui packs install sentence-embeddings`；graph 執行時不會自動下載該套件包。安裝後，這個範例就能離線在 CPU 上執行，幾秒鐘就結束。見[選用套件包](./optional-packs)。
- **Fully local RAG** 需要下載兩個項目：`rag` 套件包裡的 `qwen2.5-0.5b-instruct`，以及 `sentence-embeddings` 裡的 `multilingual-e5-small`，合計約 1.5 GB。安裝 `rag` 只會加入該套件包的 Python 套件，不含編碼器，所以還需要另外選取第二個項目。兩個項目都安裝後，文件、搜尋與生成都在本機處理，不會把資料傳送到外部。CPU 上大約每秒生成幾個 token，所以答案可能需要幾秒到幾十秒；這是依模型大小估算，不是實測值，使用 GPU 會快得多。
- **RAG with a chat API** 使用同一條檢索鏈，但最後一個節點是 `LLMChat`，所以它只需要 `multilingual-e5-small`，以及可接收 prompt 的服務。預設使用本機的 [Ollama](https://ollama.com)（先執行 `ollama pull qwen2.5:0.5b`），資料仍不會離開本機；把 `provider` 換成 hosted model 後，檢索到的內容會送給第三方，而且需要在環境變數中設定 key。
- **課程**底下有四個範例會在第一次執行時下載東西，每一個都會在自己畫布上的註記裡寫明：`deep` 套件包裡處理真實文字的範例 **Self-Attention 101**、**Multi-Head Causal Attention**、**C4-2 Self-attention on real text** 與 **C4-4 LLM inference pipeline** 會把 `cl100k_base` 的 BPE 表下載到快取目錄，之後每次執行都直接讀本機的快取。`foundations` 與 `deep` 套件包各有一個 MNIST 訓練範例，它們讀的是 CodefyUI 隨附的 MNIST，和內建範例一樣，只有在專案目錄中才會下載。這四個套件包的其他範例從第一次點下去就能離線執行。
- `eval` 套件包的範例 **A/B text classifier eval** 第一次執行時，會從 Hugging Face Hub 下載兩個情感分類模型（各約 18 MB）到 CodefyUI 自己的快取，並用 `transformers` 載入；安裝 `eval` 套件包時會一併裝上它。結果圖表與表格是 `stats` 套件包的節點。兩個載入節點都鎖定了 commit，所以之後的執行都讀快取，設定 `HF_HUB_OFFLINE=1` 時也一樣。
- `eval` 套件包的第二個範例 **VQA eval skeleton** 第一次執行時，會下載 ViLT 視覺問答模型 `dandelin/vilt-b32-finetuned-vqa`（約 470 MB）到同一個快取，回答關於套件包內附 7 張圖片的 12 個問題。它的載入節點同樣鎖定了 commit。[建立自己的評估流程](./eval-pipelines.md)說明了這兩個 eval 範例。

兩個 RL 架構 graph（**DQN on Atari pixels**、**PPO Robotics Controller**）會將合成的觀測張量（`TensorCreate`、`randn`）傳入網路，而不是連接真正的 gym 環境，所以不需要安裝 `ale-py`/`mujoco` — 若要用真實環境驅動它們，換成 `EnvWrapper` 節點即可。

## 主標籤、標籤與篩選 {/* #main-tags-tags-and-filters */}

除了主標籤，每個範例還帶有幾個**標籤**：像 `vision`、`tabular`、`rag` 或 `checkpoint` 這樣的小寫英文字，說明它用的資料、模型或示範的技巧。卡片上最多顯示兩個標籤，更多時加上 `+n`，取代原本顯示的資料夾名稱；完整的標籤列在範例圖庫的詳情窗格中。

- **新分頁上的範例集**（空白畫布 overlay 與歡迎畫面）上方有一排主標籤按鈕：**全部**，接著是每個至少有一個範例的主標籤，並附上數量。一次只能選一個，可以用方向鍵、Home 與 End 切換。第一次打開時停在**快速開始**，之後會停在你上次選的那一個（每個瀏覽器各自記住）。第一張卡片永遠是**空白圖**：在 overlay 上，它會關閉範例集，留下空白畫布給你；在歡迎畫面上，它會開一個新的空白分頁。
- **範例圖庫**有同樣的主標籤按鈕，但每次打開都停在**全部**。**依標籤篩選**會展開一排標籤，列出至少有三個範例使用的標籤，依使用次數排序；其餘的收在**更多標籤**裡。選了好幾個標籤時，只會留下同時帶有所有這些標籤的範例。詳情窗格中的每個標籤都是按鈕，點標籤或主標籤就會套用對應的篩選。沒有符合的範例時，圖庫會列出目前生效的篩選條件，並提供**清除篩選**。
- **側邊欄的範例分頁**同樣依主標籤、照相同順序分組，也有搜尋框，但沒有主標籤按鈕，也沒有標籤篩選。

範例圖庫與側邊欄的搜尋除了名稱與說明，也會比對標籤：用標籤的 id 或它在目前語言下顯示的名稱都找得到，所以搜尋 `checkpoint` 會找到所有帶這個標籤的範例。

## 載入範例 {/* #loading-an-example */}

- **在應用程式中** — 開一個新的空白分頁，範例集就會出現在畫布上。選取卡片後，graph 會載入該分頁並可立即**執行**。沒有開啟任何分頁時，在歡迎畫面上選取卡片，它會在新分頁中開啟。
- **從 CLI** — 把 `run_graph.py` 指向該 graph 的 JSON：

  ```bash
  cd backend
  python run_graph.py ../examples/Usage_Example/CNN-MNIST/TrainCNN-MNIST/graph.json
  ```

### 畫布不是空白時 {/* #when-the-canvas-is-not-empty */}

**範例圖庫**可從工具列的**範例圖庫**按鈕、空白畫布 overlay 與歡迎畫面上的**瀏覽全部範例**，以及側邊欄**範例**分頁底部的**瀏覽全部範例**按鈕開啟。圖庫依主標籤分組，主標籤按鈕與標籤篩選見[主標籤、標籤與篩選](#main-tags-tags-and-filters)；選取範例後，會顯示說明、它帶有的每個標籤、節點與連線數量，以及它是內建範例還是來自外掛。每個範例提供兩個動作：

- **在新分頁開啟**會保留目前的 graph，不做任何變動。雙擊卡片的效果相同。
- **插入目前畫布**會把範例加入正在編輯的 graph：系統會為插入的節點產生新的 id，並將它們放在目前 graph 的下方，因此不會覆蓋任何內容，而且只要復原一次就能移除。由較新版 CodefyUI 寫出的範例不會被插入（「沒有插入任何內容：…」）；請改用在新分頁開啟，它會以唯讀方式開啟。

沒有開啟任何分頁時，圖庫只提供**在新分頁開啟**。

在側邊欄的**範例**分頁中，點擊範例即可插入，也可以把它拖到畫布上想放的位置。它同樣依主標籤、照相同順序分組，搜尋也會比對標籤，但沒有主標籤按鈕或標籤篩選。

### 語言 {/* #languages */}

編輯器切到繁體中文後，每個內建範例的說明都是中文。**名稱兩種語言都保持英文**：範例的名稱就是你在這份文件、在磁碟上的 `examples/`、以及 `run_graph.py` 參數裡再次找到它的依據。搜尋兩種語言都吃，`attention` 與「注意力」會找到同一張圖。

還沒翻譯的範例（例如第三方外掛帶來的）會保留英文說明，不會變成空白。翻譯放在 `frontend/src/i18n/exampleLocales/`，以範例的路徑為 key。

## 新增範例 {/* #adding-an-example */}

一個範例就是一個放著 `graph.json` 的資料夾 — 內建範例放在 `examples/` 底下，外掛提供的則放在 `<套件包>/examples/` 底下。這個檔案裡有三件事決定它在範例集裡長什麼樣子。

**它會出現在哪一區。** 最上層可以放一個選用的 `gallery` 區塊：

```json
{
  "name": "Train and export a classifier",
  "description": "...",
  "gallery": { "main_tag": "deploy", "tags": ["tabular", "iris", "export"], "order": 10 },
  "nodes": [],
  "edges": []
}
```

- `main_tag` 只能是 `quickstart`、`training`、`llm`、`data`、`evaluation`、`research`、`deploy`、`architectures`、`concepts`、`papers` 與 `courses` 其中之一，這裡的順序就是範例集的順序；每一類放什麼，見本頁開頭的表格（`research` 顯示為**實驗**）。`plugin` 與 `other` 由伺服器自己指定，不能宣告。不在清單裡的值等於沒有宣告主標籤：內建範例會落到**其他**，外掛的範例則落到**插件範例**。
- `tags` 是一串自由填寫的字：小寫英文字母與數字，多個字之間用連字號連接（`transfer-learning`），每個 1 到 24 個字元，最多六個。不符合規則的項目只會刪掉那一項，重複的會被移除，第六個之後的會被忽略。不要重複主標籤，也不要拿其他主標籤的 id 當標籤。每個內建範例至少要有一個領域標籤（`vision`、`nlp`、`tabular`、`rl`、`diffusion`、`audio`、`timeseries` 或 `vla`）。
- `order` 是整數，在同一個主標籤內由小到大排序；在**模型架構**底下是同一個家族內，在**課程**與**插件範例**底下則是同一個套件包內。建議以十為間隔編號（`10`、`20`、`30`），之後才有空間把範例插到兩個範例之間；沒有寫 `order` 的範例排在有寫的後面，順序相同時依名稱排序。
- `family` 是**模型架構**底下的小標題，其他地方都不使用：上面那五個家族照列出的順序排在前面，其他字串接在後面依字母排序。

清單讀不懂的欄位，只有那一個欄位會變成空值，而不是讓整個範例集掛掉：不是整數的 `order` 只會讓範例排在有寫 `order` 的範例後面。

在主標籤出現之前寫的檔案，`gallery` 區塊裡只有 `"section": "papers"`（或其他主標籤 id），這樣的檔案仍然可以用：沒有可讀的 `main_tag` 時，認得的 `section` 會被當成主標籤。內建範例只用 `main_tag`，外掛的範例可以在方便時改過來；見[外掛](/advanced/plugins#examples-in-the-gallery)。

**標籤的顯示名稱。** 篩選列只列出至少有三個範例使用的標籤；比較少用的標籤收在**更多標籤**裡，搜尋時一樣找得到。標籤透過前端的 `examples.tag.<tag>` 翻譯 key 顯示名稱（`frontend/src/i18n/locales/en.ts` 與 `zh-TW.ts`），沒有翻譯 key 的標籤就照原樣顯示。主標籤的名稱放在 `examples.mainTag.<id>`。

範例所在的資料夾不等於它的主標籤。資料夾是路徑，而這份文件、翻譯表與 `run_graph.py` 的參數指的都是路徑，所以重新分組範例集時，磁碟上的東西不會動。每個內建範例的 `gallery` 區塊也列在 `backend/tests/test_examples_gallery_metadata.py` 中，這個測試讓兩邊保持一致，並禁止兩個範例佔用同一個主標籤、家族與 order 的位置。

**它可以一起帶上什麼。** 範例就是一張儲存好的 graph，所以儲存的 graph 能帶的東西它都能帶。**Five seeds, mean and spread** 與 **Learning rate x batch size** 在 `subgraphs` 裡帶著自己的[子圖](/advanced/subgraphs)定義，並把它們放了好幾次；**Compare two checkpoints** 打開時，`segmentGroups` 已經標好兩個段落比較，可直接在[教學檢視器](./teaching-inspector)中查看。

**卡片上會看到什麼。** `name` 是標籤，不是句子：最多五個英文單字、34 欄寬，而且不能和其他範例重複 — 卡片、側邊欄那一列、空白畫布 overlay 與歡迎畫面都只靠這一個名稱來分辨兩個範例。`description` 是名稱底下那一行，最多 40 欄寬，也就是二十個漢字，寫的是這張圖在做什麼、在解釋什麼。它不寫執行前的需求，也不寫成績：下載、GPU、套件包、API 金鑰、必須先跑過的另一個範例、跑完得到的準確率，這些全部改寫在畫布上的註記裡，貼在相關的節點旁邊；其中會決定範例跑不跑得起來的，也會列在本頁開頭的例外清單裡。這兩個上限就是卡片放得下的量：卡片寬 `13rem`，裡面的字級跟著根字級一起放大，所以螢幕愈寬只會讓標題變大、卡片不會跟著變寬，超出的部分就被截掉。用欄寬而不是字數計算：中文說明會夾雜英文節點名稱，而一個漢字佔兩欄。英文的兩個上限由 `backend/tests/test_example_descriptions.py` 把關，「不寫需求」這一條則由 `backend/tests/test_builtin_examples.py` 對兩個來源目錄裡的每個範例把關；中文的上限與這一條都由 `frontend/src/i18n/exampleLocales/zh-TW.test.ts` 把關 — 名稱不翻譯。

**比較長的說明放哪裡。** 這兩行放不下的內容就放到畫布上的[註記](./canvas-basics#notes)，貼在它所說明的節點旁邊。註記是 type 為 `note` 的節點：驗證會跳過它，執行不會走到它，卡片上的節點數量也不會把它算進去。每則註記在同一個註記裡寫兩次 — 先一段英文，空一行，再用繁體中文寫同一件事 — 這樣一則註記就兩種語言都能讀，也不必再多維護一份翻譯表。

## 適合的第一次執行 {/* #a-good-first-run */}

載入 **Train CNN on MNIST**，然後：

1. 先讀 `Start` 節點左邊那則註記 — 它會說這張圖在做什麼，以及跑完要看哪裡。
2. **錄製節點輸出**預設開啟，**在多次執行間保留權重**預設關閉 — 可在「設定」popover 中的**錄製與檢視**及**訓練行為**確認。
3. 點擊**執行**，並在**訓練**分頁觀看即時 loss 圖表。5 個 epoch 在 CPU 上大約一兩分鐘。
4. 點擊 `Inference` 節點，在 **[教學檢視器](./teaching-inspector)** 中查看送進去的內容（16 張測試影像的批次）與輸出的結果（每張影像 10 個 logits）。兩個 `Conv2d` 層在 `SequentialModel` 節點裡面；雙擊它即可在「模型架構」編輯器中看到。
5. 跑完之後看 **Test accuracy** 那個 `Print`，大約 0.99，算在訓練迴圈沒看過的 10,000 張測試影像上；再把 16 個預測數字和旁邊 `Visualize` 排出來的 16 張影像對照。
6. 想跨次執行持續訓練，就開啟**在多次執行間保留權重**。之後每次執行都會保留權重，下一次從那裡繼續（關閉時跑的那次沒有留下任何權重），**執行紀錄**也會註明結果包含先前的訓練。

訓練也會存下 `model_weights.pt`（在 `backend/data/models/` 底下）。之後載入 **Inference CNN on MNIST**；它會使用剛才訓練出的權重分類 `test_digit.png`（放在 `backend/data/images/` 底下的真實 MNIST 數字影像），並印出它讀到的數字。
