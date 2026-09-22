---
sidebar_position: 8
title: 節點參考
description: 所有內建節點 — 152 個節點涵蓋 16 大類別，從 CNN 與 Transformer 層到 RL、LLM、Diffusion 與傳統機器學習。
---

# 節點參考

CodefyUI 內建 **152 個節點**，涵蓋 **16 大類別**。已安裝的 [外掛包](/advanced/plugins) 與你自己的 [自訂節點](/advanced/custom-nodes) 會再加入更多。

:::tip
此表反映目前版本，但後端才是權威來源。請查看即時節點面板或 `GET /api/nodes`，確認目前安裝可用的節點。雙擊畫布可搜尋節點。
:::

| 類別 | 節點 | 數量 |
|----------|-------|------:|
| **CNN** | Conv2d、Conv1d、Conv2dExplicit、ConvTranspose2d、MaxPool2d、AvgPool2d、AdaptiveAvgPool2d、BatchNorm2d、Dropout、Activation | 10 |
| **RNN** | LSTM、GRU、RNNCell | 3 |
| **Transformer** | MultiHeadAttention、TransformerEncoder、TransformerDecoder、MoELayer | 4 |
| **RL** | DQN、PPO、EnvWrapper、RewardModel、KLDivergence、PolicyRollout、PPOClipObjective、GroupRelativeAdvantage、Discount、GridWorldEnv、PreferenceDataset、BradleyTerryLoss、BradleyTerryTrain | 13 |
| **資料 (Data)** | Dataset、ImageFolderDataset、DataLoader、DatasetBatch、Transform、HuggingFaceDataset、KaggleDataset、TensorInput、TextInput、CSVReader、ColumnSelector、RowSelector、Normalize、SyntheticDataset、SyntheticShapes、SyntheticSegmentation、SyntheticSequence、TrainTestSplit、ResizeTransform、ToTensorTransform、NormalizeTransform、RandomCrop、RandomHorizontalFlip、RandomRotation、ColorJitter、RandAugment、ComposeTransform | 27 |
| **資料流 (Data Flow)** | Map、Reduce、Switch | 3 |
| **訓練 (Training)** | Optimizer、Loss、TrainingLoop、EvaluateModel、LRScheduler、SequentialModel、BackwardOnce | 7 |
| **IO** | ImageReader、ImageWriter、ImageBatchReader、FileReader、CheckpointSaver、CheckpointLoader、ModelLoader、ModelSaver、Inference、GraphInput、GraphOutput、VideoLoad、VideoWrite | 13 |
| **控制 (Control)** | Start | 1 |
| **工具 (Utility)** | Print、Reshape、Concat、Flatten、Linear、Visualize、Embedding、PythonScript、ScatterPlot2D、DecisionBoundary | 10 |
| **正規化 (Normalization)** | BatchNorm1d、LayerNorm、GroupNorm、InstanceNorm2d | 4 |
| **張量運算 (Tensor Operations)** | Add、MatMul、Mean、Multiply、ScalarMultiply、Permute、Softmax、Argmax、Split、Squeeze、Stack、TensorCreate、Unsqueeze、MaskedFill | 14 |
| **LLM** | LLMChat、Tokenizer、WordVector、TextEmbedding、EmbeddingScatter、CosineSimilarity、AttentionMask、AttentionHeatmap、PositionalEncoding、CausalLMModel、LMCrossEntropyLoss、LMTokenizer、TextCorpusDataset、LMTokenizedDataset、DataMixDataset、PerplexityEvaluate、TextGenerate、DocumentLoader、TextChunker、VectorStore、Retriever、PromptBuilder、HFTextGenerate | 23 |
| **傳統機器學習 (Classical)** | KNN、LinearRegression、LogisticRegression、DecisionTreeClassifier、RandomForestClassifier、SVMClassifier、MLPClassifier、Accuracy | 8 |
| **Diffusion** | Upsample、TimestepEmbedding、Lerp、GaussianNoise、DDPMSampler、DiffusionUNet、DiffusionTrainingLoop | 7 |
| **VLA** | VLAModel、VLARollout、VLAActionEval、PushWorldEnv、PushWorldDemos | 5 |

## 重點節點 {/* #notable-nodes */}

- **`Start`**（控制）— 定義執行進入點。每個可執行的圖都需要一個；請參閱[你的第一個圖](./first-graph)。
- **`TensorInput`**（資料）— 提供內嵌格狀編輯器，用來輸入明確的張量值。[教學檢視器](./teaching-inspector)範例會以此節點作為輸入。
- **變換鏈**（資料）— 九個節點會組成 `transforms.Compose`：`ResizeTransform` 將圖片縮放為正方形；`ToTensorTransform` 將 PIL 圖片轉為 `[0, 1]` 張量；`NormalizeTransform` 套用逐通道正規化並提供預設組合；`RandomCrop`、`RandomHorizontalFlip`、`RandomRotation`、`ColorJitter` 與 `RandAugment` 套用資料增強；`ComposeTransform` 則合併兩條分別建構的鏈。參數與順序規則請參閱[資料與資料增強](./data-augmentation)。
- **`ImageFolderDataset`**（資料）— 依 torchvision `ImageFolder` 預期的目錄結構，從每個類別各自的目錄載入圖片。
- **`TrainingLoop`**（訓練）— 執行訓練，並將即時 loss 圖表傳至結果面板。它的選項與記錄的指標序列列在[訓練迴圈與 loss 圖表](./running-graphs#training-loops-and-loss-charts)。其**進階**區段包含記憶體控制項 `precision` 與 `accumulate_steps`；請參閱[訓練記憶體](/advanced/training-memory)。
- **`SequentialModel`**（訓練）— 以單一節點表示層堆疊。雙擊節點可開啟**模型架構**編輯器。編輯器提供依類別分組且可搜尋的層級面板，以及用來連接各層的畫布。架構必須恰好包含一個 `Input` 節點與一個 `Output` 節點，合併層則提供連接埠清單編輯器。控制項包括**吸附 ON/OFF**、由上至下的**自動排列**、JSON **匯入**與**匯出**，以及**套用**。驗證會拒絕循環。**匯出**會把架構下載為 `model_architecture.json`。**匯入**可讀取這種檔案或畫布圖的 `.json`：檔案中的層節點會轉成 `Input` → 各層 → `Output` 的鏈；若沒有層節點，則載入其中的 `SequentialModel`（有多個時，會以**選擇要匯入的 SequentialModel**詢問要匯入哪一個）。其他檔案會失敗並顯示**匯入失敗：此檔案沒有層級或 SequentialModel 節點。** 匯入只會取代編輯器目前顯示的內容；點擊**套用**後節點才會變更。
- **`EmbeddingScatter`**（LLM）— 使用 PCA 或 t-SNE 將嵌入向量投影至 2D，並顯示可縮放的散佈圖。
- **`AttentionHeatmap`**（LLM）— 傳遞 attention weight，可選擇其中一個 head，並傳遞選填的 token 標籤。節點卡片會將 weight 顯示為 heatmap，另提供完整尺寸的檢視器。
- **由套件包提供的後端**（LLM）— `WordVector` 的 `glove-50d` 與句子編碼器選項，以及整個 `TextEmbedding`，都會載入套件中心所安裝的模型。未安裝模型的選項會顯示為灰色。若缺少的選項是圖中已儲存的目前值，該選項仍可選取並會顯示警告。執行圖時不會下載這些模型。套件包大小、檔案位置與編碼器選擇方式請參閱[選用套件包](./optional-packs)。
- **語言模型鏈**（LLM）— 八個節點組成 GPT 式解碼器的訓練路徑：`TextCorpusDataset` 從 Hugging Face Hub 或本機 `.txt` 讀取文字列；`DataMixDataset` 將 2 到 6 個這類語料混合成一個資料集，可依 `weights` 的比例、以 `seed` 固定的順序抽取文字列（`interleave`），或依序讀完各個語料作為課程（`concat`）；`LMTokenizedDataset` 建立固定長度的下一詞元區塊；`DataLoader` 將批次供應給 `TrainingLoop`；`CausalLMModel` 提供模型；`LMCrossEntropyLoss` 提供 loss；`LMTokenizer` 則提供共用的 tokenizer。`PerplexityEvaluate` 會評估保留資料切分，`TextGenerate` 則從已訓練權重取樣文字。`CausalLMModel` 的預設值會產生 203,668,480 個參數；若要在筆電上訓練，請降低 `d_model` 與 `n_layers`。請參閱[範例集](./examples-gallery)中的 **Pretrain an LM on TinyStories**。
- **RAG 鏈**（LLM）— 七個節點使用指定文件作為回答脈絡：`DocumentLoader` 將目錄中的每個 `.md` 與 `.txt` 讀取為 `{text, source}`；`TextChunker` 建立可嵌入的切塊，並保留來源名稱與字元位移；`TextEmbedding` 建立向量；`VectorStore` 將一個 `[N, D]` 矩陣與切塊文字一併儲存，並以一次矩陣乘法搜尋；`Retriever` 回傳最接近的 `top_k` 個切塊與分數；`PromptBuilder` 將切塊及問題插入限定脈絡的範本；`HFTextGenerate` 則在本機執行 Qwen2.5-0.5B-Instruct。你可以用 `LLMChat` 取代 `HFTextGenerate`，改用 Ollama 或託管提供者。`TextEmbedding` 需要 `sentence-embeddings` 套件包，`HFTextGenerate` 需要 `rag` 套件包。請參閱[範例集](./examples-gallery)中的 **Fully local RAG** 與 **RAG with a chat API**，下載內容則請參閱[選用套件包](./optional-packs)。
- **VLA 鏈**（VLA）— VLA 類別包含五個專用節點。訓練與評估圖會連接 `PushWorldEnv`、`PushWorldDemos`、`VLAModel` 與 `TrainingLoop`，再使用 `VLAActionEval` 和 `VLARollout`。`PushWorldEnv` 以 torch 實作受語言條件控制的 2D 推動任務；干擾圓盤會使策略必須讀取指令才能識別目標。`PushWorldDemos` 會建立 scripted-expert 行為複製樣本與保留資料切分。`VLAModel` 結合視覺前端、位元組層級指令嵌入、Transformer 主幹與分塊動作專家。`VLAActionEval` 會測量相對於 expert 的保留資料開迴路 MSE，`VLARollout` 則以滾動時域執行測量閉迴路成功率。[**Train a VLA on PushWorld** 的 README](https://github.com/CodefyUI/CodefyUI/blob/main/examples/VLA/TrainVLA-PushWorld/README.md)建議使用 CUDA GPU，並記錄 RTX 4080 約需執行 56 分鐘。
- **RL 節點**（RL）— `GridWorldEnv` 提供不依賴 Gymnasium 的 gridworld。`PolicyRollout` 會執行策略 N 個回合，並記錄狀態、動作、獎勵與 logits。`Discount` 計算 return，`GroupRelativeAdvantage` 計算 GRPO 的群組平均 baseline，`PPOClipObjective` 則計算 PPO 的 clipped surrogate，並回傳指出哪些樣本使用 clipped branch 的 mask。`RewardModel` 會為每個 sequence 產生一個純量分數。`PreferenceDataset` 建立訓練與保留 preference pair；`BradleyTerryLoss` 計算 preference loss；`BradleyTerryTrain` 訓練 reward head，並回報訓練與保留資料的 accuracy，以呈現捷徑學習。`KLDivergence` 計算 policy 相對於 reference 的正則化項。`DQN`、`PPO` 與 `EnvWrapper` 會與 Gymnasium 整合。請參閱[範例集](./examples-gallery)中的 **RLHF reward and KL terms**。
- **Diffusion 鏈**（Diffusion）— `GaussianNoise` 建立帶 seed 的 noise，並可符合上游張量的 shape。`DiffusionUNet` 提供將 `(x, t)` 映射至 predicted noise 的 toy U-Net，`DiffusionTrainingLoop` 則訓練此模型。`DDPMSampler` 在單一節點內執行 reverse process，使圖維持無環；其詳細 trace 會記錄 trajectory snapshot。`TimestepEmbedding`、`Upsample` 與 `Lerp` 可用來明確建構相同運算。Diffusion 範例包括 **Forward diffusion on a digit**（`GaussianNoise` 與 `Lerp`），以及 **Toy reverse diffusion sampling** 和 **Mini U-Net node**（`GaussianNoise` → `DiffusionUNet` → `DDPMSampler`）。請參閱[範例集](./examples-gallery)。
- **`ModelSaver` / `ModelLoader`**（IO）— 寫入與讀取模型檔。預設的 `state_dict` 模式會儲存張量，且不要求載入任何類別。`full_model` 模式會儲存以 pickle 序列化的模組，並使用受限解序列化器載入；此解序列化器允許 torch 與 CodefyUI 的 layer class，以及 CodefyUI Transformer layer 所儲存的兩個 torch activation function。模式選擇與載入需求請參閱[儲存與載入模型](./model-files)。
- **`VideoWrite` / `VideoLoad`**（IO）— `VideoWrite` 將 `(T, C, H, W)` 或 `(T, H, W, C)` 的影格張量編碼後寫入媒體資料夾（`backend/data/media`，專案目錄模式下為 `assets/media`）。`format` 為 `auto` 時，若 `PATH` 中有 `ffmpeg` 執行檔就寫出 mp4，否則透過 Pillow 寫出 gif；選 `mp4` 則必須有 ffmpeg。相同的 `filename` 會覆寫先前的影片。執行紀錄與檢視器會直接播放 mp4，gif 則以圖片顯示；`preview` 輸出是中間影格的 PNG。`VideoLoad` 會將影片解碼為數值在 `[0, 1]` 的 `(T, 3, H, W)` 影格，並輸出 `fps` 與 `num_frames`；相對的 `path` 會從媒體資料夾讀取。mp4 與 webm 需要 `ffmpeg` 與 `ffprobe` 執行檔，gif 只需要 Pillow；`max_frames` 與 `stride` 可限制保留在記憶體中的影格數。
- **`Switch`**（資料流）— 使用條件式路由，使只有一條分支執行。

## 連接埠資料型別 {/* #port-data-types */}

連線是有型別的。內建的資料型別包括：**Tensor、Model、Dataset、DataLoader、Optimizer、Loss、Scalar、String、Image、List、Transform、Any、Trigger**。`Trigger` 型別正是 `Start` 節點所發出、用來驅動執行順序的型別，而 `Transform` 則是變換鏈節點之間互相傳遞的型別。
