---
sidebar_position: 8
title: 節點參考
description: 所有內建節點 — 128 個節點涵蓋 15 大類別，從 CNN 與 Transformer 層到 RL、LLM、Diffusion 與傳統機器學習。
---

# 節點參考

CodefyUI 內建 **128 個節點**，涵蓋 **15 大類別**。已安裝的 [外掛包](/advanced/plugins) 與你自己的 [自訂節點](/advanced/custom-nodes) 會再加入更多。

:::tip
這份清單在撰寫當下是準確的來源依據，但後端才是權威：即時的節點面板與 `GET /api/nodes` 永遠精確反映你的安裝實際有哪些節點。使用應用程式內的搜尋（在畫布上雙擊）可以快速找到節點。
:::

| 類別 | 節點 | 數量 |
|----------|-------|------:|
| **CNN** | Conv2d、Conv1d、Conv2dExplicit、Conv2dKernel、ConvTranspose2d、MaxPool2d、AvgPool2d、AdaptiveAvgPool2d、BatchNorm2d、Dropout、Activation | 11 |
| **RNN** | LSTM、GRU、RNNCell | 3 |
| **Transformer** | MultiHeadAttention、TransformerEncoder、TransformerDecoder、MoELayer | 4 |
| **RL** | DQN、PPO、EnvWrapper、RewardModel、KLDivergence | 5 |
| **資料 (Data)** | Dataset、ImageFolderDataset、DataLoader、DatasetBatch、Transform、HuggingFaceDataset、KaggleDataset、TensorInput、TextInput、CSVReader、ColumnSelector、RowSelector、Normalize、SyntheticDataset、SyntheticShapes、SyntheticSegmentation、TrainTestSplit、ResizeTransform、ToTensorTransform、NormalizeTransform、RandomCrop、RandomHorizontalFlip、RandomRotation、ColorJitter、RandAugment、ComposeTransform | 26 |
| **資料流 (Data Flow)** | Map、Reduce、Switch | 3 |
| **訓練 (Training)** | Optimizer、Loss、TrainingLoop、EvaluateModel、LRScheduler、SequentialModel、BackwardOnce | 7 |
| **IO** | ImageReader、ImageWriter、ImageBatchReader、FileReader、CheckpointSaver、CheckpointLoader、ModelLoader、ModelSaver、Inference、GraphInput、GraphOutput | 11 |
| **控制 (Control)** | Start | 1 |
| **工具 (Utility)** | Print、Reshape、Concat、Flatten、Linear、Visualize、Embedding、PythonScript、ScatterPlot2D、DecisionBoundary | 10 |
| **正規化 (Normalization)** | BatchNorm1d、LayerNorm、GroupNorm、InstanceNorm2d | 4 |
| **張量運算 (Tensor Operations)** | Add、MatMul、Mean、Multiply、ScalarMultiply、Permute、Softmax、Argmax、Split、Squeeze、Stack、TensorCreate、Unsqueeze | 13 |
| **LLM** | LLMChat、Tokenizer、WordVector、EmbeddingScatter、CosineSimilarity、AttentionMask、AttentionHeatmap、PositionalEncoding、CausalLMModel、LMCrossEntropyLoss、LMTokenizer、TextCorpusDataset、LMTokenizedDataset、PerplexityEvaluate、TextGenerate | 15 |
| **傳統機器學習 (Classical)** | KNN、LinearRegression、LogisticRegression、DecisionTreeClassifier、RandomForestClassifier、SVMClassifier、MLPClassifier、Accuracy | 8 |
| **Diffusion** | Upsample、TimestepEmbedding、Lerp、GaussianNoise、DDPMSampler、DiffusionUNet、DiffusionTrainingLoop | 7 |

## 重點節點

- **`Start`**（控制）— 執行的進入點。每個可執行的圖都需要一個；見 [你的第一個圖](./first-graph)。
- **`TensorInput`**（資料）— 一個內嵌格子編輯器，用來手動把明確指定的張量餵進管線；是 [教學檢視器](./teaching-inspector) 範例的骨幹。
- **變換鏈**（資料）— 九個可以互相串接、組出 `transforms.Compose` 的節點：`ResizeTransform`（縮放成正方形）、`ToTensorTransform`（PIL 影像轉成 `[0, 1]` 張量）、`NormalizeTransform`（每個通道做正規化，並內建預設組合）、`RandomCrop`、`RandomHorizontalFlip`、`RandomRotation`、`ColorJitter` 與 `RandAugment`（做資料增強的步驟），以及 `ComposeTransform`（合併兩條分開建立的鏈）。參數與順序規則見[資料與資料增強](./data-augmentation)。
- **`ImageFolderDataset`**（資料）— 依 torchvision `ImageFolder` 預期的結構，從「一個類別一個資料夾」載入你自己的影像。
- **`TrainingLoop`**（訓練）— 驅動訓練，並在結果面板發出即時 loss 圖表。它的進階區塊放著記憶體相關的開關（`precision`、`accumulate_steps`），詳見[訓練記憶體](/advanced/training-memory)。
- **`EmbeddingScatter`**（LLM）— 把 embedding 投影到 2D（PCA / t-SNE），畫成可縮放的散佈圖。
- **`AttentionHeatmap`**（LLM）— 把 attention 矩陣渲染成影像。
- **語言模型鏈**（LLM）— 七個節點，讓你直接在畫布上從零開始預訓練一個 GPT 式的 decoder：`TextCorpusDataset`（從 Hugging Face Hub 或本機 `.txt` 讀入原始文字列）→ `LMTokenizedDataset`（把它們打包成固定長度的下一個 token 預測區塊）→ `DataLoader` → `TrainingLoop`，模型用 `CausalLMModel`、損失函數用 `LMCrossEntropyLoss`，並由 `LMTokenizer` 提供同一個 tokenizer 物件給每個需要它的節點。訓練完成後，`PerplexityEvaluate` 在保留下來的驗證集上評分，`TextGenerate` 則從訓練好的權重採樣文字。`CausalLMModel` 的預設值描述的是一個 203,668,480 參數的模型；把 `d_model` 與 `n_layers` 調小，就能得到一堂課內筆電也訓練得完的規模。**Train a Causal LM on TinyStories** 範例把整條鏈都接好了 — 見[範例集](./examples-gallery)。
- **`Switch`**（資料流）— 條件式路由，讓只有一條分支會執行。

## 連接埠資料型別

連線是有型別的。內建的資料型別包括：**Tensor、Model、Dataset、DataLoader、Optimizer、Loss、Scalar、String、Image、List、Transform、Any、Trigger**。`Trigger` 型別正是 `Start` 節點所發出、用來驅動執行順序的型別，而 `Transform` 則是變換鏈節點之間互相傳遞的型別。
