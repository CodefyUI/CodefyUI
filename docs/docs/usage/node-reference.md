---
sidebar_position: 8
title: Node Reference
description: Every built-in node — 146 nodes across 16 categories, from CNN and Transformer layers to RL, LLM, Diffusion, and classical ML.
---

# Node Reference

CodefyUI ships **146 built-in nodes** across **16 categories**. Installed [plugin packs](/advanced/plugins) and your own [custom nodes](/advanced/custom-nodes) add more.

:::tip
This list is the source of truth at the time of writing, but the backend is authoritative: the live palette and `GET /api/nodes` always reflect exactly what your install has. Use the in-app search (double-click the canvas) to find a node fast.
:::

| Category | Nodes | Count |
|----------|-------|------:|
| **CNN** | Conv2d, Conv1d, Conv2dExplicit, ConvTranspose2d, MaxPool2d, AvgPool2d, AdaptiveAvgPool2d, BatchNorm2d, Dropout, Activation | 10 |
| **RNN** | LSTM, GRU, RNNCell | 3 |
| **Transformer** | MultiHeadAttention, TransformerEncoder, TransformerDecoder, MoELayer | 4 |
| **RL** | DQN, PPO, EnvWrapper, RewardModel, KLDivergence, PolicyRollout, PPOClipObjective, GroupRelativeAdvantage, Discount, GridWorldEnv, PreferenceDataset, BradleyTerryLoss, BradleyTerryTrain | 13 |
| **Data** | Dataset, ImageFolderDataset, DataLoader, DatasetBatch, Transform, HuggingFaceDataset, KaggleDataset, TensorInput, TextInput, CSVReader, ColumnSelector, RowSelector, Normalize, SyntheticDataset, SyntheticShapes, SyntheticSegmentation, SyntheticSequence, TrainTestSplit, ResizeTransform, ToTensorTransform, NormalizeTransform, RandomCrop, RandomHorizontalFlip, RandomRotation, ColorJitter, RandAugment, ComposeTransform | 27 |
| **Data Flow** | Map, Reduce, Switch | 3 |
| **Training** | Optimizer, Loss, TrainingLoop, EvaluateModel, LRScheduler, SequentialModel, BackwardOnce | 7 |
| **IO** | ImageReader, ImageWriter, ImageBatchReader, FileReader, CheckpointSaver, CheckpointLoader, ModelLoader, ModelSaver, Inference, GraphInput, GraphOutput, VideoLoad, VideoWrite | 13 |
| **Control** | Start | 1 |
| **Utility** | Print, Reshape, Concat, Flatten, Linear, Visualize, Embedding, PythonScript, ScatterPlot2D, DecisionBoundary | 10 |
| **Normalization** | BatchNorm1d, LayerNorm, GroupNorm, InstanceNorm2d | 4 |
| **Tensor Operations** | Add, MatMul, Mean, Multiply, ScalarMultiply, Permute, Softmax, Argmax, Split, Squeeze, Stack, TensorCreate, Unsqueeze, MaskedFill | 14 |
| **LLM** | LLMChat, Tokenizer, WordVector, TextEmbedding, EmbeddingScatter, CosineSimilarity, AttentionMask, AttentionHeatmap, PositionalEncoding, CausalLMModel, LMCrossEntropyLoss, LMTokenizer, TextCorpusDataset, LMTokenizedDataset, DataMixDataset, PerplexityEvaluate, TextGenerate | 17 |
| **Classical** | KNN, LinearRegression, LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, SVMClassifier, MLPClassifier, Accuracy | 8 |
| **Diffusion** | Upsample, TimestepEmbedding, Lerp, GaussianNoise, DDPMSampler, DiffusionUNet, DiffusionTrainingLoop | 7 |
| **VLA** | VLAModel, VLARollout, VLAActionEval, PushWorldEnv, PushWorldDemos | 5 |

## Notable nodes

- **`Start`** (Control) — the execution entry point. Every runnable graph needs one; see [Your First Graph](./first-graph).
- **`TensorInput`** (Data) — an inline grid editor to hand-feed explicit tensors into a pipeline; the backbone of [Teaching Inspector](./teaching-inspector) demos.
- **The transform chain** (Data) — nine nodes that wire into one another to build a `transforms.Compose`: `ResizeTransform` (resize to a square), `ToTensorTransform` (PIL image to `[0, 1]` tensor), `NormalizeTransform` (per-channel normalisation, with presets), `RandomCrop`, `RandomHorizontalFlip`, `RandomRotation`, `ColorJitter` and `RandAugment` (the augmentation steps), and `ComposeTransform` (join two chains built separately). Parameters and ordering rules in [Data and Augmentation](./data-augmentation).
- **`ImageFolderDataset`** (Data) — loads your own images from one folder per class, in the layout torchvision's `ImageFolder` expects.
- **`TrainingLoop`** (Training) — drives training and emits the live loss chart in the Results panel. Its Advanced section holds the memory levers (`precision`, `accumulate_steps`); see [Training Memory](/advanced/training-memory).
- **`EmbeddingScatter`** (LLM) — projects embeddings to 2D (PCA / t-SNE) for a zoomable scatter plot.
- **`AttentionHeatmap`** (LLM) — renders attention matrices as images.
- **Pack-backed backends** (LLM) — `WordVector`'s `glove-50d` and sentence-encoder options, and the whole of `TextEmbedding`, read models that the Package Center downloaded; the options that are missing one are greyed out, and a run never downloads anything itself. What each pack costs, where the files land and which encoder to pick: [Optional Packs](./optional-packs).
- **The language-model chain** (LLM) — seven nodes that pretrain a GPT-style decoder from scratch on the canvas: `TextCorpusDataset` (raw text rows from the Hugging Face Hub or a local `.txt`) → `LMTokenizedDataset` (packs them into fixed-length next-token blocks) → `DataLoader` → `TrainingLoop`, with `CausalLMModel` as the model, `LMCrossEntropyLoss` as the loss, and `LMTokenizer` supplying one tokenizer object to every node that needs one. Afterwards `PerplexityEvaluate` scores a held-out split and `TextGenerate` samples text from the trained weights. `CausalLMModel`'s defaults describe a 203,668,480-parameter model; shrink `d_model` and `n_layers` for something a laptop can train in a lesson. The **Train a Causal LM on TinyStories** example wires the whole chain up — see [Examples Gallery](./examples-gallery).
- **`ModelSaver` / `ModelLoader`** (IO) — write and read model files. Each has a `state_dict` mode (tensors; the default and the one with no conditions attached) and a `full_model` mode (the pickled module itself, read under a restricted unpickler that admits torch's and CodefyUI's own layer classes, the two torch activation functions its transformer layers store, and nothing else). Which to pick, and what a `full_model` file needs in order to load: [Saving and Loading Models](./model-files).
- **`Switch`** (Data Flow) — conditional routing so only one branch executes.

## Port data types

Edges are typed. The built-in data types include: **Tensor, Model, Dataset, DataLoader, Optimizer, Loss, Scalar, String, Image, List, Transform, Any, Trigger**. The `Trigger` type is what `Start` nodes emit to drive execution order, and `Transform` is what the transform-chain nodes pass to each other.
