---
sidebar_position: 8
title: Node Reference
description: Every built-in node — 95 nodes across 15 categories, from CNN and Transformer layers to RL, LLM, Diffusion, and classical ML.
---

# Node Reference

CodefyUI ships **95 built-in nodes** across **15 categories**. Installed [plugin packs](/advanced/plugins) and your own [custom nodes](/advanced/custom-nodes) add more.

:::tip
This list is the source of truth at the time of writing, but the backend is authoritative: the live palette and `GET /api/nodes` always reflect exactly what your install has. Use the in-app search (double-click the canvas) to find a node fast.
:::

| Category | Nodes | Count |
|----------|-------|------:|
| **CNN** | Conv2d, Conv1d, ConvTranspose2d, MaxPool2d, AvgPool2d, AdaptiveAvgPool2d, BatchNorm2d, Dropout, Activation | 9 |
| **RNN** | LSTM, GRU, RNNCell | 3 |
| **Transformer** | MultiHeadAttention, TransformerEncoder, TransformerDecoder, MoELayer | 4 |
| **RL** | DQN, PPO, EnvWrapper, RewardModel, KLDivergence | 5 |
| **Data** | Dataset, DataLoader, Transform, HuggingFaceDataset, KaggleDataset, TensorInput, TextInput, CSVReader, ColumnSelector, Normalize, SyntheticDataset, TrainTestSplit | 12 |
| **Data Flow** | Map, Reduce, Switch | 3 |
| **Training** | Optimizer, Loss, TrainingLoop, LRScheduler, SequentialModel, BackwardOnce | 6 |
| **IO** | ImageReader, ImageWriter, ImageBatchReader, FileReader, CheckpointSaver, CheckpointLoader, ModelLoader, ModelSaver, Inference | 9 |
| **Control** | Start | 1 |
| **Utility** | Print, Reshape, Concat, Flatten, Linear, Visualize, Embedding | 7 |
| **Normalization** | BatchNorm1d, LayerNorm, GroupNorm, InstanceNorm2d | 4 |
| **Tensor Operations** | Add, MatMul, Mean, Multiply, Permute, Softmax, Split, Squeeze, Stack, TensorCreate, Unsqueeze | 11 |
| **LLM** | LLMChat, Tokenizer, WordVector, EmbeddingScatter, CosineSimilarity, AttentionMask, AttentionHeatmap, PositionalEncoding | 8 |
| **Classical** | KNN, LinearRegression, LogisticRegression, DecisionTreeClassifier, SVMClassifier, MLPClassifier, Accuracy | 7 |
| **Diffusion** | Upsample, TimestepEmbedding, Lerp, GaussianNoise, DDPMSampler, DiffusionUNet | 6 |

## Notable nodes

- **`Start`** (Control) — the execution entry point. Every runnable graph needs one; see [Your First Graph](./first-graph).
- **`TensorInput`** (Data) — an inline grid editor to hand-feed explicit tensors into a pipeline; the backbone of [Teaching Inspector](./teaching-inspector) demos.
- **`TrainingLoop`** (Training) — drives training and emits the live loss chart in the Results panel.
- **`EmbeddingScatter`** (LLM) — projects embeddings to 2D (PCA / t-SNE) for a zoomable scatter plot.
- **`AttentionHeatmap`** (LLM) — renders attention matrices as images.
- **`Switch`** (Data Flow) — conditional routing so only one branch executes.

## Port data types

Edges are typed. The built-in data types include: **Tensor, Model, Dataset, DataLoader, Optimizer, Loss, Scalar, String, Image, List, Any, Trigger**. The `Trigger` type is what `Start` nodes emit to drive execution order.
