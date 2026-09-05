---
sidebar_position: 8
title: Node Reference
description: Every built-in node — 152 nodes across 16 categories, from CNN and Transformer layers to RL, LLM, Diffusion, and classical ML.
---

# Node Reference

CodefyUI ships **152 built-in nodes** across **16 categories**. Installed [plugin packs](/advanced/plugins) and your own [custom nodes](/advanced/custom-nodes) add more.

:::tip
This table reflects the current release, but the backend is authoritative. Check the live node palette or `GET /api/nodes` for the nodes available in your installation. Double-click the canvas to search for a node.
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
| **LLM** | LLMChat, Tokenizer, WordVector, TextEmbedding, EmbeddingScatter, CosineSimilarity, AttentionMask, AttentionHeatmap, PositionalEncoding, CausalLMModel, LMCrossEntropyLoss, LMTokenizer, TextCorpusDataset, LMTokenizedDataset, DataMixDataset, PerplexityEvaluate, TextGenerate, DocumentLoader, TextChunker, VectorStore, Retriever, PromptBuilder, HFTextGenerate | 23 |
| **Classical** | KNN, LinearRegression, LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, SVMClassifier, MLPClassifier, Accuracy | 8 |
| **Diffusion** | Upsample, TimestepEmbedding, Lerp, GaussianNoise, DDPMSampler, DiffusionUNet, DiffusionTrainingLoop | 7 |
| **VLA** | VLAModel, VLARollout, VLAActionEval, PushWorldEnv, PushWorldDemos | 5 |

## Notable nodes

- **`Start`** (Control) — defines the execution entry point. Every runnable graph requires one; see [Your First Graph](./first-graph).
- **`TensorInput`** (Data) — provides an inline grid for entering explicit tensor values. [Teaching Inspector](./teaching-inspector) examples use it as their input node.
- **The transform chain** (Data) — nine nodes compose a `transforms.Compose`: `ResizeTransform` resizes to a square; `ToTensorTransform` converts a PIL image to a `[0, 1]` tensor; `NormalizeTransform` applies per-channel normalisation and provides presets; `RandomCrop`, `RandomHorizontalFlip`, `RandomRotation`, `ColorJitter`, and `RandAugment` apply augmentations; and `ComposeTransform` joins two separately constructed chains. See [Data and Augmentation](./data-augmentation) for parameters and ordering rules.
- **`ImageFolderDataset`** (Data) — loads images from one directory per class, using the directory structure expected by torchvision's `ImageFolder`.
- **`TrainingLoop`** (Training) — runs training and sends the live loss chart to the Results panel. Its **Advanced** section contains the memory controls `precision` and `accumulate_steps`; see [Training Memory](/advanced/training-memory).
- **`SequentialModel`** (Training) — represents a layer stack in one node. Double-click it to open the **Model Architecture Editor**. The editor provides a searchable layer palette grouped by category and a canvas for connecting layers. An architecture must have exactly one `Input` node and one `Output` node, and merge layers have a port-list editor. The controls include **Snap: ON/OFF**, top-to-bottom **Auto Layout**, JSON **Import** and **Export**, and **Apply**. Validation rejects cycles.
- **`EmbeddingScatter`** (LLM) — projects embeddings to 2D with PCA or t-SNE and displays a zoomable scatter plot.
- **`AttentionHeatmap`** (LLM) — forwards attention weights, optionally selecting one head, and forwards optional token labels. The node card displays the weights as a heatmap, and a full-size viewer is available.
- **Pack-backed backends** (LLM) — `WordVector`'s `glove-50d` and sentence-encoder options, and all of `TextEmbedding`, load models installed by the Package Center. Options without an installed model are greyed out. If a missing option is the graph's saved value, it remains selectable and displays a warning. Graph execution does not download these models. See [Optional Packs](./optional-packs) for pack sizes, file locations, and encoder guidance.
- **The language-model chain** (LLM) — seven nodes form a training path for a GPT-style decoder: `TextCorpusDataset` reads text rows from the Hugging Face Hub or a local `.txt`; `LMTokenizedDataset` creates fixed-length next-token blocks; `DataLoader` supplies batches to `TrainingLoop`; `CausalLMModel` provides the model; `LMCrossEntropyLoss` provides the loss; and `LMTokenizer` supplies the shared tokenizer. `PerplexityEvaluate` scores a held-out split, and `TextGenerate` samples from the trained weights. The default `CausalLMModel` has 203,668,480 parameters; reduce `d_model` and `n_layers` for laptop-scale training. See the **Train a Causal LM on TinyStories** example in [Examples Gallery](./examples-gallery).
- **The RAG chain** (LLM) — seven nodes use supplied documents as answer context: `DocumentLoader` reads each `.md` and `.txt` in a directory as `{text, source}`; `TextChunker` creates embeddable chunks with source names and character offsets; `TextEmbedding` creates vectors; `VectorStore` stores one `[N, D]` matrix with the chunk text and searches it with one matrix multiplication; `Retriever` returns the nearest `top_k` chunks and their scores; `PromptBuilder` inserts the chunks and question into a context-only template; and `HFTextGenerate` runs Qwen2.5-0.5B-Instruct locally. `LLMChat` can replace `HFTextGenerate` to use Ollama or a hosted provider. `TextEmbedding` requires the `sentence-embeddings` pack, and `HFTextGenerate` requires the `rag` pack. See **RAG, fully local** and **RAG with a chat API** in [Examples Gallery](./examples-gallery), and see [Optional Packs](./optional-packs) for the downloads.
- **The VLA chain** (VLA) — the VLA category contains five specialized nodes. A training and evaluation graph connects `PushWorldEnv`, `PushWorldDemos`, `VLAModel`, and `TrainingLoop`, then uses `VLAActionEval` and `VLARollout`. `PushWorldEnv` implements a language-conditioned 2D push task in torch; distractor pucks make the instruction necessary to identify the goal. `PushWorldDemos` creates scripted-expert behaviour-cloning samples and a held-out split. `VLAModel` combines a vision stem, byte-level instruction embedding, transformer trunk, and chunked action expert. `VLAActionEval` measures held-out open-loop MSE against the expert, while `VLARollout` measures closed-loop success with receding-horizon execution. The [README for **Train a VLA on PushWorld**](https://github.com/CodefyUI/CodefyUI/blob/main/examples/VLA/TrainVLA-PushWorld/README.md) recommends a CUDA GPU and reports a run time of about 56 minutes on an RTX 4080.
- **The RL nodes** (RL) — `GridWorldEnv` provides a gridworld without a Gymnasium dependency. `PolicyRollout` runs a policy for N episodes and records states, actions, rewards, and logits. `Discount` computes returns, `GroupRelativeAdvantage` computes GRPO's group-mean baseline, and `PPOClipObjective` computes PPO's clipped surrogate and returns a mask showing which samples used the clipped branch. `RewardModel` produces one scalar score per sequence. `PreferenceDataset` creates training and held-out preference pairs; `BradleyTerryLoss` computes the preference loss; and `BradleyTerryTrain` fits a reward head and reports training and held-out accuracy to expose shortcut learning. `KLDivergence` computes the policy-to-reference regularisation term. `DQN`, `PPO`, and `EnvWrapper` integrate with Gymnasium. See **RLHF building blocks: reward + KL** in [Examples Gallery](./examples-gallery).
- **The diffusion chain** (Diffusion) — `GaussianNoise` creates seeded noise and can match an upstream tensor's shape. `DiffusionUNet` provides a toy U-Net that maps `(x, t)` to predicted noise, and `DiffusionTrainingLoop` trains it. `DDPMSampler` performs the reverse process in one node so the graph remains acyclic; its verbose trace records trajectory snapshots. `TimestepEmbedding`, `Upsample`, and `Lerp` support an explicit construction of the same operations. The Diffusion examples are **Forward Process** (`GaussianNoise` and `Lerp`) and **Toy Sampling** and **Mini U-Net** (`GaussianNoise` → `DiffusionUNet` → `DDPMSampler`). See [Examples Gallery](./examples-gallery).
- **`ModelSaver` / `ModelLoader`** (IO) — write and read model files. Their default `state_dict` mode stores tensors and imposes no class-loading requirements. Their `full_model` mode stores the pickled module and loads it with a restricted unpickler that permits torch and CodefyUI layer classes and the two torch activation functions stored by CodefyUI's transformer layers. See [Saving and Loading Models](./model-files) for selection and loading requirements.
- **`Switch`** (Data Flow) — uses conditional routing so that only one branch executes.

## Port data types

Edges are typed. The built-in data types include: **Tensor, Model, Dataset, DataLoader, Optimizer, Loss, Scalar, String, Image, List, Transform, Any, Trigger**. The `Trigger` type is what `Start` nodes emit to drive execution order, and `Transform` is what the transform-chain nodes pass to each other.
