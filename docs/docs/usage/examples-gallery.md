---
sidebar_position: 9
title: Examples Gallery
description: Pre-built example workflows — model architectures, end-to-end training, and LLM demos you can load and run.
---

# Examples Gallery

CodefyUI ships a library of ready-to-run example graphs under `examples/`. Whenever the active tab has an empty canvas, the gallery appears right on the canvas — pick a card and the graph loads into the tab, ready to **Run**. You can also run any example headless with the [CLI Graph Runner](./cli-runner).

The gallery is organized into ordered sections:

| Section | Contents |
|---------|----------|
| **Quick Start** | The three pinned starters: **Train CNN on MNIST**, **Inference CNN on MNIST**, and **Api-Function** (graph-as-a-function demo). |
| **Advanced Examples** | Every other runnable builtin example — LLM (Word Embedding Analogy with the offline `demo-16d` backend, **Sentence Similarity (zh-TW)** with a real sentence encoder, **Train a Causal LM on TinyStories**, and the two retrieval examples **RAG, fully local** and **RAG with a chat API**), Diffusion (Forward Process, Toy Sampling, Mini U-Net), Classical ML (Iris KNN, tabular pipeline), Transformer (MoE routing), RNN, RL (RLHF reward + KL), and the remaining trainers (GPT-Mini, ResNet-CIFAR10, and the measured **ResNet-18 / CIFAR-10 baseline** — see [Reproducing Baselines](./reproducing-baselines)). |
| **Plugin Examples** | Examples shipped by installed [plugins](/advanced/plugins) (and any unrecognized categories). Only shown when present. |
| **Model Architectures** | 15 classic architecture walkthroughs, always listed last: ResNet, ConvNeXt, EfficientNet, UNet, ViT, SwinTransformer, BERT, GPT, LLaMA, DiT, LSTM TimeSeries, BiGRU SpeechRecognition, Seq2Seq Attention, DQN Atari, PPO Robotics. |

On disk the examples are grouped by topic folder: `Classical/`, `Diffusion/`, `LLM/`, `Model_Architecture/`, `RL/`, `RNN/`, `Transformer/`, and `Usage_Example/`.

Every listed example runs offline out of the box, with four exceptions:

- **Train a Causal LM on TinyStories** downloads the TinyStories corpus from the Hugging Face Hub and the gpt2 BPE ranks on its first run, and needs a GPU with headroom for a 203,668,480-parameter model. Its card leads with both requirements; the full recipe, the token budgets and the memory levers are in the `README.md` beside the graph, at `examples/LLM/TrainCausalLM-TinyStories/`. Both downloads are cached, so later runs are offline too.
- **Sentence Similarity (zh-TW)** needs the `sentence-embeddings` pack, which is a one-off install from the Package Center (toolbar > Settings > Optional packs) or `cdui packs install sentence-embeddings` — a run never downloads it for you. Once the pack is in, the example runs offline on CPU in a few seconds. See [Optional Packs](./optional-packs).
- **RAG, fully local** needs two downloads rather than one: `qwen2.5-0.5b-instruct` from the `rag` pack, and the `multilingual-e5-small` item of `sentence-embeddings` — about 1.5 GB together. Installing `rag` brings that pack's Python packages but no encoder, so the second item has to be picked as well. With both in, nothing leaves the machine: the documents, the search and the generation all happen locally, at a few tokens per second on a CPU, so expect the answer to take anywhere from a few seconds to tens of seconds — an estimate from the model size rather than a measurement, and much faster on a GPU.
- **RAG with a chat API** is that same retrieval chain with `LLMChat` in the last box, so it needs only `multilingual-e5-small` — plus somewhere to send the prompt. Out of the box that is a local [Ollama](https://ollama.com) with `ollama pull qwen2.5:0.5b`, which still keeps everything on this machine; switching `provider` to a hosted model sends the retrieved chunks to a third party and needs a key in the environment.

The two RL architecture graphs (**DQN Atari**, **PPO Robotics**) feed their networks from a synthetic observation tensor (`TensorCreate`, `randn`) instead of a live gym environment, so no `ale-py`/`mujoco` install is needed — swap in an `EnvWrapper` node to drive them from a real environment.

## Loading an example

- **In the app** — open a new (empty) tab; the gallery overlay appears on the canvas. Pick a card and the graph loads into the tab, ready to **Run**.
- **From the CLI** — point `run_graph.py` at the graph's JSON:

  ```bash
  cd backend
  python run_graph.py ../examples/Usage_Example/CNN-MNIST/TrainCNN-MNIST/graph.json
  ```

## A good first run

Load **Train CNN on MNIST**, then:

1. Turn on **Record outputs** and **Persist weights between runs** in the Settings popover.
2. Click **Run** and watch the live loss chart in the **Training** tab.
3. Click a `Conv2d` node to inspect its kernels and activations in the **[Teaching Inspector](./teaching-inspector)**.
4. Run again — with weights persisted, the model keeps learning across runs.

Training also saves `model_weights.pt` (under `backend/data/models/`). After that, load **Inference CNN on MNIST** — it classifies `test_digit.png`, a real MNIST digit bundled under `backend/data/images/`, using the weights you just trained.
