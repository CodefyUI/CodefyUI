---
sidebar_position: 9
title: Examples Gallery
description: Pre-built example workflows — model architectures, end-to-end training, and LLM demos you can load and run.
---

# Examples Gallery

CodefyUI ships a library of ready-to-run example graphs under `examples/`. Load them from the **Examples** menu in the app, or run them headless with the [CLI Graph Runner](./cli-runner).

| Category | Examples |
|----------|----------|
| **Model Architecture** | ResNet, ConvNeXt, EfficientNet, UNet, ViT, SwinTransformer, BERT, GPT, LLaMA, DiT, LSTM TimeSeries, BiGRU SpeechRecognition, Seq2Seq Attention, DQN Atari, PPO Robotics |
| **Usage Example** | CNN-MNIST Training, CNN-MNIST Inference, GPT-Mini Training, ResNet-CIFAR10 Training |
| **LLM** | Word Embedding Analogy (`king − man + woman ≈ queen` with the offline `demo-16d` backend) |

The repository also groups examples by topic on disk: `Classical/`, `Diffusion/`, `LLM/`, `Model_Architecture/`, `RL/`, `RNN/`, `Transformer/`, `Usage_Example/`, and `Others/`.

## Loading an example

- **In the app** — open the **Examples** menu and pick a graph; it loads into a new tab ready to **Run**.
- **From the CLI** — point `run_graph.py` at the graph's JSON:

  ```bash
  cd backend
  python run_graph.py ../examples/Usage_Example/CNN-MNIST/TrainCNN-MNIST/graph.json
  ```

## A good first run

Load **Train CNN on MNIST**, then:

1. Turn on **Record outputs** and **Persist weights between runs** in the ⚙ Settings popover.
2. Click **Run** and watch the live loss chart in the **Training** tab.
3. Click a `Conv2d` node to inspect its kernels and activations in the **[Teaching Inspector](./teaching-inspector)**.
4. Run again — with weights persisted, the model keeps learning across runs.
