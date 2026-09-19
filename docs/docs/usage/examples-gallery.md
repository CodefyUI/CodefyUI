---
sidebar_position: 9
title: Examples Gallery
description: Pre-built example workflows — model architectures, end-to-end training, and LLM demos you can load and run.
---

# Examples Gallery

CodefyUI ships a library of ready-to-run example graphs under `examples/`. Whenever the active tab has an empty canvas, the gallery appears right on the canvas — pick a card and the graph loads into the tab, ready to **Run**. The other ways to open it are listed under [When the canvas is not empty](#when-the-canvas-is-not-empty). You can also run any example headless with the [CLI Graph Runner](./cli-runner).

The gallery is grouped by what an example is for, in this order:

| Section | Contents |
|---------|----------|
| **Quick Start** | The first three to run: **Train CNN on MNIST**, **Inference CNN on MNIST**, and **Call a graph as an API** (graph-as-a-function demo). |
| **Training** | The graphs that train a model end to end: **Train a CNN on a HuggingFace dataset**, **Train ResNet on CIFAR10**, **Train a Transformer classifier on MNIST**, the measured **ResNet-18 / CIFAR-10 baseline** (see [Reproducing Baselines](./reproducing-baselines)), **Train a Causal LM on TinyStories**, and **Train a VLA on PushWorld** — which needs a CUDA GPU and about an hour; the recipe is in its [README](https://github.com/CodefyUI/CodefyUI/blob/main/examples/VLA/TrainVLA-PushWorld/README.md). |
| **LLM and RAG** | **Word Embedding Analogy**, **Sentence Similarity (zh-TW)**, and the two retrieval examples **RAG, fully local** and **RAG with a chat API**. |
| **Concepts** | One idea per graph, small enough to read end to end: the two Iris pipelines, **RNN unrolled**, **Mixture of Experts**, the three diffusion graphs (**Forward Diffusion**, **Toy Sampling**, **Mini U-Net**), and **RLHF building blocks: reward + KL**. |
| **Model Architectures** | 15 classic architecture walkthroughs, sub-grouped by family: CNN, RNN, Transformer, Diffusion, RL. |
| **Plugin Packs** | Examples shipped by installed [plugins](/advanced/plugins), one sub-heading per pack. Only shown when present. |
| **Other** | A built-in example that declares no section, or one this list does not recognise. |

All three surfaces that list examples use this one grouping: the empty-canvas overlay, the **Template Gallery**, and the sidebar's **Templates** tab.

On disk the examples are grouped by topic folder: `Classical/`, `Diffusion/`, `LLM/`, `Model_Architecture/`, `RL/`, `RNN/`, `Transformer/`, `Usage_Example/`, and `VLA/`. A folder is not a section — see [Adding an example](#adding-an-example).

Every listed example runs offline out of the box, with these exceptions. Each one says so on its own card, in the words the card has room for — "needs a download", "needs a pack", "needs a GPU":

- The four dataset trainers fetch their data on the first run and are offline after that: **Train CNN on MNIST** and **Train a Transformer classifier on MNIST** download MNIST, **Train ResNet on CIFAR10** and the **ResNet-18 / CIFAR-10 baseline** download CIFAR-10 (about 170 MB). Both land under `backend/data/`.
- **Train a CNN on a HuggingFace dataset** downloads `AI-Lab-Makerere/beans` from the Hugging Face Hub on its first run — 1,295 photos, about 170 MB — into the Hugging Face cache. Point its three `HuggingFaceDataset` nodes at another repo and that one is fetched instead.
- **Train a Causal LM on TinyStories** downloads the TinyStories corpus from the Hugging Face Hub and the gpt2 BPE ranks on its first run, and needs a GPU with headroom for a 203,668,480-parameter model. Its card leads with both requirements; the full recipe, the token budgets and the memory levers are in the `README.md` beside the graph, at `examples/LLM/TrainCausalLM-TinyStories/`. Both downloads are cached, so later runs are offline too.
- **Sentence Similarity (zh-TW)** needs the `sentence-embeddings` pack, which is a one-off install from the Package Center (toolbar > Settings > Optional Packs & Plugins) or `cdui packs install sentence-embeddings` — a run never downloads it for you. Once the pack is in, the example runs offline on CPU in a few seconds. See [Optional Packs](./optional-packs).
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

### When the canvas is not empty

The **Template Gallery** opens from the toolbar's **Templates** button, from **Browse all templates** on the empty-canvas overlay, and from the sidebar's **Templates** tab. It groups the examples into the sections above, and selecting one shows its description, its node and connection counts, and whether it is built in or comes from a plugin. Each example offers two actions:

- **Open in new tab** leaves the current graph alone.
- **Insert into this canvas** adds the example to the graph you are editing: the inserted nodes get fresh ids and are placed below your current graph, so nothing is overwritten, and one undo removes them.

In the sidebar's **Templates** tab, click an example to insert it, or drag it onto the canvas where you want it.

### Languages

Switch the editor to 繁體中文 and every built-in example describes itself in Chinese. The **names stay English** in both languages: an example's name is how you find it again in these docs, in `examples/` on disk, and as an argument to `run_graph.py`. Search matches either language, so `attention` and 「注意力」 both find the same graph.

An example with no translation — one a third-party plugin ships — keeps its English description rather than going blank. Translations live in `frontend/src/i18n/exampleLocales/`, keyed by the example's path.

## Adding an example

An example is a directory holding a `graph.json` — under `examples/` for a built-in, under `<pack>/examples/` for one a plugin ships. Three things in that file decide how it reads in the gallery.

**Where it appears.** An optional top-level `gallery` block:

```json
{
  "name": "UNet for Image Segmentation",
  "description": "...",
  "gallery": { "section": "architectures", "family": "CNN", "order": 2 },
  "nodes": [],
  "edges": []
}
```

`section` is one of `quickstart`, `training`, `llm`, `concepts`, and `architectures`. `family` is the sub-heading inside **Model Architectures**: the five families above come first, in that order, and any other string after them alphabetically. Wherever an example declares a `family`, that is also what its card's chip shows in place of the category, except under a sub-heading that already says the same word, where the card carries no chip. `order` sorts ascending inside the section, or inside the family, and an example without one comes after those that have one. A field the list cannot read is served as nothing for that field alone rather than failing the gallery for everyone: a `section` it does not recognise puts a built-in example in **Other**, and an `order` that is not an integer simply sorts the example after the ones that have an order. An example a plugin ships is listed under its pack whatever it declares.

The folder an example sits in is not its section. Folders are paths, and the paths are what these docs, the translation tables, and `run_graph.py` arguments refer to, so they stay put when the gallery is regrouped.

**What the card says.** `description` is one line of at most 56 columns — what the card, the sidebar row, and the detail pane show without cutting. Columns rather than characters: a Chinese description mixes ideographs with Latin node names, and an ideograph is two columns wide. `backend/tests/test_example_descriptions.py` holds that rule for the English and `frontend/src/i18n/exampleLocales/zh-TW.test.ts` for the Chinese.

**Where the explanation goes.** Everything longer belongs in a [note](./canvas-basics#notes) on the canvas, beside the nodes it is about. A note is a node of type `note`: the validator skips it, a run never reaches it, and the card's node count leaves it out. Each note is written twice inside the one note — an English paragraph, a blank line, then the same thing in Traditional Chinese — so it reads in both languages with no translation table to keep in step.

## A good first run

Load **Train CNN on MNIST**, then:

1. Read the note to the left of the `Start` node — it says what the graph does and what to look at afterwards.
2. **Record node outputs** and **Persist weights between runs** are both on by default — check them in the Settings popover (**Recording & Inspection** and **Training Behavior**).
3. Click **Run** and watch the live loss chart in the **Training** tab. The first run downloads MNIST; five epochs take a minute or two on a CPU.
4. Click a `Conv2d` node to inspect its kernels and activations in the **[Teaching Inspector](./teaching-inspector)**.
5. When it finishes, read the **Test accuracy** `Print` — about 0.99, measured on the 10,000 test images the training loop never saw — and compare the 16 predicted digits with the 16 images the `Visualize` node tiles beside them.
6. Run again — with weights persisted, the model keeps learning across runs.

Training also saves `model_weights.pt` (under `backend/data/models/`). After that, load **Inference CNN on MNIST** — it classifies `test_digit.png`, a real MNIST digit bundled under `backend/data/images/`, using the weights you just trained, and prints the digit it read.
