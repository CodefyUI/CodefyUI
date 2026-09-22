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
| **Quick Start** | The first three to run: **Train CNN on MNIST**, **Inference CNN on MNIST**, and **Call a graph over HTTP** (graph-as-a-function demo). |
| **Training** | The graphs that train a model end to end: **Train a CNN on beans** (a Hugging Face dataset), **Train ResNet on CIFAR10**, **Train a Transformer on MNIST**, the measured **ResNet-18 CIFAR-10 baseline** (see [Reproducing Baselines](./reproducing-baselines)), **Pretrain an LM on TinyStories**, and **Train a VLA on PushWorld** — which needs a CUDA GPU and about an hour; the recipe is in its [README](https://github.com/CodefyUI/CodefyUI/blob/main/examples/VLA/TrainVLA-PushWorld/README.md). |
| **LLM and RAG** | **Word embedding analogy**, **Sentence similarity in zh-TW**, and the two retrieval examples **Fully local RAG** and **RAG with a chat API**. |
| **Concepts** | One idea per graph, small enough to read end to end: the two Iris pipelines, **RNN unrolled over three steps**, **Mixture of Experts top-k routing**, the three diffusion graphs (**Forward diffusion on a digit**, **Toy reverse diffusion sampling**, **Mini U-Net node**), and **RLHF reward and KL terms**. |
| **Model Architectures** | 15 classic architecture walkthroughs, sub-grouped by family: CNN, RNN, Transformer, Diffusion, RL. |
| **Plugin Packs** | Examples shipped by installed [plugins](/advanced/plugins), one sub-heading per pack. Only shown when present. |
| **Other** | A built-in example that declares no section, or one this list does not recognise. |

All four surfaces that list examples use this one grouping: the empty-canvas overlay, the [welcome screen](./tabs-persistence#the-welcome-screen) shown when no tab is open, the **Template Gallery**, and the sidebar's **Templates** tab.

On disk the examples are grouped by topic folder: `Classical/`, `Diffusion/`, `LLM/`, `Model_Architecture/`, `RL/`, `RNN/`, `Transformer/`, `Usage_Example/`, and `VLA/`. A folder is not a section — see [Adding an example](#adding-an-example).

Every listed example runs offline out of the box, with these exceptions. The card does not say so: it is one line of what the graph shows. What an example needs before it will run is written in the [note](./canvas-basics#notes) on its canvas, beside the nodes it is about — which is where there is room for how big the download is and where the pack comes from — and in the list here:

- The two CIFAR-10 trainers, **Train ResNet on CIFAR10** and the **ResNet-18 CIFAR-10 baseline**, download CIFAR-10 (about 170 MB) into `backend/data/` on their first run and are offline after that. MNIST ships with CodefyUI in `backend/data/MNIST/raw/`, so **Train CNN on MNIST** and **Train a Transformer on MNIST** run offline from the first click. In a [project directory](./project-directories) a relative `data_dir` points at the project's `assets/data/`, so there the first run of each of these four trainers downloads its dataset.
- **Inference CNN on MNIST** loads `model_weights.pt`, which **Train CNN on MNIST** writes into `backend/data/models/`. No trained weights ship with CodefyUI, so on a fresh install it stops at `ModelLoader` until the training example has run once.
- **Train a CNN on beans** downloads `AI-Lab-Makerere/beans` from the Hugging Face Hub on its first run — 1,295 photos, about 170 MB — into the Hugging Face cache. Point its three `HuggingFaceDataset` nodes at another repo and that one is fetched instead.
- **Pretrain an LM on TinyStories** downloads the TinyStories corpus from the Hugging Face Hub and the gpt2 BPE ranks on its first run, and needs a GPU with headroom for a 203,668,480-parameter model. Its overview note names both requirements, in English and in Chinese; the full recipe, the token budgets and the memory levers are in the `README.md` beside the graph, at `examples/LLM/TrainCausalLM-TinyStories/`. Both downloads are cached, so later runs are offline too.
- **Sentence similarity in zh-TW** needs the `sentence-embeddings` pack, which is a one-off install from the Package Center (toolbar > Settings > Optional Packs & Plugins) or `cdui packs install sentence-embeddings` — a run never downloads it for you. Once the pack is in, the example runs offline on CPU in a few seconds. See [Optional Packs](./optional-packs).
- **Fully local RAG** needs two downloads rather than one: `qwen2.5-0.5b-instruct` from the `rag` pack, and the `multilingual-e5-small` item of `sentence-embeddings` — about 1.5 GB together. Installing `rag` brings that pack's Python packages but no encoder, so the second item has to be picked as well. With both in, nothing leaves the machine: the documents, the search and the generation all happen locally, at a few tokens per second on a CPU, so expect the answer to take anywhere from a few seconds to tens of seconds — an estimate from the model size rather than a measurement, and much faster on a GPU.
- **RAG with a chat API** is that same retrieval chain with `LLMChat` in the last box, so it needs only `multilingual-e5-small` — plus somewhere to send the prompt. Out of the box that is a local [Ollama](https://ollama.com) with `ollama pull qwen2.5:0.5b`, which still keeps everything on this machine; switching `provider` to a hosted model sends the retrieved chunks to a third party and needs a key in the environment.
- Four examples under **Plugin Packs** fetch something on their first run, and each says so in the note on its canvas: the `deep` pack examples that tokenise real text — **Self-Attention 101**, **Multi-Head Causal Attention**, **C4-2 Self-attention on real text** and **C4-4 LLM inference pipeline** — download the `cl100k_base` BPE table into the cache, which every later run reads locally. The two MNIST trainers in the `foundations` and `deep` packs read the MNIST copy that ships with CodefyUI, and download it only in a project directory, like the built-in ones. Everything else those four packs ship runs offline from the first click.

The two RL architecture graphs (**DQN on Atari pixels**, **PPO Robotics Controller**) feed their networks from a synthetic observation tensor (`TensorCreate`, `randn`) instead of a live gym environment, so no `ale-py`/`mujoco` install is needed — swap in an `EnvWrapper` node to drive them from a real environment.

## Loading an example

- **In the app** — open a new (empty) tab; the gallery overlay appears on the canvas. Pick a card and the graph loads into the tab, ready to **Run**. With no tab open, pick a card on the welcome screen; it opens in a new tab.
- **From the CLI** — point `run_graph.py` at the graph's JSON:

  ```bash
  cd backend
  python run_graph.py ../examples/Usage_Example/CNN-MNIST/TrainCNN-MNIST/graph.json
  ```

### When the canvas is not empty

The **Template Gallery** opens from the toolbar's **Templates** button, from **Browse all templates** on the empty-canvas overlay and on the welcome screen, and from the **Browse all templates** button at the foot of the sidebar's **Templates** tab. It groups the examples into the sections above, and selecting one shows its description, its node and connection counts, and whether it is built in or comes from a plugin. Each example offers two actions:

- **Open in new tab** leaves the current graph alone. Double-clicking a card does the same.
- **Insert into this canvas** adds the example to the graph you are editing: the inserted nodes get fresh ids and are placed below your current graph, so nothing is overwritten, and one undo removes them. An example written by a newer CodefyUI is not inserted ("Nothing was inserted: …"); open it in a new tab instead, where it opens read-only.

With no tab open, the gallery offers only **Open in new tab**.

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

**What the card says.** `name` is a label, not a sentence: at most five words and 34 columns, and no two examples may carry the same one, because it is the only thing the card, the sidebar row, the empty-canvas overlay and the welcome screen all print. `description` is the single line under it, at most 40 columns — twenty Chinese characters — and it says what the graph does and what it is explaining. It carries no requirement and no score: a download, a GPU, a pack, an API key, a sibling example that has to be run first, the accuracy the run ends at — all of that goes in the note on the canvas instead, beside the nodes it is about, and anything that decides whether the example can run at all is listed among the exceptions at the top of this page as well. Both caps are what the card has room for: it is `13rem` wide, and the type inside it grows with the root font, so a wide screen makes the title bigger rather than the card, and anything over the cap is what gets cut. Columns rather than characters: a Chinese description mixes ideographs with Latin node names, and an ideograph is two columns wide. `backend/tests/test_example_descriptions.py` holds both caps for the English and `backend/tests/test_builtin_examples.py` holds the requirement rule over every shipped example in both roots; `frontend/src/i18n/exampleLocales/zh-TW.test.ts` holds the cap and the requirement rule for the Chinese — the name is not translated.

**Where the explanation goes.** Everything that does not fit those two lines belongs in a [note](./canvas-basics#notes) on the canvas, beside the nodes it is about. A note is a node of type `note`: the validator skips it, a run never reaches it, and the card's node count leaves it out. Each note is written twice inside the one note — an English paragraph, a blank line, then the same thing in Traditional Chinese — so it reads in both languages with no translation table to keep in step.

## A good first run

Load **Train CNN on MNIST**, then:

1. Read the note to the left of the `Start` node — it says what the graph does and what to look at afterwards.
2. **Record node outputs** and **Persist weights between runs** are both on by default — check them in the Settings popover (**Recording & Inspection** and **Training Behavior**).
3. Click **Run** and watch the live loss chart in the **Training** tab. Five epochs take a minute or two on a CPU.
4. Click the `Inference` node to see in the **[Teaching Inspector](./teaching-inspector)** what went in (the batch of 16 test images) and what came out (10 logits per image). The two `Conv2d` layers are inside the `SequentialModel` node; double-click it to see them in the Model Architecture editor.
5. When it finishes, read the **Test accuracy** `Print` — about 0.99, measured on the 10,000 test images the training loop never saw — and compare the 16 predicted digits with the 16 images the `Visualize` node tiles beside them.
6. Run again — with weights persisted, the model keeps learning across runs.

Training also saves `model_weights.pt` (under `backend/data/models/`). After that, load **Inference CNN on MNIST** — it classifies `test_digit.png`, a real MNIST digit bundled under `backend/data/images/`, using the weights you just trained, and prints the digit it read.
