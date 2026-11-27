---
sidebar_position: 9
title: Examples Gallery
description: Pre-built example workflows — quick starts, training and evaluation recipes, LLM demos, architectures and paper reproductions you can load and run.
---

# Examples Gallery

CodefyUI ships a library of ready-to-run example graphs under `examples/`. Whenever the active tab has an empty canvas, the gallery appears right on the canvas (an empty subgraph you have opened shows a short note instead) — pick a card and the graph opens, ready to **Run**: in that tab when the tab is empty and has nothing to undo, otherwise in a new tab, so no graph is replaced. The other ways to open it are listed under [When the canvas is not empty](#when-the-canvas-is-not-empty). You can also run any example headless with the [CLI Graph Runner](./cli-runner).

The gallery is grouped by **main tag**, what an example is for, with the practical categories first and papers and courses after them:

| Main tag | Contents |
|----------|----------|
| **Quick start** | The first four to run: **Train CNN on MNIST**, **Inference CNN on MNIST**, **Classify Iris with sklearn KNN**, and **Call a graph over HTTP** (graph-as-a-function demo). The gallery on a new tab opens here. |
| **Training** | The graphs that train a model end to end: **Train a CNN on beans** (a Hugging Face dataset), **Train ResNet on CIFAR10**, the measured **ResNet-18 CIFAR-10 baseline** (see [Reproducing Baselines](./reproducing-baselines)), **Train a Transformer on MNIST**, **Train a VLA on PushWorld** — which needs a CUDA GPU and about an hour; the recipe is in its [README](https://github.com/CodefyUI/CodefyUI/blob/main/examples/VLA/TrainVLA-PushWorld/README.md) — and three that train on generated data in under a minute on a CPU: **Segment shapes pixel by pixel**, **Train a DDPM on blobs**, and **LSTM remembers the first token**. Four show the recipes around a training loop: **Early stopping, keep the best** stops on validation loss and saves the best epoch's weights with `CheckpointSaver`, **Resume from a checkpoint** saves a run with `CheckpointSaver` and continues it with `CheckpointLoader`, **Augment, then train** wires a transform chain into a Dataset's `train_transform`, and **Transfer with a frozen backbone** freezes a pretrained backbone with `FreezeParameters`, trains only a new head, and compares it with training from scratch. |
| **LLM & RAG** | **Pretrain an LM on TinyStories**; **Fine-tune a tiny char LM**, which pretrains a character-level transformer on short stories and fine-tunes it on text you supply, measuring perplexity on both; the two retrieval chains **Fully local RAG** and **RAG with a chat API**; **Measure retrieval hit rate**, which scores a bag-of-words retriever's hit@1 and hit@3 over the bundled notes with no pack installed; **Data mix ratios**, which mixes two bundled text files by weight; **Sentence similarity in zh-TW**; and **Word embedding analogy**. |
| **Data** | Looking at the data before training on it: **Inspect a dataset first** (counts, samples and pixel statistics of the bundled digits), **Split and check for leaks** (a stratified train/validation/test split, then a check for evaluation rows that also appear in training), **Tabular Iris feature pipeline**, and **Image batch IO**, which writes a batch with `ImageWriter` and reads the folder back with `ImageBatchReader`. |
| **Evaluation** | Measuring a model rather than training it: **Five seeds, mean and spread** (one training block placed five times, reported as a mean and standard deviation), **Evaluate a saved checkpoint** (a confusion matrix and per-digit accuracy), **Compare two checkpoints** (two epochs of one run on the same test split, with the images they label differently), **Project embeddings to 2D** (a hidden layer projected with PCA), **K-fold cross-validation** (`KFoldSplit`), and **Six classifiers on two moons**. |
| **Experiments** | Small studies you would run before trusting a setting: **Find a learning rate** (an LR range test, then a run with the rate it picks), **Learning rate x batch size** (a 2x2 grid built from two blocks, each placed twice, with the cells ranked), and **Ablation: input scaling** (the same model with and without input standardization). |
| **Deploy** | A trained model as a service: **Train and export a classifier** trains on iris and saves `iris-classifier.safetensors`, and **Serve a classifier over HTTP** loads it and answers through `GraphInput` and `GraphOutput`, so it can be called as a [function over HTTP](./graph-as-a-function). |
| **Architectures** | 15 classic architecture walkthroughs, sub-grouped by family: CNN, RNN, Transformer, Diffusion, RL. |
| **Concepts** | One idea per graph, small enough to read end to end: **Tensor ops tour** (the tensor reshaping nodes), **Reduce and Switch** (the `Reduce` and `Switch` data-flow nodes), **One backward pass** (gradients you can check by hand, with Capture gradients on), **RNN unrolled over three steps**, **Conv1d classifies sequences** (a temporal CNN beside the RNN examples), **Causal attention from scratch**, **Mixture of Experts top-k routing**, the three diffusion graphs (**Forward diffusion on a digit**, **Toy reverse diffusion sampling**, **Mini U-Net node**), and the reinforcement-learning ideas: **Discounted returns** (how gamma turns rewards into returns), **Reward model from preferences** (RLHF step 1), **RLHF reward and KL terms**, **Reward hacking via a shortcut**, and **One GRPO step on rollouts**. |
| **Papers** | One published method per graph, built from the same nodes as everything else and trained end to end on a CPU in seconds. The overview note names the paper, the recipe and the number the graph is expected to reach: **Dropout narrows the overfit gap**, **BatchNorm tames a high LR**, **Skip connections train deeper**, **Adam vs SGD vs momentum**, **Label smoothing lowers confidence**, **Cosine warm restarts (SGDR)**, **Sigmoid gradients vanish in depth**, and, normalisation at a tiny batch, **GroupNorm holds at batch 2**. Sequences and regression: **Gradient clipping rescues an RNN**, **Autoencoder squeezes digits to 2-D**, **LSTM forecasts a noisy wave**, **Transformer learns to sort**, **Weight decay smooths the fit** and **Early stopping on validation loss**. Reinforcement learning, trained with the **RLTrainingLoop** node: **REINFORCE balances CartPole**, **DQN balances CartPole**, **PPO balances CartPole**, and **Q-learning crosses a trap grid**. |
| **Courses** | The examples of the course packs that ship with CodefyUI — `foundations`, `deep`, `rl`, `stats` and `eval` — one sub-heading per pack. Only shown when an enabled pack has examples; see [Plugins](/advanced/plugins). |
| **Plugin packs** | Examples of an installed [plugin](/advanced/plugins) that do not say where they belong, one sub-heading per pack. Only shown when present. |
| **Other** | A built-in example that declares no main tag, or one this list does not recognise. The shipped examples are tested to leave it empty. |

Seven of the practical examples — **Inspect a dataset first**, **Early stopping, keep the best**, **Transfer with a frozen backbone** and the four digit examples under **Evaluation** — train on `backend/data/samples/digits.csv`, the 1,797 8x8 UCI handwritten digits (CC BY 4.0), which ship with CodefyUI. Each runs offline on a CPU in a few seconds, with a fixed seed.

An example from a plugin that declares a main tag sits under that main tag, among the built-ins; only **Courses** and **Plugin packs** add a sub-heading per pack.

All four surfaces that list examples use this one grouping: the empty-canvas overlay, the [welcome screen](./tabs-persistence#the-welcome-screen) shown when no tab is open, the **Template Gallery**, and the sidebar's **Templates** tab.

On disk the examples are grouped by topic folder: `Classical/`, `Data/`, `Deploy/`, `Diffusion/`, `Evaluation/`, `LLM/`, `Model_Architecture/`, `Others/`, `Papers/`, `Research/`, `RL/`, `RNN/`, `Training/`, `Transformer/`, `Usage_Example/`, and `VLA/`. A folder is not a main tag — see [Adding an example](#adding-an-example).

Every listed example runs offline out of the box, with these exceptions. The card does not say so: it is one line of what the graph shows. What an example needs before it will run is written in the [note](./canvas-basics#notes) on its canvas, beside the nodes it is about — which is where there is room for how big the download is and where the pack comes from — and in the list here:

- The two CIFAR-10 trainers, **Train ResNet on CIFAR10** and the **ResNet-18 CIFAR-10 baseline**, download CIFAR-10 (about 170 MB) into `backend/data/` on their first run and are offline after that. MNIST ships with CodefyUI in `backend/data/MNIST/raw/`, so **Train CNN on MNIST** and **Train a Transformer on MNIST** run offline from the first click. In a [project directory](./project-directories) a relative `data_dir` points at the project's `assets/data/`, so there the first run of each of these four trainers downloads its dataset. **Augment, then train** reads the same MNIST copy, and in a project directory its first run downloads MNIST as well.
- **Inference CNN on MNIST** loads `model_weights.pt`, which **Train CNN on MNIST** writes into `backend/data/models/`. No trained weights ship with CodefyUI, so on a fresh install it stops at `ModelLoader` until the training example has run once.
- **Serve a classifier over HTTP** loads `iris-classifier.safetensors`, which **Train and export a classifier** writes into the models directory (`backend/data/models/` by default). Run the training example once first; without the file, `ModelLoader` stops with "Weights file not found: &lt;path&gt;".
- **Train a CNN on beans** downloads `AI-Lab-Makerere/beans` from the Hugging Face Hub on its first run — 1,295 photos, about 170 MB — into the Hugging Face cache. Point its three `HuggingFaceDataset` nodes at another repo and that one is fetched instead.
- **Pretrain an LM on TinyStories** downloads the TinyStories corpus from the Hugging Face Hub and the gpt2 BPE ranks on its first run, and needs a GPU with headroom for a 203,668,480-parameter model. Its overview note names both requirements, in English and in Chinese; the full recipe, the token budgets and the memory levers are in the `README.md` beside the graph, at `examples/LLM/TrainCausalLM-TinyStories/`. Both downloads are cached, so later runs are offline too.
- **Sentence similarity in zh-TW** needs the `sentence-embeddings` pack, which is a one-off install from the Package Center (toolbar > Settings > Optional Packs & Plugins) or `cdui packs install sentence-embeddings` — a run never downloads it for you. Once the pack is in, the example runs offline on CPU in a few seconds. See [Optional Packs](./optional-packs).
- **Fully local RAG** needs two downloads rather than one: `qwen2.5-0.5b-instruct` from the `rag` pack, and the `multilingual-e5-small` item of `sentence-embeddings` — about 1.5 GB together. Installing `rag` brings that pack's Python packages but no encoder, so the second item has to be picked as well. With both in, nothing leaves the machine: the documents, the search and the generation all happen locally, at a few tokens per second on a CPU, so expect the answer to take anywhere from a few seconds to tens of seconds — an estimate from the model size rather than a measurement, and much faster on a GPU.
- **RAG with a chat API** is that same retrieval chain with `LLMChat` in the last box, so it needs only `multilingual-e5-small` — plus somewhere to send the prompt. Out of the box that is a local [Ollama](https://ollama.com) with `ollama pull qwen2.5:0.5b`, which still keeps everything on this machine; switching `provider` to a hosted model sends the retrieved chunks to a third party and needs a key in the environment.
- Four examples under **Courses** fetch something on their first run, and each says so in the note on its canvas: the `deep` pack examples that tokenise real text — **Self-Attention 101**, **Multi-Head Causal Attention**, **C4-2 Self-attention on real text** and **C4-4 LLM inference pipeline** — download the `cl100k_base` BPE table into the cache, which every later run reads locally. The two MNIST trainers in the `foundations` and `deep` packs read the MNIST copy that ships with CodefyUI, and download it only in a project directory, like the built-in ones. Everything else those four packs ship runs offline from the first click.
- **A/B text classifier eval**, the `eval` pack's example, downloads two sentiment checkpoints from the Hugging Face Hub on its first run, about 18 MB each, into CodefyUI's own cache, and loads them with `transformers`, which installing the `eval` pack brings. Its outcome chart and table are `stats` pack nodes. Both loaders are pinned to a commit, so later runs read the cache, with `HF_HUB_OFFLINE=1` as well.
- **VQA eval skeleton**, the `eval` pack's second example, downloads the ViLT visual question answering checkpoint `dandelin/vilt-b32-finetuned-vqa` on its first run, about 470 MB, into the same cache, and answers 12 questions about 7 drawings the pack ships. Its loader is pinned to a commit as well. [Build Your Own Eval Pipeline](./eval-pipelines.md) explains both eval examples.

The two RL architecture graphs (**DQN on Atari pixels**, **PPO Robotics Controller**) feed their networks from a synthetic observation tensor (`TensorCreate`, `randn`) instead of a live gym environment, so no `ale-py`/`mujoco` install is needed — swap in an `EnvWrapper` node to drive them from a real environment.

## Main tags, tags and filters

Besides its main tag, an example carries a few **tags**: lowercase words such as `vision`, `tabular`, `rag` or `checkpoint`, naming its data, its model or the technique it shows. A card prints up to two of them, and `+n` when there are more, where it used to print the example's folder; the full list is in the Template Gallery's details pane.

- **The gallery on a new tab** — the empty-canvas overlay and the welcome screen — has a row of main-tag chips: **All**, then each main tag that holds at least one example, with its count. One chip is selected at a time; the arrow keys, Home and End move between them. It opens on **Quick start** and, after that, on the chip you chose last (remembered per browser). Its first card is always **Blank graph**: on the overlay it closes the gallery and leaves you the empty canvas, and on the welcome screen it opens a new empty tab.
- **The Template Gallery** has the same chips, opening on **All** every time. **Filter by tag** opens a row of the tags that at least three examples carry, most used first; the rest are behind **More tags**. Select several and only the examples carrying all of them remain. In the details pane every tag is a button, and clicking it, or the main tag, applies that filter. When nothing matches, the gallery lists the active filters and offers **Clear filters**.
- **The sidebar's Templates tab** groups by main tag in the same order and has a search box, but no chips or tag filter.

Search in the Template Gallery and the sidebar matches tags as well as names and descriptions: a tag is found by its id or by the name it is shown under in the current language, so `checkpoint` finds every example tagged with it.

## Loading an example

- **In the app** — open a new (empty) tab; the gallery overlay appears on the canvas. Pick a card and the graph loads into the tab, ready to **Run**. With no tab open, pick a card on the welcome screen; it opens in a new tab.
- **From the CLI** — point `run_graph.py` at the graph's JSON:

  ```bash
  cd backend
  python run_graph.py ../examples/Usage_Example/CNN-MNIST/TrainCNN-MNIST/graph.json
  ```

### When the canvas is not empty

The **Template Gallery** opens from the toolbar's **Templates** button, from **Browse all templates** on the empty-canvas overlay and on the welcome screen, and from the **Browse all templates** button at the foot of the sidebar's **Templates** tab. It groups the examples by main tag, with the chips and the tag filter described under [Main tags, tags and filters](#main-tags-tags-and-filters), and selecting one shows its description, every tag it carries, its node and connection counts, and whether it is built in or comes from a plugin. Each example offers two actions:

- **Open in new tab** leaves the current graph alone. Double-clicking a card does the same.
- **Insert into this canvas** adds the example to the graph you are editing: the inserted nodes get fresh ids and are placed below your current graph, so nothing is overwritten, and one undo removes them. An example written by a newer CodefyUI is not inserted ("Nothing was inserted: …"); open it in a new tab instead, where it opens read-only.

With no tab open, the gallery offers only **Open in new tab**.

In the sidebar's **Templates** tab, click an example to insert it, or drag it onto the canvas where you want it. It groups the examples by main tag in the same order, and its search matches tags as well; it has no chips or tag filter.

### Languages

Switch the editor to 繁體中文 and every built-in example describes itself in Chinese. The **names stay English** in both languages: an example's name is how you find it again in these docs, in `examples/` on disk, and as an argument to `run_graph.py`. Search matches either language, so `attention` and 「注意力」 both find the same graph.

An example with no translation — one a third-party plugin ships — keeps its English description rather than going blank. Translations live in `frontend/src/i18n/exampleLocales/`, keyed by the example's path.

## Adding an example

An example is a directory holding a `graph.json` — under `examples/` for a built-in, under `<pack>/examples/` for one a plugin ships. Three things in that file decide how it reads in the gallery.

**Where it appears.** An optional top-level `gallery` block:

```json
{
  "name": "Train and export a classifier",
  "description": "...",
  "gallery": { "main_tag": "deploy", "tags": ["tabular", "iris", "export"], "order": 10 },
  "nodes": [],
  "edges": []
}
```

- `main_tag` is one of `quickstart`, `training`, `llm`, `data`, `evaluation`, `research`, `deploy`, `architectures`, `concepts`, `papers`, and `courses`, listed in gallery order; the table at the top of this page says what each holds (`research` is shown as **Experiments**). `plugin` and `other` are assigned by the server and cannot be declared. A value outside the list counts as no main tag: a built-in example then lands in **Other**, and a plugin's in **Plugin packs**.
- `tags` is a list of free words: lowercase letters and digits, words joined by hyphens (`transfer-learning`), 1 to 24 characters each, at most six. An entry that breaks a rule is dropped on its own, duplicates are removed, and anything past the sixth is ignored. Do not repeat the main tag or use another main tag's id as a tag. Every built-in names at least one domain (`vision`, `nlp`, `tabular`, `rl`, `diffusion`, `audio`, `timeseries` or `vla`).
- `order` is an integer that sorts ascending inside the main tag, inside the family under **Architectures**, or inside the pack under **Courses** and **Plugin packs**. Number in steps of ten (`10`, `20`, `30`) so an example can later be slotted between two others; an example without one comes after those that have one, and ties sort by name.
- `family` is the sub-heading inside **Architectures** and is used nowhere else: the five families above come first, in that order, and any other string after them alphabetically.

A field the list cannot read is served as nothing for that field alone rather than failing the gallery for everyone: an `order` that is not an integer simply sorts the example after the ones that have an order.

A file written before main tags, with only `"section": "papers"` (or any other main tag id) in its `gallery` block, keeps working: when there is no readable `main_tag`, a known `section` is read as the main tag. The built-in examples use `main_tag` only, and a plugin's examples can move to it when convenient; see [Plugins](/advanced/plugins#examples-in-the-gallery).

**Tag names.** The filter row offers only the tags that at least three examples carry; a rarer tag is behind **More tags** and still works in search. A tag is displayed through the frontend's `examples.tag.<tag>` translation key (`frontend/src/i18n/locales/en.ts` and `zh-TW.ts`), and a tag without one is shown as written. Main tags have their names under `examples.mainTag.<id>`.

The folder an example sits in is not its main tag. Folders are paths, and the paths are what these docs, the translation tables, and `run_graph.py` arguments refer to, so they stay put when the gallery is regrouped. Each built-in's `gallery` block is also listed in `backend/tests/test_examples_gallery_metadata.py`, which holds the two in step and refuses two examples claiming the same main tag, family and order.

**What it brings with it.** An example is a saved graph, so it can carry anything a saved graph does. **Five seeds, mean and spread** and **Learning rate x batch size** keep their own [block](/advanced/subgraphs) definitions in `subgraphs` and place them several times; **Compare two checkpoints** opens with two Compare segments from `segmentGroups` already marked for the [Teaching Inspector](./teaching-inspector).

**What the card says.** `name` is a label, not a sentence: at most five words and 34 columns, and no two examples may carry the same one, because it is the only text that tells one example from another on the card, the sidebar row, the empty-canvas overlay and the welcome screen. `description` is the single line under it, at most 40 columns — twenty Chinese characters — and it says what the graph does and what it is explaining. It carries no requirement and no score: a download, a GPU, a pack, an API key, a sibling example that has to be run first, the accuracy the run ends at — all of that goes in the note on the canvas instead, beside the nodes it is about, and anything that decides whether the example can run at all is listed among the exceptions at the top of this page as well. Both caps are what the card has room for: it is `13rem` wide, and the type inside it grows with the root font, so a wide screen makes the title bigger rather than the card, and anything over the cap is what gets cut. Columns rather than characters: a Chinese description mixes ideographs with Latin node names, and an ideograph is two columns wide. `backend/tests/test_example_descriptions.py` holds both caps for the English and `backend/tests/test_builtin_examples.py` holds the requirement rule over every shipped example in both roots; `frontend/src/i18n/exampleLocales/zh-TW.test.ts` holds the cap and the requirement rule for the Chinese — the name is not translated.

**Where the explanation goes.** Everything that does not fit those two lines belongs in a [note](./canvas-basics#notes) on the canvas, beside the nodes it is about. A note is a node of type `note`: the validator skips it, a run never reaches it, and the card's node count leaves it out. Each note is written twice inside the one note — an English paragraph, a blank line, then the same thing in Traditional Chinese — so it reads in both languages with no translation table to keep in step.

## A good first run

Load **Train CNN on MNIST**, then:

1. Read the note to the left of the `Start` node — it says what the graph does and what to look at afterwards.
2. **Record node outputs** is on by default and **Persist weights between runs** is off — check both in the Settings popover (**Recording & Inspection** and **Training Behavior**).
3. Click **Run** and watch the live loss chart in the **Training** tab. Five epochs take a minute or two on a CPU.
4. Click the `Inference` node to see in the **[Teaching Inspector](./teaching-inspector)** what went in (the batch of 16 test images) and what came out (10 logits per image). The two `Conv2d` layers are inside the `SequentialModel` node; double-click it to see them in the Model Architecture editor.
5. When it finishes, read the **Test accuracy** `Print` — about 0.99, measured on the 10,000 test images the training loop never saw — and compare the 16 predicted digits with the 16 images the `Visualize` node tiles beside them.
6. To keep training across runs, turn **Persist weights between runs** on. Each run after that keeps its weights and the next one continues from them (a run made with it off kept nothing), and the **Execution Log** notes that the result includes that earlier training.

Training also saves `model_weights.pt` (under `backend/data/models/`). After that, load **Inference CNN on MNIST** — it classifies `test_digit.png`, a real MNIST digit bundled under `backend/data/images/`, using the weights you just trained, and prints the digit it read.
