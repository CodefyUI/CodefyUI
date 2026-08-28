---
sidebar_position: 8.5
title: Optional Packs
description: Install curated model packs to switch LLM nodes from toy demos to real embeddings, GloVe vectors and a local generator.
---

# Optional Packs

A pack is a curated bundle of Python packages and model files that a stock CodefyUI deliberately does not ship. The base install stays small enough to hand to a classroom; the four hundred megabytes a sentence-embedding lesson needs arrive only when a lesson asks for them. Everything in the catalog is small, permissively licensed, and pinned to versions this codebase is tested against.

Install a pack from the **Package Center** (toolbar > Settings > Optional packs), which lists what each one costs and follows the download with a progress bar, or from a terminal with `cdui packs install <id>`. The two are front ends over one installer and one catalog, and the catalog is an allowlist rather than a package manager: the only thing either of them accepts is an id already written into it, so no pip spec, repo id or URL travels from a request body to a subprocess.

:::note One rule, everywhere: a graph run never downloads pack contents
Pressing **Run** with a pack missing stops that node with a sentence naming the pack, and downloads nothing. Four hundred megabytes arriving mid-run, on a classroom connection, with no progress bar and no way to cancel, is not something a Run button may do. Nodes that fetch their own small assets from the Hugging Face Hub (`TextCorpusDataset`, `HuggingFaceDataset`, `Tokenizer`) are unaffected: they have their own cache, and this rule is about the Package Center's.
:::

## Why packs are optional

A stock install runs offline the moment it starts, and every lesson that can be taught without a download is. `WordVector` defaults to `demo-16d`, a hand-built 59-word vocabulary in 16 interpretable dimensions that ships inside the app: no download, and `king - man + woman = queen` comes out exact, because the vectors were written so that it would. That is the toy, and the toy is the point of it.

Adding a pack changes nothing about the base install except which options light up. A SELECT option whose download is missing is greyed out and offers the install; a node whose every backend comes from one pack (`TextEmbedding`) is greyed out as a whole. What is already installed keeps working either way, and removing a pack puts the greying back.

## The catalog

| Pack | Contents | Download | Licence | What it unlocks |
|------|----------|----------|---------|-----------------|
| `sentence-embeddings` | The `sentence-transformers` package, plus four small encoders: `all-MiniLM-L6-v2`, `paraphrase-multilingual-MiniLM-L12-v2`, `bge-small-zh-v1.5`, `multilingual-e5-small` | 90 MB, 470 MB, 95 MB, 470 MB per model (they are alternatives, not a set), plus the pip packages | Apache-2.0 for both MiniLM models, MIT for `bge` and `e5` | `TextEmbedding` (the whole node) and `WordVector`'s four sentence backends |
| `word-vectors` | `glove-wiki-gigaword-50.gz`: the real 400,000-word GloVe table in 50 dimensions. No Python packages at all | 69 MB, plus about 83 MB for the one-time conversion stored beside it | PDDL-1.0 | `WordVector`'s `glove-50d` backend |
| `rag` | `Qwen2.5-0.5B-Instruct`, a local generator small enough to run on CPU | about 1 GB | Apache-2.0 | `HFTextGenerate` and the retrieval chain around it, **next release**. Needs `sentence-embeddings` first |
| `gpu-torch` | The CUDA or ROCm PyTorch build that matches this machine | varies by variant | PyTorch's own (BSD-3-Clause) | No new nodes; every node that can use an accelerator gets one. Not installed by `cdui packs` at all: run `cdui install --gpu <variant>`, see [GPU & Device Setup](../getting-started/gpu-device.md) |

Sizes are what comes down the wire. The GloVe row is the one that costs more than it downloads: the table lists the 69 MB download, and installing also writes an 83 MB converted table beside it, so the catalog budgets the pair at 153 MB and the disk precheck asks for about 230 MB of free space (1.5 x 153 MB) before it starts.

The `rag` row is in the catalog before the nodes that use it, on purpose: the download is a separate decision from the code, and a classroom can fetch the model the day before the lesson that needs it. `HFTextGenerate` is not in this build.

## Installing and removing

**In the app.** Open the Package Center (toolbar > Settings > Optional packs). Each pack lists its items with a size and whether it is already downloaded; select the ones you want, start the install, and watch the log and the byte counter as it runs. **Cancel** takes effect immediately rather than at the end of the current file: the download stops mid-file, and the partial file is reused if you install again. One install job runs at a time.

**From a terminal.** The same installer, over the same code path:

```bash
cdui packs list                                       # every pack, its items, sizes, licences
cdui packs status                                     # ... plus this venv's PyTorch build and what to run next
cdui packs install sentence-embeddings --items all-MiniLM-L6-v2
cdui packs install word-vectors --yes
cdui packs remove word-vectors glove-50d
```

`--items a,b` downloads only those items; the default is everything the pack is missing. `--yes` skips the download-size confirmation and is required where there is no terminal to confirm at (CI, a piped run). Only ids from the catalog are accepted. Exit codes, for scripts, are listed under [Package commands](../getting-started/cli-commands.md). In a dev checkout, `uv pip install -e ".[llm-sentence]"` installs the same pinned range the `sentence-embeddings` pack does; the models are still a download.

**Where the files land.** In the CodefyUI asset cache: `%LOCALAPPDATA%\codefyui\Cache` on Windows, `~/Library/Caches/codefyui` on macOS, `~/.cache/codefyui` on Linux. Hugging Face snapshots go under `hf/` (`hf/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/<revision>/`), single-file downloads such as the GloVe table sit at the root of the cache, and one small JSON per item under `packs/state/` records that a download actually finished, since a half-written snapshot looks exactly like a complete one on disk. Setting `CODEFYUI_USER_DATA_DIR` moves all of it: the cache becomes `<dir>/cache` and the control files `<dir>/packs`. Nothing here reads or writes `HF_HOME` -- that is the whole machine's Hugging Face cache, shared with every other tool you run, and it belongs to its owner.

**Removing.** `cdui packs remove <pack> <item>`, or the delete button beside the item, deletes the download and anything derived from it (the converted GloVe npz goes with the table it came from) and forgets it. Python packages are deliberately left alone: pulling `sentence-transformers` out from under the interpreter that is running the server is not something that server may do to itself, so the command prints the line that would do it and leaves it to you:

```text
uv pip uninstall --python <path-to-venv-python> sentence-transformers
```

Run it with the server stopped.

**Installing over the network.** Every mutating `/api/packs` route is refused unless the server is bound to loopback, because starting an install runs a package manager against the interpreter serving the request. A classroom or office instance that deliberately serves a LAN opts back in with `CODEFYUI_ALLOW_REMOTE_PACK_INSTALL=1`; what may be asked for is bounded by the catalog either way. See [API Reference](../advanced/api-reference.md).

## What changes on the canvas

With a pack missing, the editor says so before anything runs: `TextEmbedding` is greyed out in the palette, and `WordVector`'s **backend** dropdown greys the options whose download is not there while `demo-16d` stays selectable. A greyed option carries the install with it, so the fix is one click away in the Package Center panel. On a build without the Package Center panel, `cdui packs list` shows the same information.

A run that reaches a node needing something missing stops at that node with the sentence that names it:

```text
Model 'all-MiniLM-L6-v2' from the Sentence embeddings pack is not downloaded. Open Package Center (toolbar > Settings > Optional packs) to download it; graph runs never download (pack=sentence-embeddings)
```

The `(pack=<id>)` suffix is machine-readable on purpose: the editor reads the id back off the message to offer exactly the download that would fix it. Nothing is fetched by the run itself, so the graph is safe to leave running on a metered connection.

## Node reference for pack-backed nodes

### WordVector

One vector per input word, from a lookup table or an encoder. Which backend you pick is the lesson:

| Backend | Needs | What it teaches |
|---------|-------|-----------------|
| `demo-16d` | nothing | 59 words in 16 hand-built dimensions (royalty, divinity, gender, animal classes, motion, vehicles, food, weather). Ships inline; the canonical analogy is exact by construction |
| `glove-50d` | `word-vectors` | The real 400,000-word GloVe table. The same analogy is only approximate here, and that gap is the point |
| `sentence-transformers/all-MiniLM-L6-v2` and the three other model ids | `sentence-embeddings` (that one model) | A modern encoder run over one word at a time. Messier still for single words, since these models are trained on sentences, but it is what real retrieval systems actually use |

`normalize` L2-normalises each row so a dot product downstream is cosine similarity. `keep_oov` emits a zero vector instead of dropping a word the table does not have, and is meaningful only for the two table backends: an encoder produces a vector for any string at all, so nothing can be out of its vocabulary.

**Retired backend names.** A graph saved against an early preview may still carry `glove-100d` or `minilm-sentence-384d`. Both raise a plain error naming their replacement rather than offering a download, because no download fixes a name that no longer exists: set the backend to `glove-50d` and to `sentence-transformers/all-MiniLM-L6-v2` respectively.

### TextEmbedding

One dense vector per text, from a real sentence encoder, so texts that mean the same thing come back pointing the same way. This is the node semantic search and RAG are built on: embed the documents once, embed the question, compare. The whole node needs `sentence-embeddings`.

Wire either `texts` (a list, for instance a chunker's output) or `text` (a single string), never both -- a graph connecting both has said two different things about what to embed. The parameters worth knowing:

- **`model`** -- which of the four encoders to load; see the table below.
- **`prefix`** -- prepended to every text before encoding. `multilingual-e5-small` was trained with `query: ` for questions and `passage: ` for documents; the other three ignore it.
- **`split_lines`** -- on by default, so each non-empty line of the text input becomes its own text. Turn it off when one multi-line document should become one vector.
- **`max_seq_length`** -- token cap per text. `0` keeps the model's own default (128 for paraphrase-multilingual, 256 for all-MiniLM, 512 for bge and e5). Longer texts are truncated, so size your chunks accordingly.
- **`normalize`** (on by default), **`batch_size`**, **`label_chars`** and **`device`** round it out.

The `embeddings` and `labels` outputs wire straight into `CosineSimilarity` and `EmbeddingScatter`. The **Sentence Similarity (zh-TW)** example in the [Examples Gallery](./examples-gallery.md) is that whole chain, ready to run once the pack is in.

### Arriving next release

The `rag` pack's `HFTextGenerate` node, and the chunk-embed-retrieve-generate chain around it, ship in the next release. The pack is already in the catalog, so the model can be downloaded before the nodes exist.

## Choosing an embedding model

| Model | Languages | Dimensions | Max tokens | Prefix needed | Download |
|-------|-----------|-----------:|-----------:|---------------|---------:|
| `sentence-transformers/all-MiniLM-L6-v2` | English | 384 | 256 | no | 90 MB |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (default) | 50+, including Traditional Chinese | 384 | 128 | no | 470 MB |
| `BAAI/bge-small-zh-v1.5` | Chinese | 512 | 512 | no | 95 MB |
| `intfloat/multilingual-e5-small` | 100+, including Traditional Chinese | 384 | 512 | `query: ` / `passage: ` | 470 MB |

- **English only, smallest download** -- `all-MiniLM-L6-v2`. It is also the fastest of the four.
- **Traditional Chinese, or a mixed-language class** -- the default. It needs no prefixes, which is one less thing to get wrong in a lesson, and it aligns languages well enough that a Chinese sentence and its English translation land near each other.
- **Chinese only, longer passages** -- `bge-small-zh-v1.5`: 512 tokens per text against the default's 128, for 95 MB.
- **Retrieval, where questions and documents are embedded separately** -- `multilingual-e5-small` with `query: ` on the question and `passage: ` on the documents. Without the prefixes it scores worse than the others and raises nothing, which is the quiet kind of wrong.

Two loaded models stay resident at a time, which is exactly what comparing an English model against a multilingual one costs; a third load evicts the least recently used one.

## Troubleshooting

- **"reports installed but `sentence_transformers` cannot be imported"** -- the sentinel says the pack is in and the interpreter disagrees, which is a broken install rather than a missing download. Reinstall the pack from the Package Center, or `cdui packs install sentence-embeddings --yes`.
- **"Model ... is not downloaded"** -- the Python side is there but that particular model is not. The four are alternatives, so installing one never brings the others: download it in the Package Center, or `cdui packs install sentence-embeddings --items multilingual-e5-small`.
- **Speed on CPU.** These are small models (22M to 118M parameters) and CPU is the expected place to run them. The first encode in a session pays a few seconds to read the weights off disk; after that a handful of sentences is well under a second. GloVe pays a one-time conversion from the downloaded text table into an npz, roughly ten seconds with a progress line saying so, and about a second per process to load afterwards.
- **Windows paths.** Hugging Face snapshot directories nest deeply. If the cache sits far down an already long path, turn on long-path support or point `CODEFYUI_USER_DATA_DIR` somewhere shallow. A removal on Windows can also report that the item is no longer registered while its files are still on disk, because something is holding them open: stop the server, then delete the directory by hand.
- **"cannot be installed while the server is running".** Every live install runs under a constraints file pinning each distribution already in this interpreter to the version it has, so an install can only ADD -- nothing the running server has already imported can be replaced under it. A pack that would have to replace something stops there instead of half-replacing it (exit code 3 from the CLI) and prints a `uv pip install` line to run with the server stopped. Stop the server with `cdui stop`, run that line, then start it again. The GPU pack is always in this class: it is installed with `cdui install --gpu <variant>`, never with `cdui packs install`.
- **Not enough disk.** Checked before the first byte is fetched, so a 470 MB download does not fail at 90% on a disk that was always too small. The message names what was needed and what is free.

## Licences

Everything in the catalog is permissively licensed, and the licence travels with the item: `cdui packs list` prints it beside every download, and `backend/app/core/packs/catalog.py` is where it is written down.

| Item | Licence |
|------|---------|
| `sentence-transformers` (the Python package) | Apache-2.0 |
| `all-MiniLM-L6-v2` | Apache-2.0 |
| `paraphrase-multilingual-MiniLM-L12-v2` | Apache-2.0 |
| `bge-small-zh-v1.5` | MIT |
| `multilingual-e5-small` | MIT |
| `glove-50d` (glove-wiki-gigaword-50) | PDDL-1.0 |
| `qwen2.5-0.5b-instruct` | Apache-2.0 |

CodefyUI itself is AGPL-3.0 with a commercial option; see [Licensing](../licensing.md). A pack's contents keep their own licence and are downloaded from their own upstream, so nothing here is redistributed by this project.
