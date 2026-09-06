---
sidebar_position: 8.5
title: Optional Packs
description: Install the optional packages and model files required by selected LLM nodes and GPU backends.
---

# Optional Packs

Optional packs contain Python packages and model files that are not part of the base CodefyUI installation. The catalog uses permissively licensed contents and pins package versions to tested ranges. The base installation remains small and works offline; install only the packs required for a graph.

Install a pack from the **Package Center** (toolbar > Settings > Optional packs, or the sidebar's **Custom & Plugins** tab > **Optional packs** > **Package Center...**) or run `cdui packs install <id>`. The Package Center displays each download size and its progress. Both interfaces use the same installer and catalog. The catalog is an allowlist and accepts only predefined ids; request bodies cannot pass a pip spec, repository id, or URL to the installer subprocess.

:::note Graph runs do not download pack contents
If a required pack is missing, **Run** stops at that node and identifies the pack without downloading it. `TextCorpusDataset`, `HuggingFaceDataset`, and `Tokenizer` can fetch their own small assets from the Hugging Face Hub and use separate caches. The restriction applies only to content managed by the Package Center.
:::

## Why packs are optional

The base installation can start and run offline. `WordVector` defaults to `demo-16d`, a bundled, hand-authored vocabulary of 59 words in 16 interpretable dimensions. It requires no download, and its vectors make `king - man + woman = queen` exact by construction.

Installing or removing a pack changes the availability of related options without changing the rest of the base installation. A `select` option with a missing download is greyed out and provides an install action. If that option is the graph's current value, it remains selectable and displays a warning so that opening the panel does not change the saved value. Nodes whose every backend requires one pack, including `TextEmbedding` and `HFTextGenerate`, are marked at the node level. Removing a pack restores its missing-content indicators.

## The catalog

| Pack | Contents | Download | Licence | What it unlocks |
|------|----------|----------|---------|-----------------|
| `sentence-embeddings` | The `sentence-transformers` package, plus four small encoders: `all-MiniLM-L6-v2`, `paraphrase-multilingual-MiniLM-L12-v2`, `bge-small-zh-v1.5`, `multilingual-e5-small` | 90 MB, 470 MB, 95 MB, 470 MB per model (they are alternatives, not a set), plus the pip packages | Apache-2.0 for both MiniLM models, MIT for `bge` and `e5` | `TextEmbedding` (the whole node) and `WordVector`'s four sentence backends |
| `word-vectors` | `glove-wiki-gigaword-50.gz`, a 400,000-word GloVe table with 50 dimensions; no Python packages | 69 MB, plus about 83 MB for the converted table stored beside it | PDDL-1.0 | The `glove-50d` backend for `WordVector` |
| `rag` | `Qwen2.5-0.5B-Instruct`, a local generator that can run on CPU | about 1 GB | Apache-2.0 | `HFTextGenerate`; requires `sentence-embeddings` first for the retrieval chain |
| `gpu-torch` | The CUDA or ROCm PyTorch build selected for this machine | varies by variant | PyTorch's BSD-3-Clause licence | Adds no nodes. Install with `cdui install --gpu <variant>`, not `cdui packs`; see [GPU & Device Setup](../getting-started/gpu-device.md) |

The Download column lists network transfer sizes. Installing `word-vectors` downloads 69 MB and writes an additional converted table of about 83 MB. Its disk precheck requires about 230 MB of free space, and removal deletes both files.

The `rag` pack depends on the Python packages from `sentence-embeddings`, but that dependency does not install an encoder model. A fully local RAG graph also needs an encoder such as `multilingual-e5-small`, for a combined download of about 1.5 GB. Install the encoder first. Because `rag` contains no Python packages of its own, installation is refused until the `sentence-embeddings` library can be imported. On a base installation, `cdui packs install rag` exits with code `2`, prints `RAG stack needs another pack first`, and identifies the required pack. The Package Center applies the same check. Use these commands in order:

```bash
cdui packs install sentence-embeddings --items multilingual-e5-small
cdui packs install rag --yes
```

In the Package Center, install the encoder item before the RAG model. You can install both before they are needed in a class.

## Installing and removing

**In the app.** Open the Package Center from toolbar > Settings > Optional packs. Each pack lists its items, size, and download status. Select items and start the installation to view its log and byte counter. **Cancel install** stops the current transfer. Model downloads resume from partial files; single-file assets such as the GloVe table restart. Only one installation job can run at a time.

**From a terminal.** The CLI uses the same installer and catalog:

```bash
cdui packs list                                       # every pack, its items, sizes, licences
cdui packs status                                     # ... plus this venv's PyTorch build and what to run next
cdui packs install sentence-embeddings --items all-MiniLM-L6-v2
cdui packs install word-vectors --yes
cdui packs remove word-vectors glove-50d
```

`--items a,b` downloads only the listed items; without it, installation downloads every missing item in the pack. `--yes` skips download-size confirmation and is required when no terminal is available, including CI and piped commands. Only catalog ids are accepted. See [Package commands](../getting-started/cli-commands.md#package-commands) for exit codes. In a development checkout, `uv pip install -e ".[llm-sentence]"` installs the package versions used by `sentence-embeddings`, but it does not download the models.

**File locations.** A server started by `cdui start` or `cdui dev` uses `<install dir>/.codefyui_dev/cache/`; the default installation directory is `~/CodefyUI`. The same user-data root stores the session token and plugin lockfile. See [Project Directories](./project-directories.md#6-create-an-api-key-invoke-needs-one). A manually started `uvicorn app.main:app` process instead uses `%LOCALAPPDATA%\codefyui\Cache` on Windows, `~/Library/Caches/codefyui` on macOS, or `~/.cache/codefyui` on Linux. Hugging Face snapshots are stored under `hf/`, for example `hf/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/<revision>/`. Single-file downloads such as the GloVe table are stored at the cache root. A JSON file under `packs/state/` records completion for each item so a partial snapshot is not treated as complete. If `CODEFYUI_USER_DATA_DIR` is set, the cache is `<dir>/cache` and control files are under `<dir>/packs`. CodefyUI does not read or write `HF_HOME` for pack operations.

**Removing an item.** Run `cdui packs remove <pack> <item>` or use the item's delete button. This deletes the download, derived files such as the converted GloVe `.npz`, and its state record. Python packages remain installed because a running server cannot safely remove packages from its own interpreter. The command instead prints the uninstall command:

```text
uv pip uninstall --python <path-to-venv-python> sentence-transformers
```

Run it with the server stopped.

**Network installation.** Every mutating `/api/packs` route requires the server to be bound to loopback because installation runs a package manager against the serving interpreter. Set `CODEFYUI_ALLOW_REMOTE_PACK_INSTALL=1` to permit installation on a server intentionally exposed to a LAN. The catalog allowlist applies in both cases. See [API Reference](../advanced/api-reference.md).

### Installs that restart the server

Some installations must replace packages already imported by the server. **GPU PyTorch** always requires this mode. A pack also requires it when the live-install constraints detect a resolver conflict and stop before replacing any package. A server managed by `cdui start` handles these installations by recording the request, starting a detached helper, and shutting down. The helper waits for the server to exit, runs the installation, records the result, and starts the server with the original `cdui start` arguments. The helper does not use the constraints file applied to live installations, so it can replace torch when a pack requires a different build.

**Package Center behavior.** The **GPU PyTorch** card displays the detected GPU and installed build, and adds the recommended build when it differs. Its control row contains the build selector and **Install and restart**. The caption states, "It will not start while a graph is running." The equivalent command remains available under **Manual install command**. When restart mode is unavailable, including under `cdui dev`, the card omits the button, displays the command, and states why restart mode is unavailable.

After confirmation, for example "Install cu128 and restart the server?", a **Server restarting** overlay blocks the page and displays elapsed seconds. The page reloads when a different server process responds. It then displays either "Server restarted. GPU PyTorch is ready." or "The server restarted, but installing GPU PyTorch failed:" followed by the reason or the installer's final output. A live installation stopped by a resolver conflict offers **Restart the server and install** in its activity banner.

Restart mode installs a torch wheel or a pack's Python packages. It does not download model items because the detached helper does not contain the application's downloader. If the selected pack also contains models, run a normal pack installation after the restart.

**Availability and refusal conditions.** `GET /api/packs` returns `restart_available`, and the button appears only when it is `true`. Restart mode is available only when all of these conditions hold:

- The server was started by `cdui start`. A `cdui dev` process cannot relaunch itself and displays the manual command instead.
- The launcher still exists at its original path.
- `CODEFYUI_ENABLE_RESTART_INSTALL` is not `0`. Setting it to `0` disables restart installations on that machine and displays the manual command.

Even when restart mode is available, the request is refused before any state is written or package is installed if a graph is running or queued, another installation is active, or another restart installation is pending. The helper requires 3 GB of free space for a torch wheel or 1 GB for a pack's Python packages on the volume containing the virtual environment. Insufficient space produces a failed job record without a partial installation. The helper always relaunches the server in background mode, including when the original command was `cdui start -f`, because it has no console to attach.

**Restart state files.** A managed server stores restart state under `<install dir>/.codefyui_dev/`, or under `<dir>` when `CODEFYUI_USER_DATA_DIR` is set. The pending claim identifies the requested installation and its helper process.

A claim is **finishing** while the helper is running. Before the helper records its process id, a claim less than 60 seconds old is also treated as finishing. In this state, `cdui start` does not start a second server while the helper modifies the virtual environment and instead directs you to `cdui status`. `cdui update` and `cdui dev` also refuse with exit code `1`; `cdui start` returns `0`. The Package Center refuses another restart installation while the originating server is alive and the claim is less than 15 minutes old.

A claim is **abandoned** after the helper exits, or when no helper records a process id within 60 seconds. `cdui start` deletes an abandoned claim and starts normally. A server also clears its claim during startup, and a new restart installation can replace an abandoned claim.

Each user-data root supports one restart claim for one managed server. Running two managed servers against the same root, such as `cdui start -f` beside a background `cdui start`, is unsupported. Set a separate `CODEFYUI_USER_DATA_DIR` for the second server.

```text
<user data>/packs/pending_restart.json      requested installation and helper state
<user data>/packs/last_restart_job.json     outcome read by the reloaded page and by cdui status for one hour
<user data>/packs/logs/restart-<job>.log    complete installer output
```

**If the server does not return.** If the original server is still responding after 30 seconds, the overlay displays "The server did not restart. Run this command, then reload:". If no server returns within 10 minutes, it displays "The server has not come back after 10 minutes." Both states show the command when the API supplied one and provide **Reload now**. Closing or timing out the overlay does not stop the helper or remove its outcome record.

Run `cdui status` to inspect restart state. Its `Restart install` line identifies the pack and reports *finishing* while the helper runs or *abandoned* after it stops without clearing the claim. For one hour after completion, `Last restart` displays the recorded outcome. If installation succeeds but relaunch fails, the record retains the installation status, adds `relaunch: failed`, and includes the log path; `cdui status` reports the overall restart as failed. Complete installer output is stored in `packs/logs/restart-<job>.log`.

When a claim is *abandoned*, run `cdui start` to delete the claim and start the server. When it is *finishing*, `cdui start` refuses to start another process and directs you to `cdui status`.

## What changes on the canvas

When a pack is missing, the editor indicates it before execution. `TextEmbedding` and `HFTextGenerate`, which require packs for every backend, display a **Needs pack** chip in the palette but remain draggable. A placed node displays a **PACK** badge that opens the Package Center at the required pack. In `WordVector`, unavailable **backend** options are greyed out while `demo-16d` remains selectable, and an **Install pack** link appears below the field. On a build without the Package Center, use `cdui packs list` to view availability.

A run that reaches a node with missing content stops at that node and identifies the requirement:

```text
Model 'all-MiniLM-L6-v2' from the Sentence embeddings pack is not downloaded. Open Package Center (toolbar > Settings > Optional packs) to download it; graph runs never download (pack=sentence-embeddings)
```

The `(pack=<id>)` suffix is machine-readable. The editor extracts the id and displays an error notification with an **Open Package Center** button focused on the required pack. The run does not fetch pack content.

## Node reference for pack-backed nodes

### WordVector

`WordVector` returns one vector per input word from a lookup table or encoder. Select the backend according to the representation required:

| Backend | Needs | Behavior |
|---------|-------|----------|
| `demo-16d` | nothing | Built-in set of 59 words in 16 hand-authored dimensions for royalty, divinity, gender, animal classes, motion, vehicles, food, and weather. The canonical analogy is exact by construction. |
| `glove-50d` | `word-vectors` | A 400,000-word GloVe table. The canonical analogy is approximate. |
| `sentence-transformers/all-MiniLM-L6-v2` and the other three model ids | the selected model from `sentence-embeddings` | A sentence encoder applied to one word at a time. These models are trained on sentences and are also used by retrieval systems. |

`normalize` applies L2 normalisation to each row, so a downstream dot product is cosine similarity. `keep_oov` returns a zero vector instead of omitting a word absent from a lookup table. It applies only to `demo-16d` and `glove-50d`; sentence encoders return a vector for any string.

**Retired backend names.** Graphs saved with an early preview may contain `glove-100d` or `minilm-sentence-384d`. These values return an error that identifies the replacement; no download can restore them. Replace them with `glove-50d` and `sentence-transformers/all-MiniLM-L6-v2`, respectively.

### TextEmbedding

`TextEmbedding` returns one dense vector per input text. Semantic-search and RAG graphs use it to embed documents and questions for comparison. The node requires `sentence-embeddings`.

Connect either `texts`, for a list such as a chunker's output, or `text`, for one string. Connecting both is invalid. Its main parameters are:

- **`model`** selects one of the four encoders listed below.
- **`prefix`** is added before each text. `multilingual-e5-small` expects `query: ` for questions and `passage: ` for documents; the other three models do not require these prefixes.
- **`split_lines`** is enabled by default and encodes each non-empty input line separately. Disable it to encode a multiline document as one vector.
- **`max_seq_length`** sets the token limit for each text. `0` uses the model default: 128 for paraphrase-multilingual, 256 for all-MiniLM, and 512 for bge and e5. Longer text is truncated.
- **`normalize`** is enabled by default. Other controls are **`batch_size`**, **`label_chars`**, and **`device`**.

The `embeddings` and `labels` outputs connect to `CosineSimilarity` and `EmbeddingScatter`. The **Sentence Similarity (zh-TW)** example in [Examples Gallery](./examples-gallery.md) uses this path and requires the pack.

### The RAG chain

A retrieval-augmented generation graph contains seven nodes. Two require downloads:

```text
DocumentLoader -> TextChunker -> TextEmbedding -> VectorStore -> Retriever -> PromptBuilder -> HFTextGenerate
                                                                                           (or LLMChat)
```

| Node | Behavior | Needs |
|------|----------|-------|
| `DocumentLoader` | Reads every `.md` and `.txt` file in a directory and returns `{text, source}` for each file. PDF, HTML, and DOCX are unsupported. `recursive` includes subdirectories, and `max_docs` limits the file count. Set `source` to `uploaded_file` to read one `.txt` selected with `file`. | nothing |
| `TextChunker` | Splits each document for embedding and prompt construction. `characters` uses fixed windows and does not depend on word boundaries. `sentences` and `paragraphs` use author-defined boundaries and combine units up to `chunk_size`. Each chunk includes its source, `start_char`, and `end_char`; `text[start_char:end_char]` equals the chunk. | nothing |
| `TextEmbedding` | Creates one vector per chunk and one for the question. Both uses must select the same model. | `sentence-embeddings` |
| `VectorStore` | Stores chunk vectors in one `[N, D]` matrix with their text and metadata. Rows are unit-normalised, so cosine search uses one matrix multiplication. The store is in memory and can be rebuilt from cached embeddings. | nothing |
| `Retriever` | Scores the question against every row, returns the highest `top_k` rows above `min_score`, and includes each chunk's source. Its log records every returned score. | nothing |
| `PromptBuilder` | Inserts retrieved chunks and the question into a template that restricts the answer to that context. The template must contain `{context}` and `{question}`. Connect `TextInput` to `template` to supply a custom template. | nothing |
| `HFTextGenerate` | Runs Qwen2.5-0.5B-Instruct locally, applies its chat template, and reports generation progress by token. | `rag` |

`LLMChat` can replace `HFTextGenerate` at the end of the chain. It sends the same prompt to Ollama or a hosted provider and requires no optional pack.

**Sample corpus.** `backend/data/samples/rag` contains five short notes about CodefyUI, nodes and edges, training, embeddings and RAG, and optional packs. Each note has English and Traditional Chinese sections. The examples can therefore run without corpus setup and demonstrate a multilingual encoder. To use other material, set `DocumentLoader.directory` to a directory of `.md` and `.txt` files.

**e5 prefixes.** `multilingual-e5-small` was trained with `query: ` on questions and `passage: ` on indexed documents. The two `TextEmbedding` nodes must use those different prefixes and the same `model`. Embeddings from different models occupy different vector spaces, but the graph cannot detect this mismatch; retrieval still returns `top_k` results with invalid similarity scores.

**Context-only answers.** The local 0.5B model has little built-in information about CodefyUI. The sample notes supply that information in the prompt without fine-tuning. `PromptBuilder` instructs the model to answer only from retrieved context. `Retriever` always returns the nearest chunks unless `min_score` filters them, even when the corpus lacks the answer, so the prompt instruction is responsible for refusing unsupported answers.

**CPU performance.** On a laptop CPU, generation is typically a few tokens per second, so an answer can take several to tens of seconds. The first run can add a few seconds to load weights from disk. The bundled question usually completes before its 160-token limit. These are estimates based on model size, not benchmark measurements. The node reports progress for each token. A GPU is faster; `device` follows the global selection unless overridden on the node.

Both graphs are available in [Examples Gallery](./examples-gallery.md), with a `README.md` in each example directory. **RAG, fully local** (`examples/LLM/RAG-Local-Offline`) requires `qwen2.5-0.5b-instruct` and `multilingual-e5-small` and makes no provider request. **RAG with a chat API** (`examples/LLM/RAG-LLMChat-API`) uses the same retrieval nodes and replaces the final node with `LLMChat`; it requires the encoder and either Ollama or a provider key. Running both with the same question keeps the retrieved context constant when comparing generators.

## Choosing an embedding model

| Model | Languages | Dimensions | Max tokens | Prefix needed | Download |
|-------|-----------|-----------:|-----------:|---------------|---------:|
| `sentence-transformers/all-MiniLM-L6-v2` | English | 384 | 256 | no | 90 MB |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (default) | 50+, including Traditional Chinese | 384 | 128 | no | 470 MB |
| `BAAI/bge-small-zh-v1.5` | Chinese | 512 | 512 | no | 95 MB |
| `intfloat/multilingual-e5-small` | 100+, including Traditional Chinese | 384 | 512 | `query: ` / `passage: ` | 470 MB |

- **English only and smallest download** — `all-MiniLM-L6-v2`; also the fastest of these four models.
- **Traditional Chinese or mixed-language use** — the default, `paraphrase-multilingual-MiniLM-L12-v2`. It requires no prefixes and aligns equivalent text across languages.
- **Chinese only with longer passages** — `bge-small-zh-v1.5`; it supports 512 tokens per text instead of the default model's 128 and downloads 95 MB.
- **Questions and documents embedded separately for retrieval** — `multilingual-e5-small`, with `query: ` for questions and `passage: ` for documents. Omitting the prefixes reduces retrieval quality without producing an error.

The process keeps at most two models resident. Loading a third evicts the least recently used model.

## Troubleshooting

- **"reports installed but `sentence_transformers` cannot be imported"** — the state record exists, but the active interpreter cannot import the package. Reinstall `sentence-embeddings` from the Package Center or run `cdui packs install sentence-embeddings --yes`.
- **"Model ... is not downloaded"** — the Python package is installed, but the selected model is not. Each of the four encoder models is installed separately. Install the specified model in the Package Center or run `cdui packs install sentence-embeddings --items multilingual-e5-small`.
- **Slow encoding on CPU.** The models contain 22M to 118M parameters and support CPU execution. The first encode can take a few seconds while weights load; later batches of a few sentences usually complete in less than a second. GloVe requires a one-time text-to-`.npz` conversion that takes a few seconds and reports progress, followed by about one second to load per process.
- **Slow generation.** `HFTextGenerate` decodes one token at a time. A 0.5B model on a laptop CPU typically produces a few tokens per second, so long answers can take tens of seconds. To reduce time, lower `max_new_tokens`; lower `Retriever.top_k` or `PromptBuilder.max_context_chars`; or set `device` to `cuda` when available. Replacing the final node with `LLMChat` moves generation to Ollama or a hosted provider.
- **The answer ignores context.** Check the per-result scores in the `Retriever` log. A highest score near 0.3 usually means the corpus lacks the answer. If the returned chunks contain the answer but the prompt omits the required passage, increase `top_k`. If no chunks are returned, lower `min_score`; `0` retains every chunk. With no context, `PromptBuilder` writes `(no context retrieved)` and emits a warning. Also confirm that both `TextEmbedding` nodes use the same model; different models can produce plausible-looking but invalid retrieval results without an error.
- **Windows paths.** Hugging Face snapshot paths are deeply nested. Enable Windows long-path support or set `CODEFYUI_USER_DATA_DIR` to a short path. If removal reports that an item is unregistered but its files remain because another process holds them open, stop the server and delete the directory manually.
- **"cannot be installed while the server is running"** — live installations use a constraints file that fixes every distribution already loaded by the interpreter. They can add packages but cannot replace loaded ones. A conflicting installation stops without replacing anything, returns CLI exit code `3`, and prints a `uv pip install` command. Stop the server with `cdui stop`, run that command, and restart it. GPU PyTorch always requires restart mode and uses `cdui install --gpu <variant>`, not `cdui packs install`.
- **The server stopped during an installation and did not return.** This can occur only in restart mode. Run `cdui status`. If `Restart install` reports *finishing*, wait; `cdui start` will not start a second process. If it reports *abandoned*, or `Last restart` appears, inspect `<user data>/packs/logs/restart-<job>.log` and run `cdui start`; startup removes the stale claim. A relaunch failure is recorded as `relaunch: failed` without replacing the installation's own status. The Package Center displays an outcome only while its overlay is still tracking the same job. After the overlay times out or the tab is opened later, use the `Last restart` line, which remains for one hour, and the log.
- **Insufficient disk space.** Space is checked before downloading. The error reports the required and available space.

## Licences

Every catalog item has a permissive licence. `cdui packs list` displays the licence for each download; the catalog is defined in `backend/app/core/packs/catalog.py`.

| Item | Licence |
|------|---------|
| `sentence-transformers` (the Python package) | Apache-2.0 |
| `all-MiniLM-L6-v2` | Apache-2.0 |
| `paraphrase-multilingual-MiniLM-L12-v2` | Apache-2.0 |
| `bge-small-zh-v1.5` | MIT |
| `multilingual-e5-small` | MIT |
| `glove-50d` (glove-wiki-gigaword-50) | PDDL-1.0 |
| `qwen2.5-0.5b-instruct` | Apache-2.0 |

CodefyUI is available under AGPL-3.0 or a commercial licence; see [Licensing](../licensing.md). Pack contents retain their own licences and are downloaded directly from their upstream sources. CodefyUI does not redistribute them.
