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

Adding a pack changes nothing about the base install except which options light up. A SELECT option whose download is missing is greyed out and offers the install; a node whose every backend comes from one pack (`TextEmbedding` and `HFTextGenerate`, both of which come only from packs) is greyed out as a whole. What is already installed keeps working either way, and removing a pack puts the greying back.

## The catalog

| Pack | Contents | Download | Licence | What it unlocks |
|------|----------|----------|---------|-----------------|
| `sentence-embeddings` | The `sentence-transformers` package, plus four small encoders: `all-MiniLM-L6-v2`, `paraphrase-multilingual-MiniLM-L12-v2`, `bge-small-zh-v1.5`, `multilingual-e5-small` | 90 MB, 470 MB, 95 MB, 470 MB per model (they are alternatives, not a set), plus the pip packages | Apache-2.0 for both MiniLM models, MIT for `bge` and `e5` | `TextEmbedding` (the whole node) and `WordVector`'s four sentence backends |
| `word-vectors` | `glove-wiki-gigaword-50.gz`: the real 400,000-word GloVe table in 50 dimensions. No Python packages at all | 69 MB, plus about 83 MB for the one-time conversion stored beside it | PDDL-1.0 | `WordVector`'s `glove-50d` backend |
| `rag` | `Qwen2.5-0.5B-Instruct`, a local generator small enough to run on CPU | about 1 GB | Apache-2.0 | `HFTextGenerate`, and with it a retrieve-then-generate chain that answers from your own notes without a network. Needs `sentence-embeddings` first |
| `gpu-torch` | The CUDA or ROCm PyTorch build that matches this machine | varies by variant | PyTorch's own (BSD-3-Clause) | No new nodes; every node that can use an accelerator gets one. Not installed by `cdui packs` at all: run `cdui install --gpu <variant>`, see [GPU & Device Setup](../getting-started/gpu-device.md) |

Sizes are what comes down the wire. The GloVe row is the one that costs more than it downloads: a 69 MB download plus the 83 MB converted table the install writes beside it, so the disk precheck asks for about 230 MB of free space before it starts, and removing the item gives all of it back.

The `rag` row costs two downloads, not one. It depends on `sentence-embeddings`, and that dependency brings the Python packages, **not** an encoder: the retrieval half of a RAG graph still needs a model to embed with. So a fully local RAG lesson wants `qwen2.5-0.5b-instruct` from `rag` AND the `multilingual-e5-small` item of `sentence-embeddings` -- about 1.5 GB together, with the encoder going in first. `rag` ships no Python packages of its own, so it is refused until `sentence-embeddings`' library is importable: `cdui packs install rag` on a stock install exits 2 with `RAG stack needs another pack first` and prints the pack to install instead, and the Package Center refuses that install for the same reason, since the two are front ends over one rule. So `cdui packs install sentence-embeddings --items multilingual-e5-small` goes first -- one command, which installs the Python packages and then the encoder -- and `cdui packs install rag --yes` follows it. Picking the two items in the Package Center works the same way round: the RAG stack unlocks once the sentence-embeddings library is in. The download is a separate decision from the code either way, so a classroom can fetch both the day before the lesson.

## Installing and removing

**In the app.** Open the Package Center (toolbar > Settings > Optional packs). Each pack lists its items with a size and whether it is already downloaded; select the ones you want, start the install, and watch the log and the byte counter as it runs. **Cancel** stops the transfer mid-file. A model download resumes from the partial file next time; an asset download such as the GloVe table starts over. One install job runs at a time.

**From a terminal.** The same installer, over the same code path:

```bash
cdui packs list                                       # every pack, its items, sizes, licences
cdui packs status                                     # ... plus this venv's PyTorch build and what to run next
cdui packs install sentence-embeddings --items all-MiniLM-L6-v2
cdui packs install word-vectors --yes
cdui packs remove word-vectors glove-50d
```

`--items a,b` downloads only those items; the default is everything the pack is missing. `--yes` skips the download-size confirmation and is required where there is no terminal to confirm at (CI, a piped run). Only ids from the catalog are accepted. Exit codes, for scripts, are listed under [Package commands](../getting-started/cli-commands.md#package-commands). In a dev checkout, `uv pip install -e ".[llm-sentence]"` installs the same pinned range the `sentence-embeddings` pack does; the models are still a download.

**Where the files land.** In the CodefyUI asset cache: `%LOCALAPPDATA%\codefyui\Cache` on Windows, `~/Library/Caches/codefyui` on macOS, `~/.cache/codefyui` on Linux. Hugging Face snapshots go under `hf/` (`hf/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/<revision>/`), single-file downloads such as the GloVe table sit at the root of the cache, and one small JSON per item under `packs/state/` records that a download actually finished, since a half-written snapshot looks exactly like a complete one on disk. Setting `CODEFYUI_USER_DATA_DIR` moves all of it: the cache becomes `<dir>/cache` and the control files `<dir>/packs`. Nothing here reads or writes `HF_HOME` -- that is the whole machine's Hugging Face cache, shared with every other tool you run, and it belongs to its owner.

**Removing.** `cdui packs remove <pack> <item>`, or the delete button beside the item, deletes the download and anything derived from it (the converted GloVe npz goes with the table it came from) and forgets it. Python packages are deliberately left alone: pulling `sentence-transformers` out from under the interpreter that is running the server is not something that server may do to itself, so the command prints the line that would do it and leaves it to you:

```text
uv pip uninstall --python <path-to-venv-python> sentence-transformers
```

Run it with the server stopped.

**Installing over the network.** Every mutating `/api/packs` route is refused unless the server is bound to loopback, because starting an install runs a package manager against the interpreter serving the request. A classroom or office instance that deliberately serves a LAN opts back in with `CODEFYUI_ALLOW_REMOTE_PACK_INSTALL=1`; what may be asked for is bounded by the catalog either way. See [API Reference](../advanced/api-reference.md).

### Installs that restart the server

Two installs cannot happen underneath a running server, because they replace something that server has already imported. **GPU PyTorch** is always one of them: swapping the CUDA or ROCm wheel out from under the interpreter that is serving the request is the one thing no process can do to itself. The other is any pack whose live install hits the resolver conflict described under [Troubleshooting](#troubleshooting) -- the constraints file stops it rather than half-replacing anything, and a restart is what would finish it. For both, a server started with `cdui start` can do the install across the gap where it does not exist: it writes down what to install, starts a helper that outlives it, and shuts itself down. The helper waits for the process to go, runs the install, records how it went, and starts the server again with exactly the arguments `cdui start` was given -- so the address the browser is still pointing at comes back. The install the helper runs is deliberately NOT under the constraints file every live install is pinned by, which is the whole point of doing it here: a pack whose packages need a different torch than this venv has may move torch to the build they require, rather than stopping at the conflict.

**What the panel does.** The GPU PyTorch card offers a build, the note "The server restarts after this install. It refuses to start one while a graph is running.", and an **Install and restart** button, with the terminal command still printed underneath as the manual alternative. Pressing it asks you to confirm ("Install cu128 and restart the server?"); the page is then covered by a blocking **Server restarting** overlay with a seconds counter, because every other surface in the app is a lie while the server is mid-exit. The page reloads by itself as soon as a *different* server process answers, and the reloaded page reports the outcome as a toast -- "Server restarted. GPU PyTorch is ready.", or "The server restarted, but installing GPU PyTorch failed:" with the reason, followed by the installer's last output when there is no reason to give. A live install stopped by a resolver conflict is offered the same thing from its activity banner: a **Restart the server and install** button beside the command block. A restart installs a pack's **Python packages**, or the torch wheel, and **never a model item** -- the helper runs from an interpreter with none of this app's downloader in it -- so a pack that also has model files needs an ordinary install afterwards to download them, which is what the confirmation says.

**When it is offered, and when it is refused.** The offer is the server's, not the browser's: `GET /api/packs` carries `restart_available`, and the button exists only where that is true. Three things have to hold at once. The server must have been started by `cdui start`, which is the only thing that knows how to start it again -- `cdui dev` reloads in place and has nobody to relaunch it, so the card says so and gives you the command instead. That launcher must still be on disk: a checkout moved or deleted since the server started cannot bring anything back. And `CODEFYUI_ENABLE_RESTART_INSTALL` must not be set to `0`, which is the kill switch for a machine where the restart does not come back cleanly -- set it and every restart-mode install is refused with the command to type instead, on that machine, without downgrading anything. Even with all three, the request is refused, having written nothing down and installed nothing, while a graph is running or queued (the restart would take the run with it), while another install is already running, or while another restart-mode install is already pending. Two more things are worth knowing before you press it: the helper checks free space on the volume the venv lives on -- 3 GB for the torch wheel, 1 GB for a pack's Python packages -- and records a job that failed for want of disk rather than half-installing one; and the server always comes back as a background daemon, even when it was started in the foreground with `cdui start -f`, because the helper has no console to hand over.

**Where the files are.** Three files carry a restart across the gap, under the user data root -- `%LOCALAPPDATA%\codefyui` on Windows, `~/Library/Application Support/codefyui` on macOS, `~/.local/share/codefyui` on Linux, or `<dir>` when `CODEFYUI_USER_DATA_DIR` is set. The pending file is the claim, and it names the helper doing the work. While that helper is alive -- or, before it has stamped its own pid into the claim, while the file is less than sixty seconds old -- the restart is still **finishing**: `cdui start` says a restart install is finishing, declines to start a second server into the venv that helper is rewriting, and points at `cdui status`; the panel refuses another restart-mode install while the server that wrote the claim is alive and the claim is under fifteen minutes old. Once the helper is gone -- or it never arrived and those sixty seconds have passed -- the claim is **abandoned**: `cdui start` deletes it and starts normally, a server that reaches its own startup clears it there, and a fresh restart-mode install overwrites it. There is one claim per user data root, and one server is what it is written for: running two managed servers against the same root -- a foreground `cdui start -f` beside a daemon `cdui start` -- is not supported, and a restart-mode install under that arrangement is not something this design can get right. Give the second one its own `CODEFYUI_USER_DATA_DIR`.

```text
<user data>/packs/pending_restart.json      what the restart was asked to install
<user data>/packs/last_restart_job.json     how it went; read by the page that comes back, and by `cdui status` for an hour
<user data>/packs/logs/restart-<job>.log    everything the installer printed
```

**If the server does not come back.** The overlay gives up in two ways. After thirty seconds in which the server never even stopped answering, nothing picked the restart up, and it says "The server did not restart. Run this command, then reload:". After ten minutes it says "The server has not come back after 10 minutes." Both show the command, when the server sent one, and a **Reload now** button, and nothing about the install is lost by giving up on the overlay: the helper records its outcome whether anybody is watching or not. `cdui status` is the next place to look, and it tells the two claim states apart: a `Restart install` line naming the pack, reading *finishing* while the helper is still working and *abandoned* once it is not, plus a `Last restart` line for an hour after one finished, carrying the message from the outcome record. A relaunch that failed does not overwrite how the install went: the record keeps the install's own status, adds `relaunch: failed`, and appends the log path to its message, which is enough for `cdui status` to show that `Last restart` line as failed even when the package itself installed cleanly. The installer's whole output is in `packs/logs/restart-<job>.log`. Once the claim reads *abandoned*, `cdui start` brings the server back by hand and clears the claim on the way up; while it still reads *finishing*, `cdui start` declines and points you at `cdui status`.

## What changes on the canvas

With a pack missing, the editor says so before anything runs: `TextEmbedding` and `HFTextGenerate` are greyed out in the palette -- the two nodes that come only from a pack -- and `WordVector`'s **backend** dropdown greys the options whose download is not there while `demo-16d` stays selectable. A greyed option carries the install with it, so the fix is one click away in the Package Center panel. On a build without the Package Center panel, `cdui packs list` shows the same information.

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

### The RAG chain

Retrieval-augmented generation is seven nodes in a row, and only two of them need a download:

```text
DocumentLoader -> TextChunker -> TextEmbedding -> VectorStore -> Retriever -> PromptBuilder -> HFTextGenerate
                                                                                           (or LLMChat)
```

| Node | What it does | Needs |
|------|--------------|-------|
| `DocumentLoader` | Reads every `.md` and `.txt` in one folder (no PDF, no HTML, no DOCX) and emits `{text, source}` per file, so every chunk downstream can still name the file it came from. `recursive` adds subfolders, `max_docs` caps how many files are read; switch `source` to `uploaded_file` and it reads the one `.txt` picked in `file` instead, uploaded with the button beside it | nothing |
| `TextChunker` | Cuts each document into pieces small enough to embed and to fit in a prompt. `characters` is fixed windows and is the language-neutral one, since Chinese has no spaces; `sentences` and `paragraphs` cut where the author did and pack up to `chunk_size`. Every chunk carries its source and its `start_char`/`end_char`, and `text[start_char:end_char]` is exactly the chunk, so a citation is checkable rather than decorative | nothing |
| `TextEmbedding` | One vector per chunk, and one for the question. Both sides must name the same model | `sentence-embeddings` |
| `VectorStore` | Stacks the chunk vectors into one `[N, D]` matrix and keeps the chunk texts and metadata beside them. That is the "database" of a RAG system; rows are stored unit-length, so a cosine search is one matrix multiply. In memory only, rebuilt in milliseconds from the cached embeddings | nothing |
| `Retriever` | Scores the question against every row, keeps `top_k`, drops anything below `min_score`, and hands on the chunk texts with the file each came from. Its log line prints the score of every hit | nothing |
| `PromptBuilder` | Pastes the retrieved chunks and the question into a template that tells the model to answer only from that context. `{context}` and `{question}` are required; wire a `TextInput` into the `template` input to write your own | nothing |
| `HFTextGenerate` | Qwen2.5-0.5B-Instruct answers locally, applying the model's chat template for you and reporting progress token by token | `rag` |

`LLMChat` is the drop-in alternative in the last box: the same prompt, sent to a local Ollama or a hosted provider instead of loaded into this process, and it needs no pack at all.

**The sample corpus.** `backend/data/samples/rag` ships five short notes -- what CodefyUI is, nodes and edges, training basics, embeddings and RAG, and optional packs -- each with an English half and a Traditional Chinese half, so a graph runs with no setup and a multilingual encoder has something to prove. `DocumentLoader` ingests every `.md` and `.txt` in the folder it is pointed at, so the fastest way to make the lesson about your own material is to point `directory` at a folder of your own notes; nothing else in the graph changes.

**The e5 prefixes are not decoration.** `multilingual-e5-small` was trained asymmetrically: `query: ` on the asking side, `passage: ` on the indexed side. The two `TextEmbedding` nodes therefore carry different prefixes and the same `model` -- and the model is the invariant the graph cannot survive breaking. Two models are two vector spaces, and a cosine between them is a number with no meaning: change the encoder on one side only and the search still returns `top_k` chunks, they are simply arbitrary ones, and nothing in the graph can reject them.

**Answer only from the context is the whole teaching point.** A 0.5B model knows almost nothing about CodefyUI. Shown five notes about it, it answers correctly anyway -- not because it was fine-tuned, but because the right paragraphs were pasted in front of the question. That gap is the argument for retrieval, and the template is what keeps it honest: ask something the corpus cannot answer and the `Retriever` still returns its nearest chunks, because "nearest" is always defined, while the instruction is what makes the model say it does not know instead of inventing something from them.

**What CPU generation feels like.** A few tokens per second on a laptop, so an answer takes anywhere from a few seconds to tens of seconds -- the local example stops when the model ends its turn, which for the shipped question is well short of its 160-token ceiling -- plus a few seconds on the first run to read the weights off disk. Those are estimates from the model size, not stopwatch figures. The node reports progress token by token, so it is visibly working rather than merely pending. A GPU is much faster; `device` follows the global selector unless you set it on the node.

Both graphs are in the [Examples Gallery](./examples-gallery.md), with a `README.md` beside each one. **RAG, fully local** (`examples/LLM/RAG-Local-Offline`) needs both downloads -- `qwen2.5-0.5b-instruct` and `multilingual-e5-small` -- and then sends nothing off the machine. **RAG with a chat API** (`examples/LLM/RAG-LLMChat-API`) is the same retrieval chain node for node, with `LLMChat` in the last box, so it needs only the encoder plus a running Ollama or a provider key. Running both on one question is the comparison worth making: the retrieved contexts are identical, so every difference in the answer is the generator.

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
- **Speed on CPU.** These are small models (22M to 118M parameters) and CPU is the expected place to run them. The first encode in a session pays a few seconds to read the weights off disk; after that a handful of sentences is well under a second. GloVe pays a one-time conversion from the downloaded text table into an npz, a few seconds with a progress line saying so, and about a second per process to load afterwards.
- **Generation is slow.** `HFTextGenerate` decodes one token at a time, and a 0.5B model on a laptop CPU manages a few per second, so a long answer can take tens of seconds and there is nothing to fix. The levers, cheapest first: lower `max_new_tokens`, which is an upper bound on how long the node runs rather than a target; lower the `Retriever`'s `top_k` or cap `PromptBuilder`'s `max_context_chars`, since every retrieved character is read before the first token is written; set `device` to `cuda` where there is a GPU. Swapping the last node for `LLMChat` moves the generation off this process entirely.
- **The answer ignores the context.** Read the `Retriever`'s log line before blaming the model -- it prints a score per hit. A top hit near 0.3 means the corpus probably does not contain the answer, and no prompt fixes that. If the hits look right but the answer wanders, raise `top_k` so the paragraph that answers the question is actually in the prompt. If nothing came back at all, `min_score` filtered it: `PromptBuilder` then writes `(no context retrieved)` and warns, and lowering the floor (0 keeps everything) puts the chunks back. If the scores themselves look arbitrary, check that both `TextEmbedding` nodes name the same model -- that failure produces plausible nonsense rather than an error.
- **Windows paths.** Hugging Face snapshot directories nest deeply. If the cache sits far down an already long path, turn on long-path support or point `CODEFYUI_USER_DATA_DIR` somewhere shallow. A removal on Windows can also report that the item is no longer registered while its files are still on disk, because something is holding them open: stop the server, then delete the directory by hand.
- **"cannot be installed while the server is running".** Every live install runs under a constraints file pinning each distribution already in this interpreter to the version it has, so an install can only ADD -- nothing the running server has already imported can be replaced under it. A pack that would have to replace something stops there instead of half-replacing it (exit code 3 from the CLI) and prints a `uv pip install` line to run with the server stopped. Stop the server with `cdui stop`, run that line, then start it again. The GPU pack is always in this class: it is installed with `cdui install --gpu <variant>`, never with `cdui packs install`.
- **The server went away during an install and did not come back.** Only a restart-mode install stops the server on purpose -- GPU PyTorch, or a pack whose live install hit the conflict above -- and the helper that does it relaunches the server whether the install worked or not, so a server that is still missing means the helper died too, or the relaunch itself failed (in which case the outcome record keeps the install's own status, adds `relaunch: failed` and appends the log path to its message, so `cdui status` shows it as failed). Run `cdui status` first. A `Restart install` line reading *finishing* means the helper is still working and the thing to do is wait: `cdui start` declines while it says so -- it reports that a restart install is finishing and points at `cdui status` -- rather than starting a second server into a venv something else is rewriting. Once it reads *abandoned*, or a `Last restart` line has appeared, read `<user data>/packs/logs/restart-<job>.log` for what the installer actually printed and run `cdui start` -- it deletes the leftover claim on the way up, so the next attempt is not turned away by the panel with "A restart is already pending. Wait for the server to come back." The panel reports an outcome only when its own overlay was still waiting and the record names the job it was waiting for: the note that says which pack and which job to report on is dropped the moment the overlay gives up, so a tab reloaded after that -- or one that was never watching -- says nothing about it. `cdui status` is where the answer is instead, on the `Last restart` line for an hour after the install finished, and the log has the rest.
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
