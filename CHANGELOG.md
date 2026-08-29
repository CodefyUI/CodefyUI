# Changelog

All notable changes to CodefyUI are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

**Why this file exists.** Release notes previously lived only in the annotated
git tag, which `release-build.yml` copies into the GitHub release body. That is
fine for reading a published release and useless for the question that actually
matters between releases: *what has landed on `main` that nobody has yet?* Eight
commits — including three plugin-sandbox security fixes — sat unreleased before
this file was added, with nothing in the repository saying so.

The `Unreleased` section is the answer to that question. `.github/RELEASING.md`
step 1 is to promote it.

**Versions before this file.** Their notes were written as tag annotations and
were never in the repository. Rather than reconstruct them here from commit
history — which would produce a different document than the one users actually
received — each links to the release it was published as.

## [Unreleased]

### Added

- **Package Center — the optional four hundred megabytes a lesson needs now
  install from inside the app.** A stock CodefyUI is deliberately small
  enough to hand to a classroom, which means the parts a sentence-embedding
  lesson needs — `sentence-transformers`, small embedding models, a real
  400k-word GloVe table, an accelerated PyTorch build — were not in it, and
  the only way to get them was a terminal, a pip spec, and a guess at which
  versions this build was tested against. A curated catalog
  (`backend/app/core/packs/`) is now the whole allowlist — no pip spec, repo
  id or URL from a request body ever reaches a subprocess — and five routes
  under `/api/packs` start an install, follow its log and byte-by-byte
  progress over a long poll, stop it mid-file, and delete a downloaded model
  to get the disk back. Every live install runs under a constraints file
  pinning each distribution already in this interpreter to the version it
  has, so an install can only ADD: nothing already imported by the running
  server can be replaced, which on Windows would fail halfway through and
  everywhere else would leave the process running code that is no longer on
  disk. A pack that *must* replace something imported (the CUDA/ROCm torch
  build) refuses to run live and prints the command to type instead.
  Mutating routes are refused unless the server is bound to loopback —
  starting an install runs a package manager against the interpreter serving
  the request — with `CODEFYUI_ALLOW_REMOTE_PACK_INSTALL=1` to opt a
  deliberate classroom or office instance back in.

- **`cdui packs list|status|install|remove`** — the same installer from a
  terminal, over the same code path as the panel, for the packs that cannot
  be installed while the server is running and for anyone who would rather
  not. `status` also reports which PyTorch build this venv has and what to
  run next; `remove` deletes one downloaded model and prints the
  `uv pip uninstall` line for the packages it deliberately leaves alone.

- **A node can declare the pack it needs, and a dropdown can declare one per
  option.** `/api/nodes` now carries `requires_pack` on each node definition
  and `option_packs` on each SELECT param (option value → pack id), so an
  editor can show that an option is one download away instead of letting the
  run find out. The first nodes to set them ship in this same release —
  `WordVector`'s new backends and `TextEmbedding`, below — and what the
  editor shows is only half of the promise:
  `require_pack()` raises naming the pack, and a graph run never downloads
  — four hundred megabytes arriving mid-run, on a classroom connection, with
  no progress bar and no way to cancel, is not something a Run button may do.

- **`/api/health` reports `boot_id`**, the identity of the process that
  answered. A client waiting out a restart cannot tell "the server came back"
  from "it never went down" by whether the route answers; a changed `boot_id`
  is the only proof, and a restart-mode pack install is the first thing that
  needs to know.

- **`WordVector` looks words up in real embeddings, not only the toy table.**
  The `backend` dropdown gained `glove-50d` — the actual 400,000-word GloVe
  table out of the `word-vectors` pack — and the four sentence-transformer
  encoders out of `sentence-embeddings`, beside the `demo-16d` vocabulary
  that still ships inline, still needs no download, and still makes
  `king - man + woman = queen` come out exact because its 59 words were
  written so that it would. That gap is the lesson: the same analogy is only
  approximate on real GloVe, and messier still through an encoder built for
  sentences rather than single words. Each option carries the download it
  needs, so an install without the pack sees it greyed out rather than
  finding out at run time, and the 400k-word text table is converted to an
  `.npz` once on install — ten seconds of parsing that no longer happens on
  every run. Two backend names from an early preview are retired: a graph
  still carrying `glove-100d` or `minilm-sentence-384d` gets a plain error
  naming the option to pick instead (`glove-50d`,
  `sentence-transformers/all-MiniLM-L6-v2`), because no download fixes a name
  that no longer exists.

- **`TextEmbedding`** — one dense vector per text from a real
  sentence-transformer, so texts that mean the same thing come back pointing
  the same way. This is the encoder semantic search and RAG are built on:
  embed the documents once, embed the question, compare. It takes a list
  (`texts`) or a single string (`text`, one text per line unless
  `split_lines` is off), reports progress and honours Stop between batches —
  keeping the rows it already has — and carries the `prefix` the e5 models
  were trained with (`query: ` / `passage: `). Its four models are the
  `sentence-embeddings` pack's, so the whole node is greyed out until that
  pack is installed, and two loaded encoders stay resident at a time, which
  is what comparing an English model against a multilingual one costs.

- **Sentence Similarity (zh-TW), a gallery example that runs on meaning
  rather than spelling.** Eight Traditional Chinese sentences in four pairs —
  weather, food, the stock market, machine learning, the last pair split
  across Chinese and English — encoded, ranked by `CosineSimilarity` and
  projected to 2D by `EmbeddingScatter`. Rank 2 is each sentence's partner
  though the two share few characters, and the zh/en pair shows the
  model aligning languages. Needs the `sentence-embeddings` pack; runs
  offline on CPU in a few seconds once it is there.

- **A retrieval chain on the canvas: six nodes that answer a question out of
  your own documents instead of out of the model's memory.**
  `DocumentLoader` reads every `.md` and `.txt` in a folder as
  `{text, source}` so a citation survives to the end, stripping the UTF-8
  byte-order mark Notepad writes rather than letting an invisible character
  ride into the first chunk, its embedding and the first citation;
  `TextChunker` cuts them into pieces small enough to embed (character
  windows, sentences or paragraphs, all capped by `chunk_size`) carrying
  the source and the character offsets, with `text[start_char:end_char]`
  guaranteed to be exactly the chunk; `VectorStore` stacks the vectors
  `TextEmbedding` produced into one `[N, D]` matrix with the chunk texts
  and their sources beside them, so a cosine search over the whole corpus
  is a single matrix multiply; `Retriever` scores a question against every
  row, keeps `top_k` and drops anything under `min_score`, printing the
  score of each hit that clears the floor; and `PromptBuilder` pastes the
  winners into a template that instructs the model to answer only from that
  context. `HFTextGenerate` closes the chain with Qwen2.5-0.5B-Instruct from
  the `rag` pack, applying the model's chat template and reporting progress
  token by token, and `LLMChat` is the drop-in alternative that sends the
  same prompt to a server. Only two boxes in the chain need a download, so
  the head of it runs on any install.

- **Two RAG gallery examples, and the corpus they read.** **RAG, fully
  local** (`examples/LLM/RAG-Local-Offline`) retrieves and generates on this
  machine, and needs two downloads rather than one: `qwen2.5-0.5b-instruct`
  from `rag` plus the `multilingual-e5-small` item of `sentence-embeddings`,
  because the pack dependency brings that pack's Python packages and not an
  encoder. **RAG with a chat API** (`examples/LLM/RAG-LLMChat-API`) is the
  same retrieval chain node for node with `LLMChat` in the last box, aimed
  at a local Ollama by default, so running both on one question compares
  generators and nothing else — a test asserts the shared half stays
  identical, and that no `SECRET` param is ever committed in the file. Both
  read `backend/data/samples/rag`: five short notes about CodefyUI and ML
  basics, each with an English and a Traditional Chinese half, so the
  examples run with no setup and a multilingual encoder has something to
  prove.

- **`CODEFYUI_PACK_NETWORK_TESTS=1` runs the pack-backed examples for real.**
  A faked encoder returns whatever the fake was written to return, so the
  suite that asks whether these graphs still DO what their cards claim runs
  against the downloaded models or not at all. It is opt-in twice over:
  the variable is the outer gate, and each test skips on its own if the exact
  model it needs is not in the cache. Nothing is downloaded either way.
  `test_rag_local_example_answers_for_real` joins it for the fully local RAG
  graph, asking twice — the encoder and the generator are separate downloads,
  and one being absent must not hide the other — then running the graph and
  asserting that the top three chunks all come from the two notes that contain
  the answer and that `02-nodes-and-edges.md` is among them, and that the
  generated answer is not empty.

- **An Optional Packs page in the docs**, English and Traditional Chinese
  (`docs/docs/usage/optional-packs.md`) — the catalog with sizes
  and licences, both ways to install one, where the files land per OS, what
  changes on the canvas, which of the four embedding models to pick, and what
  to do when a pack reports installed but will not import. It carries the RAG
  chain too: one line per node, the sample corpus, the e5 prefix rule, what
  CPU generation feels like, and what to try when generation is slow or the
  answer ignores the context.

- **The Package Center, on screen.** A panel that lists every optional pack
  with what is already on this machine, what a download would cost, and one
  button that changes it: per-model progress bars driven by the byte counts
  the long poll carries, a live install log beside them, a Cancel that stops a
  download mid-file, and a Remove that deletes one model and gives the disk
  back. It is a pure view of a store that lives above it, so closing the
  window — or opening a graph, or switching tabs — does not interrupt a
  two-gigabyte download, and a job started in another browser tab is adopted
  rather than duplicated. The GPU/accelerated-PyTorch pack gets a card of its
  own: it cannot be installed while the server is running, so instead of a
  button that would fail it names the detected card, the build this venv
  currently has, and the exact `cdui install --gpu` line to run, ready to
  copy. Reachable from Settings (with an at-a-glance "2 of 4 packs
  installed") and from the sidebar's Custom tab.

- **A dropdown says which of its options are one download away.** A SELECT
  whose options declare `option_packs` greys out exactly the ones that are not
  installed, suffixes each with the pack or model it needs, and puts an
  "Install pack" link under the field that opens the Package Center already
  scrolled to that pack. The value the node currently holds is never disabled
  — a `<select>` whose selected option is disabled has its selection dropped
  by the browser, which would silently rewrite a saved graph the moment its
  config panel opened — so it stays selectable and gets the warning instead.
  A node with a whole-pack requirement gets the same treatment as a banner
  above its parameters, which stay editable: a graph saved where the pack was
  installed still has to be readable where it is not.

- **Badges where the node is, not where the failure is.** A palette entry
  whose node needs a missing pack carries a "Needs pack" chip, and the node
  card on the canvas carries a `PACK` button in its header that opens the
  Package Center focused on what it needs. Everything above answers through
  one helper with one rule, and every unknown resolves to *available*: an
  unloaded catalog, a pack id this build has never heard of, and a server
  with no Package Center at all grey out nothing, because wrongly hiding a
  feature that works costs more than one clear error naming the pack.

- **A run that dies of a missing pack says so, and offers the fix.** The
  backend's `PackMissingError` becomes "This node needs the Word vectors
  pack. Install it from the Package Center." on the node and in the run log,
  and the failure toast carries an "Open Package Center" button that lands on
  the right card — rather than a Python exception naming a module the reader
  has no way to install.

- **The Package Center can now install the two things it could only ever
  print a command for: it restarts the server to do them.** A CUDA/ROCm
  PyTorch wheel replaces something the running server has already imported,
  and so does any pack whose live install stops at a resolver conflict —
  neither can happen underneath the interpreter serving the request, which is
  why both used to end in a line to paste into a terminal. A server started
  by `cdui start` now offers to do the install across the gap where it does
  not exist: it writes the job down (`<user data>/packs/pending_restart.json`),
  starts a detached `cdui packs-run-pending` helper, and shuts itself down;
  the helper waits for the process to go, installs into the venv, and starts
  the server again with exactly the arguments that `cdui start` was given, so
  the address the browser is still pointing at comes back. The GPU PyTorch
  card gains an **Install and restart** button with the terminal command kept
  underneath as the manual alternative, a live install stopped by a conflict
  gains a **Restart the server and install** retry on its banner, and the page
  is held behind a blocking "Server restarting" overlay until a *different*
  server process answers, at which point it reloads itself. The offer belongs
  to the server, not the browser: `GET /api/packs` carries
  `restart_available`, true only under `cdui start`, with the launcher it
  recorded still on disk and `CODEFYUI_ENABLE_RESTART_INSTALL` not set to `0`
  — a kill switch for a machine where the restart does not come back cleanly.
  It is refused, having written nothing down and installed nothing, while a
  graph is running or queued (the restart would take the run with it), while
  another install is running, and while another restart is already pending.
  Only the Python packages, or the torch wheel, are installed this way, never
  a model item — the helper runs from an interpreter with none of this app's
  downloader in it — so a pack's model files still arrive by an ordinary
  install afterwards.

- **An install nobody could watch still reports how it went.** The helper
  writes `<user data>/packs/last_restart_job.json` and keeps the installer's
  whole output in `<user data>/packs/logs/restart-<job>.log`, so the page that
  comes back from the automatic reload can toast "Server restarted. GPU
  PyTorch is ready." or the failure with its reason — and, when the record
  carried no reason at all, a second toast with the installer's last output,
  which is the only account of a subprocess that died with the old server.
  `cdui status` reads the same two files: a `Restart install` line naming the
  pack, saying *finishing* while the helper it recorded is still working and
  *abandoned* once it is not, and a `Last restart` line for an hour after one
  finished. A relaunch that itself failed leaves the install's own status
  alone — that is the field the panel reports — and adds `relaunch: failed`
  plus the log path on the end of the message, which is what makes
  `cdui status` show that line as failed even when the package installed
  cleanly. The overlay gives up in two ways rather than spinning
  forever — after thirty seconds in which the server never even stopped
  answering ("The server did not restart. Run this command, then reload:"),
  and after ten minutes — and both leave the command, when the server sent
  one, and a **Reload now** button. Free space on the volume the venv is on
  is checked before the install starts (3 GB for the torch wheel, 1 GB for a
  pack's Python packages), and a shortfall is recorded as a failed job that
  still brings the server back.

### Changed

- **`cdui start` hands the server the launcher that can start it again.**
  Nothing inside the server knows how it was launched, so `start` now exports
  `CODEFYUI_LAUNCHER` — the OUTER interpreter plus `scripts/dev.py`, as JSON
  so a Windows path with a space in it survives the round trip, and not the
  `cdui` shim, whose whole job is to find an interpreter and which a detached
  child would get a second chance to resolve differently — together with
  `CODEFYUI_RELAUNCH_ARGV`, the arguments this start was given. Those two are
  what make a restart-mode pack install possible at all; without them
  `restart_available` is false and the Package Center still prints the command
  to type, which remains the right answer for a `uvicorn app.main:app` somebody
  started by hand. `start` also learns about the claim file: while a restart
  install is *finishing* — the helper it recorded is alive, or the claim is
  under sixty seconds old with no helper pid stamped into it yet — it refuses
  to start a second server into the venv that helper is rewriting, reporting
  that a restart install is finishing and pointing at `cdui status`; once the
  helper is gone, or it never arrived and those sixty seconds have passed, the
  claim is *abandoned* and `start` deletes it and starts normally.
  And a server always comes back from a restart-mode install as a background
  daemon, even one started in the foreground with `cdui start -f`, because the
  helper that relaunches it has no console to hand over.

- **The Word Embedding Analogy example says what each backend teaches.** Its
  description explained the analogy on `demo-16d` and said nothing about the
  real thing; it now says the toy vocabulary is 59 words and makes the
  analogy exact by construction, and that `backend=glove-50d` swaps in
  400,000 real words, where queen still wins but other analogies go
  approximate — which is the point of running it twice. It is also short
  enough now that the gallery card stops cutting it off mid-word.

- **`TextEmbedding` counts as a slow node in the example test suite.**
  `_SLOW_NODE_TYPES` gains it, so the fast smoke suite validates the Sentence
  Similarity graph's shape without executing it: CI has no pack cache and
  would fail at the gate, and a machine that does have the pack would load
  half a gigabyte of weights inside a test that is supposed to be quick.

- **`HFTextGenerate` and `LLMChat` join it.** `_SLOW_NODE_TYPES` now covers
  the generation half of the chain as well, so the smoke suite validates
  both RAG graphs structurally and executes neither: one would read a
  gigabyte of Qwen2.5 weights and then decode on the CPU at a few tokens a
  second, and the other would open a socket to Ollama or to a hosted
  provider — a network round trip, a key CI does not have, and somebody's
  money.

### Fixed

- **The engine puts a node's inputs on the device that node runs on**
  ([#359]). `ExecutionContext.device` promised one device for a whole run and
  enforced half of it: `StatefulModuleMixin` moved every layer module
  centrally, while getting the *tensors* there was left to each node — and
  ten of the fourteen nodes that build a tensor from nothing never learned
  it. On a CPU-only machine the two halves cannot disagree, so the gap was
  invisible during development and showed up only on someone else's GPU:
  this repo's own `server.log` has two graphs dying of it, one fixed inside
  `PolicyRollout` by [#347] and the identical failure back the next day
  through `Conv2d`. Alignment now happens in `graph_engine.invoke_node`, the
  single function every node call goes through, so it covers plugin nodes and
  a teacher's custom node as well as the builtin set. Modules are left alone
  deliberately — `nn.Module.to()` is in-place, and relocating a model handed
  across a wire would flip weights out from under the node that owns it — as
  are datasets, DataLoaders and environments, so a dataset stays lazy.
- **A node whose work is host-side is no longer dragged onto the GPU**
  ([#359]). Aligning inputs is wrong for a node that hands them straight to
  numpy, sklearn or matplotlib, because `Tensor.numpy()` raises on anything
  but the CPU. Those nodes declare `align_inputs = False`, alongside
  `cacheable`; `TrainTestSplit` is the builtin case, and without it nine of
  the forty-one shipped example graphs — `Supervised-Learning-101` among them
  — stopped running the moment a device was selected. `Map` opts out for a
  different reason: aligning a LIST port would copy the whole collection
  on-device before the first body node ran, when the point of iterating it is
  that only one element is resident at a time.
- **Relocating a value no longer changes what it is** ([#359]). Moving a
  tensor across devices used to cost a single-field namedtuple its contents
  (`B(x=tensor)` came back `B(x=[tensor])`, silently, because calling the type
  with one list *succeeds* there), a `state_dict()` its `OrderedDict` type and
  the `_metadata` `load_state_dict` reads, a `defaultdict` its factory, and an
  `nn.Parameter` both its class and its leaf-ness — so `SGD([t])` answered
  "can't optimize a non-leaf Tensor". A container whose contents are already
  in place now comes back as the *same object*, which is what keeps a no-op
  alignment from breaking the in-place-mutation contract `PythonScript`
  documents. Tensors inside a `set` are aligned too, and the walk is
  depth-capped and cycle-guarded, because it runs inside the node's own `try`
  where a `RecursionError` would be reported as the node failing.
- **A node pinned to one device gets its weights and its inputs there**
  ([#359]). `get_or_build_module` placed the module by the run's device while
  the engine aligned inputs by the node's own `device` param, so pinning a
  stateful node produced a "must be on the same device" raise from the one
  node that had asked for something specific. Both read the pin through
  `node_target_device` now — which also stops a plugin whose `device` param
  means a serial port or a camera index from pulling every input off the
  accelerator, by requiring the node to actually declare the param and the
  value to actually name a device.
- **The startup device is the one you chose, not the best one present**
  ([#359]). The frontend adopted the backend's best available device at
  startup for anyone who had never opened Settings ([#235]). That made the run
  device a property of the hardware rather than of a decision, and a run that
  moves to a GPU on its own is a run whose failure modes nobody asked for. CPU
  is the baseline again and an accelerator is opt-in; the trade is
  discoverability, which the device dropdown in Settings currently carries
  alone. Note that an exported graph is unaffected: `python graph.py` still
  defaults to `--device auto`, and the docs now say so.

- **Two nodes reading different ports of the same upstream no longer share one
  cache entry** ([#360]). Running CF201 a second time drew the *same* picture in
  both `Visualize` nodes hanging off a `Split` — vertical edges where the
  horizontal-edge map belonged — and reported both as `cached`. The execution
  cache key was built from the *source node ids* of the incoming edges and
  nothing else, so `Split.chunk_0` and `Split.chunk_1` contributed the identical
  `split` key: two same-typed, same-param siblings hashed to one entry, and
  whichever executed first became the answer for both. `compute_key` then
  `sorted()` those ids, which erased input-port order too, so
  `MatMul(a=A, b=B)` and `MatMul(a=B, b=A)` collided as well — that one returned
  each other's numbers on every re-run, silently and with no error. Each entry
  now names its whole edge (target handle, source handle, source key) via the
  new `ExecutionCache.upstream_ref`, so the sort still normalises the order
  edges are listed in while the ports stay part of the identity. Cold runs were
  always correct; only re-runs were wrong, which is why this survived so long.

- **A typed failure keeps its type on the way to the browser.** `node_status`
  error frames now carry `error_type`, the exception's class name, which the
  engine had recorded and the run service then dropped. Every rule that maps a
  raw Python exception into a sentence a beginner can act on keys off that
  field — the missing-pack sentence with its "Open Package Center" button, and
  the pre-existing `KeyError` ("this node needs an input on `<port>`") and
  `ValueError` rules — so all of them were unreachable outside DEBUG, where a
  traceback happens to name the class in its last line. The message itself
  never carried it: `str(KeyError('tensor'))` is just `"'tensor'"`.

- **Paging through `GET /api/apps/{slug}/runs` no longer drops every run that
  shares a timestamp with the cursor** ([#372]). The `before` cursor carried
  the previous page's own `created_at` into `AND created_at < ?`, so a page
  boundary landing inside a group of runs recorded in the same microsecond
  tick excluded that whole group — including the runs the client had not seen
  yet. Two invokes in quick succession share a tick often enough that this
  endpoint's own test hit it about one run in five, and the affected rows were
  unreachable through the API entirely: no page ever returned them. The list
  now takes a composite keyset cursor, `before` plus `before_id`, which are
  the `created_at` and `run_id` of the last row of the previous page and name
  one row in the `created_at DESC, rowid DESC` ordering [#371] settled. An
  anchor that no longer resolves — pruned by retention, or belonging to
  another app — degrades to the old `created_at < ?`, which stays exact for
  the retention case because retention deletes a timestamp whole. `before`
  alone still works for clients written against the earlier contract and keeps
  its original meaning; `before_id` without `before` is a 422.

- **A follow-up pass over the Package Center installer and its launcher**
  ([#380]). The catalog now validates what its tests had only assumed; the
  byte meter cannot freeze a bar or pass 100%; an asset's sentinel is
  re-checked against its catalog entry; and an interrupted asset download
  takes its `.part` file with it rather than leave behind 69 MB nothing
  counts and nothing removes. A late event from a finished job no longer
  stamps a step on the next one, a claim stamped in the future cannot wedge
  a restart, a helper's pid stamp survives a concurrent read on Windows, and
  a claim from another checkout is refused but left alone. An interrupted
  helper waits for the old server before relaunching, a crash in it keeps its
  own message, `cdui update` and `cdui dev` stand down while a restart
  install is finishing, and two `cdui packs` prompts stop misreporting.

## [2.4.1] — 2026-08-22

Four fixes that landed on `main` in the hours after 2.4.0, and nothing else:
one that stopped the Windows installer at its very first step, three on the
canvas.

### Fixed

- **The Windows installer no longer gives up when winget's `msstore` source is
  unreachable** ([#354], contributed by [@oyea0801]). On some machines — school
  computer labs, corporate networks — `msstore` fails certificate pinning with
  `0x8a15005e`, typically because a proxy or antivirus performs TLS
  interception, or because App Installer is too old for Microsoft's rotated
  certificates. `Git.Git` was sitting in the working `winget` source the whole
  time, but with no `--source` passed, winget called the partial source failure
  an ambiguous result and exited non-zero — and the installer stopped there, at
  step one. The install call now pins `--source winget`, which skips `msstore`
  entirely. For a machine where winget is absent or its own source is also
  unreachable, a new portable-Git fallback extracts Git for Windows' PortableGit
  into `%LOCALAPPDATA%\CodefyUI\PortableGit` and puts it on the user PATH — no
  administrator rights, no winget involvement. It downloads over the Windows
  certificate store, so an inspection proxy whose root CA is already on the lab
  image does not break it.

- **A media port shows its picture or its clip, not its base64 or its `repr`**
  ([#355], contributed by [@oyea0801]). Double-clicking a Visualize node and
  opening Outputs showed its `image` port as a wall of base64 text, seconds
  after the same picture had rendered correctly in the execution log; VideoWrite's
  `video` port had the same problem dressed as a Python dict `repr`. The
  declarations were right and the backend was already sending both — the rows
  were reading the wrong source. `/api/execution/outputs` truncates every string
  at 4000 characters and a matplotlib loss curve is ~34 600 base64 characters,
  so the captured value was not merely ugly, it was a fragment that decodes to
  nothing; a video value is a reference dict the endpoint can only describe as a
  `repr`. The rows now read the media from the tab's log, where the `node_status`
  entry already names the port it came from, and render the picture or the clip
  in place of the captured value. It works with *Record outputs* off, too, when
  the port fetches have nothing at all. No backend change — the data was already
  on the wire and already correct.

- **The quick-add palette is dismissable by click and by Escape** ([#356],
  contributed by [@oyea0801]). Double-clicking the canvas opens the quick node
  search. Clicking the canvas to dismiss it did nothing, and after one such
  click Escape stopped working too, leaving a palette on screen with no way out
  at all. One root cause wearing two hats: React Flow's pane is panned by
  d3-zoom, whose mousedown handler calls `stopImmediatePropagation` so it can own
  the drag, so the event died at the pane and the bubble-phase outside-click
  listener never fired — which is why the palette closed for a click on the
  toolbar but not on the canvas, the first place anyone dismissing it clicks.
  Escape was the downstream consequence: it lived on the input's own
  `onKeyDown`, and the click had blurred the input. Both listeners now sit on
  the document, the pointer one in the capture phase, which reaches the document
  before the pane's own handler runs.

- **Connection handles are big enough to hit on the first try** ([#357],
  contributed by [@latteine1217]). A port dot was 10px across with a 2px ring in
  `--surface-raised`, the same colour as the node card behind it, so only a 6px
  core ever read as the dot — on a projector, in front of a class, a target
  teachers miss. The dot is now 17px with a 13px visible core, and every handle
  carries a transparent 20px press ring so the target can outgrow the dot
  without the diagram gaining weight; port rows go from 4px to 8px of vertical
  padding, because at the old 24px row pitch those rings overlap and
  neighbouring ports steal each other's presses. The Start and trigger diamonds
  come *down*, from 12px and 14px to 13px: rotating a square 45° stretches its
  widest span to 1.41× its side, and its ring is `--text-primary`, part of the
  shape rather than camouflage, so at nominally similar numbers the diamond read
  about five times heavier than the dot. Measured in the running app the two are
  now 169px² against 133px². Handle geometry lives in three tokens, which
  PresetNode and the layers-editor nodes read as well, instead of six inline
  copies.

## [2.4.0] — 2026-08-22

### Added

- **Load now asks where the graph should land, and its list can be searched**
  ([#352]). The Load menu opens on two destinations instead of straight onto
  the list of saved graphs. *Load into this canvas tab* replaces what is on the
  canvas and binds the tab to nothing, so the next Save asks where it should go;
  *Load and save* is the load this menu has always performed, which adopts the
  file so Save writes straight back over it. The two were one action before,
  which meant opening a saved graph to look at it silently took over where the
  tab saves — and since a Save in project mode overwrites the bound file in
  place with no prompt, the graph you opened was the graph you overwrote.
  Because the unbound path replaces live work, it confirms first; an empty
  canvas has nothing to lose, so it does not ask. Each destination opens the
  same flyout beside it: a search box over the saved graphs, matching the file
  name as well as the label, and switching destination neither refetches the
  list nor throws away what has been typed.

### Fixed

- **A long list of saved graphs is reachable again** ([#352]). The list rendered
  straight into the shared menu panel, which clips at `overflow: hidden` under
  no height cap at all — so past a screenful of saved graphs, every row below
  the fold could be neither scrolled to nor clicked, on an install with 64 of
  them. The list now lives in the flyout above, which scrolls.

## [2.3.0] — 2026-08-21

### Added

- **`cdui plugin sync` — one command to catch up on built-in packs, and an
  uninstall that is finally remembered** ([#175]). A built-in pack is activated
  by a lockfile entry, and an update never writes one. So a release that *adds*
  a pack — `stats` did — puts its files on every machine and loads them on
  none: the nodes are installable and invisible at the same time, a class
  follows the update instructions, the new chapter's palette is empty, and the
  only cure was typing an id list nobody knew existed. `cdui plugin sync`
  installs every built-in pack this install has made no decision about, with
  `--dry-run` to just list them, `--yes` for scripts and classroom images, and
  `--prune` to drop lockfile entries for packs that no longer ship (which
  discovery had been skipping in silence). Failures are per pack: one pack whose
  `python_deps` will not download on a school network reports and the rest still
  install, because a batch that aborts on the first failure is a batch nobody
  can use offline.

  Two shapes were rejected on the way here, and the reason is the same one:
  syncing at startup, and prompting during `cdui update`, both activate code the
  user never asked for because a release shipped it. That is a consent decision,
  so it stays a verb they type. Which exposed the real bug — `uninstall` popped
  the lockfile entry outright, making "this install has never heard of the pack"
  and "the user threw it away" the same state, and those are precisely the two
  states a catch-up command has to tell apart. Uninstalling a built-in pack now
  leaves a **tombstone** (a `removed` map beside `plugins`, so every existing
  reader of `plugins` keeps its exact meaning), `sync` names it as skipped
  rather than re-installing it, the `cdui start` and `cdui plugin list` notices
  stop nagging about it, and installing the pack by name clears the record. A
  lockfile written before this field loads unchanged; only built-in packs are
  tombstoned, since they are the only ones sync could ever put back uninvited.

- **Plugin apiVersion 4: `api.graph.getView()` — a plugin can now tell which
  level of the graph the user is looking at** ([#200] item 7). A graph nests, and
  entering a block swaps its insides onto the one canvas. `getGraph()` has always
  flattened that and answered with the whole graph; `applyOperations()` has always
  written to the canvas in front of the user. Both are defensible, but a plugin
  had no way to tell the two apart, so a batch issued while a block happened to be
  open landed inside the block — `clear_graph` emptied the block instead of the
  graph — and nothing in the API said so. `getView()` returns `{ depth, path,
  atTopLevel }`, read-only and live, so a plugin can refuse, warn, or wait
  instead of writing blind, and the write rule is now documented API rather than
  an accident. Purely additive: nothing about where writes land changed, no
  installed plugin needs a line, and `apiVersion` is how you feature-check
  (`api.apiVersion >= 4`). The apiVersion 3 contract is now frozen under test
  too, the same way v2 already was.

- **A language model can now be pretrained on the canvas** ([#289], [#290],
  [#291]). Seven LLM nodes close the gap epic [#292] named: the catalog could
  describe a transformer but had no path from raw text to trained weights, so
  the only way to train a decoder was to leave CodefyUI.

  `CausalLMModel` is a pre-LN GPT-style decoder — learned/sinusoidal/RoPE
  positions, LayerNorm or RMSNorm, optional weight tying and gradient
  checkpointing — whose forward is `input_ids (B, T) -> logits (B, T, V)`, so
  `Optimizer`, `TrainingLoop`, `CheckpointSaver` and `ModelSaver` need to know
  nothing about language models. Its defaults build 203,668,480 parameters.
  `LMCrossEntropyLoss` flattens the batch and time axes before
  `F.cross_entropy` — deliberately *not* a subclass of `nn.CrossEntropyLoss`,
  because `TrainingLoop`'s classification gate would then have run
  `argmax(dim=1)` over the time axis and reported a meaningless accuracy that
  early stopping can be asked to monitor.

  Data comes in through `TextCorpusDataset` (text rows from the Hugging Face
  Hub or a local `.txt`) and `LMTokenizedDataset`, which concatenates every
  document into one token stream and cuts fixed-length blocks whose labels are
  the inputs shifted by one — no padding, and a disk cache keyed on everything
  that changes the stream, because tokenising a real corpus is minutes that
  produce the same bytes every time. `LMTokenizer` supplies one tokenizer
  object to every node that needs one, over a duck-typed `ANY` port
  (`encode`/`decode`/`eos_id`/`vocab_size`) rather than a new wire type.

  `PerplexityEvaluate` answers the question `EvaluateModel` cannot: a language
  model is wrong most of the time and should be, so it is scored on the
  probability it put on the token that actually came next. The mean is
  token-weighted, so the number does not move when `batch_size` does.
  `TextGenerate` samples text with temperature / top-k / top-p, slides the
  context window past `max_seq_len`, and draws on the CPU so one seed gives
  the same text on a laptop and on a GPU box.

  A new example, **Train a Causal LM on TinyStories**
  (`examples/LLM/TrainCausalLM-TinyStories`), wires the whole chain up at bf16
  with an effective batch of 32 sequences, then scores a held-out split and
  writes a sample. It ships with a `README.md` carrying the recipe, both token
  budgets and the memory levers, because the card cannot: the canvas gallery
  truncates a description at 80 characters and shows no tooltip, so the card
  leads with the only two facts that decide whether the graph can run at all —
  a 16 GB GPU, and one first-run download. A test asserts those two survive
  that truncation, and a second asserts every number the README quotes is
  still derivable from the graph's params — each of them is a relationship
  between two nodes, which is exactly what graph validation cannot see.

- **The optimizer and loss applicability tables are now checked against the
  installed torch** ([#189]). `#134` declares which algorithm accepts which
  hyperparameter rather than inferring it from `inspect.signature`, because
  inferring it would forward `eps` to Adagrad — whose torch default is 1e-10
  against the Adam family's 1e-8 — and silently retune every existing Adagrad
  graph. The cost of declaring is that the table can stop describing torch
  without anything saying so. Two tests per node close that: one asserts each
  declared set still equals "accepts it *and* agrees with our default", the
  other fails when any new keyword appears in a torch signature. Neither
  forwards anything automatically.

- **OOM crossed with determinism** ([#193]). Both paths reach for
  process-global state and their interaction had no test. Two now pin it: OOM
  recovery runs *inside* the determinism scope, and an OOM unwinds that scope
  rather than stranding it.

- **A vision-language-action policy can now be trained, evaluated and watched
  on the canvas** ([#311], [#312]). Five nodes in a new `VLA` category close
  the wave epic [#309] scoped: the catalog could build a transformer and,
  since the LLM wave, train a decoder — but there was no environment to act
  in, no demonstrations to clone and no closed-loop number to report, so the
  loop every VLA paper is built around had to happen outside CodefyUI.

  `PushWorldEnv` is that environment, and it installs nothing. A white agent
  disc, one to four coloured pucks and one or two coloured ring targets on a
  96px canvas, all of it pure torch: dynamics are circle-overlap resolution,
  rendering is anti-aliased distance fields on a cached meshgrid, and there
  is no pygame, no pymunk and no gymnasium behind any of it. Every episode
  carries an instruction of the form `push the {color} puck to the {color}
  target`, and that sentence is load-bearing by construction — from
  `n_distractors: 1` upward there are two targets and at least two pucks with
  the colours freshly shuffled per episode, so nothing in the pixels says
  which puck or which target is the goal. What travels on the wire is a
  factory rather than a live episode, so `(seed, config)` reproduces an
  episode tensor-exactly and two consumers can draw from disjoint seed
  streams without coordinating.

  `PushWorldDemos` rolls a scripted expert through it — approach the point
  behind the puck, orbit if caught on the wrong side, then push through it,
  recomputed every step so it self-corrects, and pinned in the suite at 99 of
  100 seeds solved, in under 40 steps on average across the ones it solves —
  and emits behavior-cloning samples shaped `((image, instruction bytes,
  action chunk), action chunk)`. The chunk rides inside `data` as well as the
  target so a flow-matching model can noise it inside `forward` while
  `TrainingLoop`'s contract stays untouched; instructions are UTF-8 bytes
  zero-padded to 48, because at this scale a 256-row byte embedding is the
  whole language stack and BPE would buy a 50k-row table. `demo_noise`
  defaults to 0.5 and is DART rather than decoration: with probability one
  half per step the action *executed* in the environment is perturbed while
  the recorded label stays the expert's, because pure on-trajectory
  demonstrations contain no recovery states and a cloned policy that drifts
  one pixel off the manifold has never seen the situation it is now in. The
  prototype measured that as 4% closed-loop success trained clean against 24%
  with noise 0.5, everything else equal, before any architecture change. The
  held-out split runs on a seed stream offset by 1,000,000 so it can never
  share an episode with training at any `episodes` value, and `demo_video`
  hands the first few episodes straight to `VideoWrite`.

  `VLAModel` is the policy: a vision stem and a 256-row byte embedding of the
  instruction feed a pre-LN bidirectional trunk over `[vision; text]` tokens,
  and an action expert — self-attention over chunk queries, cross-attention
  into the trunk — predicts H actions at once, 3,339,938 parameters at the
  defaults. `head_type` is the research knob the node is shaped around:
  `flow_matching` is the pi0 / SmolVLA family (noise the chunk, learn the
  velocity field, Euler-integrate `flow_steps` at inference), `regression` is
  direct MSE behavior cloning, and the trunk, the data and every other knob
  are held fixed across the two, so the dominant continuous-action paradigms
  can be compared instead of argued about.

  The loss comes out of a port. A flow head trains on the residual `v_pred -
  v_target` against zero, a regression head on MSE against the chunk, and a
  generic loss node wired to the wrong head would train the wrong objective
  with nothing anywhere reporting it — so there is no separate VLA loss node
  to mismatch. `VLAModel` emits the mode-matched `loss_fn` on an output port
  beside the model, and the mistake cannot be built. (The issue sketched a
  standalone `VLALoss`; the deviation is deliberate and recorded on [#312].)
  The emitted loss reduces in float32 so bf16 autocast cannot degrade it, and
  is deliberately not an `nn.CrossEntropyLoss` subclass, for the same reason
  `LMCrossEntropyLoss` is not: `TrainingLoop`'s classification gate keys on
  that `isinstance`.

  Evaluation arrives in both halves the literature reports. `VLARollout` runs
  fresh episodes from a seed stream offset by 2,000,000 — so a default-wired
  graph never evaluates on initial states it trained on — and reports
  closed-loop `success_rate`, mean episode length, a per-episode text report,
  per-episode metric series, and the rollout frames bordered green or red by
  outcome, ready for `VideoWrite`. Two of its params are experiments rather
  than settings. `execute_k` is the receding horizon (predict H, execute k,
  re-plan): measured on one trained policy at 46% for k=2, 34% for k=4 and
  20% for the full chunk of 8 — the open-loop compounding-error curve as a
  single knob. `instruction_mode: swapped` hands the policy a distractor
  puck's colour instead of the real goal's, where a pixels-only policy scores
  exactly what it scored before and a language-reading one collapses (46% to
  2% on the same prototype policy). `VLAActionEval` is the fast complement —
  chunk MSE against the expert over the held-out demos, seconds rather than
  minutes and deterministic per seed — and a low action MSE beside a low
  success rate is precisely the compounding-error signature that `demo_noise`
  and `execute_k` exist to manage.

  A new example, **Train a VLA on PushWorld**
  (`examples/VLA/TrainVLA-PushWorld`), ships the wave's acceptance graph as a
  one-click start ([#332]): 13 nodes carrying 2,400 DART-noised demonstration
  episodes into a 3.3M-parameter regression-head policy, AdamW at 1e-3 under
  a `warmup_cosine` schedule stepped per optimizer step, 110 epochs, then
  closed-loop rollout, open-loop MSE and two `VideoWrite` artifacts that play
  inline in the editor. Reported from an RTX 4080 run of about 56 minutes
  under 3 GB of VRAM: `success_rate` 0.967 (29 of 30), mean 45.6 steps of a
  120-step budget, held-out action MSE 0.588 — and 0.033 (1 of 30) with the
  instruction swapped and the weights unchanged, which is the row the example
  exists for. The `README.md` beside the graph carries the recipe, the
  ablation playbook and a scale-down configuration, because the card cannot:
  the gallery truncates a description at 80 characters, so this one leads
  with the GPU and the hour.

  One claim in that node set was corrected against a controlled experiment
  before any of it shipped ([#338]). `vision_stem` was written describing the
  conv stem as the better one and citing the "early convolutions help
  transformers see" result, an attribution taken from prototype notes in
  which the stem and the dataset size had changed in the same iteration — so
  it measured neither. A two-arm study on the same data, the same seed and
  the same 1,200-episode / 45-epoch budget, with the stem as the only
  variable, put `patchify` ahead: 0.85 success against 0.45. The param text
  now states that result, records that only `conv` has been run at the full
  2,400-episode / 110-epoch budget (0.97), and says the knob exists to settle
  exactly this. What is not here: discretized action tokens (the OpenVLA
  family) would be a third `head_type` rather than a rewrite, and nobody has
  written it; and `VLA` is in neither the sidebar's curated category order
  nor the empty-canvas gallery's, so the nodes sort in after the taught
  categories and the example lands in the gallery's catch-all section.

- **Video on the canvas — a run can write a clip and the results panel plays
  it** ([#310], the first slice of epic [#309]). One `node_status` event is
  capped at `RUN_EVENT_PAYLOAD_CAP_BYTES` — 128 KB, two orders of magnitude
  under a clip — so video cannot ride the event stream the way
  `MEDIA_IMAGE`'s base64 PNG does. It travels by *reference* instead: nodes
  write files under a new `settings.MEDIA_DIR` (`<data>/media`,
  `assets/media` in project mode), the new `GET /api/media/{path}` serves
  them inline with a real `Content-Type` — the existing download routes force
  `application/octet-stream`, which a `<video>` element cannot play — and the
  new `MEDIA_VIDEO` port kind carries a small validated dict of `path` /
  `url` / `format` plus optional `fps`, `frames`, `width`, `height`, `bytes`.
  `FileResponse` answers Range requests, so seeking in the player works.

  Zero new Python dependencies, on the tensorboard precedent that turning a
  feature on should cost an install nothing — torchvision removed its video
  API in 0.26 and vendoring PyAV for one feature is not the trade.
  `VideoWrite` accepts `(T,C,H,W)` or `(T,H,W,C)`, float `[0,1]` or uint8,
  gray or RGB, and encodes gif through Pillow — always available — or mp4 by
  piping rawvideo to an `ffmpeg` binary when one is on `PATH`. `format: auto`
  picks mp4 exactly when ffmpeg is there, and the error when it is not names
  the gif fallback. It also emits a middle-frame PNG on a `MEDIA_IMAGE` port,
  capped at 256 px on its longest side, so a clip has a face in any client
  that has not learned the video kind. `VideoLoad` is the other direction —
  gif via Pillow, mp4/webm via `ffmpeg`/`ffprobe` — returning `(T, 3, H, W)`
  float `[0,1]`, with `max_frames` to stop the decode early and `stride` that
  scales the reported fps down so playback duration holds. Both are
  `cacheable = False` for `ImageWriter`'s reason: the file on disk *is* the
  output, so a cache hit would hand back a reference to a file that may no
  longer exist.

  The frontend `LogKind` union gains `'video'`, and the panel plays mp4 in
  `<video controls loop>` and gif in `<img>` — a gif is an animated image,
  not a valid `<video>` source. The wire validator forwards only a closed
  list of reference keys and refuses a path that is not relative, judged
  under `PurePosixPath` *and* `PureWindowsPath` because `Path.is_absolute()`
  is platform-shaped in both directions: POSIX waves `C:/leak/a.mp4` through
  as a filename and Windows waves `/leak/a.mp4` through as drive-relative.
  `..` is refused as any component, not only as a prefix. `/api/media` is
  read-only — files arrive only by a node writing one, there is no upload and
  no delete — so its GETs stay unauthenticated reads like every other
  download route, and the extension allowlist means a file written with any
  other suffix is unreachable through it.

- **`CausalLMModel` can now be ablated: grouped-query attention, qk-norm, and
  a bias switch** ([#299]). The node shipped able to train exactly one
  architecture. Studying architecture needs knobs that change it, and these
  three land as advanced params whose defaults leave today's model
  bit-for-bit unchanged.

  `n_kv_heads` (0 = `n_heads`) is grouped-query attention. At the default the
  fused `qkv` projection is kept exactly as it was, so every existing config
  keeps its parameter shapes and its initialization stream; below `n_heads`
  the module switches to split `q` / `kv` projections and SDPA's
  `enable_gqa`, and `n_kv_heads=1` is multi-query attention. It has to divide
  `n_heads` evenly, and the error names both parameters rather than only the
  one that was typed. On the default 203,668,480-parameter configuration the
  trade is readable straight off the node's `param_count` output: 184,775,680
  at `n_kv_heads=4`, 180,052,480 at 1.

  `qk_norm` RMS-normalizes each head's queries and keys before the attention
  dot product — the standard intervention for training at high learning rates
  — for 1,536 extra parameters on that same configuration, all of them 1-D
  gains the initialization pass deliberately leaves at one. `bias=false`
  gives Llama-style bias-free attention and MLP linears, 110,592 parameters
  lighter. All three join `structural_params`, so editing one honestly
  discards the weights persisted for that node rather than trying to load a
  checkpoint into a different shape. The scope is training-time architecture:
  GQA's other well-known payoff is a smaller KV cache at decode time, and
  `TextGenerate` deliberately keeps no KV cache at all, so what moves on the
  canvas is parameter count, memory and quality — not generation speed.

- **`DataMixDataset` — an eighth LLM node, for mixing corpora** ([#300]).
  What ratio of TinyStories to wikitext? Does easy-then-hard ordering help?
  Neither question could be asked on the canvas, because `LMTokenizedDataset`
  takes one corpus. `DataMixDataset` takes two to six — dynamic
  `corpus_1..corpus_N` ports driven by a `sources` param, the same
  `resolve_count_param` convention `ComposeTransform` uses — and emits one
  dataset of raw text rows for `LMTokenizedDataset` to tokenize, plus a
  `num_rows` scalar and a per-source row-count breakdown in the run log.

  `interleave` draws rows proportionally to `weights`, seeded and without
  replacement, so the same seed over the same inputs reproduces the same
  mixture exactly; picks are drawn in chunks of 8,192 so a million-row corpus
  does not put a Python loop around a kernel launch. Rows within one source
  keep their relative order — shuffling stays `DataLoader`'s decision,
  downstream. `concat` is the curriculum: `corpus_1` in full, then
  `corpus_2`, and so on. Two behaviours are documented on the params rather
  than left to be discovered: a source that empties stops being drawn and the
  remaining weights renormalize, so the tail of a mixture is whatever corpora
  still have rows; and a blank `weights` means equal weights, the only
  default that is valid at every source count. The mixture itself stores only
  `(source, row)` index pairs and reads through a new `MixedTextDataset`
  adapter, so mixing two Hugging Face corpora never materializes their text.
  The node is `cacheable = False`, since it consumes live dataset handles a
  fingerprint cannot describe.

- **The training loop can now run on the optimizer-step clock, and report on
  itself while it does** ([#297], [#298]). `TrainingLoop` stepped its LR
  scheduler once per epoch, full stop. A step-budgeted language-model run —
  `epochs=1` with `max_steps=1500` over a packed corpus — therefore took one
  scheduler step for the entire run: the acceptance run for epic [#292]
  declared `OneCycleLR(total_steps=1500)`, traversed about a tenth of a
  percent of it, and trained at an effectively constant learning rate. It
  still converged, to a validation perplexity of 19.17, which is exactly what
  makes it worth fixing — the schedule the graph declared and the schedule
  that ran were different objects, and nothing said so.

  `TrainingLoop.scheduler_step` picks the clock: `epoch` is the historical
  behaviour and stays the default, `optimizer_step` advances the scheduler
  exactly once per applied optimizer step. `ReduceLROnPlateau` is
  metric-driven and stays per-epoch in both modes, because there is no
  per-step metric for it to react to. `LRScheduler` gains the three shapes a
  pretraining run actually asks for — `warmup_cosine`, `warmup_linear` and
  `constant_with_warmup` — each a `SequentialLR` composing a linear ramp from
  ~0 over `warmup_steps` with, respectively, a cosine decay, a linear decay,
  or a hold, all denominated in scheduler steps over `total_steps`.

  Four telemetry switches ([#298]) make the run something you can study
  rather than only watch, all off by default and all thinned by
  `log_interval`. `log_grad_norm` records the pre-clip global gradient norm
  as a `grad_norm` series — free when clipping is on, because
  `clip_grad_norm_` returns that number anyway, and measured through an
  infinite-threshold clip call when it is not, so the fp16 unscaling is
  identical either way; with `grad_clip_norm` set, a `grad_norm_clipped`
  series makes the clipping pressure visible. `log_update_ratio` records
  `||lr * grad|| / ||weights||`, the classic learning-rate health signal,
  around 1e-3 on a healthy run. `val_every_steps` runs the wired
  `val_dataloader` mid-epoch and records its own `val_loss_step` series
  against optimizer steps — for an `epochs=1` run that is the only way to get
  a validation curve instead of one terminal point — and the pass consumes no
  RNG, so the training stream is undisturbed. `checkpoint_every_steps` writes
  step milestones through the existing periodic-checkpoint path, each stamped
  with its step count so the milestones are distinct files rather than a
  rolling latest. That stamp is the trade: the checkpoint's `epoch` field
  carries the step count, so resuming `start_epoch` from one of them is not
  meaningful, and the parameter says so. Every series goes to the run's
  metric store and to TensorBoard through the same call, so the two cannot
  disagree about what was recorded.

- **`TrainingLoop` now hands out the optimizer it actually trained with**
  ([#148]). `CheckpointSaver.optimizer` could only be fed from the
  `Optimizer` node, and the checkpoint it wrote was correct for two reasons,
  neither of them visible on the canvas: `_prepare_optimizer` usually mutates
  that same object in place, and the `train.model -> save.model` edge happens
  to force `TrainingLoop` to run first. The `rebuilt` branch breaks the first
  one outright — when the optimizer on the port does not line up with the
  model, the loop constructs a fresh one and the object the graph saved never
  trains at all, so the checkpoint stored optimizer state that had never seen
  this model. No error, and a file that looks entirely valid. The new
  `optimizer: OPTIMIZER` output carries whichever optimizer the run settled
  on, on every path including an interrupted one — which is precisely when
  the optimizer state matters — so the save path becomes an edge you can draw
  instead of an assumption you have to know about. Additive: existing graphs
  are unaffected, and no shipped example is rewired here.

- **The layers editor can now build a recurrent or attention model**
  ([#346]). The backend's `_build_layer` already knew how to construct
  `LSTM`, `GRU`, `MultiHeadAttention`, `TransformerEncoder`,
  `TransformerDecoder` and `SelectIndex`, but none of them were in the
  editor's palette — the only way to get one into a layer-editor model was to
  hand-edit the serialized spec. All six are now draggable, plus a brand-new
  `RNN`, and `graphSerialization` learns eight new type names (those seven
  and `Reshape`, which is recognised on load but is still not in the palette)
  so a file containing one stops round-tripping as `Unknown`.

  `RNNBlock` is new on the backend beside `LSTMBlock` and `GRUBlock`: the
  plain-tanh cell the gated ones exist to improve on, without which the
  controlled comparison that motivates gating at all — same graph, same data,
  same seed, swap the recurrent layer — could not be built here. All three
  recurrent wrappers now default to `batch_first=True`, because inside a
  layer-editor model the input always arrives from a DataLoader, which yields
  `(batch, seq, feature)`; torch's own default would read that as `(seq,
  batch, feature)` and silently transpose the two, training on nonsense with
  no error anywhere. The editor's param form is int/float only, so this could
  not be a checkbox — it is the default instead, and an explicit
  `batch_first` in the layer spec still wins. `RNNBlock` joins the curated
  `full_model` unpickling allowlist ([#288]) alongside its siblings, so a
  model containing one loads back. The two new palette groups reuse hues
  rather than adding them: Recurrent takes the blue the canvas already paints
  an RNN node and Attention the purple it paints a Transformer node, so a
  layer inside `SequentialModel` is the same colour as the node it
  corresponds to — the pairing table went from eleven roles to thirteen, not
  from seven hues to nine.

- **`SyntheticSequence` and `MaskedFill` — the two nodes a sequence lesson
  could not be built without** ([#346]). The Data category's three synthetic
  generators are all spatial or tabular — `SyntheticDataset` makes 2D points,
  `SyntheticSegmentation` an image plus a mask, `SyntheticShapes` an image —
  so the recurrent nodes had no zero-download dataset to train against at
  all: one forward pass on a hand-typed `TensorInput` was the whole story.
  `SyntheticSequence` emits the standard *memory* benchmark — a
  length-`seq_len` sequence of distractor tokens with the answer hidden at
  one end and a label obtainable only by carrying it across. The
  `recall_first` / `recall_last` pair is the point: identical generator,
  identical shapes, identical task difficulty, dependency length `T` versus
  1, so flipping one dropdown isolates *distance* as the variable and shows a
  plain RNN's gradient dying. Answers occupy tokens `0 .. n_classes-1` (they
  are also the labels) and distractors the range above, and a `vocab_size`
  output reports the `num_embeddings` a downstream `Embedding` needs. Each
  sample is `(sequence (T,) int64, label int)`, which drops straight into
  DataLoader then TrainingLoop with no adapter, and the whole tensor is
  materialised once under one seed so a shuffling DataLoader with any worker
  count stays deterministic.

  `MaskedFill` applies a boolean mask — `AttentionMask`'s `True = blocked`
  convention, broadcasting by torch's normal rules so a `[seq, seq]` mask
  covers `[batch, seq, seq]` or `[batch, heads, seq, seq]` scores unchanged,
  and a non-boolean mask is read as non-zero = blocked. Until now
  `scores.masked_fill(mask, -inf)` was reachable only from *inside* a
  packaged attention node, so a graph that spells attention out one node per
  step — `MatMul`, `ScalarMultiply`, `Softmax`, `MatMul` — could build a
  causal mask and had no way to apply it: the masking half of the mechanism
  was the one step that could not be shown on the canvas. The default fill is
  `-inf` and the node belongs *before* the softmax, which is what makes a
  blocked position get exactly zero probability while the survivors
  renormalise to sum to 1. `zero` and `custom` fills are offered and
  documented as not giving correct attention weights.

- **`batch_first` on `MultiHeadAttention` and `PositionalEncoding`**
  ([#346]). Both took torch's transformer layout, `(seq, batch, embed)`,
  while the recurrent nodes work in `(batch, seq, feature)` — so the two
  halves of a sequence model disagreed about which axis was which. Both nodes
  now carry a `batch_first` param that switches the accepted 3D layout to
  batch-first; 2D input is `[seq, D]` either way. It defaults to `False`
  because existing graphs were built against the old layout. On
  `MultiHeadAttention` it is listed in `structural_params`, so flipping it
  drops the persisted module and builds a fresh one rather than reusing a
  differently-shaped `nn.MultiheadAttention`, and the verbose step trace
  stops transposing for display when the input is already batch-first.

- **Eight RL nodes: the canvas can now run a policy in an environment, score
  it, and train a reward model on preferences** ([#347]). The RL category had
  `EnvWrapper` (gymnasium's CartPole), `DQN`, `PPO`, `RewardModel` and
  `KLDivergence` — and no way to execute the sentence every algorithm starts
  with, "use the current policy to collect a batch of trajectories". Rollout
  was faked with a `TensorInput`, which meant the one thing that makes
  reinforcement learning *reinforcement* learning was the one thing a student
  never saw run. The change is purely additive: eight new node classes, eight
  new test files, and zh-TW descriptions — no existing file was modified.

  `GridWorldEnv` is the textbook N-by-N grid with no gymnasium dependency:
  start top-left, goal bottom-right, traps where you put them, reward sparse
  and terminal by default (0 per step, +1 at the goal, -1 in a trap, 30-step
  cap). It is a plain object exposing only `reset()` and `step(action)` —
  anything more is surface a lesson has to explain away — and state is a
  one-hot over cells, so a single `Linear` is an exactly-expressive tabular
  policy. `PolicyRollout` runs a policy in it for `episodes` rollouts and
  returns the batch flattened: `states`, `actions`, `rewards`, `logits`,
  `log_probs`, plus per-episode `returns`, `episode_lengths`, `episode_ids`,
  a `success_rate` and two text tables. Actions are *sampled* from
  `softmax(logits / temperature)`, never argmaxed — that is both the
  exploration dial and the entire basis of the group baseline below — and
  `log_probs` is recorded at sampling time because that number is PPO's
  `log_probs_old` and is unrecoverable once the policy updates.

  `Discount` folds a reward sequence backwards into `G_t = r_t + gamma *
  G_{t+1}` — one right-to-left pass, linear time — and restarts the fold at
  each boundary named by `episode_ids`, so one episode's ending cannot leak
  into the previous one's returns. `PPOClipObjective` returns every part of
  `min(r*A, clip(r, 1-eps, 1+eps)*A)` rather than just the value: the
  unclipped term, the clipped term, the minimum, a mask of which samples the
  clipped branch actually decided, the ratio used, the scalar loss and a
  `clip_fraction`. *Which branch won* is the part a lesson is trying to show,
  and the mask is deliberately "the clipped branch decided this", not "the
  ratio left the interval" — at zero advantage the two branches agree and
  nothing was truncated. `GroupRelativeAdvantage` is GRPO's one change to PPO
  on its own: `A_i = r_i - mean(r)` within each group, with an optional
  `expand_index` input that broadcasts a per-episode advantage back over
  every step.

  `PreferenceDataset` and the two Bradley-Terry nodes make reward hacking
  reproducible rather than anecdotal. The dataset spreads true quality thinly
  across `signal_dims` coordinates and plants one loud *shortcut* coordinate
  that tracks quality in the training split and is pure noise in the holdout;
  nothing else about the two splits differs. `BradleyTerryLoss` is the
  arithmetic alone — `P(w beats l) = sigmoid(r_w - r_l)`, `loss = -log P`,
  computed through `softplus(-diff)` so a huge score gap does not overflow —
  which is what lets a lesson show that *only the difference matters*, and
  therefore why two reward models' scores are not comparable at all.
  `BradleyTerryTrain` fits the model and measures accuracy on *both* splits
  every epoch, because the gap is the only thing that can see the shortcut:
  at the defaults training accuracy reads 1.0000 either way, while holdout
  accuracy is 0.9609 with `shortcut_strength = 0` and 0.7773 with the
  shortcut planted, an ordering asserted across five seeds.

  Honest limits, taken from the nodes' own docstrings. The reward-hacking
  demonstration is a *gap*, not a divergence curve — at 512 synthetic pairs
  the fit is essentially immediate, so there is no "learns it right, then
  learns it wrong" phase to watch, and `shortcut_strength = 0` is the control
  that attributes the gap to the shortcut rather than to ordinary
  overfitting. A GRPO group where every sample scores the same yields
  all-zero advantages and teaches nothing, which is a constraint on the task
  rather than a bug. `GridWorldEnv`, `PolicyRollout` and `BradleyTerryTrain`
  are `cacheable = False`, each for a stated reason — an env carries position
  state, a rollout is stochastic experience, a trained model would come back
  already fitted. A device-safety test pins the rest: no RL node declares a
  device or reaches for a GPU, `PolicyRollout` forwards on the model's own
  device and records back on the CPU so no downstream node has to know where
  the policy sat, and the numbers do not move with the device. No example
  graph ships with them.

- **The settings popover shows what this server is running, and how much its
  caches are holding** ([#193] item 2). `/api/health` has reported the
  running version, the registry counts and a per-store cache byte breakdown
  since #135, and the frontend read exactly one field of it — `project`,
  once, at bootstrap — and dropped the rest. A user on a bounded memory
  budget had no way to see how close they were, and a bug reporter had no way
  to state their version from inside the editor. A "This Server" section at
  the bottom of the gear popover now shows the version, the node and preset
  counts, and each cache store's usage against its budget. The backend is
  untouched.

  It reads `/api/health` when the popover opens — `SettingsPopover` renders
  nothing while closed, so "fetch on open" needed no plumbing — with a
  Refresh button rather than a poll: the numbers do move during a run, but a
  timer costs a request per interval for figures nobody is necessarily
  reading. A read that fails leaves an inline line and *keeps* whatever
  numbers it already had, which says which half of the panel is
  untrustworthy; a toast would have outlived the popover it belongs to. The
  three stores do not share a shape, so `caches` is read as an open map of
  numbers rather than a named-field interface: the budget is `max_bytes` for
  the run-output and node-state stores but `max_bytes_each` for the execution
  cache (one instance per WebSocket, so the total has no single ceiling), and
  a store this build has never heard of is listed under its raw name rather
  than dropped. A configured budget of zero means *unbounded* to every store,
  so it renders as a bare size rather than as "1.5 GB of 0 B", which would
  say catastrophically over the limit when it means the exact opposite.
  `formatBytes` steps by 1024 and still says KB/MB/GB, because the budgets it
  prints are configured that way (`EXECUTION_CACHE_MAX_MB * 1024 * 1024`) —
  dividing by 1000 would render a configured 512 MB ceiling as "536.9 MB" and
  read as the app misreporting the setting — and it promotes on the *rounded*
  value, so 1048575 B prints "1.0 MB" and not "1024.0 KB". The caption says
  what a cache is and stops short of the tidy version of it: the weight store
  holds *trained* weights, so clearing that one costs training time rather
  than a recompute, unless a checkpoint was saved.

### Changed

- **A `full_model` file CodefyUI wrote now loads back into CodefyUI**
  ([#288]). [#222] had made the loader work by widening torch's restricted
  unpickler to `torch.nn`'s own classes; [#283] had made the saver work. The two
  never met: CodefyUI's own module classes are not in `torch.nn`, so the product
  wrote a valid file it then refused — including every layer-editor model, all
  of which are a `GraphModelModule`. That was a trust decision rather than a
  bug, and #288 took it: a CodefyUI instance is a localhost or intranet
  deployment, so the gate widens by one **curated list of exact classes**, never
  a wildcard and never a module prefix (`app.custom_nodes.*` is code the user
  uploaded; an `app.` prefix would have admitted it). `weights_only=True` stays
  on and #222's detonating-payload test still passes.

  The admission rule has two halves, because admitting a class by name lets a
  file both restore its attributes AND call its constructor: torch's restricted
  unpickler runs `func(*args)` for any allowed name, so `cls(...)` is reachable
  with file-chosen arguments — which was already true of `torch.nn`'s classes
  under [#222]. So an admissible class defines no `__reduce__` / `__setstate__` /
  `__getnewargs__`, *and* has a constructor that touches no files, no network
  and no global state (a local `torch.Generator` is fine; `torch.manual_seed` is
  not). All 24 of CodefyUI's `nn.Module` classes were audited against both
  halves, the result is recorded per class next to the list, and two tests keep
  it honest: one re-derives what is mechanically checkable in each half (no
  pickle hooks; no dangerous call in the constructor or the module-level helpers
  it reaches), the other fails when a new `nn.Module` is added to `app.nodes`
  and not audited, so the list cannot rot back into the trap.

  One edge remains, stated at save time and in the new
  [Saving and Loading Models](https://docs.codefyui.com/usage/model-files) page:
  a `full_model` file is no longer self-contained — an older CodefyUI refuses
  it, and plain torch needs `weights_only=False` plus the backend package
  importable. `ModelSaver`'s save-time note says which of the three cases a file
  is, derived from the same allowlist the loader reads, and now inspects
  function-valued attributes as well as module classes so it cannot promise a
  round trip the loader will refuse. `state_dict` mode is byte-for-byte
  unchanged. (A second edge — a `TransformerEncoder` / `TransformerDecoder`
  layer that still did not load — was closed by the entry below before either
  reached a release.)

- **The transformer demos' `full_model` checkpoints load back too** (follow-up
  to [#288]). #288's allowlist admitted CodefyUI's own module *classes* and no
  functions, which left the shape it was opened about still broken one level
  down: `nn.TransformerEncoderLayer` and `nn.TransformerDecoderLayer` store
  their activation as a plain callable attribute, so every transformer
  checkpoint — a `TransformerEncoder` or `TransformerDecoder` wrapper, and any
  layer-editor graph containing one — saved fine and was refused on the way back
  in. The gap was pinned by two *strict* xfails so it could not pass unnoticed.

  The maintainer's ruling, 2026-08-12: torch-owned activation functions are
  admitted **by exact identity**, because a pure tensor function invoked with
  arbitrary file-chosen arguments has strictly *less* surface than the
  already-admitted classes' constructors — which the same REDUCE path can also
  invoke. Admitting `nn.Linear` already admits `nn.Linear(*file_chosen_args)`;
  admitting `F.relu` admits a call that returns a tensor or raises. #288 had
  admitted the larger surface and refused the smaller one, which was
  inconsistent rather than cautious. A function is admissible when it is
  torch-owned, a pure tensor operation, free of filesystem / network / process
  side effects, and free of global-state mutation.

  Enumerated from the save side, like the class list: a sweep over every layer
  the layer editor builds, every admitted module family and the graph model
  finds exactly one stored callable — `F.relu`, torch's default activation, as
  no CodefyUI node exposes the choice. `torch._C._nn.gelu` is admitted beside it
  because for a layer built the documented way — `activation` as a *string* — it
  is the only other value that attribute can hold, so over string-constructed
  layers the enumeration is complete rather than merely current. Passing a
  callable straight in (`activation=torch.tanh`) bypasses that mapping and
  stores it verbatim; such a layer is *not* admitted, and gets the save-time
  warning naming what it stored and a refusal by name on load — the widening is
  two identities, not an activation slot that stopped being checked. Two names;
  never the `torch.nn.functional` namespace, which also holds
  `handle_torch_function` and would admit whatever torch adds there next. Three
  tests keep it honest: the
  criterion is re-derived per entry, the save-side sweep must agree with the list
  in *both* directions (a stored function that is not admitted, and an admitted
  name nothing stores), and a sibling function from the same module is still
  refused by name. The save-time note stops warning about a file that now loads,
  because it reads the same widened allowlist.

- `components/SubgraphEditor/` is now `components/LayersEditor/`. It never
  edited a nested graph — it edits one node's `layers` param — and since real
  graph nesting landed, two unrelated features had been reading as one. The
  store surface (`layersModalNodeId`, `openLayersModal`, `closeLayersModal`,
  `updateNodeLayers`) and the 34 layers-editor i18n keys (`layersEditor.*`,
  out of the `subgraph.*` namespace nesting also occupies) move with it. No
  behaviour change. (#199)

- **A `Map` body ran with no execution context, so augmentation inside one was
  silently unseeded** ([#196]). The loop called `instance.execute(inputs,
  params)` directly — the last node-execute call site in the repository that
  did not go through `invoke_node` — so every node in a Map sub-graph saw
  `context=None`. `seed_pipeline` returns the pipeline unchanged when there is
  no context, so a seeded run's augmentation inside a Map was neither
  reproducible nor isolated, with no error and no warning. The body also lost
  its cancellation flag, its metric/artifact sink and its device.

  The body now gets a per-node **copy** of the run's context (the rule the
  engine has followed since [#253]), with the inner node id qualified by the
  Map's own — `map1__node_0` — because preset internals are named `node_0`,
  `node_1`, … and three things key off that field: the transform seed label,
  `StatefulModuleMixin`'s weight key, and the node id on every signal. The
  body deliberately does *not* get the Map's progress callback, which the
  engine binds to the Map node's id.

- **A preset built from a node with param-driven ports lost the extra ports**
  ([#196]). Preset extraction and the preset registry both read the static
  `define_inputs()` / `define_outputs()`, which answer for the *default*
  params. A `ComposeTransform(steps=5)` therefore reported two ports however
  many it had, so `step_3`..`step_5` never reached `exposed_inputs` and edges
  into them had nowhere to reattach when the preset was dropped back onto a
  canvas. `PythonScript` was affected in both directions. Both sites now use
  the dynamic form, which the params were already in scope for.

- **The metrics CSV export was open to formula injection, and unreadable in
  Excel** ([#196]). `csv.writer` quotes commas, quotes and newlines correctly,
  but quoting is not what stops a formula: a spreadsheet evaluates a cell
  beginning `=`, `+`, `-`, `@`, tab or CR. Two columns are user-influenced —
  `name`, which any plugin or custom node picks when it calls
  `context.log_metric`, and `node_id`, straight off the graph JSON. Text cells
  are now prefixed with an apostrophe when they lead with one of those;
  numeric cells are untouched, so a negative value stays a number. The body
  also gains a UTF-8 BOM: `charset=utf-8` is not enough for Excel on Windows,
  which reads a BOM-less CSV in the ANSI codepage and mangles every non-ASCII
  name — in a product that ships a zh-TW locale.

- **`ImageFolderDataset` validated the directory tree and never a single
  file** ([#197]). `ImageFolder` accepts anything with an image extension, so
  a truncated, zero-byte or merely renamed `.png` built fine and then raised
  `PIL.UnidentifiedImageError` from inside a DataLoader worker, potentially
  minutes into training, naming nothing actionable. The node now opens an even
  spread of at most 32 files at build time — a fixed cost that does not grow
  with the dataset — and names the first unreadable one.

- **The three port-count params disagreed with the canvas about anything with
  a trailing character** ([#197]). `ComposeTransform`'s `steps`,
  `PythonScript`'s `input_ports`/`output_ports` and `Split`'s `chunks` each
  spelled their own `int(raw)`, while the canvas reads all three with
  `parseInt(String(raw), 10)`, which takes a numeric *prefix*. So
  `{"steps": "5.7"}` drew five handles and validated two, and an edge into
  `step_3` was refused as an invalid input port. All three now share one
  helper that mirrors `parseInt` — including on `"1e3"`, where the obvious
  `int(float(raw))` fix would have traded one divergence for another. Only
  reachable through hand-edited or externally-generated graph JSON.
- **A graph with a non-numeric value on a bounded parameter crashed the
  validator instead of being refused** ([#193]). `validate_graph` compared
  every value straight against its declared `min_value`/`max_value`, so
  `"abc" < 0.0` — or `null < 0.0` — raised a `TypeError` that escaped the
  function entirely. `POST /api/graph/run` turns that function's *return
  value* into its 409 `invalid_graph` response, so the client got a 500
  naming nothing at all. Every parameter with a declared bound was reachable
  this way, on every node, not just the training ones.

  `null` is the case the canvas can produce by itself: clearing a numeric
  input yields `NaN`, which `JSON.stringify` writes as `null` into the saved
  graph. A non-comparable value is now reported as an ordinary validation
  error naming the parameter.

- **The execution WebSocket read `"deterministic": "false"` as a request for
  deterministic kernels** ([#189]). The socket coerced the flag with
  `bool()` while passing its sibling `seed` through untouched, so a client
  bug became a silently different run rather than the error the canvas
  already knows how to display. Both halves of the reproducibility pair now
  follow the same rule, and a non-boolean is refused with a message saying
  so.

- **Determinism could stay applied after the run that asked for it had gone**
  ([#190]). `execute_graph` enters the refcounted determinism scope by hand
  and pairs it in a `finally`, and two statements sat outside that pairing:
  creating the outbox pump task, and awaiting it during teardown. Either one
  raising left the scope open, holding `torch.use_deterministic_algorithms`
  at the failed run's setting and suppressing every later run's baseline
  capture. Both are now inside the guard.

  Scope, measured rather than assumed: on CPython the depth does come back
  once the traceback is released, because finalising the suspended generator
  runs the `finally` that was skipped. So this is bounded by whatever holds
  the traceback — a stored exception, an unretrieved task — rather than
  permanent, which is less severe than [#190] states. It is still a
  correctness property resting on an interpreter detail nothing in the code
  states, on a setting that is process-global.

- **A failing wake-up on the run gate replaced the exception the run
  actually failed with** ([#190]). `_RunExclusion._wake` guarded the one line
  that cannot realistically fail and left the following `await` bare, and
  both callers release inside an `@asynccontextmanager`'s `finally` — where a
  raise substitutes itself for the body's exception. The hold was always
  given back correctly; only the reported error was lost.

- **`ModelLoader` with `load_mode="full_model"` could not load anything, by
  construction** ([#222]). The branch called `torch.load(...,
  weights_only=True)` on a file that is by definition a pickled `nn.Module`,
  and restricting the unpickler to tensors and a small allowlist is the
  entire point of that flag — so the mode failed for every input it was
  written to accept, and no test covered it, which is why the suite stayed
  green. It is a dropdown entry that has never worked, not a broken main
  path: `state_dict` is the default and no shipped example selects
  `full_model`.

  Fixed without weakening anything. `weights_only=True` stays on and is
  widened, for that one call, to the `nn.Module` subclasses `torch.nn`
  itself defines, via `torch.serialization.safe_globals`. A model made of
  stock torch layers now round-trips through `ModelSaver(save_mode=
  "full_model")` and back; a pickle naming `os.system` is still refused,
  because `os.system` is not a torch layer. The allowlist is *derived* — a
  walk over the loaded subclasses of `nn.Module`, filtered to the ones torch
  defines — so there is no list to maintain and it tracks the installed
  torch rather than the torch that was current when this was written.

  The other half of [#222] was that the failure was unreadable: a
  legitimate-looking option produced a raw unpickler traceback with no hint
  that a safe default was involved. A file the allowlist does not cover now
  names the class it stopped on, says that a full-model file is a pickle and
  loading one runs it, and gives the two ways out — re-save as a
  `state_dict`, or convert the file once outside CodefyUI. Anything built
  from a class that is not torch's (a custom node's, a plugin's, an
  attacker's) is refused rather than executed, which is the same line
  `plugin_validator` already draws for third-party code.

- **`value_bytes` now says when it stops measuring** ([#193]). The
  `MAX_WALK_ITEMS` cap already logged that its total was a lower bound;
  `MAX_WALK_DEPTH` returned a smaller number in silence, which makes an
  under-count indistinguishable from a genuinely small value. The module
  docstring also claimed over-counting as *the* safe direction — true of
  cross-measurement sharing, and not true of the three things that make the
  walk under-count, which is the direction that costs memory.

- **The canvas renders only the nodes the viewport can show** ([#162]). After
  #125 a 300-node drag still measured 49.3ms p95 against a 32ms budget, and
  the residue was React Flow re-rendering all 300 node components on every
  `pointermove`, including the ones nobody can see. React Flow's
  `onlyRenderVisibleElements` is now set on the main canvas.

  The three risks the issue named were checked against `@xyflow/react`
  12.10.1's own selectors rather than assumed: the MiniMap draws from
  `s.nodes` and not from the visible set, so it stays complete; an edge
  survives while the box spanning its two endpoints overlaps the viewport at
  all, so a wire with one endpoint off-screen still draws; box selection runs
  `getNodesInside` over the whole `nodeLookup`, and a node being dragged is
  force-rendered; and a node is force-rendered until it has been measured, so
  every node is laid out once and its size is known to layout and to the
  minimap even if it is never looked at.

  Two limits worth stating. The saving is proportional to how much of the
  graph is off-screen, so it is *zero* at the zoom `fitView` picks — a fitted
  graph is on-screen by construction — and pays at the zoom someone actually
  works at. Measured against the built app on a 320-node graph: 320 mounted
  at fit view (zoom 0.103), 21 at zoom 1.0, 11 at zoom 2.0, each equal to the
  geometrically on-screen set. And no unit test can watch culling work,
  because jsdom gives an unmeasured node zero area and zero area counts as
  visible — 300 of 300 still render there with the flag on — so the tests pin
  the wiring and a browser pass covers the behaviour. The store half is
  load-bearing and now pinned too: `onNodesChange` must keep applying
  `dimensions` changes, or `measured` never reaches our nodes, `@xyflow`
  drops `handleBounds` on the next commit, every node becomes force-rendered,
  and culling dies with the flag still set and every test green.

  One behaviour widens with it: a card that leaves the viewport unmounts, so
  React state local to it resets on the way back — the viz nodes' `expanded`
  toggle, and, until this change, a note's unsaved text. `NoteNode` kept
  typed text in the contentEditable DOM and wrote it to the store only on
  blur, which held while every way a mounted note could disappear began with
  a pointer press somewhere else, because a press blurs the note first.
  Culling adds the first press-free path — `zoomOnScroll` is on, so a wheel
  or pinch zoom can carry a focused note out of the viewport — and an unmount
  is not a blur, so the text was simply gone. The card now commits its draft
  on unmount while editing, and only text typed during that edit.

- **`TRANSFORM` wires are a lighter amber, so a dichromat can tell them from
  `DATASET`** ([#197] item 5). `#FFC107` and `DATASET`'s `#FF9800` meet at
  every `train_transform` / `eval_transform` port — they are always drawn
  touching — and sat 14.5 dE00 apart, nearly all of it on the red-green axis.
  Simulate deuteranopia and that collapses to 6.1 dE00 from `DATASET` and 2.5
  from `LIST`, which made `TRANSFORM` the closest pair in the entire type
  palette for a dichromat. `#FFE082` (Material Amber 200) keeps the hue — 91
  degrees in Lab, against the old 83 and `DATASET`'s 68 — and buys the
  distance in lightness instead, ~18 L* above `DATASET` rather than ~9.5:
  21.9 dE00 in normal vision, 12.6 simulated deuteran, 16.6 protan, and
  `TRANSFORM` is no longer any dichromat's closest pair. Lightness is the
  axis every viewer keeps, which is why the fix is a lighter amber and not a
  different hue. The light-export twin `--diagram-light-type-transform`
  deliberately keeps `#b78901`: that palette is drawn on white against a 3:1
  floor no lighter amber clears, and it is the same hue darkened, which is
  what a light-export colour is.
- `DATA_TYPE_COLORS` now lists its keys in the backend `DataType` declaration
  order, so the `PythonScript` per-port type dropdown — which is
  `Object.keys` of that map — reads in the same sequence as the enum, with a
  test pinning the order against a hand transcription of it. Membership is
  untouched; `TRIGGER` still has no entry, because it is control flow rather
  than a data port. (#197)

### Fixed

- **Clicking a sidebar example destroyed the graph you were working on, with
  nothing to undo** ([#348]). The Templates tab sent every click through
  `openExample`, which replaces the active tab's nodes, edges, subgraphs,
  description, save binding and tab name in one commit and — by design —
  pushes no undo frame. So a stray click on a list of ~30 examples took
  however long you had spent on the canvas with it, and Ctrl+Z did nothing,
  because from the undo stack's point of view nothing had happened. The hint
  under the list advertised exactly that behaviour: "Click an example to open
  it". Examples now **join** the canvas the way every other palette item
  does. Drag one out of the sidebar and it lands where you release the
  pointer, without the camera moving off the gesture that just finished;
  click one and it lands clear of the graph already there, with the viewport
  brought onto it. Either way the ids are remapped so nothing on the canvas
  can be overwritten, the tab keeps its own name, description and save
  target, and one Ctrl+Z takes the whole block back off. A template written
  by a newer CodefyUI is still refused, on the drag path as well as the
  click.

- **An unhandled `/api` path answered `405 Method Not Allowed` on every real
  installation, and `404` in CI** ([#285]). The catch-all that serves the built
  frontend is registered only when `frontend/dist/index.html` exists, and it
  accepts `GET` on any path. Starlette does not stop at the first route whose
  *path* matches: a route matched by path but missed by method is recorded as a
  partial match, and a partial match with no full match anywhere is answered
  405. So `DELETE /api/files/../../etc/passwd` — which no API route can match,
  because `{filename}` does not span `/` — reached the SPA handler, matched it
  by path, missed it by method, and came back "the resource exists, your verb
  is wrong". Both halves of that were false. The exclusion now lives in the
  route's *pattern* rather than in the handler body, where it could never have
  helped: the 405 is produced by the router, before any handler runs.
  Wrong-method requests to real endpoints still answer 405, which is why this
  is a lookahead on the catch-all and not an all-methods `/api` 404 route in
  front of it.

  The reason nobody saw it is the more interesting half. Four traversal
  assertions in `test_api_data_files.py` had been deterministically red for
  anyone who had run `pnpm build` and green in CI for as long as the catch-all
  has existed, because CI's checkout has no `frontend/dist` — so CI was testing
  the *less* representative environment, the no-Node release path having made a
  built `dist` universal for real users. A new `pytest (built frontend)` job
  runs the whole backend suite against a real `pnpm build`, with a step that
  fails if the SPA routes are not registered so the job cannot quietly decay
  into a fifth copy of the matrix; the existing matrix keeps covering the
  dist-absent state, which is what `cdui dev` runs. A sweep of the full suite
  in both states found those four and nothing else. Three SPA cache-header
  tests that had been skipped in CI since they were written now run there too.

- **`ModelSaver(save_mode="full_model")` died with a raw pickle error on any
  graph using Reshape, SelectIndex or a transformer block** ([#283]). Those
  layers — and four more with the same defect: TransformerDecoder, LSTM, GRU,
  MultiHeadAttention — were built from classes defined *inside a function*, and
  pickle stores a class by name, so there was nothing to write. The mode failed
  by construction on exactly these layers with `AttributeError: Can't pickle
  local object 'Reshape.__new__.<locals>.Mod'`, which names an internal and
  offers no way out — the same complaint [#222] made about the loader. The
  closure turned out to be incidental: each already took its configuration
  through `__init__` arguments, and the nesting only kept `import torch` off
  the node module's import path. They are now ordinary module-scope classes in
  `app.nodes.utility.sequential_modules`, imported inside the builder so the
  lazy-torch property is unchanged, and their attribute names are unchanged so
  existing `state_dict` checkpoints still load. What is still unpicklable — a
  custom node or plugin that builds its module in a function — is refused
  *before* the write, naming the class and pointing at `state_dict`, instead of
  leaving a half-written file.

  Saving is only half a round trip, and [#222]'s loader accepts `torch.nn`'s own
  classes only, so a full-model file containing CodefyUI's classes still will
  not come back through `ModelLoader` — including `GraphModelModule`, which
  every layer-editor model is. That was previously unreachable (nothing got as
  far as saving one); it is now reachable, so the saver says so at save time
  rather than letting the user discover it one node later. The file is valid
  and `torch.load(..., weights_only=False)` reads it outside CodefyUI;
  `state_dict` remains the round trip that works.

- **A published app's OpenAPI document advertised `http://` even when it was
  fetched over HTTPS** ([#275]). `servers[].url` was built with a literal
  scheme, which was true of the deployment CodefyUI was written for — one
  machine, loopback, no TLS — and stopped being true the moment the documented
  way to deploy it became a reverse proxy terminating HTTPS. A browser blocks
  the resulting call as mixed content, so Swagger UI's "Try it out" failed, and
  a generated client got the wrong base URL. The scheme now comes from the
  request, which uvicorn rewrites from `X-Forwarded-Proto` when it is run with
  `--proxy-headers` (reachable since [#272]). The header is deliberately *not*
  read by the application: doing so would let any client forge the URL a
  published app advertises to every integrator who fetches the document.

- **The canvas WebSocket refused graphs the HTTP routes accept, and said so
  only by hanging up** ([#274]). `WS /ws/execution` was never uncapped —
  uvicorn's `ws_max_size` bounded it, enforced while fragments are assembled —
  but that 16 MB was an inherited library default: nothing in this repository
  chose it, no launch path passed it, no document mentioned it, and it was four
  times *stricter* than the 64 MB `MAX_RUN_BODY_BYTES` the HTTP paths use, so a
  graph between the two was accepted by `POST /api/graph/run/{name}` and
  refused by the socket the editor actually uses. There is now a
  `WS_MAX_MESSAGE_BYTES` setting that defaults to `MAX_RUN_BODY_BYTES` — one
  graph ceiling, both transports — which `cdui start` and `cdui dev` hand to
  uvicorn as `--ws-max-size`. The refusal is also legible now: the close frame
  always carried code 1009 and a reason, and the editor threw both away and ran
  its generic reconnect, so "your graph is too large" reached the user as
  "Connection lost" followed by "Connection restored" and an unexplained
  failure on the next Run click. It now says the graph was too large.

- **The layers editor had its own colour scheme, and it was the old one.** The
  modal that edits a `SequentialModel`'s layers carried a layer-type palette in
  two hand-synced copies, on the pre-lift Material tones the rest of the app
  moved off when they were measured too dark to read on a dark surface. The app
  therefore had two purples, two blues, two reds and two blue-greys, and which
  one you saw depended on whether a modal was open. There is now one table, in
  `tokens.css` as its own `--layer-*` group, and the contrast gate checks it —
  393 colour relationships across 186 tokens, up from 337 across 175. Adding
  the gate found a real failure it then fixed: a layer node's header was the
  raw hue with a white title on it, 2.16:1 to 3.09:1 on all seven hues, and is
  now built the way a canvas node's header is. (#228)
- The non-convex collapse refusal now tells two same-named blockers apart by
  their canvas position — a message reading `Conv, Conv` named neither of them.
  (#200)
- A plugin's `onGraphChanged` now fires when only the subgraph definitions
  change. Renaming a block, or editing its insides and stepping back out, moved
  bytes that `graph.getGraph()` reports while telling a watching plugin
  nothing. (#200)
- The publish pre-flight now scans portable preset definitions. A SECRET-typed
  value baked into `presets[].nodes[].params` — the third and last place a
  graph file can carry a node — published cleanly, while the identical value
  one level up was refused. (#200)
- A node whose `"type"` is explicitly `null` no longer crashes the secret
  walkers with an `AttributeError`. Reachable only through the publish gate,
  which reads a file straight off disk with no validation; it failed closed
  (a 500, nothing written or leaked). (#200)

- **A stale `addToolbarButton` disposer removed the button that had replaced
  it** ([#186]). Re-adding a toolbar button id replaces the button, but the
  remove function returned by the *superseded* registration was keyed by id,
  so calling it took the live replacement down instead of doing nothing. A
  plugin that re-registers a button when its own state changes accumulates
  exactly those disposers, and the host tracks every one of them for teardown.
  The disposer is now scoped to the registration that produced it — the same
  discipline `nodes.registerRenderer` already used. `removeToolbarButton(id)`
  is unchanged and still removes whatever currently holds the id; both
  behaviours are now in the published contract and the plugin docs.

- **Opening an example carried the previous graph's description onto the tab,
  and could save the example over the file that was already open** ([#200]
  items 4 and 8). Three separate readers open a graph document — the
  Toolbar's Load, the Toolbar's Import, and the examples path in
  `openExample` — and each hand-sequenced five or six `tabStore` setters. A
  sequence written out three times is only ever as right as its worst copy,
  and the third copy was missing three things. It never wrote `description`,
  so the description of the graph that had been there stayed on the tab — and
  `description` is persisted through save, so the leftover was written to
  disk as the new graph's own. It skipped the `format_version` gate entirely,
  so an example or a plugin-shipped template written by a newer CodefyUI
  opened fully *editable* — the one direction that gate must never fail — and
  the next save silently down-converted it; the two Toolbar readers had the
  gate, the third did not. And it never touched the save binding, so an
  example opened into a tab bound to `foo.json` inherited `foo.json` as its
  save target and the next Save overwrote that file with no overwrite prompt,
  the prompt being skipped precisely because a bound tab is the one case Save
  is allowed to overwrite in place.

  There is one door now. `tabStore.loadGraphDocument(doc)` installs a whole
  document — nodes, edges, subgraph definitions, Teaching Inspector overlays,
  tab name, description, save binding and the read-only verdict — in a single
  state update, and all three readers call it. The version verdict is
  computed inside the action from the raw, untrusted `format_version` and
  returned to the caller, which owns only the notice it shows: a reader can
  no longer open a newer-format document editable by forgetting a line, and
  the gate fails closed, with a missing or non-numeric field reading as
  current-format. `description` and the overlays are written whether or not
  the file carries them, so nothing from the previous graph survives an open,
  and `activeSegment` is nulled with its list because an overlay naming
  head/tail ids the new graph does not have is a dangling reference.
  `boundFile` is a *required* field rather than an optional one, so the
  compiler asks every reader the question the third one never knew existed —
  Load binds to the sanitized file stem, Import and an example both unbind.
  And the ordering that used to be load-bearing — nodes before definitions,
  because `setSubgraphs` drops the sub-canvas stack without putting a canvas
  back — is gone along with the sequence: one `set`, so no subscriber can
  observe a half-installed graph.

  Opening a document still deliberately pushes no undo frame. A frame carries
  no description, no read-only flag and no tab name, so an undo that restored
  the previous graph's nodes under the new graph's description would be a
  worse lie than not offering the step at all. ([#348] has since taken the
  sidebar's click off the replacing path entirely; the Toolbar's Load and
  Import, the empty-canvas example list and Open in new tab still go through
  it.)

- **A template written by a newer CodefyUI merged straight into an editable
  graph** ([#200]). The two paths that *open* a document answer a too-new
  `format_version` by opening it read-only; `insertExample` — the
  merge-into-the-canvas path — never looked at the field at all. Read-only
  exists so an older build can never write back fields it does not
  understand, and merging those fields into an editable graph reaches the
  same place by a shorter road, because the next save writes the result.

  Read-only is not an available answer for a merge: there is no separate
  document to mark, and marking the tab would punish the user's own graph for
  what the template is. So the merge is refused outright, with a toast (en
  and zh-TW) naming the version and both ways forward — open it to view it
  read-only, or update CodefyUI. The gate reads the raw payload *before*
  resolution, because resolution is not pure: it merges the template's
  unknown presets into the palette, which a refusal decided any later would
  already have done. A refused template leaves the canvas, the tab's
  read-only flag and the preset list exactly as they were. This is the
  refusal [#348] later carried onto the drag path as well.

- **Undo put a deleted node back and left the Teaching Inspector overlay it
  had swallowed deleted** ([#200] item 2). An undo frame carried nodes, edges
  and subgraph definitions, and nothing about the segment groups. Every
  action that takes a node off the canvas also prunes the segments naming it
  — `deleteNode`, `collapseSelectionToSubgraph`, `expandSubgraphInstance` —
  because a segment whose head or tail is gone can never resolve a path, and
  that pruning was one-way: Ctrl+Z restored the node and left the overlay
  gone, with nothing on screen to say the step had only half happened. Since
  `segmentGroups` is persisted through save, the loss then reached the file.
  The bubble's own close button was worse — it had no undo entry at all, so a
  single misclick was an unrecoverable edit to the graph document.

  A frame now carries `segmentGroups` and `activeSegment`. The highlight
  travels with the list because it points *into* it: restoring the groups
  alone brings the bubble back unfocused, restoring the highlight alone
  leaves it naming a group that is not there. Creating an overlay and
  clearing one are undo steps of their own, both being single deliberate
  clicks; merely focusing one still is not, because that is a change of view,
  like selecting a node, and neither is the bulk `setSegmentGroups`, because
  opening a document is deliberately not undoable. Leaving a sub-canvas also
  pushes its exit frame when only the overlays changed in there, not just
  when a definition did: an overlay-only visit leaves every definition
  byte-identical, so the exit used to push nothing, throw away the inner
  stack holding that overlay's own undo entry, and hand back an outer stack
  whose top frame still carried the *pre-entry* overlays — at which point the
  next unrelated Ctrl+Z wiped the overlay as a side effect of undoing
  something else. Both compares are structural rather than by identity, so
  merely looking inside a block still costs no undo step.

  Underneath, the four near-copies of the frame builder that
  `pushUndoSnapshot`, `undo`, `redo` and `closeFrameHistory` each kept are
  now one `undoFrameOf`, and both consumers spread the frame instead of
  listing its fields — so a future field cannot be captured by one producer
  and quietly ignored by the other, which is the exact shape of this bug. Two
  tests assert that `undo` and `redo` apply *every* field a frame carries.
  Undo stacks are never persisted (`PersistedTab` has no `undoStack`), so no
  in-flight session holds a frame of the old shape and there is nothing to
  migrate.

- **Closing a canvas tab discarded the graph in it, permanently, with no
  confirmation** ([#331]). `removeTab` takes the tab's undo and redo stacks
  with it, so there was nothing left to undo from either — one misclick on
  the wrong tab and a whole graph was gone. The tab bar now asks first when
  the tab has anything in it, using the house confirm (`dialogStore` /
  `DialogContainer`, `variant: 'danger'` — the same family as the save-name
  prompt and the overwrite confirm), naming the tab and its node count so the
  user can see they clicked the wrong one. An *empty* tab still closes
  silently: asking about nothing is the noise that trains people to click
  through the dialog that matters.

  It asks for every non-empty tab rather than only the unsaved ones, and that
  is deliberate. `dirtyNodeIds` looks like a dirty flag but answers a
  different question — it is the partial-re-execution hint, `clearDirty`
  empties it at the start of every run, and `addNode` never adds to it at all
  — so a graph that was dragged together and run but never saved reads as
  perfectly clean, which is exactly the tab whose loss hurts most. Nothing in
  the store records "matches what is on disk": `currentGraphFile` says which
  file a tab is *bound* to, not that the two are identical, and there is no
  snapshot of the last save to diff against. "Saved and unchanged" is not a
  state this codebase can currently prove, and guessing it wrong is
  unrecoverable, so the extra dialog on a genuinely-saved tab is the cheap
  side of the trade until a saved-state signature exists.

  The node count sums the open `subgraphStack` frames as well as the visible
  canvas, for the same reason `buildPersistedTab` flushes the stack before
  persisting: `enterSubgraph` swaps the graph's nodes out into a frame, so
  standing inside a freshly created block leaves `tab.nodes` empty with the
  whole graph stashed one level up, and a naive `nodes.length` check would
  wave that tab through as empty. A *running* tab keeps its existing single
  question — the stronger one, since closing kills the run too — and now
  carries the graph warning as that dialog's body, so one click never
  produces two dialogs. Only the explicit per-tab close is intercepted;
  window unload and the last-remaining-tab path are untouched.

- **The schedule-length advisory told correctly configured per-step runs to
  break themselves** ([#308]). The check that compares an `LRScheduler`'s
  length against the run's was written when there was only one possible
  answer to "how long is this run" — `TrainingLoop.epochs` — and [#297]'s
  `scheduler_step=optimizer_step` never reached it. So the exact
  configuration that mode's own description recommends, `OneCycleLR` with
  `total_steps = max_steps`, emitted a `run_warning` on every single run
  telling the user to set `total_steps` to the epoch count instead. Wrong
  advice is worse than none: it is indistinguishable from right advice, and
  following it breaks a schedule that was already correct.

  The check now carries a clock through every branch. Per-epoch stepping
  still measures against `epochs` — the epoch-mode assertions are untouched,
  so nothing that read correctly before reads differently now — while
  per-step stepping measures against the run's optimizer-step budget, which
  has to be derived because no single parameter holds it: `max_steps` when it
  actually binds, otherwise `epochs * ceil(batches / accumulate_steps)`,
  rounding up because the loop applies an epoch's short accumulation tail as
  a step of its own. A budget that cannot be known before the run starts — an
  `IterableDataset` with no `max_steps` — produces no advisory at all rather
  than a guess, which would be the same failure in a new costume.

  The warmup families also got their first length check of any kind. They
  compose a `SequentialLR`, which matched none of the check's `isinstance`
  branches — so the one family whose entire point is a step-denominated total
  was the only one never checked, in either mode. The composed total is now
  recovered from the object itself, the last milestone plus the tail's own
  `T_max` or `total_iters`, and a tail that declares no length declines
  rather than inventing one. Three real traps come out of it: a ramp that
  never finishes, which the node's own default `warmup_steps=100` produces
  against a five-epoch run and which means the run never trains at the
  learning rate that was set; a total longer than the run, so the decay never
  finishes; and a total shorter than it, where `warmup_cosine` turns back
  *up* and the tail of the run trains at a rising learning rate while
  `warmup_linear` sits at ~0, spending compute without learning.
  `constant_with_warmup` has no cycle length to disagree about, so only its
  ramp is checked and the correct configuration stays silent. `T_max`,
  `total_steps`, `step_size` and `warmup_steps` on `LRScheduler` and
  `scheduler_step` on `TrainingLoop` now describe both modes instead of
  flatly contradicting each other one node over, with zh-TW twins and mode
  guards in the unit tests so the one-unit claim cannot quietly come back.

- **A resumed run put its per-step schedule back on the wrong clock**
  ([#316]). Where [#308] was wrong advice, this one was a wrong learning
  rate. `_fast_forward_scheduler` repositions a resumed scheduler by
  replaying one `scheduler.step()` per completed epoch — exactly right while
  the loop also steps once per epoch, and exactly wrong under
  `scheduler_step=optimizer_step`. A `warmup_cosine(total_steps=1500)`
  resumed after 1000 optimizer steps was replayed by a handful of
  epoch-steps and went back to training on a learning rate from the warmup
  ramp — still rising — for a run two thirds finished. Nothing warned,
  because from the scheduler's side a replay is a replay. Checkpoints
  carrying `scheduler_state_dict` take the state-restore path and were never
  affected; the fast-forward is the legacy fallback.

  Nothing stores the completed step count. A checkpoint carries `epoch` and
  no step field of any kind — `build_checkpoint` is the single writer both
  engine paths go through — and the run's own `metrics["total_steps"]` is an
  output with no port back into a resumed loop. So the replay count is
  derived instead, as `start_epoch * steps per epoch`, which is by
  construction where the loop leaves a per-step schedule at the start of
  epoch `start_epoch`. When there is nothing to derive it from — an
  `IterableDataset` or a hand-rolled generator, where the loader has no
  length — the replay is refused outright rather than falling back to the
  epoch count, with an advisory naming `CheckpointSaver.lr_scheduler` and
  `CheckpointLoader.lr_scheduler` as the route that needs no batch count and
  is exact in either mode. Guessing there would have reproduced this bug in a
  new shape.

  The restored-state branch reads the same clock, because it had the same bug
  pointed the other way: a per-step schedule comes back at `last_epoch=1000`
  for a run resuming at epoch 10, and comparing that against an epoch index
  accused the one always-exact resume route of holding an inconsistent
  checkpoint, on every per-step run. The derivation's limits are written down
  where the derivation lives rather than left to be found — an earlier leg
  over a different dataset size or `accumulate_steps`, an fp16 step skipped
  on an overflowing gradient, and a leg that `max_steps` cut off mid-epoch,
  which over-counts by up to one epoch's worth of steps (recorded at 24
  samples, `batch_size=6`, `epochs=5`, `max_steps=6`: a real 6 steps against
  a derived 8). Epoch mode is unchanged — same replay count, same log lines,
  same advisory text.

- **Retention finally sweeps the TensorBoard directories it has been
  logging** ([#196]). `RunStore.prune` collected `checkpoint`-kind artifact
  rows only, so every `tensorboard`-kind row's *directory* outlived the run
  it belonged to. Nothing else scans the runs tree, so those directories
  became unreachable litter the moment their row was pruned — one per
  training node per run, for the life of the install, and invisible because a
  default install gitignores the data root. The sweep now collects them too,
  by the same keep-last window, in the same transaction, with the delete
  dispatched off the event loop exactly as the checkpoint unlink already was.
  No new knob: retention is still only `RUN_RETENTION_KEEP_LAST`.

  The delete guard is [#224]'s `owned_checkpoint_path` reasoning applied to a
  tree. A row is a *claim*, not evidence — `kind` is free text and
  `ExecutionContext.log_artifact`, which the plugin API reaches, writes both
  the kind and the path — so a row labelled `tensorboard` must not be able to
  hand an unattended `rmtree` an arbitrary directory. The path is re-derived
  from `run_logdir(run_id)` for that row's *own* run, and anything resolving
  outside it, any symlink (checked before `resolve()`, so the delete can
  never resolve *through* a link), any `..` escape and any other run's log
  directory is skipped and logged, never raised: one odd row must not fail a
  whole prune. A review catch tightened it further — the expected root now
  resolves only the shared `runs/` ancestor and keeps the run id and `tb`
  literal, because resolving the whole prefix let a symlink planted at
  `runs/<id>` or `runs/<id>/tb` move both sides of the containment check to
  the same foreign place and pass. That is the same split [#224] already used
  for checkpoints. The parents the delete just emptied are then rmdir'd,
  stopping at the shared `runs/` root and at the first directory still
  holding anything, so the sweep does not trade a growing tree for a growing
  skeleton of empty directories.

  Two deliberate differences from the checkpoint sweep, both recorded in the
  docstrings where the code makes them: the delete is `shutil.rmtree`, so the
  guard is correspondingly stricter (per-run, not merely per-shape); and the
  `interrupted` exemption does *not* apply. That exemption exists to protect
  a resume point — event files resume nothing, and exempting them would leave
  unreferenced directories for precisely the runs a crashed server produces,
  which is the leak this closes. The augmentation docs and their zh-TW twin
  now also say how long the logs last, since a curve worth keeping past the
  window has to be copied out. The issue as filed named `Database.prune_runs`;
  that is the *publish* `runs` table, which cannot produce a TensorBoard
  directory at all — `open_run_writer` returns `None` unless the run can
  record artifacts, which the REST contract runner cannot — so the fix
  follows the correction comment on the issue rather than its body. Items 1-3
  of [#196] shipped in #280 and are listed above; this was item 4.

- **`Print` could kill a whole run over one character the console could not
  spell** ([#346]). `print` encodes with `sys.stdout.encoding`, which on
  Windows is the machine's ANSI codepage — cp950 on a Traditional Chinese
  install, cp932 on Japanese. Those cover Han characters fine; what they do
  not cover is the long tail of Unicode a graph legitimately carries. A
  superscript `T` (U+1D40) in a label is enough, and so is one stray
  character in text a language model generated. The resulting
  `UnicodeEncodeError` came straight out of `execute`, so the node failed and
  took the run with it — losing a five-minute training run at the final
  `Print` because the console could not render one glyph. The console now
  gets a lossy rendering (`errors="replace"`), and any other stdout failure —
  a closed or redirected stream included — is swallowed rather than raised,
  because echoing to a console is not worth failing a run for. The `__log__`
  string the UI displays never passes through that path and keeps the exact
  text, since the browser has no such limitation, and the value on the output
  port was always passed through untouched.

- **A plugin whose manifest `id` contains a hyphen registered its nodes under
  a different id than the rest of the product used** ([#350]).
  `official-template` is the id in the manifest, in the install directory, in
  `cdui plugin list`, in the examples route's `plugin:<id>` prefix, and in
  the `"type"` string that `qualify()`'s own docstring documents as the
  saved-graph form. The registry said `official_template:HelloPlugin`,
  because the id never reached discovery — it was re-derived from the Python
  package the pack is imported under, and that step is not reversible. A
  hyphen cannot appear in a module path, so `cdui_plugins.official_template`
  is the only importable spelling and nothing in it records that the
  underscore used to be a hyphen.

  Why it shipped green is the more useful half. `registry.get()` falls back
  to a suffix scan, so such a graph still resolved, validated and executed —
  every server-side check passes. The canvas has no such fallback:
  `resolveSerializedNodes` does an exact lookup and substitutes an empty
  definition on a miss, so the node renders as a blank box with no ports, and
  because those handles do not exist every edge attached to it is silently
  dropped on load. An example that is visibly broken while nothing anywhere
  reports a problem — which is exactly how the official plugin template's two
  `Demo` examples shipped.

  The manifest id is now threaded through: `install_plugin_finder` returns a
  named `PluginNamespace(nodes_dir, package_name, plugin_id)` rather than a
  pair that threw the id away, and `NodeRegistry.discover` takes an explicit
  `plugin_id`, with the lossy derivation kept as a documented fallback so the
  built-in discoveries are untouched. The three plugin call sites — the
  server lifespan, `rediscover_all` behind `POST /api/plugins/reload`, and
  `scripts/project.py` — each kept their own copy of the discovery loop,
  which is the mechanism that let one subsystem's spelling drift from the
  others; they now share one `discover_plugin_nodes()`. `/api/nodes`'s
  `provider` field reads the id off the registry key instead of the module
  path, so a single node entry no longer contradicts itself. The regression
  guard is in `test_chapter_examples.py`: every in-repo pack example's node
  types are checked against *exact* registry keys — deliberately not
  `registry.get()`, whose suffix scan would mask the mistake being guarded.
  There is no compatibility alias, because a second entry would duplicate
  every kebab-named node in the palette and no graph in the repository uses
  the underscored spelling. The companion fix is in the plugin template
  repository; both are needed for those two `Demo` examples to open.

### Internal

- **A codegen test asserted randomness by sampling for it** ([#277]).
  `test_no_seed_leaves_an_exported_run_on_torchs_own_entropy` ran the same
  exported graph twice with `--no-seed` and required the two stdouts to
  differ. That is probabilistic by construction, and on that graph the draw
  was eight independent coin flips — `RandomHorizontalFlip(p=0.5)` over a
  batch of eight — so a collision had probability 1/256 per run. It duly
  collided on the 2.2.0 release PR, whose entire diff was three version
  strings, and cost real time to establish that a version bump had not broken
  the code generator. `--no-seed` itself was never broken.

  The test now asserts the contract instead of sampling for it, against a
  purpose-built two-node probe graph: the first node installs a sentinel
  seed, the second reports `torch.initial_seed()`. Because the export
  re-seeds before *every* node, the sentinel is overwritten exactly when
  seeding is active and survives untouched exactly when it is not — so
  `--no-seed` must report the sentinel and the baked seed must report
  `derive_seed(4321, "probe")`. Both are equalities against values known
  before either run, so the test passes on every run or fails on every run.
  The seeded sibling above it is untouched: it compares two fixed outputs
  from two different explicit seeds, which is the shape that was already
  safe.

- **The plugin host's teardown belt could not survive being needed** ([#186]).
  Three latent gaps, none reachable from the shipped UI, all in the paths that
  run when a plugin is unloaded or hot-reloaded:

  A panel is deleted from the registry *before* its element is detached and
  the change is published *after*, so a throw in between left the published
  snapshot holding a panel the registry no longer had — a dock tab whose
  lookup returns `undefined`. The detach is now contained, so every caller
  publishes. The sweep that drops whatever a plugin left registered is
  likewise no longer able to abort: it used to propagate out of the teardown
  loop, skipping every later plugin's sweep and the widget-stack clear. It had
  never been observed to fail only because the tracked cleanups had already
  emptied the registries by the time it ran — the second line of defence was
  correct by accident, and would have failed in precisely the case it exists
  for. Both are now driven by tests that make them fail.

  The plugin event stream's tab tap also keyed on tab id alone, so a tab whose
  socket was replaced would have kept a handler on the discarded one and seen
  nothing from the live one — silently, with every count still looking right.
  It compares the socket now.

- **Two claims in the plugin execution-event docs were not true** ([#186]).
  The module presented its `rejected: true` and `reason: "not_running"` drops
  as load-bearing, and told plugin authors those frames "still consume a
  cursor" in the durable log. They do neither: the WebSocket handler answers
  both itself and sends no `cursor`, so the no-cursor rule already drops them
  and they occupy nothing in the log — which matters, because `cursor` is what
  the contract sells for detecting gaps. The guards stay (a plugin host runs
  against whatever server version is installed, and the day a refusal carries
  a cursor, forwarding it reports a still-running run as failed), but the docs
  now say which rule actually fires, and a test pins the real wire shapes.

- **`TrainingLoop`'s nested-batch device path finally has a caller, and a
  test** ([#312]). `to_device` has always mapped lists, tuples and dicts
  element-wise, so a batch whose `data` is itself a tuple was supposed to
  reach the GPU intact — but every dataset in `app/nodes` emitted a flat
  `(tensor, tensor)` sample, so the recursion had never carried a real
  training run and nothing would have failed if it stopped working.
  `PushWorldDemos` is the first shipped dataset whose sample is `((image,
  tokens, chunk), chunk)`, and an integration test now trains a `VLAModel`
  through the stock loop on that shape — both heads, a training loader and a
  validation loader — so the contract fails loudly if it regresses.
  `training_loop_node.py` itself is untouched.

- **Backend and frontend checks now run on every PR, not just path-matched
  ones** ([#270]). The `main` ruleset requires eight status checks to report
  — `pytest` on Python 3.10, 3.11 and 3.12, on Windows 3.12, and against a
  built frontend on 3.11; `ruff check`; `pnpm build + tsc + test`; and the
  raw-control-byte scan. A required check that a path filter kept from
  running never reports at all, which leaves the PR's merge button waiting
  forever — a docs-only PR would have been unmergeable. The `pull_request`
  path filters are dropped from `backend-test.yml` and `frontend-build.yml`;
  `byte-scan.yml` never had one, for a version of the same reason. The
  push-to-main triggers keep their filters, since nothing gates on those
  runs.

- **Issue and PR templates that ask for the graph up front** ([#141]).
  `.github/` had workflows and `RELEASING.md` but no issue forms and no PR
  template, and a bug report with no exported graph JSON attached is close to
  useless for reproduction — nothing on the issue-open path even asked for
  it. `bug_report.yml` makes that field required and front-loads it with a
  note pointing at the canvas toolbar's Export button, alongside CodefyUI
  version (`cdui --version` / `build-info.json`), install flavour (uv-only
  versus a dev install with pnpm), OS and Python, and an optional logs field
  pointing at `.codefyui_dev/server.log`. `feature_request.yml` is
  problem-first — what you cannot do today, and who runs into it — with the
  proposed shape optional. `config.yml` keeps `blank_issues_enabled: true`,
  because a lot of house issues here are free-form root-cause write-ups that
  fit no structured form, and adds contact links to CONTRIBUTING and the docs
  site. `PULL_REQUEST_TEMPLATE.md` spells out the closing-keyword negation
  trap explicitly — "does not close #N" still closes #N, so say "Part of #N
  (stays open)" — asks for the commands actually run as test evidence, and
  checklists DCO sign-off, the zh-TW docs and node-description twins, and no
  pictographic emoji. Nothing in CI validates issue-form YAML and this does
  not add a validator: the four files were checked against GitHub's
  issue-forms schema by hand and parsed with a structural assertion that
  every non-markdown body item carries a `type`, an `id` and a `label`.
  [#141] stays open — good-first-issue curation is handled separately.

## [2.2.0] — 2026-08-10

The release that makes a re-run mean something.

The headline is a correctness bug nobody had noticed: **a second Run of a
shipped training example did no training at all, and reported success.**
Measured on `plugins/foundations/examples/C2-5/MLP-MNIST-Training`, counting
real `TrainingLoop.execute()` calls across three runs against one cache: `1`,
`0`, `0`. Four of the six shipped graphs that train were affected, two of them
teaching-plugin examples students open in class. It stayed invisible because a
preset reported `completed` whether its contents ran or came from cache — both
halves of that are fixed here.

The rest divides into three: the same cache-correctness family (a hit must not
skip a side effect, a mutation, or a file write), a security pass driven by
CodefyUI now being evaluated for shared company servers rather than one
student's laptop, and the quality gates that stop the next round of this being
found by hand.

### Fixed

- **A path typed into a node parameter could overwrite the run database, and
  an unattended prune could delete a file it never wrote** ([#224]). One
  write-scoped rule — "stay inside the project data directory" — guarded both
  directions, and on a default install the database is inside that directory.

  *Write.* With the default `cdui start` and no `--project`, `PROJECT_DIR` is
  `None`, the project-mode derivation never runs, and `MODELS_DIR` stays
  `backend/data/models` — one level below `codefyui.db`. So `../codefyui.db`
  as a `CheckpointSaver` or `ModelSaver` path resolved to the live database
  and passed the guard, and a training run wrote a `.pt` payload over it. No
  plugin and no mislabelled row required. Project mode was never affected:
  there `MODELS_DIR` is `<project>/assets/models` and the database, which
  stays install-global, is outside the data root entirely. `ImageWriter`
  shares the same rule but was not reachable this way — it forces the file
  extension to match its `format`, so `../codefyui.db` was rewritten to
  `codefyui.png` and landed beside the database; what it could overwrite was
  any file under the data root ending in an image extension.

  *Delete.* `RunStore.prune` unlinks the file of every pruned artifact row
  whose `kind` is `checkpoint`. `kind` is a free-text column and the plugin
  API can log artifacts, so a row claiming `kind="checkpoint"` with a path
  pointing at anything else under the data root had that file removed by a
  background sweep, with no user action and no confirmation.

  Both now have their own rule, because they ask different questions. Writes
  stay permissive — the data directory is a node's to write into — minus
  CodefyUI's own storage: the database and its SQLite `-wal` / `-shm` /
  `-journal` sidecars, derived from `DB_PATH` at call time and compared with
  case folding so a differently-cased spelling cannot slip past on Windows.
  Deletes no longer trust the row at all: retention removes a file only where
  it can prove it wrote it — a generated checkpoint filename under
  `MODELS_DIR/interrupted/` or `MODELS_DIR/periodic/`, the only two places
  the interrupt and periodic writers put anything. Everything else is skipped
  and logged, and the row still goes either way.

  Nothing about ordinary use changes: checkpoints, saved models and written
  images work exactly as before in both modes, including to arbitrary
  sub-directories under the data root. A `CheckpointSaver` file the user named
  themselves was never touched by retention before and still is not.

- **Every request-body size cap could be walked past by chunking, and four
  routes had no cap at all** ([#265], [#242]). Three routes capped their body
  by comparing `Content-Length` to `MAX_RUN_BODY_BYTES`. A chunked request —
  `Transfer-Encoding: chunked` — declares no `Content-Length`, so all three
  checks were skipped in full by any client that chose to chunk. They were
  advisory, not enforcing, on every route that had one.

  Four more routes took a body with no cap at all (`POST /api/graph/save`,
  `/validate`, `/export`, `/api/presets/create`), and the four upload routes
  read the whole file into memory *before* comparing it to `MAX_UPLOAD_SIZE`,
  so a request far larger than the limit was buffered in full and only then
  refused.

  All of it is replaced by one mechanism that counts bytes as they arrive on
  the ASGI receive channel, so it holds whether or not a length was declared
  and covers every route rather than the three that remembered to ask. Two
  ceilings still apply and are resolved per path: `MAX_RUN_BODY_BYTES`
  (64 MB) everywhere, `MAX_UPLOAD_SIZE` (500 MB) on the upload routes.

  User-visible where it was not before: `POST /api/graph/save` with a 70 MB
  graph is now a 413. The 413 on `/api/graph/run/{name}` and
  `/api/apps/{slug}/invoke` keeps the 9-key envelope those routes promise, so
  clients generated from the per-app OpenAPI document are unaffected.
  WebSocket messages are unchanged — they are bounded by uvicorn's
  `ws_max_size` (16 MB) at the transport.

- **The second Run of a training graph did no training, and reported success**
  ([#253]). `TrainingLoop` inherited the default `cacheable = True`, and a
  cache hit replays a node's recorded outputs without calling it — so no
  weights were updated, no metrics were written, and the canvas said
  `completed` anyway. On four of the six shipped examples that train, run 2
  and run 3 executed the training loop zero times. Two of those are teaching
  plugin examples students open in class (`C2-5 MLP-MNIST-Training`,
  `C3-1 LeNet-MNIST-Training`). Change a hyperparameter, press Run, and the
  loss curve you get back is the previous run's, with nothing to tell you.

  The other two shipped training graphs were protected only by accident, and
  the accident has since been removed ([#259], below): their `ModelSaver` may
  only write under `./data`, `Dataset`'s cache fingerprint walked all of
  `./data`, so the write happened to invalidate the dataset. This fix does
  not depend on it, which is what let the walk be scoped.

  `SequentialModel` was the other half. It owns the weights `TrainingLoop`
  mutates, declared no inputs, and was cacheable — a root nothing could ever
  invalidate, so run 2 was handed run 1's *already trained* module. Making
  only `TrainingLoop` re-run would therefore have traded "no training at all"
  for "every run silently continues from the last run's weights", which is
  arguably worse because the numbers still look plausible. It now carries
  `StatefulModuleMixin` like every other weight-owning node, so **Settings →
  Training Behavior → Persist weights between runs** and **Reset all weights
  now** decide which of the two happens, and the node writes which one it did
  to its Log tab on every run.

  Also fixes a latent engine race this exposed: nodes at one topological
  level run concurrently on an unseeded run and shared one mutable
  `current_node_id`, so a node's persisted weights could be filed under a
  concurrent sibling's node id — out of reach of "Reset all weights now".
  Each node now gets its own view of that field.

- **A learning-rate schedule whose length did not match the run said nothing**
  ([#252]). `TrainingLoop` steps the scheduler once per epoch, so every
  cycle-length parameter on `LRScheduler` counts epochs — while PyTorch counts
  batches for several of the same parameters, and nothing reconciled the two.
  A schedule could therefore be entirely wrong without one thing going red: no
  exception, no warning, a loss curve that looks normal, and an accuracy a few
  points short.

  Worst was `OneCycleLR.total_steps`, whose default of 1000 is a plausible
  batch count and an impossible epoch count: at 20 epochs the run traverses 2%
  of the cycle, so the learning rate warms up and never anneals — for a user
  who changed nothing. `CosineAnnealingLR.T_max` disagreeing with
  `TrainingLoop.epochs` is the same trap one step less silent.

  `TrainingLoop` now compares the two before the first epoch runs and says so
  in the server log, in the run log the Runs panel reads back, and in the
  canvas Log tab. Advisory only: it never changes the schedule and never fails
  the run, because a truncated schedule is a legitimate choice. The default of
  1000 is deliberately unchanged — a new default would rewrite what every
  already-saved graph does, silently, on update. `CosineAnnealingWarmRestarts`
  is inverted rather than exempt: it reuses the same value as `T_0`, where
  equality would mean no restart ever happens.

- **A cache hit skipped the mutation that was the node's whole point**
  ([#254]). [#253] fixed the case where the mutated thing was the weights a
  training loop updates. Five more nodes are handed a live `model` or
  `optimizer`, write into it, and hand it back — and a cache hit returns
  their recorded outputs without calling them, so the write never happens
  while the node reports success. Measured across three runs against one
  shared cache, each node ran once and then not at all:

  - `ModelLoader` — "load pretrained weights, then fine-tune" quietly became
    one long run: the file's weights reached training only on run 1
    (0.05 → 0.085 → 0.157).
  - `CheckpointLoader` — the resume did not happen; the model kept whatever
    the previous run had left on it.
  - `LRScheduler` — `TrainingLoop` received a scheduler already at the end of
    the schedule it ran last time. With `StepLR(step_size=1, gamma=0.1)` over
    two epochs it saw learning rates 0.1, then 0.001, then 1e-05.
  - `EvaluateModel` — one `eval_accuracy` point logged across three runs, so
    the chart was empty on runs the canvas called successful.
  - `Inference` — the prediction of a network that had since been trained
    further.

  All five are now `cacheable = False`, and so are `Optimizer`, `DQN`, `PPO`
  and `RewardModel`: those four hand out a live model or optimizer with no
  input that could ever invalidate them, which is what made the five
  reachable in the first place — `DQN`, `PPO` and `RewardModel` are
  `SequentialModel`'s [#253] bug one package over.

  This does take back the `cacheable = True` that [#144] gave `ModelLoader`
  and `CheckpointLoader`, and it costs less than that sounds. Opting out
  propagates downstream, and both take their model from a weight-owning
  node, which is already non-cacheable — measured on that shape, both
  executed on all three runs *before* this change. [#144]'s fingerprint
  mechanism is untouched and still serves `CSVReader`, `FileReader`,
  `ImageReader`, `ImageBatchReader`, `Dataset` and `ImageFolderDataset`.

  Two registry-wide invariants now hold the line: no cacheable node may hand
  out a `MODEL` or `OPTIMIZER`, and no cacheable node's `execute` may record
  a metric or write a checkpoint. The hand audit found five nodes; the
  invariants find the next one.

- **Resuming a checkpoint could throw away its learning-rate schedule
  silently** ([#149]). `CheckpointLoader.lr_scheduler` is an optional input.
  Leave it unwired and a checkpoint that stores the schedule's position has
  nothing to restore it into, so the position is discarded and the schedule
  is rebuilt by replaying epochs instead. For `StepLR` or
  `CosineAnnealingLR` that replay is exact; for a metric-driven
  `ReduceLROnPlateau` it cannot be — measured on an 8-epoch checkpoint whose
  last five epochs were a plateau, `best` and `num_bad_epochs` came back as
  `inf`/`0` instead of `0.8`/`5`, postponing indefinitely a decay that was
  one epoch away. It was reported at INFO, i.e. nowhere a canvas user looks.

  Both halves are now advisories on all three surfaces [#252] built —
  server log, the run log the Runs panel reads back, and the node's Log tab:
  `CheckpointLoader` says it is discarding a stored schedule position and
  names the input to connect, and `TrainingLoop` says when it could not put
  a schedule back where it was. Advisory, never fatal.

  The mechanism this issue was filed for — a scheduler built on a restored
  optimizer starting from a decayed `base_lrs` — does not occur, and the
  measurements are on the issue. `initial_lr` survives an optimizer state
  round trip and `LRScheduler.__init__` reads it with `setdefault`; the
  `initial_lrs` checkpoint key added since makes that independent of torch's
  behaviour rather than dependent on it.

- **`Dataset` re-read itself whenever anything else under `data_dir`
  changed** ([#259]). The cache fingerprint walked the whole directory
  recursively. In a project directory every dataset shares one
  `assets/data/`, so downloading a second dataset invalidated the first, and
  a large tree paid a full recursive stat every run whether or not that
  dataset had moved — the opposite of what [#144] restored the caching for.
  It is now scoped to the directory the dataset itself lives in (`MNIST/`,
  `cifar-10-batches-py/`, and so on), read off torchvision's own class
  attributes so an upstream rename is a changed answer rather than a
  fingerprint that covers nothing. An unrecognised name still walks the tree,
  because over-invalidation costs a re-read and under-invalidation costs a
  wrong experiment.

  This was blocked on [#253], because the over-invalidation was the only
  thing keeping the shipped `ModelSaver` graphs from skipping training
  entirely: the saver may only write under `./data`, and that write dirtied
  the dataset's key. Measured on three shipped graphs run three times each
  against one shared cache, counting real `TrainingLoop.execute()` calls:
  1/1/1 on all three, with the dataset now a cache hit on runs 2 and 3.

### Changed

- **The declared torch floor is 2.5, up from 2.0.0** ([#252]). It is the
  version the code already assumed: `torch.OutOfMemoryError` arrived there, and
  the OOM classifier had been carrying a second lookup for the pre-2.5 spelling
  that nothing exercised. That branch is gone; the message match stays, since
  it is how MPS and out-of-tree backends report an OOM on any torch. Installs
  on torch 2.0-2.4 are no longer supported.

### Security

- **A SECRET param typed into a graph was written in clear text into the run
  database** ([#251]). Every other path that persists a graph already scrubbed
  SECRET values — save, export, publish pre-flight, preset creation, Python
  codegen — but the run path wrote `exec_runs.graph_snapshot` exactly as
  submitted. An API key typed into an `LLMChat` node therefore landed verbatim
  in the shared SQLite file, and run history is pruned by COUNT (the newest
  200), not by age, so on a quiet install it never aged out. The column is not
  reachable over the API, so this was a leak at rest: in the database file, in
  any backup of it, and in any copy of the install tree.

  The snapshot is now stored scrubbed. A queued run's secrets are held in
  process memory keyed by run id and re-injected when the run is promoted off
  the queue, so a queued graph still executes with the key its submitter typed
  and nothing changes for the user. Holding them in memory is safe for one
  specific reason: the queue is in memory too, and `recover_interrupted`
  retires every `queued` row a dead process left behind — so a snapshot can
  only ever be read back by the same process that wrote it, which is exactly
  how long the values are kept.

  Startup also sweeps values written by older builds out of finished runs and
  logs how many it removed, and the connection now sets `PRAGMA
  secure_delete=ON` so a deleted page is zeroed instead of being recycled with
  its contents intact. That second part matters more than it looks, and it
  covers both statements: retention DELETEs old runs continuously, and a
  shrinking UPDATE — which is exactly what blanking a secret out of a stored
  graph is — releases overflow pages, so the sweep above depends on the pragma
  too. Without it, measurement put the old bytes in the main database file in
  both cases. Cost is unmeasurable at this project's write volume (write
  throughput inside noise; about 1 ms on a prune of 40 runs, 1k events and 4k
  metric rows).

  **One window remains, and it is the reason to rotate.** Runs that retention
  had already pruned *before* you upgrade are beyond both fixes: the rows are
  gone, so the sweep cannot reach them, and on the old build their freed pages
  kept the key until a `VACUUM`. If you ran a graph carrying a SECRET param on
  an earlier version, rotate that key. For anything written from here on there
  is no such caveat.

- **Documented that ambient credentials are instance-wide** ([#251]). A new
  [Shared Instances](https://docs.codefyui.com/usage/shared-instances) page
  states what a shared box actually shares: once one person completes the
  ChatGPT sign-in every graph bills to their personal account and anyone can
  sign them out, an `LLMChat` node with an empty key silently spends whatever
  is in `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`, and `KaggleDataset` uses the
  service account's credentials. No behaviour changed — this is correct for a
  single-user install, and the point is that nothing said so.

## [2.1.1] — 2026-08-08

A small release cut for one reason: the CSVReader file picker landed on `main`
two commits after the 2.1.0 tag, so nobody running a released version had it.
The textbook's I1-1 lab instructs students to upload `grades.csv` through that
picker, which made the instructions false for every installed copy. Shipping it
is the fix.

### Added

- **`CSVReader` gets a file picker, like `ImageReader` already had** ([#239],
  contributed by [@oyea0801]). `path` was a free-text field defaulting to
  `data/samples/iris.csv`, so using your own CSV meant knowing where CodefyUI
  keeps its data directory and typing a path relative to it. It is now a
  dropdown plus an upload button: pick the file off your machine and it stays
  in the dropdown for every later graph. The default is now empty — a starter
  graph should not name a file that only exists on the author's computer.
  Adds `/api/files` for listing and uploading data files.

  The eight shipped example graphs that read a CSV all set `path` explicitly
  with a directory component, so the changed default does not affect them.

### Fixed

- **The release guide gave advice that was wrong in one detail** ([#241]).
  It said `git tag -m "..."` was unaffected by the comment-stripping that eats
  markdown headings from tag annotations. It is not — `-m` loses them exactly
  as `-F` does, verified both ways. Only `--cleanup=verbatim` preserves them.

## [2.1.0] — 2026-08-06

A classroom-readiness release. The headline items are not new features: they
are the things that stopped a room full of students from getting started at
all, several of which had been failing silently.

### Security

- **The install and update path pointed at a repository the project no longer
  owns.** `install.sh`, `install.ps1`, `scripts/dev.py` and the two README
  one-liners all fetched from `treeleaves30760/CodefyUI`, which survived only
  on a GitHub rename redirect. That redirect dies permanently the moment
  anyone creates a repo at the old path, and the old owner name stays
  claimable — at which point every install and every `cdui update` would clone
  and unpack `frontend-dist.tar.gz` from a repository the maintainer does not
  control, with no signature or checksum verification anywhere in the path.
  All 54 references repointed, and a test now asserts the four copies agree
  and that no tracked file names the old owner. ([#231])
- Closed three bypasses in the plugin and custom-node AST gate: `import nt` /
  `import posix` (the C-level modules `os` itself imports from and re-exports —
  neither had ever been enumerated), a skip-directory scan gap, and an
  allowlisted-library attribute escape. ([#221])

### Added

- A real design-token layer (`src/styles/tokens.css`) with a contrast gate that
  runs as the first step of `pnpm build`. There was no token layer at all
  before: 69 of 142 measured elements failed WCAG AA, 70 rendered under 13px,
  and the welcome screen's own headline measured 2.66:1. Node titles sat between
  1.85:1 and 2.69:1 on twelve of the fourteen categories — on every node, in
  every graph. Now 2 of 125 fail, and both are disabled controls, which SC 1.4.3
  exempts. ([#229])
- `cdui --version`, a `version` field on `/api/health`, and the version in the
  OpenAPI document. Nothing reported a version before, so a bug report from a
  classroom could not say which build it came from. A test now asserts the three
  hand-edited version fields agree. ([#236])
- `cdui plugin list` and the `cdui start` banner now name built-in packs that
  shipped on disk but were never installed. A release can add a pack and
  `cdui update` places its files, but the server loads only what the lockfile
  records — so the pack is fully installable and completely invisible.
  Measured on a real install: all five current packs were uninstalled while the
  superseded `c1`–`c6` remained. Discoverability only; nothing is auto-enabled.
  ([#233])
- This changelog, and a "Before you tag" section in `RELEASING.md`. ([#237])
- Periodic checkpointing on `TrainingLoop` (`checkpoint_every`), so a run killed
  by SIGKILL, an OOM kill or a restart keeps its completed epochs instead of
  losing all of them. Checkpoints were previously written only after the loop
  returned, or on a cooperative stop that needs the process alive. ([#226])
- Per-epoch `val_accuracy` and a `monitor` option for early stopping.
  `TrainingLoop` recorded six metric series and not one of them was accuracy, so
  the curve people actually read for a classifier did not exist. ([#218])

### Fixed

- **Serving the editor on a LAN address rendered a blank page.** `generateId()`
  called `crypto.randomUUID()` unguarded; that API is secure-context only, so
  it is `undefined` over plain HTTP — exactly `cdui start --host <lan-ip>`, the
  way one machine is pointed at a room. The call happens during module
  evaluation, so `createRoot().render()` was never reached and `<div id="root">`
  stayed empty, with only a console `TypeError` to show for it. The teacher saw
  a working URL; every student saw white. ([#230])
- **A shared GPU machine trained every student on the CPU.** The global device
  fell back to `cpu` for anyone who had never opened Settings, while the backend
  already computed the best available device and served it — the frontend simply
  never read that field. An explicit choice, including an explicit `cpu`, still
  wins. ([#235])
- **The beginner-friendly error messages had never run in production.** Every
  rule in `friendlyError` matched a `KeyError:` / `ValueError:` prefix, but the
  backend sends `str(exc)`, which never contains the class name —
  `str(KeyError('tensor'))` is `"'tensor'"`. The tests passed because they fed
  it strings the backend cannot emit. A student who forgot a connection saw the
  literal text `'tensor'`. The exception class now travels as its own field, and
  the messages are localized. Added the most common first-CNN mistake, which
  appeared nowhere: `mat1 and mat2 shapes cannot be multiplied` now names the
  two numbers and which one to change. ([#234])
- The plugin catalog advertised fourteen nodes that no pack installs — `deep`
  claimed twelve and ships five, `rl` claimed four and ships one. `cdui plugin
  search` prints that prose verbatim, so it is where a student learns what to
  type into a palette that then returns nothing. ([#232])
- `cdui` could hang forever on a locked-down network: `_ensure_uv()` runs before
  every command and its installer had no timeout, so a network that drops
  packets was indistinguishable from a slow one. Now bounded, with an error that
  names the ways out. ([#231])
- `LRScheduler`'s epoch units are now stated where the mistake is made. The
  scheduler steps once per epoch, so `T_max` should normally equal
  `TrainingLoop.epochs` — the two live on different nodes with nothing between
  them, and getting it wrong costs a few points of accuracy while looking like
  an architecture problem. Documented rather than enforced, since a truncated
  schedule is legitimate and `CosineAnnealingWarmRestarts` reuses the same value
  as `T_0`, where equality would mean no restart ever happens. ([#238])
- `GET /api/execution/outputs/...` returned 500 for any tensor containing NaN or
  Inf, taking the inspector's I/O tab down with it — Starlette renders with
  `allow_nan=False`. Also: writers that always write, content-aware cache keys,
  and a validate/execute agreement. ([#225])
- A graph submitted with `{"device": "cuda"}` trained on the GPU and then
  silently evaluated on the CPU: `EvaluateModel.device` was the one selector in
  the training path that could not follow the run device. ([#212])
- Delete removed the last-*clicked* node rather than the one on screen, because
  the store's `selectedNodeId` and React Flow's per-node `selected` flag were two
  competing sources of truth. Also: log attribution to the event's own tab, an
  untranslated `Cancel` in every dialog, and a silent IndexedDB→localStorage
  fallback that brought back the 5 MB ceiling unannounced. ([#213])
- Layer-editor auto-layout was all-or-nothing: one node with a position turned
  the gate off and the other sixty-nine stacked at the origin, which reads as
  data loss rather than a layout bug. ([#214])

### Internal

- Test and CI hygiene: fixture paths (eight tests errored when pytest ran from
  the repo root, which is how the contribution docs invoke it), test isolation, a
  byte-scan guard, and two deflaked timing tests. ([#217])

## Released

Notes for these live in their GitHub release, written as the tag annotation:

| Version | Date | Notes |
|---|---|---|
| 2.1.0 | 2026-08-06 | [release](https://github.com/CodefyUI/CodefyUI/releases/tag/2.1.0) |
| 2.0.0 | 2026-08-05 | [release](https://github.com/CodefyUI/CodefyUI/releases/tag/2.0.0) |
| 1.4.2 | 2026-07-20 | [release](https://github.com/CodefyUI/CodefyUI/releases/tag/1.4.2) |
| 1.4.1 | 2026-07-20 | [release](https://github.com/CodefyUI/CodefyUI/releases/tag/1.4.1) |
| 1.4.0 | 2026-07-18 | [release](https://github.com/CodefyUI/CodefyUI/releases/tag/1.4.0) |
| 1.3.0 | 2026-06-13 | [release](https://github.com/CodefyUI/CodefyUI/releases/tag/1.3.0) |
| 1.2.1 | 2026-06-04 | [release](https://github.com/CodefyUI/CodefyUI/releases/tag/1.2.1) |
| 1.2.0 | 2026-06-02 | [release](https://github.com/CodefyUI/CodefyUI/releases/tag/1.2.0) |
| 1.1.2 | 2026-06-02 | [release](https://github.com/CodefyUI/CodefyUI/releases/tag/1.1.2) |
| 1.1.1 | 2026-06-02 | [release](https://github.com/CodefyUI/CodefyUI/releases/tag/1.1.1) |
| 1.1.0 | 2026-06-01 | [release](https://github.com/CodefyUI/CodefyUI/releases/tag/1.1.0) |
| 1.0.3 | 2026-05-05 | [release](https://github.com/CodefyUI/CodefyUI/releases/tag/1.0.3) |
| 1.0.2 | 2026-05-05 | [release](https://github.com/CodefyUI/CodefyUI/releases/tag/1.0.2) |
| 1.0.1 | 2026-05-05 | [release](https://github.com/CodefyUI/CodefyUI/releases/tag/1.0.1) |
| 1.0.0 | 2026-05-05 | [release](https://github.com/CodefyUI/CodefyUI/releases/tag/1.0.0) |

Release candidates before 1.0.0 are on the
[releases page](https://github.com/CodefyUI/CodefyUI/releases).

[#212]: https://github.com/CodefyUI/CodefyUI/pull/212
[#213]: https://github.com/CodefyUI/CodefyUI/pull/213
[#214]: https://github.com/CodefyUI/CodefyUI/pull/214
[#217]: https://github.com/CodefyUI/CodefyUI/pull/217
[#218]: https://github.com/CodefyUI/CodefyUI/pull/218
[#221]: https://github.com/CodefyUI/CodefyUI/pull/221
[#225]: https://github.com/CodefyUI/CodefyUI/pull/225
[#226]: https://github.com/CodefyUI/CodefyUI/pull/226
[#229]: https://github.com/CodefyUI/CodefyUI/pull/229
[#230]: https://github.com/CodefyUI/CodefyUI/pull/230
[#231]: https://github.com/CodefyUI/CodefyUI/pull/231
[#232]: https://github.com/CodefyUI/CodefyUI/pull/232
[#233]: https://github.com/CodefyUI/CodefyUI/pull/233
[#234]: https://github.com/CodefyUI/CodefyUI/pull/234
[#235]: https://github.com/CodefyUI/CodefyUI/pull/235
[#236]: https://github.com/CodefyUI/CodefyUI/pull/236
[#237]: https://github.com/CodefyUI/CodefyUI/pull/237
[#238]: https://github.com/CodefyUI/CodefyUI/pull/238
[#239]: https://github.com/CodefyUI/CodefyUI/pull/239
[#241]: https://github.com/CodefyUI/CodefyUI/pull/241
[#144]: https://github.com/CodefyUI/CodefyUI/issues/144
[#149]: https://github.com/CodefyUI/CodefyUI/issues/149
[#252]: https://github.com/CodefyUI/CodefyUI/pull/252
[#251]: https://github.com/CodefyUI/CodefyUI/issues/251
[#253]: https://github.com/CodefyUI/CodefyUI/issues/253
[#254]: https://github.com/CodefyUI/CodefyUI/issues/254
[#259]: https://github.com/CodefyUI/CodefyUI/issues/259
[#242]: https://github.com/CodefyUI/CodefyUI/issues/242
[#265]: https://github.com/CodefyUI/CodefyUI/issues/265
[#196]: https://github.com/CodefyUI/CodefyUI/issues/196
[#197]: https://github.com/CodefyUI/CodefyUI/issues/197
[#189]: https://github.com/CodefyUI/CodefyUI/issues/189
[#190]: https://github.com/CodefyUI/CodefyUI/issues/190
[#193]: https://github.com/CodefyUI/CodefyUI/issues/193
[#224]: https://github.com/CodefyUI/CodefyUI/issues/224
[#272]: https://github.com/CodefyUI/CodefyUI/pull/272
[#274]: https://github.com/CodefyUI/CodefyUI/issues/274
[#275]: https://github.com/CodefyUI/CodefyUI/issues/275
[#222]: https://github.com/CodefyUI/CodefyUI/issues/222
[#277]: https://github.com/CodefyUI/CodefyUI/issues/277
[#186]: https://github.com/CodefyUI/CodefyUI/issues/186
[#283]: https://github.com/CodefyUI/CodefyUI/issues/283
[#285]: https://github.com/CodefyUI/CodefyUI/issues/285
[#289]: https://github.com/CodefyUI/CodefyUI/issues/289
[#290]: https://github.com/CodefyUI/CodefyUI/issues/290
[#291]: https://github.com/CodefyUI/CodefyUI/issues/291
[#288]: https://github.com/CodefyUI/CodefyUI/issues/288
[#292]: https://github.com/CodefyUI/CodefyUI/issues/292
[#175]: https://github.com/CodefyUI/CodefyUI/issues/175
[#348]: https://github.com/CodefyUI/CodefyUI/issues/348
[#200]: https://github.com/CodefyUI/CodefyUI/issues/200
[#141]: https://github.com/CodefyUI/CodefyUI/issues/141
[#148]: https://github.com/CodefyUI/CodefyUI/issues/148
[#162]: https://github.com/CodefyUI/CodefyUI/issues/162
[#270]: https://github.com/CodefyUI/CodefyUI/issues/270
[#297]: https://github.com/CodefyUI/CodefyUI/issues/297
[#298]: https://github.com/CodefyUI/CodefyUI/issues/298
[#299]: https://github.com/CodefyUI/CodefyUI/issues/299
[#300]: https://github.com/CodefyUI/CodefyUI/issues/300
[#308]: https://github.com/CodefyUI/CodefyUI/issues/308
[#309]: https://github.com/CodefyUI/CodefyUI/issues/309
[#310]: https://github.com/CodefyUI/CodefyUI/issues/310
[#311]: https://github.com/CodefyUI/CodefyUI/issues/311
[#312]: https://github.com/CodefyUI/CodefyUI/issues/312
[#316]: https://github.com/CodefyUI/CodefyUI/issues/316
[#331]: https://github.com/CodefyUI/CodefyUI/issues/331
[#332]: https://github.com/CodefyUI/CodefyUI/pull/332
[#338]: https://github.com/CodefyUI/CodefyUI/pull/338
[#346]: https://github.com/CodefyUI/CodefyUI/pull/346
[#347]: https://github.com/CodefyUI/CodefyUI/pull/347
[#350]: https://github.com/CodefyUI/CodefyUI/pull/350
[#352]: https://github.com/CodefyUI/CodefyUI/pull/352
[#354]: https://github.com/CodefyUI/CodefyUI/pull/354
[#355]: https://github.com/CodefyUI/CodefyUI/pull/355
[#356]: https://github.com/CodefyUI/CodefyUI/pull/356
[#357]: https://github.com/CodefyUI/CodefyUI/pull/357
[#359]: https://github.com/CodefyUI/CodefyUI/pull/359
[#360]: https://github.com/CodefyUI/CodefyUI/issues/360
<<<<<<< HEAD
[#380]: https://github.com/CodefyUI/CodefyUI/issues/380
=======
[#371]: https://github.com/CodefyUI/CodefyUI/pull/371
[#372]: https://github.com/CodefyUI/CodefyUI/issues/372
>>>>>>> origin/main
[@oyea0801]: https://github.com/oyea0801
[@latteine1217]: https://github.com/latteine1217
[Unreleased]: https://github.com/CodefyUI/CodefyUI/compare/2.4.1...main
[2.4.1]: https://github.com/CodefyUI/CodefyUI/compare/2.4.0...2.4.1
[2.4.0]: https://github.com/CodefyUI/CodefyUI/compare/2.3.0...2.4.0
[2.3.0]: https://github.com/CodefyUI/CodefyUI/compare/2.2.0...2.3.0
[2.2.0]: https://github.com/CodefyUI/CodefyUI/compare/2.1.1...2.2.0
[2.1.1]: https://github.com/CodefyUI/CodefyUI/compare/2.1.0...2.1.1
[2.1.0]: https://github.com/CodefyUI/CodefyUI/compare/2.0.0...2.1.0
