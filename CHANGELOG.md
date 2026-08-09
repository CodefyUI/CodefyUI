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

### Fixed

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

### Changed

- **`value_bytes` now says when it stops measuring** ([#193]). The
  `MAX_WALK_ITEMS` cap already logged that its total was a lower bound;
  `MAX_WALK_DEPTH` returned a smaller number in silence, which makes an
  under-count indistinguishable from a genuinely small value. The module
  docstring also claimed over-counting as *the* safe direction — true of
  cross-measurement sharing, and not true of the three things that make the
  walk under-count, which is the direction that costs memory.

### Added

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
[#222]: https://github.com/CodefyUI/CodefyUI/issues/222
[#277]: https://github.com/CodefyUI/CodefyUI/issues/277
[@oyea0801]: https://github.com/oyea0801
[Unreleased]: https://github.com/CodefyUI/CodefyUI/compare/2.2.0...main
[2.2.0]: https://github.com/CodefyUI/CodefyUI/compare/2.1.1...2.2.0
[2.1.1]: https://github.com/CodefyUI/CodefyUI/compare/2.1.0...2.1.1
[2.1.0]: https://github.com/CodefyUI/CodefyUI/compare/2.0.0...2.1.0
