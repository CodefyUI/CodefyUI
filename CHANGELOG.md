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

### Changed

- **The declared torch floor is 2.5, up from 2.0.0** ([#252]). It is the
  version the code already assumed: `torch.OutOfMemoryError` arrived there, and
  the OOM classifier had been carrying a second lookup for the pre-2.5 spelling
  that nothing exercised. That branch is gone; the message match stays, since
  it is how MPS and out-of-tree backends report an OOM on any torch. Installs
  on torch 2.0-2.4 are no longer supported.

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
[#252]: https://github.com/CodefyUI/CodefyUI/pull/252
[@oyea0801]: https://github.com/oyea0801
[Unreleased]: https://github.com/CodefyUI/CodefyUI/compare/2.1.1...main
[2.1.1]: https://github.com/CodefyUI/CodefyUI/compare/2.1.0...2.1.1
[2.1.0]: https://github.com/CodefyUI/CodefyUI/compare/2.0.0...2.1.0
