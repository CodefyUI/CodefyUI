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

### Security

- Closed three bypasses in the plugin and custom-node AST gate: `import nt` /
  `import posix` (the C-level modules `os` itself imports from and re-exports —
  neither had ever been enumerated), a skip-directory scan gap, and an
  allowlisted-library attribute escape. ([#221])

### Added

- Periodic checkpointing on `TrainingLoop` (`checkpoint_every`), so a run killed
  by SIGKILL, an OOM kill or a restart keeps its completed epochs instead of
  losing all of them. Checkpoints were previously written only after the loop
  returned, or on a cooperative stop that needs the process alive. ([#226])
- Per-epoch `val_accuracy` and a `monitor` option for early stopping.
  `TrainingLoop` recorded six metric series and not one of them was accuracy, so
  the curve people actually read for a classifier did not exist. ([#218])

### Fixed

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
[Unreleased]: https://github.com/CodefyUI/CodefyUI/compare/2.0.0...main
