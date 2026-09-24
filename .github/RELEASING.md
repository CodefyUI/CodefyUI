# Releasing CodefyUI

Quick reference for cutting a new release. The CI does the heavy lifting; the
maintainer's job is to push the tag and check the result before publishing.

## TL;DR — happy path

```bash
# 1. Promote CHANGELOG.md's [Unreleased] section to the new version, bump the
#    three version fields, stamp any "unreleased" docs placeholder with the
#    new number, and check the plugin template's SDK copy (see "Before you
#    tag" below).
# 2. From main, once that commit is in, tag it with the release notes in
#    notes.md -- an annotated tag, kept verbatim (see "Then on GitHub"):
git tag -a X.Y.Z --cleanup=verbatim -F notes.md
# The check Release Build runs; fails on a wrong heading, link or date:
python scripts/check_changelog.py --tag X.Y.Z
git push origin X.Y.Z
```

## Before you tag

Four things are done by hand. Steps 1 and 2 are checked on every PR
(`test_check_changelog.py`, `test_all_version_fields_agree` and
`uv lock --check`), and step 1 again when the tag is pushed. Steps 3 and 4
are not checked.

1. **Promote `CHANGELOG.md`.** Rename `## [Unreleased]` to
   `## [X.Y.Z] — YYYY-MM-DD`, open a fresh empty `## [Unreleased]` above it,
   update the `[Unreleased]` compare link at the bottom to start at the new
   tag, and add
   `[X.Y.Z]: https://github.com/CodefyUI/CodefyUI/compare/<previous>...X.Y.Z`
   below it so the new heading links like the others.

   > The date is the tag's **UTC** date — what the tag object and GitHub's
   > `published_at` both record, and what 2.8.0 and 2.5.0 used. On a UTC+8 box
   > that differs from the local date for eight hours out of every day.
   >
   > **Re-read it just before you push the tag.** The heading is written when
   > the release PR is opened; the tag goes up when that PR is merged, and
   > nothing recomputes the date in between. 2.8.2 was promoted as 2026-09-16
   > and tagged eleven hours later at 2026-09-17T05:22Z, which took a
   > follow-up PR to correct.

   `scripts/check_changelog.py` checks this step on every PR, through
   `backend/tests/test_check_changelog.py`: `## [Unreleased]` on top and only
   once; below it the newest version, named after the version in
   `backend/pyproject.toml`, written with an em dash and dated no earlier than
   the heading below it; the versions newest first; one `[Unreleased]` link,
   starting at that version; and one compare link per version, starting at
   the release below it. `--tag X.Y.Z` reads `CHANGELOG.md` and
   `backend/pyproject.toml` at a tag you created locally, and also requires
   the newest heading to name the tag and carry its UTC date, and
   `[Unreleased]` to hold no entries: renaming `## [Unreleased]` moves them
   all, while adding the new heading below them leaves them there. Run it
   before `git push`: the tag push runs the same check, and a refusal caught
   locally saves deleting the tag on origin and a red Release Build run. A
   wrong heading still needs a fix PR, because the heading lives on `main`.

   The tag annotation — which becomes the GitHub release body — should say the
   same thing; the changelog is what answers "what is on main that nobody has
   yet" *between* releases, which the tag cannot.

2. **Bump the version in all three files**, which are edited by hand:
   `backend/pyproject.toml`, `backend/uv.lock` (regenerate with `uv lock`), and
   `frontend/package.json`. `test_all_version_fields_agree` fails a PR whose
   three fields disagree, and `uv lock --check` fails one that bumps
   `pyproject.toml` without regenerating the lock.

3. **Stamp every docs placeholder that is waiting for this version number.**

   ```bash
   git grep -n "stamp-on-release" docs/
   ```

   A docs page that promises a feature "from the next release" has to name the
   release once there is one, so every such spot carries the marker
   `{/* stamp-on-release */}` — an **MDX** comment, which renders to nothing at
   all (verified: the string does not appear in the built HTML of either locale).
   Replace the placeholder with the version you are tagging and delete the marker
   with it.

   > Use `{/* ... */}`, not `<!-- ... -->`. Docusaurus compiles these `.md` pages
   > as MDX, where an HTML comment is a syntax error — `pnpm build` in `docs/`
   > fails with "MDX compilation failed" rather than quietly ignoring it.

   The marker is ASCII **on purpose**: the placeholder text itself is
   translated (`*next release (unreleased)*` in `docs/docs/`,
   `*下一個版本（尚未發布）*` in `docs/i18n/zh-TW/`), so grepping for the English
   words finds the English row and silently walks past the Chinese one — which
   reproduces the bug in the locale nobody proofreads. One marker, both locales,
   one command. Add it to any new placeholder you write, in every locale.

   No page carries the marker at the moment, so the command should print
   nothing. The place it is made for is the plugin `apiVersion` table and its
   availability note (`docs/.../advanced/plugin-frontend-extensions.md` + the
   zh-TW twin), whose version column says which CodefyUI release shipped each
   `apiVersion`: a new row lands as a placeholder because the number does not
   exist when the PR is written. That is not hypothetical: the apiVersion 3
   row said "1.5.0" from 2.0.0 through 2.2.0 — a version never tagged —
   because it was written before 2.0.0 was the number, and nothing brought
   anyone back to it.

4. **Check the plugin template's SDK copy.** The README and the plugin docs
   tell authors to fork
   [CodefyUI-Plugin-Official](https://github.com/CodefyUI/CodefyUI-Plugin-Official),
   which vendors the plugin SDK in `ui/src/sdk/`. CI cannot see that
   repository, so compare a local checkout of it with this release:

   ```bash
   git clone https://github.com/CodefyUI/CodefyUI-Plugin-Official ../CodefyUI-Plugin-Official  # once
   git -C ../CodefyUI-Plugin-Official pull
   python scripts/sync_plugin_sdk.py --template ../CodefyUI-Plugin-Official --check
   ```

   Exit 1 lists the stale files. The same command without `--check` rewrites
   them, `types.ts` from `frontend/src/plugins/contract.ts` and the rest from
   the `cdui plugin new` scaffold's `ui/src/sdk/`; open a pull request with
   them in the template repository. The template's copy fell three API
   versions behind before this step existed (#461).

Then on GitHub:
1. Wait for **Release Build** to finish (≈2 min) — produces a draft release.
   The workflow already sets:
   - `frontend-dist.tar.gz` attached
   - body = your tag annotation — **always pass `--cleanup=verbatim`**
     > `git tag` strips every line beginning with `#` as a comment, so a
     > markdown annotation silently loses all its headings and the release
     > body arrives as one unbroken wall of text.
     >
     > **This applies to `-m "..."` just as much as to `-F notes.md`** —
     > verified both ways; only `--cleanup=verbatim` preserves them:
     > ```bash
     > git tag -a X.Y.Z --cleanup=verbatim -F notes.md   # or -m "..."
     > ```
     > The failure is invisible from the command line: the tag exists, Release
     > Build succeeds, the asset attaches, and the body is non-empty. You only
     > see it by reading the rendered release page. This ate all five headings
     > of 2.1.0's notes on the first attempt.
   - `prerelease` = true if the tag matches `rc` / `beta` / `alpha` / `dev`
   - `make_latest` is not set: GitHub refuses it on a draft
2. Open the draft, **edit notes if needed**.
3. Click **Publish**. A stable release becomes **Latest** by GitHub's default.
   A prerelease cannot be Latest, so `releases/latest/...` (what the
   installers and `cdui update` download) stays on the previous stable
   release; install an rc with `CODEFYUI_RELEASE_TAG=<tag>`.
4. **Install Check** workflow fires automatically and end-to-ends `install.sh` /
   `install.ps1` against the just-published asset on Linux/macOS/Windows.

## Workflows that gate the release

| Workflow | Triggers | Catches |
|----------|----------|---------|
| `frontend-build.yml` | every PR; push to `main` touching `frontend/**`, `examples/**` or `backend/tests/fixtures/**` | broken `pnpm build` / `tsc` / `vitest`, and Vite chunk warnings, before merge |
| `backend-test.yml` | every PR; push to `main` touching the backend, examples, plugins, scripts or the frontend files the backend tests read | pytest on 3.10 / 3.11 / 3.12, on Windows 3.12 and against a built frontend; a half-promoted `CHANGELOG.md` (`test_check_changelog.py`); `uv lock --check`; ruff |
| `byte-scan.yml` | every PR and push to `main` | raw C0 control bytes in tracked files |
| `release-build.yml` | tag push, `release: created`, manual | tag without a fresh asset; on a tag push, a tag `CHANGELOG.md` does not name or dates on another UTC day, or entries left under `[Unreleased]` |
| `install-check.yml` | `release: published`, manual | install flow regression on real OS runners |

## When CI surprises you

- **Release Build failed** — fix the cause (lockfile mismatch, build error)
  and re-push the tag (`git tag -d X && git tag -a X --cleanup=verbatim -F
  notes.md && python scripts/check_changelog.py --tag X && git push --delete
  origin X && git push origin X`). The re-created tag gets a new date, and the
  check compares it with the heading before origin is touched: a refusal
  leaves the old tag on origin, and the same chain runs again once the cause
  is fixed. The workflow concurrency block cancels the prior run.
- **Release Build failed at "Check CHANGELOG.md against the tag"** — the step
  runs before the release is created, so this run made no draft. Delete the
  tag locally and on origin (`git tag -d X && git push --delete origin X`),
  fix what the message names, then tag, check and push again as in the
  TL;DR:
  - *tag X is not the newest version*: the tag is on the wrong commit or has
    the wrong name. On the wrong commit, tag the commit that promotes
    `[Unreleased]` to X; with the wrong name, tag the same commit with the
    version the newest heading names. No PR is needed.
  - *the heading has to carry another date*: the heading lives on `main`, so
    fix it in a PR and tag the new merge commit. A re-created tag gets a new
    date, so re-check the heading against the UTC clock (`date -u +%F`)
    first.
  - *`[Unreleased]` still holds entries*: if the release PR added
    `## [X] — ...` below them instead of renaming `## [Unreleased]`, move
    them under `## [X]` in a PR and tag the new merge commit. If the tag is
    on a commit after the release, tag the release commit instead.
  - Anything else is a `CHANGELOG.md` problem the release PR's checks should
    have stopped: fix it in a PR and tag the new merge commit.
- **Install Check failed after publish** — the asset is still attached, but
  `install.sh` / `install.ps1` broke. Check the failing job's log; usually a
  Node version or network issue.
- **A published rc is not "Latest"** — expected: GitHub never marks a
  prerelease as Latest, and `/releases/latest/download/...` follows that flag,
  not the tag's chronological order. Point an install at the rc with
  `CODEFYUI_RELEASE_TAG=<tag>`, or publish it without the pre-release flag.

## Manual rebuild of an existing release

```text
Actions → Release Build → Run workflow → Tag: X.Y.Z
```

This rebuilds `frontend-dist.tar.gz` from the tag and replaces the asset, and
it also turns the release back into a draft with its body reset to the tag
annotation, so notes edited on GitHub are lost. If it was the latest release,
`releases/latest` (what the installers and `cdui update` download) resolves to
the one before it until you publish it again; publishing runs Install Check
again. Useful if a release was published before `frontend-build.yml` existed
and the asset is missing.

Leaving **Tag** blank does not produce an artifact, although the form says
"leave blank to upload as artifact only": the run fails at the **Read tag
annotation as release body** step (`fatal: ambiguous argument ''`) before the
frontend is built, and no release is changed.
