# Releasing CodefyUI

Quick reference for cutting a new release. The CI does the heavy lifting; the
maintainer's job is to push the tag and check the result before publishing.

## TL;DR — happy path

```bash
# 1. Promote CHANGELOG.md's [Unreleased] section to the new version, bump the
#    three version fields, and stamp any "unreleased" docs placeholder with the
#    new number (see "Before you tag" below).
# 2. From main, once that commit is in:
git tag 1.0.0rcN
git push origin 1.0.0rcN
```

## Before you tag

Three things are done by hand, and nothing else in the pipeline checks them for
you:

1. **Promote `CHANGELOG.md`.** Rename `## [Unreleased]` to
   `## [X.Y.Z] — YYYY-MM-DD`, open a fresh empty `## [Unreleased]` above it, and
   update the `[Unreleased]` compare link at the bottom to point at the new tag.
   The tag annotation — which becomes the GitHub release body — should say the
   same thing; the changelog is what answers "what is on main that nobody has
   yet" *between* releases, which the tag cannot.

2. **Bump the version in all three files**, which are edited by hand and which
   nothing reconciles: `backend/pyproject.toml`, `backend/uv.lock` (regenerate
   with `uv lock`), and `frontend/package.json`. A mismatch between them ships
   silently — the frontend claiming one version while the backend claims
   another — so check all three before tagging.

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

   Today the marker sits on the plugin `apiVersion` table and its availability
   note (`docs/.../advanced/plugin-frontend-extensions.md` + the zh-TW twin),
   whose version column says which CodefyUI release shipped each `apiVersion`;
   a new row lands as a placeholder because the number does not exist when the
   PR is written. That is not hypothetical: the apiVersion 3 row said "1.5.0"
   from 2.0.0 through 2.2.0 — a version never tagged — because it was written
   before 2.0.0 was the number, and nothing brought anyone back to it.

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
   - `make_latest` = true (overrides GitHub's "skip prereleases for /latest"
     so `releases/latest/download/...` resolves to this rc)
2. Open the draft, **edit notes if needed**.
3. Click **Publish** — no manual flag toggles required.
4. **Install Check** workflow fires automatically and end-to-ends `install.sh` /
   `install.ps1` against the just-published asset on Linux/macOS/Windows.

## Workflows that gate the release

| Workflow | Triggers | Catches |
|----------|----------|---------|
| `frontend-build.yml` | PR + push to `main` (frontend changes) | broken `pnpm build` / `tsc` / `vitest` before merge |
| `release-build.yml` | tag push, `release: created`, manual | tag without a fresh asset |
| `install-check.yml` | `release: published`, manual | install flow regression on real OS runners |

## When CI surprises you

- **Release Build failed** — fix the cause (lockfile mismatch, build error)
  and re-push the tag (`git tag -d X && git push --delete origin X && git tag X
  && git push origin X`). The workflow concurrency block cancels the prior run.
- **Install Check failed after publish** — the asset is still attached, but
  `install.sh` / `install.ps1` broke. Check the failing job's log; usually a
  Node version or network issue.
- **Only the rc is "latest" but Github keeps showing the previous stable** —
  toggle *Set as the latest release* on the rc; `/releases/latest/download/...`
  follows that flag, not the tag's chronological order.

## Manual rebuild of an existing release

```text
Actions → Release Build → Run workflow → Tag: 1.0.0rcN
```

This re-builds and replaces `frontend-dist.tar.gz` on the existing release
without touching anything else. Useful if a release was published before
`frontend-build.yml` existed and the asset is missing.
