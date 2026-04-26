# Releasing CodefyUI

Quick reference for cutting a new release. The CI does the heavy lifting; the
maintainer's job is to push the tag and check the result before publishing.

## TL;DR — happy path

```bash
# from main, after the version-bump commit is in
git tag 1.0.0rcN
git push origin 1.0.0rcN
```

Then on GitHub:
1. Wait for **Release Build** to finish (≈2 min) — produces a draft release with
   `frontend-dist.tar.gz` attached.
2. Open the draft, **edit notes if needed**.
3. **For pre-release tags (`*rc*`)**: tick *Set as a pre-release* AND *Set as
   the latest release* — without the latter, `releases/latest/download/...`
   skips the rc and end-user installers fall through to local build.
4. Click **Publish**.
5. **Install Check** workflow fires automatically and end-to-ends `install.sh` /
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
