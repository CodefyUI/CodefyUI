# Contributing to CodefyUI

Thanks for wanting to help. This document covers the three things you need before your first pull request: how to sign your commits, how to get a working development environment, and what the review conventions in this repository actually are.

> 中文摘要：送 PR 前請先讀「Signing your work」一節——每個 commit 都要用 `git commit -s` 加上 DCO 簽署，否則因為本專案採雙軌授權，你的程式碼無法被納入商業授權路徑。開發環境設定與各項檢查指令列在下面，指令都是從 `scripts/dev.py` 與 CI workflow 直接核對過的，不是憑印象寫的。

**Table of contents**

- [Signing your work (DCO)](#signing-your-work-dco)
- [Development environment](#development-environment)
- [Running the checks](#running-the-checks)
- [Branches and pull requests](#branches-and-pull-requests)
- [House style](#house-style)

---

## Signing your work (DCO)

**Every commit must carry a `Signed-off-by` trailer.** Add it automatically:

```bash
git commit -s -m "fix(engine): ..."
```

That appends one line to the commit message, using your configured `user.name` and `user.email`:

```
Signed-off-by: Random J Developer <random@developer.example.org>
```

Use your real name and a real, reachable email address. Anonymous and pseudonymous sign-offs cannot serve the purpose the sign-off exists for.

Forgot it? Fix the last commit with `git commit --amend -s --no-edit`, or a whole branch with:

```bash
git rebase --signoff main
```

Then force-push your branch. This rewrites commit hashes, which is fine on your own PR branch.

### Why a dual-licensed project needs this

CodefyUI is published under **AGPL-3.0-only**, and the maintainers also offer a **separate commercial license** to organizations that cannot work under copyleft terms (see [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md)). That second path only functions if the project can account for where every line of code came from and under what terms it arrived.

Put concretely: if someone pastes code they do not own into a pull request and it lands in `main`, the project has distributed code it had no right to distribute — and every downstream user inherits the problem. The sign-off is how you state, on the record and per commit, that this is not what happened.

The [Developer Certificate of Origin](https://developercertificate.org/) is the lightweight instrument for this. It is not a copyright assignment and it is not a CLA. **You keep the copyright in your contribution.** You are certifying provenance, not handing over ownership, and the certification lives in the git history rather than in a signed document somebody has to file and track.

### Inbound licensing

Two things are true when you sign off on a commit here.

1. **You certify the DCO 1.1 terms below.** That is the standard, unmodified text used across the Linux kernel and hundreds of other projects.
2. **You agree your contribution may be distributed by the project on either licensing path** — under AGPL-3.0-only, and under the commercial license offered by the copyright holder named in [NOTICE](NOTICE).

Point 2 is stated here as an inbound term of contributing, not as a separate agreement you sign. It is spelled out explicitly because the DCO on its own certifies *provenance* and submission under the project's open source license; it does not by itself say anything about a second, proprietary license. Rather than leave that gap implicit, the project states its position where you can read it before you contribute.

If you are contributing on behalf of an employer, make sure you actually have the authority to agree to both points. If you are not comfortable with point 2, say so in the pull request before doing the work and the maintainers will discuss it with you rather than merge something on an unclear basis.

### Developer Certificate of Origin 1.1

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

---

## Development environment

### Prerequisites

| Tool | Needed for | Notes |
|---|---|---|
| `git` | everything | |
| [`uv`](https://github.com/astral-sh/uv) | the Python backend | Replaces `pip` and `venv`. Every backend command below goes through it. |
| `pnpm` + Node.js 24+ | frontend work | **Optional.** Only needed if you touch `frontend/`, run `cdui dev`, or build the bundle yourself. |

### The short path

```bash
git clone https://github.com/CodefyUI/CodefyUI.git
cd CodefyUI
./cdui install --dev          # Windows: .\cdui.cmd install --dev
```

`--dev` is not optional for contributors. Without it the backend is installed as `.` rather than `.[dev]`, `pytest` never lands in the virtualenv, and `cdui test` exits with "not found". If you already installed without it, re-run with `--dev`.

`cdui install` creates `backend/.venv`, installs the backend editable, and then either builds `frontend/dist` (if pnpm is present) or downloads the prebuilt bundle from the latest release. Useful flags:

| Flag | Effect |
|---|---|
| `--dev` / `--no-dev` | install the `[dev]` extra (pytest, httpx, httpx-ws, tensorboard) |
| `--gpu auto` | detect your driver and pick the matching PyTorch wheel |
| `--gpu cpu` \| `cu118` \| `cu121` \| `cu124` \| `cu126` \| `cu128` \| `rocm6.1` \| `rocm6.2` \| `mps` \| `skip` | pick the wheel yourself; `skip` leaves an existing torch alone |
| `--yes` / `-y` | non-interactive; same as `--gpu auto --no-dev` |

With no flags on a TTY you get an interactive menu. See [GPU and Device Setup](https://docs.codefyui.com/getting-started/gpu-device) for how to choose.

> `make install`, `make dev`, `make test`, `make clean` and `make stop` are thin wrappers around `./cdui` and add nothing. They hardcode the POSIX launcher, so on Windows use `.\cdui.cmd <command>` directly.

### Doing it by hand

If you would rather not use the task runner:

```bash
cd backend
uv venv --python 3.11
source .venv/bin/activate     # Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"
```

### Watch out: uv will pick a Python newer than CI uses

`uv venv --python 3.11` pins the version. **Bare `uv` commands in a fresh clone or worktree do not** — they resolve to whatever the newest CPython on your machine is, which today can be 3.14. CI never tests that:

- `backend/pyproject.toml` declares `requires-python = ">=3.10"`.
- `.github/workflows/backend-test.yml` runs a matrix of **3.10, 3.11, 3.12** only (on ubuntu, plus one Windows job on 3.12).
- `cdui install` and both launchers (`cdui`, `cdui.cmd`) hardcode **3.11**.

So a test that passes locally on 3.14 tells you nothing about 3.10, and a syntax feature you reach for might be above the declared floor. Always create the venv with an explicit `--python 3.11` (or `3.10` if you want to develop against the floor), and let CI cover the rest.

### Running the app

```bash
./cdui dev      # backend :8000 with --reload + Vite :5173 with HMR. Needs pnpm.
./cdui start    # one uvicorn on :8000 serving the prebuilt frontend/dist. No Node needed.
```

`cdui dev` streams both processes in the foreground with `[backend]` / `[frontend]` prefixes; Ctrl+C stops both. `cdui start` daemonizes by default — use `cdui status` and `cdui stop` to manage it, or `--foreground` to attach.

Use `cdui dev` when you are changing frontend code. Use `cdui start` when you are only touching the backend, or when you want to see what an end user sees. If you changed frontend source and want `cdui start` to reflect it, rebuild first with `./cdui build`.

More detail: [Dev Install](https://docs.codefyui.com/getting-started/dev-install) and [CLI Commands](https://docs.codefyui.com/getting-started/cli-commands).

---

## Running the checks

There is a linter (`ruff`) and there is no formatter — do not go looking for one. There is also no pre-commit hook, so **nothing runs automatically**. These are the real gates, and you run them yourself.

### `cdui test` runs both halves

It used to run only `pytest` in `backend/`, which meant a green `cdui test` said nothing about the 138 frontend test files. Since core#245 it runs both and prints a summary naming each half:

```
=== Test summary ===
  backend   PASS
  frontend  PASS
```

Three things worth knowing:

- **A missing `pnpm` is a skip, not a failure.** CodefyUI has a deliberate no-Node install path (the release ships a prebuilt `frontend-dist.tar.gz`), so a machine with no Node is a healthy machine. The frontend half is reported as `SKIPPED` — never as a pass — and CI's `frontend-build.yml` runs those tests regardless.
- **Both halves always finish.** A red backend does not stop the frontend from running; you get both answers in one pass. The exit code is 1 if either failed.
- **`--backend` / `--frontend`** narrow the run when you know what you touched. Anything else is rejected rather than ignored — `cdui test -k foo` used to run the whole suite while looking like it had filtered.

### The full local set

Run every line that applies to what you touched:

```bash
# Repository root -- always. Guards against stray C0 control bytes,
# which no test, type check or grep catches (ripgrep silently skips them).
python scripts/check_control_bytes.py

# Repository root -- always. Same rule set CI runs (see ruff.toml).
uvx ruff@0.14.4 check .

# Backend + frontend. Add --backend if you only touched Python.
./cdui test

# Backend -- if you edited backend/pyproject.toml. CI runs this BEFORE
# installing anything, so a drifted lock fails all three matrices in
# about ten seconds. Regenerate the lock in the SAME commit as the
# dependency change.
cd backend && uv lock --check

# Frontend -- if you touched frontend/. `cdui test` covers `pnpm test`;
# these are the type-check and build gates it does not run.
cd frontend && pnpm install
cd frontend && pnpm exec tsc -b
cd frontend && pnpm build
```

Two notes on the frontend commands:

- **`tsc -b`, not `tsc --noEmit`.** `frontend/tsconfig.json` is a solution-style config with `"files": []` and project references, so `tsc --noEmit` against it checks **zero files** and passes no matter what is broken. Build mode follows the references and actually type-checks `src/`.
- **`pnpm build` includes the contrast gate.** The build script is `node scripts/check-contrast.mjs && tsc -b && vite build` — the first step re-derives every WCAG contrast relationship claimed by `frontend/src/styles/tokens.css` and fails the build if a token pair drops below threshold. Run it alone with `pnpm contrast` when you are editing colours.

### The linter

`ruff.toml` at the repo root covers `backend/`, `scripts/`, `plugins/` and `examples/` — the same blast radius `backend-test.yml` uses. It runs the rule set ruff itself defaults to (`E4`, `E7`, `E9`, `F`): unused imports, undefined names, unused locals, `== None`, bare `except`, syntax errors.

It is deliberately narrow. `E501` (line length), `I001` (import order) and the `B` / `SIM` / `UP` families are **not** enabled, and the file says why for each, with the finding count from the first run. If you want to turn one on, that is a welcome PR — one rule family at a time, with the fixes in the same diff.

There is no frontend linter yet. That is a bigger argument because it drags a formatting decision along with it; see core#245.

### What CI runs that you cannot easily run locally

- **`backend-test.yml`** runs the whole suite on Python 3.10, 3.11 and 3.12 on ubuntu, **plus one Windows job on 3.12**, plus `uv lock --check`, a smoke import (`from app.main import app`) that catches import-time syntax errors, and `ruff check`. The Windows job is not decoration: CPython 3.12 replaced `os.path.exists` / `isdir` / `isfile` / `islink` with `nt` C fast paths **on Windows only**, and `ntpath` guards that behind `try: from nt import ... except ImportError:` — so on ubuntu the fallback always wins and no Python version in an ubuntu-only matrix can ever see the difference (core#258). If you change anything that touches paths, processes or file locking, expect Windows to have an opinion.
- **`byte-scan.yml`** runs `scripts/check_control_bytes.py` over every tracked file on every PR, with no path filter.
- **`frontend-build.yml`** runs install, `tsc -b`, `pnpm build`, a `dist/` sanity check, then `pnpm test` — on `frontend/**` changes only.

### Tests are required

A pull request that changes behaviour needs a test that fails without the change. This is not negotiable for bug fixes: the test is the evidence that the bug was real and that it is now gone. For UI changes, unit tests are necessary but not sufficient — maintainers additionally verify UI-touching changes in a real browser before merging.

If a change genuinely cannot be tested, say so in the PR body and explain why, rather than leaving reviewers to notice.

---

## Branches and pull requests

**Never push to `main`.** Every change lands through a pull request, including one-line documentation fixes and CI repairs. There are no exceptions to this and no size below which it stops applying.

```bash
git switch -c fix/short-description main
# ... work, with `git commit -s` ...
git push -u origin fix/short-description
gh pr create
```

### Commit subjects

Conventional-commit format: `type(scope): imperative summary`.

```
fix(engine): NaN-safe outputs when a loss diverges mid-epoch
feat(training): periodic checkpointing so a killed server does not lose the run
docs(licensing): state what AGPL section 13 actually requires
ci: pin pyarrow below 25 until the segfault is fixed upstream
```

Types actually in use here, by frequency: `feat`, `fix`, `docs`, `chore`, `test`, `refactor`, `ci`, `perf`. Common scopes follow the area touched — `frontend`, `backend`, `ui`, `plugins`, `training`, `cli`, `nodes`, `security`, `examples`. The scope is optional; the type is not.

Keep the subject under about 72 characters, write it in the imperative, and make it say what the change *does* rather than what area it touches. `fix(cache): a second Run of a training graph must actually train` is a real subject from this repo and a good model: a reader who never opens the diff still knows what was broken.

### PR bodies

Bodies here are **prose, not a checklist**. Look at any recently merged PR for the shape. What reviewers expect:

- **A `> 中文摘要：` blockquote at the top.** One paragraph of Traditional Chinese summarising what changed and why. It goes first, before the English prose, because it is what the maintainer reads to decide whether the framing is right.
- **`Closes #NNN`** for every issue the PR actually resolves. Do not use a closing keyword on an issue the PR only partly addresses — describe what you delivered and what remains, and leave it open.
- **Why the change matters**, with the concrete failure it fixes. "From the issue: a server process died 13 minutes in and all 101 completed epochs were unrecoverable" is worth more than "improves reliability".
- **What changed**, as prose or a short list, including the decisions you made and rejected alternatives. If you did something non-obvious, explain why the obvious thing was wrong.
- **What you verified**, with the commands you actually ran and their outcome. Claims of "tests pass" without the command are not evidence.
- **What you deliberately did not do**, if a reviewer might expect it.

Detail is welcome. Under-explaining costs a review round trip; over-explaining costs nobody anything.

---

## House style

### No pictographic emoji

Do not put emoji in code, in log output, in the UI, in commit messages, or in PR bodies. The reason is concrete, not aesthetic: CodefyUI runs on Windows consoles where the active code page is often cp950 or cp1252, and a single emoji in a print statement crashes the process with a `UnicodeEncodeError`. It has happened repeatedly.

Use ASCII markers (`[OK]`, `[WARN]`, `->`) instead. A small set of functional glyphs is fine and already used throughout: check and cross marks, gear and bullet, arrows, and box-drawing or bar characters for terminal tables and charts.

This is a convention, not a CI gate — `byte-scan.yml` checks for raw C0 control bytes, not for emoji. Nobody will catch it for you.

### Documentation and translations

`docs/` is a Docusaurus site with a full Traditional Chinese translation under `docs/i18n/zh-TW/docusaurus-plugin-content-docs/current/`. **If you change an English page, change its zh-TW counterpart in the same PR.** A missing translation silently falls back to English, so drift is invisible until a reader hits a half-translated section.

Build the site before pushing docs changes — `onBrokenLinks` is set to `throw`, so a bad relative link fails the build rather than shipping:

```bash
cd docs && pnpm install && pnpm build
```

### User-facing strings are bilingual

The app ships English and Traditional Chinese. New user-facing text — node descriptions, parameter labels, error messages, CLI output — needs both.

### Where to start

Read the [Architecture](https://docs.codefyui.com/advanced/architecture) page first. The single most important thing to know is that CodefyUI is **backend-authoritative**: `GET /api/nodes` returns every node definition and one React component renders all of them, so adding a node is a backend-only change.

Browse the [issue tracker](https://github.com/CodefyUI/CodefyUI/issues) for something to pick up. If an issue is not clear, ask in a comment before writing code — a question costs a day, a wrong implementation costs a week.

---

## License

By contributing you agree that your contributions are licensed as described in [Signing your work](#signing-your-work-dco) above: AGPL-3.0-only, and available to the copyright holder for the commercial path. See [LICENSE](LICENSE), [NOTICE](NOTICE), and [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md).
