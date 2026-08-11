<!--
Delete this comment block before submitting. See CONTRIBUTING.md#pr-bodies
for the full convention this template is a shortcut for.
-->

> 中文摘要：（一段中文摘要，說明改了什麼、為什麼要改）

## What / Why

<!-- What changed, and the concrete problem it fixes. "From the issue: ..." -->

## Closes / relates to

<!--
Use a closing keyword ONLY when this PR fully resolves the issue:
  Closes #NNN
For partial work, say so plainly and do NOT use a closing keyword nearby —
GitHub closes the issue on merge regardless of surrounding negation, e.g.
"does not close #NNN" still closes #NNN. Write instead:
  Part of #NNN (stays open — <what remains>)
-->

## Test evidence

<!--
The commands you actually ran and their outcome, not "tests pass". e.g.:
  python scripts/check_control_bytes.py   -> clean
  uvx ruff@0.14.4 check .                 -> clean
  cd backend && uv run pytest -q          -> N passed
  cd frontend && pnpm test                -> N passed
If this genuinely can't be tested, say why instead of leaving it blank.
-->

## Checklist

- [ ] Commits are signed off (`git commit -s`) per CONTRIBUTING.md's DCO section.
- [ ] If I changed a docs page or a node-param description, I updated its
      zh-TW twin in the same PR (docs/i18n/zh-TW/... or the matching
      nodeLocales entry).
- [ ] No pictographic emoji in code, logs, UI strings, or this PR body
      (Windows consoles crash on them — ASCII markers like `[OK]` instead).
- [ ] I explained what I deliberately did not do, if a reviewer might expect it.
