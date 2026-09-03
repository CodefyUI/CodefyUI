"""Reading ``git status --porcelain=v2 --branch --show-stash -z``.

One pure function over a byte string. No subprocess, no runner import,
nothing to mock: the awkward part of the Source Control tab is this format,
and keeping it a function of bytes means every awkward case -- a filename
with a newline in it, a rename, a conflict, an upstream that no longer
exists -- is a two-line test instead of a temporary repository.

Why this format, and why these rules:

* **Porcelain v2, not v1.** v1 packs everything into two letters and then
  quotes paths it does not like; v2 gives every entry named fields, tells a
  rename from a copy WITH its similarity score, and marks submodules. It is
  also the format git documents as stable for scripts, which matters for a
  parser that ships to users who upgrade git without asking us.
* **NUL, not newline.** A filename may legally contain a newline. ``-z`` is
  what makes that safe: git stops quoting and escaping paths altogether --
  a CJK path arrives as its own bytes -- and terminates every record with a
  NUL instead. Which means the NUL is also the only safe thing to split on,
  and reading this input as "lines" would be a bug waiting for the first odd
  filename. (The runner's ``core.quotepath=false`` says the same thing for
  the commands that are not ``-z``.)
* **The first N fields, then the rest is the path.** Each record has a fixed
  number of space-separated fields before its path, so the split is
  ``split(" ", N)`` -- single-space, with a maximum -- and never a generic
  whitespace split, which would eat the leading space of a file actually
  named `` leading.txt``.
* **``MM`` is two entries.** The staged group is built from X and the
  unstaged group from Y, so a file modified in both places appears in both,
  which is what VS Code shows and the only way the tab can offer "unstage"
  and "discard" on it at once. The ``xy`` on both entries stays ``MM``.
* **An unknown record is skipped, never raised.** This runs on every status
  poll while the tab is open. If a future git grows a record type, the cost
  of skipping it is one file missing from a list; the cost of raising is a
  panel that shows nothing at all and an error the user cannot act on. Same
  for a truncated record: it is not evidence that the rest is wrong.

``merge_in_progress`` / ``rebase_in_progress`` are parameters rather than
something read here, because porcelain v2 does not mention them: the answer
is whether ``MERGE_HEAD``, ``rebase-merge`` or ``rebase-apply`` exist under
the git directory, which needs a ``rev-parse --git-path`` -- the service's
job, right after it calls this.

``upstream_gone`` has the same shape of limitation and is worth stating
plainly: when a tracked remote branch is deleted and pruned, git keeps
printing ``# branch.upstream`` and simply STOPS printing ``# branch.ab``
(verified against git 2.53). "Configured but uncounted" is therefore the
whole signal this format carries, and it is what the flag is computed from.
G3, which runs ``for-each-ref``, can see the real answer -- ``%(upstream:
track)`` prints ``[gone]`` -- and may refine the flag there.

Every record shape below was read off real ``git status`` output rather than
the manual page, including the rare ones (``2 C.`` for a copy, ``u DU`` for
delete-vs-modify, ``S...`` for a submodule).
"""

from __future__ import annotations

from .models import FileKind, GitFile, GitStatus

#: git's status letter -> the kind the tab draws an icon from.
_KIND_BY_LETTER: dict[str, FileKind] = {
    "M": "modified",
    "A": "added",
    "D": "deleted",
    "R": "renamed",
    "C": "copied",
    "T": "typechange",
}

#: The XY porcelain v2 does NOT print for untracked files. Porcelain v1
#: spells that state ``??`` and so does every UI built on git, so untracked
#: entries carry it here: every :class:`~app.core.git.models.GitFile` the
#: frontend sees then has two characters in ``xy``, and nothing downstream
#: needs a special case for the one group that would otherwise have none.
UNTRACKED_XY = "??"

#: Porcelain v2's "nothing happened on this side" (v1 used a space). This
#: parser only ever reads v2 -- the runner always passes ``--porcelain=v2``.
_UNCHANGED = "."

#: Fields BEFORE the path in each record type, including the record letter
#: itself. ``1 <XY> <sub> <mH> <mI> <mW> <hH> <hI> <path>`` is 8 then the
#: path; ``2`` adds the ``<X><score>`` field; ``u`` carries three stage
#: modes and three stage hashes.
_ORDINARY_FIELDS = 8
_RENAMED_FIELDS = 9
_UNMERGED_FIELDS = 10

#: Where the ``<X><score>`` field sits in a ``2`` record: last before the
#: path. Spelled as an offset rather than as ``8`` so it stays right if a
#: future git adds a field.
_SCORE_FIELD = _RENAMED_FIELDS - 1


def kind_from_letter(letter: str) -> FileKind:
    """One of git's status letters as a :data:`~app.core.git.models.FileKind`.

    An unrecognised letter is reported as ``modified``, deliberately. Git
    could grow a letter (it has before), and the three alternatives are all
    worse: raising turns one odd file into a failed status read, dropping it
    hides a change the user has made, and widening ``FileKind`` would send
    the frontend a kind it cannot draw. "Something changed here" is the one
    thing every status letter has in common, and the exact letters travel
    alongside in ``GitFile.xy`` for anyone who wants the truth.
    """
    return _KIND_BY_LETTER.get(letter, "modified")


def _count(token: str, sign: str) -> int | None:
    """``+3`` / ``-1`` from a ``# branch.ab`` header, or ``None``.

    Strict about the sign because the sign is what says which side of the
    pair this is; anything else means the header is not what we think it
    is, and half an answer here becomes a wrong badge in the UI.
    """
    if not token.startswith(sign):
        return None
    try:
        count = int(token[1:])
    except ValueError:
        return None
    return count if count >= 0 else None


def _score(field: str) -> int | None:
    """The number in git's ``R100`` / ``C75`` rename-similarity field."""
    digits = field[1:]
    try:
        score = int(digits)
    except ValueError:
        return None
    return score if score >= 0 else None


def parse_porcelain_v2(
    data: bytes,
    *,
    merge_in_progress: bool = False,
    rebase_in_progress: bool = False,
) -> GitStatus:
    """Parse one ``--porcelain=v2 --branch --show-stash -z`` payload.

    *data* is git's raw stdout. Each NUL-separated token is decoded on its
    own as UTF-8 with ``errors="replace"``: under ``-z`` git prints the
    bytes of a filename exactly as the filesystem holds them, and on Windows
    or in a repository made on another machine those bytes are not always
    valid UTF-8. One unreadable name must cost that name a few replacement
    characters, not cost the user their whole status panel.

    The two in-progress flags are passed straight through; see the module
    docstring for why they are not read here.
    """
    tokens = [chunk.decode("utf-8", errors="replace")
              for chunk in data.split(b"\x00")]

    branch: str | None = None
    detached = False
    head: str | None = None
    unborn = False
    upstream: str | None = None
    ahead: int | None = None
    behind: int | None = None
    stash_count = 0
    staged: list[GitFile] = []
    unstaged: list[GitFile] = []
    untracked: list[GitFile] = []
    conflicted: list[GitFile] = []

    index = 0
    while index < len(tokens):
        token = tokens[index]
        index += 1
        # Every record git writes is a letter, a space, then the rest. The
        # empty token after the final NUL fails this, and so does anything
        # else that is not a record -- both are skipped rather than guessed
        # at.
        if len(token) < 2 or token[1] != " ":
            continue
        record = token[0]

        if record == "#":
            key, _, value = token[2:].partition(" ")
            if key == "branch.oid":
                if value == "(initial)":
                    unborn = True
                else:
                    head = value or None
            elif key == "branch.head":
                if value == "(detached)":
                    detached = True
                else:
                    branch = value or None
            elif key == "branch.upstream":
                upstream = value or None
            elif key == "branch.ab":
                first, _, second = value.partition(" ")
                ahead = _count(first, "+")
                behind = _count(second, "-")
                # Half a pair is not an answer: report neither.
                if ahead is None or behind is None:
                    ahead = behind = None
            elif key == "stash":
                try:
                    stash_count = max(int(value), 0)
                except ValueError:
                    pass
            continue

        if record == "1":
            parts = token.split(" ", _ORDINARY_FIELDS)
            if len(parts) <= _ORDINARY_FIELDS or len(parts[1]) < 2:
                continue
            xy = parts[1]
            path = parts[_ORDINARY_FIELDS]
            if xy[0] != _UNCHANGED:
                staged.append(GitFile(path=path, kind=kind_from_letter(xy[0]),
                                      xy=xy))
            if xy[1] != _UNCHANGED:
                unstaged.append(GitFile(path=path,
                                        kind=kind_from_letter(xy[1]), xy=xy))
            continue

        if record == "2":
            parts = token.split(" ", _RENAMED_FIELDS)
            # The path git renamed FROM is the next token, not a field of
            # this one -- under ``-z`` the TAB of the human format becomes a
            # separator, so a rename is two tokens.
            if index >= len(tokens):
                continue
            orig_path = tokens[index]
            index += 1
            # Consumed before the record is checked: a malformed ``2`` must
            # not leave its second token behind to be read as a record.
            if not orig_path:
                continue
            if len(parts) <= _RENAMED_FIELDS or len(parts[1]) < 2:
                continue
            xy = parts[1]
            score = _score(parts[_SCORE_FIELD])
            path = parts[_RENAMED_FIELDS]
            for letter, group in ((xy[0], staged), (xy[1], unstaged)):
                if letter == _UNCHANGED:
                    continue
                kind = kind_from_letter(letter)
                # Only the side that IS the rename carries where it came
                # from. The worktree half of an ``RM`` is an edit to the new
                # path, and showing "old -> new" against it would say the
                # file moved twice.
                moved = kind in ("renamed", "copied")
                group.append(GitFile(
                    path=path,
                    orig_path=orig_path if moved else None,
                    kind=kind,
                    xy=xy,
                    score=score if moved else None))
            continue

        if record == "u":
            parts = token.split(" ", _UNMERGED_FIELDS)
            if len(parts) <= _UNMERGED_FIELDS or len(parts[1]) < 2:
                continue
            # ``xy`` is the whole answer here: ``UU`` and ``DU`` are both a
            # conflict, and they need different resolve buttons.
            conflicted.append(GitFile(path=parts[_UNMERGED_FIELDS],
                                      kind="conflict", xy=parts[1]))
            continue

        if record == "?":
            path = token[2:]
            if path:
                untracked.append(GitFile(path=path, kind="untracked",
                                         xy=UNTRACKED_XY))
            continue

        # ``!`` is an ignored file, which only appears when a caller asks
        # for ``--ignored`` -- ours never does, and there is no group to
        # put it in. Anything else is a record type git grew after this was
        # written. Both are skipped; see the module docstring.

    return GitStatus(
        branch=branch,
        detached=detached,
        head=head,
        unborn=unborn,
        upstream=upstream,
        ahead=ahead,
        behind=behind,
        # git prints no ``# branch.ab`` for an upstream whose ref is gone.
        upstream_gone=upstream is not None and ahead is None,
        staged=staged,
        unstaged=unstaged,
        untracked=untracked,
        conflicted=conflicted,
        stash_count=stash_count,
        merge_in_progress=merge_in_progress,
        rebase_in_progress=rebase_in_progress,
    )
