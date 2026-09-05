---
sidebar_position: 7.75
title: Source Control
description: Commit, branch, stash, push and review a CodefyUI project from the editor's Source Control tab, with a diff view that says what changed in the graph.
---

# Source Control

The **Source Control** tab -- the branch icon in the sidebar rail -- is git for the project directory the server was started on. It shows what has changed since the last commit, stages and commits it, creates and switches branches, adds a remote and pushes to it, settles a merge conflict, lists the history, and opens any file's changes as a diff. A graph is a pair of plain JSON files on disk, so all of this is ordinary git against ordinary files: everything the tab does you can also do at a command line in the same directory, and everything you do at that command line shows up in the file lists, on the branch line and in every list you have open within fifteen seconds.

Two things have to be true before the tab can work. The server must have been started on a **project directory** -- `cdui project init my-project`, then `cdui start --project my-project`; a server started without one says "Source control needs a project directory." and prints those two commands, because the project is an argument to the server and no button in the browser can supply it. See [Project directories](./project-directories) for what a project is. And **git 2.23 or newer** must be installed on the computer that runs the server, not on the computer with the browser: without it the tab says "git is not installed on the server computer.", and with an older one `git {version} is too old; 2.23 or newer is required.` A project directory that is not a repository yet gets one button, **Initialize Repository**.

The server runs git as a subprocess, as its own operating-system user, and never interactively. Prompting is turned off outright -- `GIT_TERMINAL_PROMPT=0`, an emptied `GIT_ASKPASS`, `ssh -oBatchMode=yes` where you have no `GIT_SSH_COMMAND` of your own -- so a git that decides to ask for a password fails in a second instead of hanging until you close the tab, and the panel reports it as "The server computer has no saved credentials for this remote." CodefyUI stores no tokens, passwords or keys of its own. Whatever credential helper and SSH keys that user already has are what every fetch and push uses, which is why authentication is set up once, in a terminal on that machine, and never in the app.

## Set up git and GitHub

Do this once per computer that runs a server.

**Install git.** On Windows, `winget install Git.Git` -- Git for Windows bundles Git Credential Manager, which is the piece that remembers an HTTPS login. On macOS, `xcode-select --install` for Apple's git, or `brew install git` for a current one. On Ubuntu and Debian, `sudo apt install git`. The server rechecks for git about every half minute, so a fresh install is normally picked up on its own -- the screen's "Install it, then restart the server." is more cautious than it needs to be.

**Say who you are.** Commits carry a name and an email, and git refuses to make one without them:

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

The tab can do the same thing: **More actions** > **Commit identity...** opens a small form that shows what git currently answers with and whether it comes "from global git config" or "for this project". Saving from that form writes into this repository's own config, so it is the right place for a project that needs a different address from your usual one, and the wrong place for the machine-wide default -- for that, run the two commands above.

**Give the server a way to authenticate.** The first credential capture has to happen in a terminal on the server computer, because the server itself never prompts. Three ways, and one is enough:

- **HTTPS with the GitHub CLI.** `winget install GitHub.cli` on Windows, or `brew install gh` on macOS; on Ubuntu, `gh` is not in every release's default repositories, so follow GitHub's own apt instructions at cli.github.com. Then `gh auth login`; answer yes when it offers to authenticate git with your GitHub credentials. It writes a credential helper into your global git config, and every push after that is silent.
- **HTTPS with one manual push.** Clone or push once from a terminal in the project directory. The credential manager stores what you type, and from then on the server's pushes find it.
- **SSH instead.** `ssh-keygen -t ed25519 -C "you@example.com"`, add the public key on GitHub under Settings > SSH and GPG keys, then run `ssh -T git@github.com` once from a terminal. That last step is not optional here: the server runs ssh in batch mode, and an unknown host key is a question ssh cannot ask, so a first connection made from the server would simply fail. Use the `git@github.com:owner/repo.git` form of the URL when you add the remote.

**Connect a repository.** Create an empty repository on GitHub -- no README, no `.gitignore`, no licence, so that its history is empty and your first push is not a merge. Then, in the tab, open **Remotes** and press the **+** (**Add Remote...**) on its heading: the name is `origin` unless you have a reason, and the URL is the one GitHub shows (`https://github.com/owner/repo.git`, or the SSH form). The header's second row then offers **Publish Branch**, which pushes the current branch and sets it to track the remote one. From then on that row is a **Sync (pull, then push)** button instead.

## The daily loop

Save the graph (`Ctrl/Cmd` + `S`). Both files it writes -- `graphs/<name>.graph.json` and `layout/<name>.layout.json` -- appear under **Changes** at once rather than at the next poll. Stage what belongs in the commit with **Stage** on a row or **Stage All** on the group heading, write a message, and press **Commit** or `Ctrl/Cmd` + `Enter`. If you would rather not stage first, **Commit All (stages every change, including new files)** sits behind the chevron beside the button, and **Amend Last Commit** beside it replaces the previous commit -- always in the menu and refused rather than withheld: greyed out while the branch has no commit at all, and renamed to "Cannot amend: the last commit is already pushed" once it has been.

The **Commit** button says why it is unavailable rather than merely being grey: "Enter a message" or "Nothing staged", in the tooltip and to a screen reader.

The header's branch row carries the state of the branch. `↑2 ↓1` means two commits to push and one to pull; the same fact is spelled out for a screen reader and in the tooltip as "2 to push, 1 to pull", and a branch that is level with its upstream draws nothing at all. "Not published" means the branch has no upstream yet, "Upstream deleted" means it had one and the remote no longer has it, "No commits yet" is an empty repository and "Detached HEAD" is a checkout that is on no branch.

**Sync (pull, then push)** is the everyday button. **Fetch**, **Pull** and **Push** are in the **More actions** menu for the times you want one half of it; Pull there is fast-forward only, so a divergence is reported rather than merged behind you. A row that cannot be pressed says why rather than merely greying out -- "No remote yet.", "Not published", "No commits yet", "Detached HEAD", or the operation already running. Network operations run in a lane of their own, so a slow fetch does not block the commit you were about to make, and neither can be started twice.

**Refresh** in the title row re-reads everything on screen: the status, every section that is open, and the history whenever it still holds a page -- collapsing History keeps the pages you loaded, so they are kept true as well. A closed section with nothing behind it is not read.

## Branches, remotes and stashes

Under the file groups are four collapsible lists: these three, and History, which has its own section below. Each is remembered between sessions, each is read when it opens, and a closed one costs nothing.

**Branches** lists the local branches with how far each is from what it tracks; the name is the button that switches to it, and the one you are on says "Current" instead. **New Branch...** on the heading creates one and switches to it. Each row's menu has **Rename** and **Delete**; deleting is absent on the current branch, because git refuses it. Deleting a branch whose commits are on no other branch asks a second time, in git's own words: `{name} has unmerged commits. Delete anyway?` Under the local branches, **Remote branches** lists what the remotes have, and one press makes a local branch that tracks one.

**Remotes** lists each remote with the URL git fetches from, plus **Change URL** and **Remove**; **Add Remote...** is the **+** on the heading. Only `https://`, `ssh://`, `file://` and the `git@host:owner/repo.git` form are accepted, and the server allows git exactly those three protocols -- the transports whose "URL" is a command line to run cannot be configured at all.

**Stashes** is the stash stack, newest first, each row showing the message you gave it, the branch it was made on and how long ago. **Stash Changes...** in the **More actions** menu asks for a message and takes untracked files with it, which is what makes a stash enough to free the tree for a checkout; leave the box empty and git writes its own subject. Rows offer **Pop**, **Apply** and **Drop**, and only Drop asks first -- it is the one that throws work away.

## Conflicts

A merge that stops with conflicts puts a **Merge Changes** group above the other file groups, with the banner "Merge in progress: resolve each file, then commit." Every row offers git's three answers: **Keep mine** overwrites the file with your side, **Take incoming** with theirs, and **Mark resolved** stages the file exactly as it is on disk -- which is the one to use after you have edited the conflict markers out yourself, in the editor or in any other tool. There is no discard on these rows: git refuses to discard a conflicted path, so a button for it could only ever show an error.

**Abort Merge** on the group heading puts the tree back the way it was, after asking "Abort the merge and discard what you have resolved?". The other way out is a commit, and the **Commit** button stays live even when settling every file left nothing staged -- resolving everything as "mine" changes no file, and that is exactly the merge git still wants a commit for.

When a push is refused with "Local and remote branches have diverged.", the error line carries a **Merge remote changes** button beside Dismiss; it pulls with a merge, which is the step the fast-forward pull declined to take. A refusal that carries git's own output has a **Details** disclosure with the raw tail in it, redacted of anything that looked like a credential.

## History and the diff view

**History** is the fourth section, below Stashes, closed until you open it. Opening it reads the first thirty commits: one row each, with the short commit id, the subject, how long ago it was made and the author. Expanding a row lists the files that commit changed, and **Copy commit id** in the row menu puts the full hash on the clipboard. **Load more** appears while the server says there is another page.

History is deliberately not on the fifteen-second poll -- a poll would throw away the pages you loaded. It is re-read by **Refresh**, by a commit or an amend, and by anything that moves the branch: a pull, a push, a sync, a checkout, a branch created or deleted.

Any file row anywhere in the tab opens that file's changes: rows under **Changes** show the unstaged side, rows under **Staged Changes** show what the next commit will contain, and a file row under a commit in History shows what that commit did to it. The view opens over the editor with the path in its title and the scope under it -- "Unstaged changes", "Staged changes" or `Commit {sha}` -- and closes with **Close** or Escape.

**Unified** and **Side by side** are the two ways to read the patch. Side by side is derived from the same patch rather than from the whole file, so it shows the changed hunks with their context, paired left and right; a conflicted file offers unified only, because a file full of conflict markers has no two sides to pair -- git answers an unmerged path with all of its sides at once, and that is what the window draws, markers and all. Three things about the patch as a whole are said in words instead of lines: a binary file gets "Binary file; no text diff."; a patch the server cut at 1 MiB carries `Diff truncated at {kb} KB.` above what did fit; and a file with no textual difference says "No changes". Two more appear inside the patch: git's own "No newline at end of file" beside the line it is about -- otherwise that change is two lines whose text is identical -- and, past two thousand lines, a note under the last line drawn, because laying out more than that freezes the tab for a second and nobody reads it in a side panel.

### What changed in the graph

A JSON diff of a saved graph is a wall of braces in which "I changed `k` from 5 to 7" is invisible. So for `graphs/<name>.graph.json` and `layout/<name>.layout.json` a short summary sits above the patch, read from the two sides of the diff:

- nodes added and removed, counted on the canvas: dropping one preset block is one node, not the six inside the definition it brought with it;
- a node whose type changed;
- a parameter whose value changed, under the node's label or its id where it has none, with long values clipped;
- edges added and removed, compared by their endpoints and handles rather than by their ids, because copy and paste regenerates ids;
- node positions moved, which can only appear on a `.layout.json` diff -- a `.graph.json` has no coordinates in it.

At most eight lines are shown, then a line counting the rest ("and 3 more"). "No logic change" means the two sides say the same thing and the difference is text only: key order, whitespace, an array written in another order, a regenerated id. "Could not parse as a graph" means one side is not readable JSON of that kind.

Some real changes have no line yet, and the summary is then empty rather than reassuring: segment groups, note geometry, a subgraph definition, a preset definition arriving or leaving (the instance on the canvas is still counted), a graph's name or description, and a preset instance's per-instance overrides. None of them is ever reported as "No logic change" -- the patch below is what shows them. A non-project graph saved as a single `<name>.json` gets no summary at all.

## When the files change under an open graph

A pull, a checkout, a stash pop, a discard, an abort or a resolution can put different bytes under a graph you have open. The tab does not reload it behind you. It raises one sticky toast saying how many open graphs the write landed under, with a **Reload** button; pressing it asks once for all of them ("Unsaved edits in those tabs are lost.") and then re-reads each from disk. A tab whose file does not exist on the branch you switched to keeps what it is showing -- which is now the only copy of it anywhere -- and says so.

## Security

Every write carries the editor's session token, the same as every other mutating call in the app: init, stage, unstage, discard, commit, the identity, branches, remotes, stashes, merge, resolve, and everything that talks to a remote. The reads -- status, history, a commit's files, a diff, a file at a ref, the config and the three lists -- are open GETs like the rest of the API.

There is no loopback gate on these routes, unlike a pack install. Access control for an instance you deliberately serve to a LAN is the deploying organisation's job: put it behind something that authenticates, as [Deployment behind a reverse proxy](./deployment) describes. What runs here is the server user's own git against the directory they opened, with the credentials they already have.

Two rules are enforced below the API and cannot be turned off. Anything named `.env` or `.env.<anything>` is refused at every ref, before git is started -- the diff and the file read both, at the working tree, in the index and in any commit -- so a dotenv committed by accident cannot be read back through the editor. `.env.example` is exempt, because it exists to be read. And git's stderr is redacted once on its way out, since `https://user:token@github.com/...` is a URL people really do paste into a remote.

Keep secrets out of the graphs themselves as well: an API key typed into an LLM node's parameter field is saved verbatim into the graph JSON. [Version control your graphs](./version-control-graphs#secrets-keep-keys-out-of-your-graphs) has the environment variables to use instead.

## Limits

- **No merge editor.** A conflict is settled by taking one whole side, or by editing the file yourself and pressing **Mark resolved**.
- **No force push, no rebase, no cherry-pick, no tags, no clone.** Those are a terminal's job; the tab is the everyday half of git, not all of it.
- **One project per server.** The project directory is chosen when the server starts, and there is no way to switch to another from the editor.
- **No hunk staging.** Staging is per file.
- **Commit signing is not supported.** A repository configured with `commit.gpgsign = true` fails with "Commit signing is not supported from the app.", because gpg cannot ask for a passphrase in a process with no terminal. Turn signing off for this repository, or make that commit from a terminal.
- **The diff is the patch.** Side by side pairs the patch's hunks; a whole-file comparison of both revisions is a follow-up.
- **History pages by offset.** Thirty commits at a time, so a commit made between two pages can shift the window; **Refresh** re-reads the pages you have loaded, from the first.

## Troubleshooting

| What the panel says | What it means | What to do |
|---|---|---|
| "This branch is not published yet." | The branch has no upstream, so there is nothing to pull from or push to. | Press **Publish Branch**. The refusal itself carries that button when the state allows it. |
| "Local and remote branches have diverged." | Both sides have commits the other does not; a fast-forward is impossible. | Press **Merge remote changes** beside the message, settle any conflicts, then push. |
| "The server computer has no saved credentials for this remote." | git asked for a password and the server has no way to answer. | On the server computer, in a terminal: `gh auth login` for HTTPS, or set up an SSH key and run `ssh -T git@github.com` once. Then retry. |
| "Your git push configuration refuses this push (push.default or the upstream branch name)." | git's own `push.default` rules, not the remote's. | Read git's sentence under **Details**; usually `git config --global push.default current` or a matching upstream branch name fixes it. |
| `git {version} is too old; 2.23 or newer is required.` | The server computer's git predates the status format the tab reads. | Upgrade git there. |
| "Not a git repository." | The project directory has no `.git` in it. | Press **Initialize Repository** on the tab's empty screen, or start the server on the directory you meant. |
| "This file is ignored by git." | You opened the changes for a path git ignores, or for a `.env`-shaped file the server never serves. | Nothing to fix; the file is deliberately not readable through the editor. |
| "Another git action is still running." | Two writes at once, or a write during a long one. | Wait for the progress bar under the header to finish. |
| `git did not finish within {seconds}s.` | The command passed its deadline -- 10 s for a status, 20 s for a read, 30 s for a local write, 130 s for anything over the network (git's own 120 s plus the request's grace). Usually an index lock held by another process. | Close whatever else is running git in that directory, then press **Refresh**. |

For the wider picture -- what belongs in the repository, what must never be committed, and how to validate every graph in CI -- see [Version control your graphs](./version-control-graphs) and [Project directories](./project-directories).
