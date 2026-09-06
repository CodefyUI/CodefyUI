---
sidebar_position: 8.6
title: Plugin Center
description: Install teaching node packs and GitHub plugins from inside the editor, and read what each one asks for before it goes in.
---

# Plugin Center

The Plugin Center installs plugins without leaving the editor. It and `cdui plugin install` are two front ends over one installer, so what an install does, what counts as a failure, and what a failure is called are decided once for both.

Open it from the sidebar's **Custom & Plugins** tab -- the **Plugins** section header carries a **Plugin Center...** button -- or from toolbar > **Settings** > **Plugins** > **Open**. The Package Center next to it is a different window for a different thing: model files and Python packages rather than nodes. See [Optional Packs](./optional-packs.md).

## The list

The window is three parts: a box for a GitHub source at the top, the list of plugins under it, and a column on the right reporting whatever install is running.

**All**, **Installed** and **Available** filter the list. Available is what is not here -- a plugin you have never installed, and one you uninstalled. Everything else is under Installed, including a plugin you have switched off and one whose install is still running.

A card carries the plugin's name, a status pill (**Installed**, **Disabled**, **Not installed**, **Removed**, **Installing** or **Files missing**) and its version, then the description, then two lines of small facts. The first says where it came from: **Built-in**, **Official** or **Linked folder**, the `owner/repo` it was fetched from as a link, and the commit it is pinned to. The second says what it brings -- the lessons it covers, how many nodes it registers, and the Python packages installing it would add. The buttons come last, and every one of them is off while anything is happening to that row.

## Installing a built-in pack

Five packs ship inside this release: `edu`, `foundations`, `deep`, `rl` and `stats`. Find the row and press **Install**. A built-in pack is not a third party, so there is nothing to agree to and nothing else to press.

When the job ends, the panel re-reads the catalog, the node definitions and the plugin UIs, so the new nodes are in the palette without reloading the page. They are qualified by the pack they came from -- `edu:FilterRows`, `foundations:Edu-KNN` -- so searching the palette for the pack id finds them.

## Installing from GitHub

The box at the top takes `owner/repo`, `owner/repo@ref` (a tag, a branch or a commit), or a GitHub URL. **Review** reads the manifest at one resolved commit and installs nothing. What comes back is a card at the top of the list, and the install acts on that card rather than on whatever the branch holds a minute later.

The card names the plugin, its author, its version and commit, and the Python packages it would install. Under that is what it is asking for:

- **This plugin asks for:** one line per declared capability -- `network`, `filesystem` or `process-env` -- each saying what granting it costs. The note under the list is the part to read: *Granting is a declaration, not a sandbox: the plugin may use these modules and will not be asked again.* The tick box is **Grant these capabilities**.
- **I trust this author. Allows: ...** is a second box, for the modules this plugin asks to have the security scan turned off for. That is a decision about the author, not about a feature.
- A warning -- *Ships JavaScript that runs in this editor with full access.* -- appears when the plugin has a UI. Browser code is covered by neither box above.

**Install** stays off until every box the manifest raised is ticked, and **Cancel** throws the review away. A plugin that is already installed comes back as **Reinstall**, which replaces the installed copy with this one.

## Enable, disable, update, uninstall

**Disable** keeps the plugin on disk and takes its nodes out of the registry; **Enable** puts them back. Neither downloads anything.

**Update** is offered only on a plugin fetched from a repository, and answers one of three ways: the plugin is up to date, the new version asks for something the installed one did not (so a review card opens, with the added capabilities on it), or there is nothing new to agree to and the job starts. A built-in pack has no Update button: it ships with the release and updates with `cdui update`. Neither has a directory you linked with `cdui plugin link`, which is already whatever is on your disk.

**Uninstall** asks first, and the question names the plugin and both halves of what removing it does: graphs that use its nodes will stop running, and its Python packages stay installed. A downloaded plugin's directory is deleted; a built-in keeps its files and is remembered as removed, so it stays out until you install it by name again. Nothing here removes a plugin's Python packages: pulling a package out from under the interpreter that imported it is not something a running server may do to itself. When the uninstall is done, the right-hand column names the packages it left behind and the `uv pip uninstall` line that removes them once the server is stopped.

## Watching an install

The right-hand column names the plugin, the step it is on (*Resolving the source*, *Downloading*, *Unpacking*, *Checking the code*, *Installing Python packages*, *Copying files*, *Recording the install*, *Loading the nodes*) and how far the whole job has got, and under that the installer's own log. **Cancel install** stops it. Closing the window does not: the job belongs to the server, so reopening the panel -- or another tab, or a reloaded page -- picks the same one back up. One install runs at a time across this panel and the Package Center both.

A job that failed offers the same install in a terminal, under **Or install from a terminal:**, as the `cdui plugin install` line that reproduces exactly that row. See [Plugin commands](../getting-started/cli-commands.md#plugin-commands).

## Installing over the network

Inspect, install, update, cancel and uninstall are refused unless the server is bound to loopback, on top of the session token every write already needs: each of them decides what code this machine will import. Serve the editor on a LAN address and those buttons are off in every browser, including the one on the server itself, with the footer saying *Installing is only allowed from the computer that runs the server.* A classroom or lab server that deliberately serves a LAN opts back in with `CODEFYUI_ALLOW_REMOTE_PLUGIN_INSTALL=1`. Enable and disable are not behind that gate -- they act on code this machine already has and you already agreed to -- so a remote browser with the token can still switch a plugin on and off.

## Where the files land

A downloaded plugin goes to `<USER_DATA>/plugins/<id>/`, and `<USER_DATA>/plugins/installed.json` records every install, including the capabilities you granted, so the next `cdui start` finds them again. `<USER_DATA>` is `%LOCALAPPDATA%\codefyui` on Windows, `~/Library/Application Support/codefyui` on macOS and `~/.local/share/codefyui` on Linux, or `<dir>` when `CODEFYUI_USER_DATA_DIR` is set. Built-in packs are not copied anywhere: they live in `plugins/<id>/` inside the release and are activated in place, so what the lockfile holds for one is a record rather than a copy.

## Troubleshooting

- **"Enter a catalog name, `owner/repo[@ref]` or a GitHub URL."** The box refused the string without asking the server, because it is not a source: what it takes is one of the names this build ships, an `owner/repo` with an optional `@ref`, or a GitHub URL. A bare word that is not a pack this build knows is refused by the server instead, and the names that would have worked are printed under it.
- **"The install stopped before changing anything."** A plugin's Python packages are installed add-only, under a constraints file pinning every package the running server has already loaded, so nothing a plugin asks for can replace what your session is holding open. When that cannot be done live, the job stops before staging or recording anything -- the plugin is not installed, and nothing on disk changed. Stop the server, run the `uv pip install` line the panel prints, and install again. Asking the same running server a second time ends the same way.
- **An uninstall that says the directory is still there.** Something has a file open, which on Windows is the ordinary cause. Nothing was removed: the lockfile entry stays and the plugin stays installed. Close whatever is using the directory, or stop the server, and uninstall again.
- **Files missing.** The lockfile has the plugin and its directory is gone. **Install** fetches it again; **Uninstall** clears the record.
- **"GitHub's request limit was reached."** Unauthenticated GitHub requests are capped per address, which a classroom behind one NAT can exhaust in a morning. Set `CODEFYUI_GITHUB_TOKEN` on the server, or try again later.

For the routes, the security tiers, and every refusal by name, see [Plugins](/advanced/plugins#plugin-center).
