---
sidebar_position: 8.6
title: Plugin Center
description: Install teaching node packs and GitHub plugins from inside the editor, and read what each one asks for before it goes in.
---

# Plugin Center

The Plugin Center installs plugins without leaving the editor. It and `cdui plugin install` are two front ends over one installer, so what an install does, what counts as a failure, and what a failure is called are decided once for both.

Open it from the sidebar's **Custom & Plugins** tab -- the **Plugins** section header carries a **Plugin Center...** button -- or from toolbar > **Settings** > **Optional Packs & Plugins** > **Plugin Center** > **Open**. The Package Center next to it is a different window for a different thing: model files and Python packages rather than nodes. See [Optional Packs](./optional-packs.md).

## The list

The window is three parts: the list of plugins, a folded **Install from a source** section above it, and a column on the right reporting whatever install is running.

**All**, **Installed** and **Not installed** filter the list. Not installed is what is not here -- a plugin you have never installed, and one you uninstalled. Everything else is under Installed, including a plugin you have switched off and one whose install is still running.

A card carries the plugin's name, a status pill (**Installed**, **Disabled**, **Not installed**, **Removed**, **Installing** or **Files missing**) and its version, then the description, then two lines of small facts. The first says where it came from: **Built-in**, **Official**, **Linked folder** or **Local copy** (with the path it was copied from), the `owner/repo` it was fetched from as a link, and the commit it is pinned to. The second says what it brings -- the lessons it covers, how many nodes it registers, and the Python packages installing it would add. The buttons come last, and every one of them is off while anything is happening to that row.

## Installing a built-in pack

Six packs ship inside this release: `edu`, `foundations`, `deep`, `rl`, `stats` and `eval`. Find the row and press **Install**. A built-in pack is not a third party, so there is nothing to agree to and nothing else to press.

When the job ends, the panel re-reads the catalog, the node definitions and the plugin UIs, so the new nodes are in the palette without reloading the page. They are qualified by the pack they came from -- `edu:FilterRows`, `foundations:Edu-KNN` -- so searching the palette for the pack id finds them.

## Installing from GitHub

Open **Install from a source** above the list. Its field takes a catalog name, `owner/repo`, `owner/repo@ref` (a tag, a branch or a commit), a GitHub URL, or a path on the server (see [the next section](#installing-from-a-folder-or-an-archive)); the line under the field lists the shapes. **Review** reads the manifest at one resolved commit and installs nothing. What comes back is a card at the top of the list, and the install acts on that card rather than on whatever the branch holds a minute later.

The card names the plugin, its author, its version and commit, and the Python packages it would install. Under that is what it is asking for:

- **This plugin asks for:** one line per declared capability -- `network`, `filesystem` or `process-env` -- each saying what granting it costs. The note under the list is the part to read: *Granting is a declaration. Nothing is enforced at runtime, and the plugin will not ask again.* The tick box is **Grant these capabilities**.
- **Trust this author to import ...**, followed by the module names, is a second box, for the modules this plugin asks to have the security scan turned off for. That is a decision about the author, not about a feature.
- A warning -- *Ships JavaScript that runs in this editor with full access.* -- appears when the plugin has a UI. Browser code is covered by neither box above.

**Install** stays off until every box the manifest raised is ticked, and **Cancel** throws the review away. A plugin that is already installed comes back as **Reinstall**, which replaces the installed copy with this one.

## Installing from a folder or an archive

For a plugin handed out on a USB stick or a shared drive, with no GitHub access, type its path into the same field: a plugin folder, or a `.zip`, `.tar.gz` or `.tgz` of one. The path is read on the machine the **server** runs on, so a folder on another laptop is out of reach. Four spellings are accepted:

- an absolute path: `/Volumes/USB/my-plugin`, `D:\plugins\my-plugin.zip`;
- a relative path starting with `./` or `../`, resolved against the server's working directory (`plugins/x` alone means a GitHub `owner/repo`);
- a path under your home directory: `~/Downloads/my-plugin.tgz`;
- a `file://` URL: `file:///srv/share/my-plugin`.

An archive may hold the manifest at its root or inside its one top-level folder; the `__MACOSX` folder macOS adds when it compresses is ignored.

**Review** copies the folder (or unpacks the archive) into a temporary directory and reads the manifest there. The same limits as a GitHub download apply: 100 MB for an archive, 500 MB and 20,000 files unpacked, no symbolic links, no device files, and no member that would land outside the plugin's folder. `.git`, `__pycache__`, `.venv`, `venv` and `node_modules` folders are left out of a folder copy. The card says **Local folder:** or **Local archive:** followed by the path, and asks the same questions a repository does. Installing copies the source again, through the security scan, the dependency install and staging, into `<USER_DATA>/plugins/<id>/`. Your folder is not changed and is not loaded from afterwards.

The install is pinned to what the review read: the sha256 of the archive file, or of every file in the copied folder. A folder edited between **Review** and **Install** is refused with *changed after it was reviewed*; review it again. There is no **Update** for a local copy, since there is nothing to fetch from. Install the newer copy over it with **Reinstall** (or `cdui plugin install <path> --force`).

## Enable, disable, update, uninstall

**Disable** keeps the plugin on disk and takes its nodes out of the registry; **Enable** puts them back. Neither downloads anything.

**Update** is offered only on a plugin fetched from a repository, and answers one of three ways: the plugin is up to date, the new version asks for something the installed one did not (so a review card opens, with the added capabilities on it), or there is nothing new to agree to and the job starts. A built-in pack has no Update button: it ships with the release and updates with `cdui update`. Neither has a directory you linked with `cdui plugin link`, which is already whatever is on your disk, or a plugin copied from a local folder or archive.

**Uninstall** asks first, and the question names the plugin and both halves of what removing it does: graphs that use its nodes will stop running, and its Python packages stay installed. A downloaded or copied plugin's directory is deleted (the folder or archive it was copied from stays where it is); a built-in keeps its files and is remembered as removed, so it stays out until you install it by name again. Nothing here removes a plugin's Python packages: pulling a package out from under the interpreter that imported it is not something a running server may do to itself. When the uninstall is done, the right-hand column names the packages it left behind that nothing else still needs, and the `uv pip uninstall` line that removes them once the server is stopped.

## Watching an install

The right-hand column names the plugin, the step it is on (*Resolving the source*, *Downloading*, *Unpacking*, *Checking the code*, *Installing Python packages*, *Copying files*, *Recording the install*, *Loading the nodes*) and how far the whole job has got, and under that the installer's own log. **Cancel install** stops it. Closing the window does not: the job belongs to the server, so reopening the panel -- or another tab, or a reloaded page -- picks the same one back up. One install runs at a time across this panel and the Package Center both.

A job that failed offers the same install in a terminal, under **Or install from a terminal:**, as the `cdui plugin install` line that reproduces exactly that row. See [Plugin commands](../getting-started/cli-commands.md#plugin-commands).

## Installing over the network

Inspect, install, update, cancel and uninstall are refused unless the server is bound to loopback, on top of the session token every write already needs: each of them decides what code this machine will import. Serve the editor on a LAN address and **Review**, **Install**, **Update** and **Uninstall** are off in every browser, including the one on the server itself, with the footer saying *Installing works only from the computer that runs the server.* **Cancel install** stays pressable, but the server refuses it. A classroom or lab server that deliberately serves a LAN opts back in with `CODEFYUI_ALLOW_REMOTE_PLUGIN_INSTALL=1`. Enable and disable are not behind that gate -- they act on code this machine already has and you already agreed to -- so a remote browser with the token can still switch a plugin on and off.

## Where the files land

A downloaded plugin, and one copied from a local folder or archive, goes to `<USER_DATA>/plugins/<id>/`, and `<USER_DATA>/plugins/installed.json` records every install, including the capabilities you granted, so the next `cdui start` finds them again. For a server started with `cdui start` or `cdui dev`, and for every `cdui plugin` command, `<USER_DATA>` is `<install dir>/.codefyui_dev/` (default install: `~/CodefyUI/.codefyui_dev/`), or `<dir>` when `CODEFYUI_USER_DATA_DIR` is already set. The platform directory -- `%LOCALAPPDATA%\codefyui` on Windows, `~/Library/Application Support/codefyui` on macOS, `~/.local/share/codefyui` on Linux -- applies only to a `uvicorn app.main:app` started by hand without that variable. Built-in packs are not copied anywhere: they live in `plugins/<id>/` inside the release and are activated in place, so what the lockfile holds for one is a record rather than a copy.

## Troubleshooting

- **"Enter owner/repo[@ref], a GitHub URL, or a folder, .zip or .tar.gz path on the server ..."** The field refused the string without asking the server, because it is not a source: what it takes is one of the names this build ships, an `owner/repo` with an optional `@ref`, a GitHub URL, or a path in one of the spellings under [Installing from a folder or an archive](#installing-from-a-folder-or-an-archive). A bare word that is not a pack this build knows is refused by the server instead, and the names that would have worked are printed under it.
- **"Nothing was installed: ...'s Python packages would replace one the server is using."** A plugin's Python packages are installed add-only, under a constraints file that pins every package installed in the server's environment to its current version (editable installs, such as CodefyUI itself, are left out), so nothing a plugin asks for can replace a package the server already has. When that cannot be done live, the job stops before staging or recording anything -- the plugin is not installed, and nothing on disk changed. Stop the server, run the `uv pip install` line the panel prints, and install again. Asking the same running server a second time ends the same way.
- **An uninstall that says the directory is still there.** Something has a file open, which on Windows is the ordinary cause. Nothing was removed: the lockfile entry stays and the plugin stays installed. Close whatever is using the directory, or stop the server, and uninstall again.
- **Files missing.** The lockfile has the plugin and its directory is gone. **Install** opens a review card saying the plugin is already installed; press **Reinstall** on it to fetch the files again. **Uninstall** clears the record.
- **"Nothing at that path on the server."** The path is read on the server's machine. Check that the drive is mounted there and that a relative path starts from the server's working directory; an absolute path avoids the question.
- **"This plugin contains a link, a device file or a path outside its folder."** and the other refusals about size, file count or layout name the rule the folder or archive broke. Nothing was copied.
- **"... changed after it was reviewed."** The files differ from the ones the review read. Press **Review** again and install what that shows.
- **"GitHub's request limit was reached."** Unauthenticated GitHub requests are capped per address, which a classroom behind one NAT can exhaust in a morning. Set `CODEFYUI_GITHUB_TOKEN` on the server, or try again later.

For the routes, the security tiers, and every refusal by name, see [Plugins](/advanced/plugins#plugin-center).
