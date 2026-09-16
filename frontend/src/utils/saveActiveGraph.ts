import { saveGraph, listGraphs } from '../api/rest';
import { useTabStore } from '../store/tabStore';
import { useProjectStore } from '../store/projectStore';
import { useToastStore } from '../store/toastStore';
import { useI18n } from '../i18n';
import { confirm, prompt } from './dialog';
import { announceWorktreeWrite } from './worktreeWrite';
import { announceGraphsWrite } from './graphsWrite';
import { sanitizeGraphName, findGraphNameCollision } from './index';

/**
 * Save the active tab's graph.
 *
 * - A tab BOUND to a saved graph overwrites THAT FILE, always: the binding
 *   travels on the request as `file` and decides where the bytes land, while
 *   `name` is only the graph's title. Normally there is no prompt (inside a
 *   project, git is the undo -- ID9); a tab carrying a file but no title is
 *   asked for the title once, and what it answers is the title of the file it
 *   is already on, never a new address.
 * - An unbound tab -- a gallery example, an import, a canvas that has never
 *   been saved -- takes the prompt path, as does every explicit Save As: a
 *   name is asked for, no `file` is sent so the server derives the address
 *   from that name exactly as it always has, and the collision guard that
 *   stops one graph quietly replacing another still runs.
 *
 * The binding is now the whole of that rule. It used to read "project mode +
 * bound + !saveAs", with "non-project mode: ALWAYS prompt for a name (legacy
 * behavior, Toolbar.tsx)" alongside it, and outside a project directory that
 * made every save of an already-saved graph ask for its name again -- and
 * then, because the entered name matches the file it was read from, put a
 * second dialog in the way asking whether to overwrite it. Two questions to
 * write a file that already has a name is not a save button, and the save
 * icon in the toolbar has to be one. The binding is not a hidden state being
 * guessed at on the user's behalf either: the Graphs panel marks the bound
 * row "Current", so the row about to be overwritten is already on screen
 * beside the icon that overwrites it.
 *
 * Task 13 extends this with project-origin stamping + cross-project refusal.
 * Task 16 (ID8) refuses outright when the active tab is read-only (its graph
 * was loaded from a newer format_version than this build writes).
 *
 * Everything after the dialogs addresses the tab this save STARTED from, by
 * id, and never "the active tab". A save can stop for a name and a list read,
 * and the user can switch tabs or close one while it waits -- so by the time
 * the canvas is serialized, "the active tab" may be a different graph
 * entirely. Reading it there wrote the OTHER tab's nodes into the file this
 * save named, under a success toast, and nothing on screen said so. The
 * addressed calls below are the whole of the fix, and they are load-bearing
 * on every path that stops for a dialog -- the prompted path, and a bound
 * tab's one-time question about its own title.
 */
export async function saveActiveGraph(opts: { saveAs?: boolean } = {}): Promise<void> {
  const t = useI18n.getState().t;
  const addToast = useToastStore.getState().addToast;
  const store = useTabStore.getState();
  const tab = store.tabs.find((tb) => tb.id === store.activeTabId);
  if (!tab) return;
  if (tab.readOnly) {
    // A graph written by a newer CodefyUI opens read-only so an older build
    // can never round-trip-drop fields it does not understand (ID8).
    useToastStore.getState().addToast(t('project.readOnly.saveBlocked'), 'error');
    return;
  }
  const projectDir = useProjectStore.getState().projectDir;
  const projectMode = projectDir !== null;

  // Cross-project footgun guard (ID10): a tab stamped with a DIFFERENT project
  // must never overwrite into the currently-open project.
  if (projectMode && tab.projectOrigin != null && tab.projectOrigin !== projectDir) {
    addToast(t('project.save.crossProjectRefused', { origin: tab.projectOrigin }), 'error');
    return;
  }

  // ONE rule, and the binding is the whole of it: a bound tab's save targets
  // THE FILE IT IS BOUND TO. Not the file the title sanitizes into, not the
  // file the user names at a dialog -- that file.
  //
  // The address and the title are two fields because they are two things,
  // and `POST /api/graph/save` used to take one string for both: it
  // sanitized `name` into the file stem AND stored `name` in the file as the
  // graph's title. Nothing can satisfy both jobs at once, which is why three
  // rounds of fixing this from inside this function each traded one bug for
  // the other.
  //
  // Sending the STEM as the name kept the address and renamed the graph:
  // `GET /api/graph/list` reads the stored name back out, so "My Graph"
  // became "My_Graph" in the panel and in the toast the first time the save
  // icon was pressed, permanently, as the act of saving it -- sanitizing is
  // lossy, so nothing downstream could undo it. Sending the TITLE as the name
  // kept the title and moved the address instead: whenever a file's stem is
  // not `sanitize(its own stored title)` the write landed somewhere else
  // entirely. Reproduced end to end against the real backend -- a tab bound
  // to `Beta.json`, whose stored name was "Alpha", overwrote `Alpha.json`,
  // and in project mode `write_graph_pair`'s legacy path DELETED it.
  //
  // With the address on the request neither is possible. `file` is never
  // written into the graph file; it is where this save goes, and the title is
  // only a title.
  //
  // The title travels ON THE TAB rather than being fetched when it is needed.
  // Asking `GET /api/graph/list` for it was tried and is worse in three ways,
  // each reproduced against the real backend: a case-insensitive filesystem
  // leaves the tab's stem and the listed file differing by case, so no row
  // matches; the route answers 409 in a project holding both a legacy and a
  // canonical copy of a base, which is what checking out an older commit
  // produces; and the await itself opens the tab-switch window described
  // above on the one path that had never had one.
  const boundFile = tab.currentGraphFile;
  const inPlace = !!boundFile && !opts.saveAs;

  // The address to write to, or null for "let the server derive it from the
  // name" -- which is exactly today's behaviour, byte for byte, and what a
  // graph being named for the first time wants.
  let targetFile: string | null = null;
  let targetName: string;
  // The file the collision guard matched and the user agreed to overwrite, in
  // the SERVER's spelling -- see the clear at the end of this function for why
  // that spelling, and not the one this save produces, is the one to act on.
  // Null on every path that overwrote nothing, which is most of them.
  let overwrittenFile: string | null = null;

  if (inPlace && tab.currentGraphName) {
    // The one-click Save: both halves of the binding are known, so there is
    // nothing to ask and nothing to check. The file about to be overwritten
    // is already on screen -- the Graphs panel marks its row "Current",
    // beside the icon that overwrites it.
    targetFile = boundFile;
    targetName = tab.currentGraphName;
  } else {
    const entered = await prompt({ title: t('toolbar.save.prompt'), placeholder: 'graph-name' });
    const trimmed = entered?.trim();
    if (!trimmed) return;
    targetName = trimmed;
    if (inPlace) {
      // A tab restored from a record 2.8.0 wrote: a file and no title,
      // because that build persisted the stem alone. Those are exactly the
      // tabs holding the graphs whose titles needed sanitizing, so falling
      // back to the stem would rename precisely the graphs this protects --
      // they are asked once instead, and never again, because the save below
      // writes both halves back onto the tab.
      //
      // The address stays pinned to the binding and the collision guard does
      // not run, because the question here is "what is this file called?",
      // not "where should this go?". While the typed name chose the address
      // this prompt FORKED the graph: type anything but the exact original
      // title and a NEW file was written, the bound one left stale, the tab
      // silently moved onto the copy. And the guard would have been a second
      // dialog asking whether to overwrite the file the tab is already on.
      targetFile = boundFile;
    } else {
      // Unbound, or an explicit Save As. The name IS the address on this
      // path -- no `file` is sent -- so the guard that stops one graph
      // quietly replacing another is the only thing standing between a typed
      // name and somebody else's file. The tab's own file is excepted inside
      // `findGraphNameCollision`: re-saving a graph under its own name is not
      // a collision to warn about.
      let existing: { name: string; file: string }[] = [];
      try {
        const r = await listGraphs();
        if (Array.isArray(r)) existing = r;
      } catch {
        /* list unavailable -- proceed without the overwrite check */
      }
      const colliding = findGraphNameCollision(trimmed, existing, tab.currentGraphFile);
      if (colliding !== null) {
        const okConfirm = await confirm({
          // The TITLE, because that is the only form of this graph the user
          // has ever seen -- the stem beside it is an implementation detail
          // of where it is stored.
          title: t('toolbar.save.overwriteConfirm', { name: colliding.name }),
          confirmText: t('toolbar.save'),
          variant: 'danger',
        });
        if (!okConfirm) return;
        // Remembered past the dialog for the clear at the end of this
        // function. Set only here, after a yes: a refused overwrite writes
        // nothing, so there is nothing for any other tab's binding to have
        // become wrong about.
        overwrittenFile = colliding.file;
      }
    }
  }

  // The tab this save started on, re-read AFTER the dialogs above. `tab` is
  // a snapshot from before them, and serializing it would throw away every
  // edit the user made while the name dialog was open -- which is most of
  // what they were doing with the seconds the dialog cost them. The live
  // object is the one to save; the id is what keeps it the RIGHT one.
  const liveStore = useTabStore.getState();
  const liveTab = liveStore.tabs.find((tb) => tb.id === tab.id);
  if (liveTab === undefined) {
    // The tab was closed while the dialog was open, so there is no longer a
    // graph to write. Silently, and that is a decision rather than an
    // omission: closing a tab is already this app's answer to "discard what
    // is on this canvas" -- `removeTab` takes it away with no confirmation
    // and no toast -- so a message here would arrive over a DIFFERENT tab's
    // canvas to report a loss the user had just chosen. What must never
    // happen is the alternative this return exists to prevent: serializing
    // whichever tab is active NOW and writing it to disk under the name
    // typed for the tab that is gone.
    return;
  }

  try {
    // `getSerializedGraphOf(liveTab)`, never `getSerializedGraph()`: the
    // latter serializes the ACTIVE tab, which after a dialog is not
    // necessarily the tab being saved.
    //
    // `subgraphs` (core#137) is NOT optional dressing: a collapsed block's
    // instance node only stores `subgraph:<id>`, and the definition holding
    // its inner nodes/edges lives solely in this list. Drop it here and the
    // saved file keeps the instance while losing everything inside it —
    // unrecoverably, since nothing else on disk has a copy. The bug was
    // invisible in normal use because the tab's IndexedDB autosave DOES
    // persist `subgraphs`, so blocks survived browser reloads and were only
    // lost when the user did the deliberate, trust-building thing: Save.
    const { nodes, edges, presets, segmentGroups, subgraphs, settings } =
      liveStore.getSerializedGraphOf(liveTab);
    const result = await saveGraph({
      nodes, edges, name: targetName,
      // The address, and only when this tab is bound to one. An omitted
      // `file` is the server's original behaviour -- the stem comes from the
      // name -- which is what a graph named a moment ago at the prompt above
      // is asking for. `name` above is the title in both cases, and the
      // server never writes this field into the file.
      ...(targetFile !== null ? { file: targetFile } : {}),
      // From the live tab too: the description is edited in the Inspector,
      // which is on screen the whole time the name dialog is not.
      description: liveTab.description ?? '', presets, segmentGroups, subgraphs,
      // Only when the graph assigns a device: the serializer emits the block
      // only then, and a file with no assignment stays byte-identical.
      ...(settings ? { settings } : {}),
    });
    // What the server just wrote, AS THE SERVER SPELLS IT. The route answers
    // with the stem it sanitized and wrote, and taking it from there is the
    // only way to be sure: `sanitizeGraphName` is a replica of the backend's
    // `_sanitize_name`, and a replica of a rule is not the rule. This one
    // tests `[\p{L}\p{N}]` against Node's ICU tables where the backend uses
    // CPython's `str.isalnum()`, and the two Unicode versions disagree on 16
    // BMP code points, plus astral ones. The write LANDED correctly either
    // way -- the server re-sanitizes whatever `file` it is handed -- but a
    // binding built from the replica's answer names a stem `GET
    // /api/graph/list` will never report, so the Graphs panel marks no row
    // Current and the exact-match clear below finds no tab.
    //
    // The fallback is for a frontend built from source meeting a backend
    // older than the field, which is the same reason `renameGraph`'s caller
    // in `GraphsTab` carries one: the address we sent (so a bound tab's
    // binding still never moves as a side effect of being saved), else the
    // replica's best guess at the stem derived from the name. A BLANK `file`
    // takes that fallback too rather than being believed -- an empty stem is
    // not an address, and a tab bound to one would send `file: ""` on its
    // next save, which the route reads as "no address given".
    const savedFile =
      typeof result?.file === 'string' && result.file
        ? result.file
        : targetFile ?? sanitizeGraphName(targetName);
    // Both halves of the binding, so the NEXT save of this tab writes the
    // name the user just typed rather than the stem it was sanitized into.
    // This is what makes the 2.8.0-restored tab's one-time prompt a one-time
    // prompt.
    liveStore.setTabGraphFile(tab.id, savedFile, targetName);
    if (savedFile !== liveTab.currentGraphFile) {
      // This save wrote a file the SAVING tab was not bound to a moment ago,
      // and a confirmed Save As is allowed to write over a file another tab
      // IS bound to. That other tab's binding is now a lie: the file holds
      // this graph, not the one that tab is showing. Left alone, its next
      // Save is promptless and in place, so it writes its own older graph
      // back over the work just saved here -- and the Graphs panel's "this
      // graph is already open, raise its tab" branch finds it first (array
      // order) and raises the stale tab instead of this one. Clearing the
      // binding makes that tab ask for a name on its next Save rather than
      // write, which is the same treatment the panel's rename and delete
      // already give every bound tab, for the same stated reason. The saving
      // tab is excepted because its binding is the one this save just made
      // true. A save that targeted the tab's own binding does not normally
      // reach here at all: `savedFile` IS that binding, so nothing about any
      // other tab changed.
      liveStore.rebindGraphFile(savedFile, null, tab.id);
    }
    if (overwrittenFile !== null && overwrittenFile !== savedFile) {
      // The same clear, for the file the collision guard MATCHED rather than
      // the one this save produced -- and they are not always the same
      // string, which is the whole of this branch.
      //
      // `findGraphNameCollision` folds case on purpose: NTFS and APFS are
      // case-insensitive, and a silent overwrite on the majority platform is
      // the worse failure. `rebindGraphFile` compares with `===`, also on
      // purpose: on a case-sensitive filesystem the two stems really are two
      // files, and folding them there would raise the wrong tab from the
      // Graphs panel and mark the wrong row Current. The gap between the two
      // is a tab that still believes it owns the file:
      //
      //   tab A saves "My Graph" -> `My_Graph.json`, binds `My_Graph`
      //   tab B's Save As of "my graph" matches that row, the user accepts,
      //     and the save's `os.replace` re-spells the ENTRY to
      //     `my_graph.json`; tab B binds `my_graph`
      //   the clear above looks for `my_graph` and never matches `My_Graph`
      //   tab A's next Save is promptless, sends `file: "My_Graph"`, and on
      //     NTFS lands on that same physical file -- over tab B's graph, with
      //     no dialog
      //
      // So the app warned that two names are one file and then treated them
      // as two. Reproduced end to end over real HTTP, in both non-project and
      // project mode. The saving tab is excepted for the reason it is above:
      // its binding is the one this save just made true.
      liveStore.rebindGraphFile(overwrittenFile, null, tab.id);
    }
    if (projectMode) {
      liveStore.stampTabProject(tab.id, projectDir);
      // A project save writes two files into the repository, and the Source
      // Control tab polls on a fifteen-second interval. This is what makes
      // them appear under Changes while the hand is still on the keyboard.
      // Nothing is listening unless that tab is open, which is why this file
      // needs no knowledge of it -- see `utils/worktreeWrite.ts`.
      announceWorktreeWrite();
    }
    // The sidebar's Graphs panel is listing the very file this call just
    // wrote: a prompted save adds a row, an in-place save moves that row's
    // timestamp. It sits in the success half of the try because a refused,
    // cancelled or failed save changed nothing on the server, and re-reading
    // the list for one would only make the panel flicker. Unlike the
    // worktree signal above it is NOT gated on project mode -- the panel
    // lists saved graphs wherever they live, so a save with no project open
    // still owes it a row. Nothing is listening unless that panel is
    // mounted, which is why this file needs no knowledge of it -- see
    // `utils/graphsWrite.ts`.
    announceGraphsWrite();
    addToast(t('toolbar.save.success', { name: targetName }), 'success');
  } catch (e) {
    addToast(t('toolbar.save.fail', { error: (e as Error).message }), 'error');
  }
}
