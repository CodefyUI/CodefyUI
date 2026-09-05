import { useCallback, useEffect, useId, useRef, useState } from 'react';
import { REF_KINDS, isLayoutFile, useGitStore, type GitRefKind } from '../../store/gitStore';
import { useI18n } from '../../i18n';
import { docsUrl } from '../../utils/docsUrl';
import { prompt } from '../../utils/dialog';
import { MoreHorizontalIcon, RefreshIcon, SyncIcon } from '../shared/Icons';
import { ActionMenu, type ActionMenuItem } from '../shared/ActionMenu';
import { ProgressBar } from '../shared/ProgressBar';
import shell from '../Sidebar/NodePalette.module.css';
import styles from './SourceControl.module.css';
import { refSectionIds } from './RefSection';
import { aheadBehindGlyphs, errorSentence, followUpFor, gitOpKey } from './scm';

/**
 * The documentation page the Setup guide link and menu row point at.
 *
 * Project directories, not a source-control page: the tab's own page is
 * written in the part of this track that adds history and diffs, and a link
 * to it now would 404. This one exists, and it is the right answer for the
 * screen the link matters most on -- "Source control needs a project
 * directory", whose two commands are that page's subject. It moves to the
 * tab's page when that page lands.
 */
export const SCM_DOCS_PATH = '/usage/project-directories';

/**
 * The two places focus can be parked when the row that held it is gone.
 *
 * A `data-` attribute rather than a ref threaded through four components: the
 * thing that loses focus (a file row) and the thing that catches it (the
 * message box, or failing that the panel title) are in different subtrees, and
 * the panel is a singleton — the sidebar mounts exactly one open tab.
 */
export const SCM_FOCUS = { commit: 'commit', title: 'title' } as const;

/**
 * Put focus back somewhere sensible in the panel.
 *
 * Called when a group heading has been unmounted by the very action that was
 * pressed — discarding the last change empties the panel — so that focus lands
 * on the message box rather than falling to the document body, where the next
 * Tab starts from the top of the page.
 */
export function focusScmFallback(): void {
  const target = document.querySelector<HTMLElement>(
    `[data-scm-focus="${SCM_FOCUS.commit}"]`,
  ) ?? document.querySelector<HTMLElement>(`[data-scm-focus="${SCM_FOCUS.title}"]`);
  target?.focus();
}

/**
 * Put focus back on one reference section's heading.
 *
 * Called after a row that held focus has gone -- a branch deleted, a stash
 * dropped, a remote removed -- so it lands on the heading of the list it left
 * rather than on the document body, where the next Tab starts from the top of
 * the page.
 *
 * By id rather than through a ref, for the same reason the branch button
 * points `aria-controls` at one: `refSectionIds(kind)` is fixed per kind, the
 * panel is a singleton, and the row that loses focus and the heading that
 * catches it are drawn by two different components. On the next frame, because
 * the store's update has been applied but React has not necessarily rendered
 * yet -- and the whole SECTION can be what disappears, when the panel switches
 * screens under it.
 */
export function focusRefSection(kind: GitRefKind): void {
  requestAnimationFrame(() => {
    const heading = document.getElementById(refSectionIds(kind).headingId);
    if (heading !== null && heading.isConnected) {
      heading.focus();
      return;
    }
    focusScmFallback();
  });
}

/** The four overflow rows that talk to a remote, in the order they are drawn. */
type RemoteAction = 'fetch' | 'pull' | 'push' | 'publish';

/**
 * How many times one panel asks for the remote list before it gives up.
 *
 * The read is automatic -- nobody pressed anything -- so a server that keeps
 * refusing it would otherwise be asked again every fifteen seconds, for as
 * long as the tab is open, for an answer it has already refused three times.
 * Three attempts covers a server that was restarting while the panel opened;
 * past that the list is read again by anything that opens the Remotes section,
 * writes a remote, presses Publish, or reopens the tab.
 */
const REMOTES_READ_TRIES = 3;

/**
 * The tab's title row, the branch line, the busy bar and the error line.
 *
 * Drawn in EVERY repository state, not only in `ready`. Two reasons: the title
 * row is the shell the other four sidebar tabs share, so dropping it would
 * leave the fifth tab's header a few pixels off everyone else's; and a server
 * that stops answering after a good first read has to be able to say so, which
 * means the error line cannot live inside the branch of a switch that a
 * failed read never reaches.
 */
export function ScmHeader() {
  const { t } = useI18n();
  const repoState = useGitStore((s) => s.repoState);
  const status = useGitStore((s) => s.status);
  const busyOp = useGitStore((s) => s.busyOp);
  const netOp = useGitStore((s) => s.netOp);
  const remotes = useGitStore((s) => s.remotes);
  const sections = useGitStore((s) => s.sections);
  const branchesOpen = sections.branches;
  const lastError = useGitStore((s) => s.lastError);
  const loadError = useGitStore((s) => s.loadError);
  const hideLayout = useGitStore((s) => s.hideLayout);
  const refresh = useGitStore((s) => s.refresh);
  const refreshRefs = useGitStore((s) => s.refreshRefs);
  const setSectionOpen = useGitStore((s) => s.setSectionOpen);
  const setHideLayout = useGitStore((s) => s.setHideLayout);
  const openIdentityForm = useGitStore((s) => s.openIdentityForm);
  const dismissError = useGitStore((s) => s.dismissError);
  const runFetch = useGitStore((s) => s.fetch);
  const runPull = useGitStore((s) => s.pull);
  const runPush = useGitStore((s) => s.push);
  const runSync = useGitStore((s) => s.sync);
  const runPublish = useGitStore((s) => s.publish);
  const runStash = useGitStore((s) => s.stashPush);
  const [detailsOpen, setDetailsOpen] = useState(false);
  // The Details toggle names what it opens, so the reader it was announced to
  // can go straight there instead of hunting for what changed.
  const stderrId = useId();

  // What the filter is swallowing RIGHT NOW: zero while it is off, so the
  // count appears as it is switched on and the menu stays open to show it.
  const hiddenCount = hideLayout
    ? [...(status?.unstaged ?? []), ...(status?.untracked ?? [])].filter((file) =>
      isLayoutFile(file.path),
    ).length
    : 0;

  const ready = repoState === 'ready' && status !== null;
  const detached = status?.detached === true;
  const unborn = status?.unborn === true;
  const hasUpstream = status !== null && status.upstream !== null;
  // `null` is "not read yet" and NEVER "none": treating it as none would hide
  // Publish on a repository that has a remote.
  const noRemotes = remotes !== null && remotes.length === 0;
  const clean = status !== null
    && status.staged.length === 0
    && status.unstaged.length === 0
    && status.untracked.length === 0
    && status.conflicted.length === 0;
  // Publishing needs a branch to publish and somewhere to put it.
  const publishable = ready && !unborn && !detached && !hasUpstream && !noRemotes;

  // Attempts spent on the remote list, for the life of this panel.
  const remoteTries = useRef(0);
  // Whether one of those attempts is still out. StrictMode runs an effect
  // twice on mount and a slow answer leaves the state that triggered this
  // one unchanged, so without it two reads could spend two of the three
  // attempts on the same question -- and a server that was restarting would
  // have the budget gone before it came back.
  const remotesInFlight = useRef(false);

  useEffect(() => {
    // The Sync / Publish / neither decision needs to know whether there IS a
    // remote, and nothing else in the tab asks until a section is opened. One
    // read is normally all it takes: every write that can change the list
    // refreshes it afterwards.
    //
    // A read that FAILS leaves `remotes` null, and null hides Publish -- so
    // `status` is a dependency here, as a clock rather than a value: each
    // status the poll brings back is both evidence that the server is
    // answering and another chance at the read that was not. Without it, one
    // refused read hid Publish for the life of the panel.
    if (repoState !== 'ready') {
      // Another repository, or none: the next one gets a fresh budget.
      remoteTries.current = 0;
      return;
    }
    if (status === null || remotes !== null) return;
    // The Remotes SECTION reads the same list, on mount and on every poll
    // (`refreshExpandedRefs`), because it draws it. Two readers and one
    // question: while the section is open its read is the one that answers,
    // and this one went out before it came back -- so opening a tab whose
    // Remotes section was remembered open sent two GET /remotes for one
    // list. Closing the section runs this effect again, and this time the
    // header is the only reader there is.
    if (sections.remotes) return;
    if (remotesInFlight.current) return;
    if (remoteTries.current >= REMOTES_READ_TRIES) return;
    remoteTries.current += 1;
    remotesInFlight.current = true;
    void refreshRefs('remotes').finally(() => {
      remotesInFlight.current = false;
    });
  }, [refreshRefs, remotes, repoState, sections.remotes, status]);

  /**
   * Publish this branch, resolving WHICH remote first.
   *
   * `remotes` is null until something reads it, and a publish that names no
   * remote is refused with a 400 whenever there is more than one to choose
   * between -- so the list is read before the decision rather than after the
   * refusal. Exactly one remote publishes straight to it; none sends nothing
   * at all, which is what keeps a repository with no remote from asking the
   * same refused question twice.
   *
   * Several remotes are the picker's business, and every control that offers
   * Publish with a list already in hand IS the picker. The last line is what
   * is left for the one case that cannot be: a button pressed in the moment
   * before the list landed, on a repository that turns out to have several. It
   * sends the publish the server will refuse, because the refusal carries the
   * follow-up -- which, by the time it is drawn, is that picker.
   *
   * When the list resolves to NEITHER a remote nor an answer, nothing is sent
   * -- and something is still said. A refused refs read writes `refsError`,
   * which is only drawn inside its section, so a bare return left the More
   * menu's Publish doing nothing at all with nothing on screen to explain it.
   * Opening the Remotes section puts that line where it can be read, reads the
   * list once more on the way in, and lands on the Add Remote... that answers
   * the other half of it.
   */
  const publishBranch = useCallback(async () => {
    let list = useGitStore.getState().remotes;
    if (list === null) {
      await refreshRefs('remotes');
      list = useGitStore.getState().remotes;
    }
    if (list === null || list.length === 0) {
      setSectionOpen('remotes', true);
      // ...and go there. A section that opens somewhere below the fold, with
      // the keyboard left on a Publish that did nothing, is the same dead end
      // with one more thing on screen.
      focusRefSection('remotes');
      return;
    }
    if (list.length === 1) {
      await runPublish(list[0].name);
      return;
    }
    await runPublish();
  }, [refreshRefs, runPublish, setSectionOpen]);

  /**
   * Read everything that is on screen, which is what Refresh means.
   *
   * The status AND every OPEN section's list. A hidden tab runs no poll at
   * all, so the panel a user comes back to can be minutes old in every one of
   * its four parts -- and a Refresh that read the status alone left the
   * Stashes count and the branch rows exactly as stale as they were, next to
   * a header that had just updated. A closed section is not read: nothing of
   * it is on screen to be wrong.
   *
   * The status is read in EVERY repository state, and the lists only where
   * there is a repository to read them from -- the same guard the poll's own
   * walk uses (`refreshExpandedRefs` in the store), because this is the other
   * way into that loop. The three ref routes can only be REFUSED against a
   * project that is not a repository, the tab draws no section there, so the
   * `refsError` each refusal writes is not on screen either; and the open
   * flags are remembered per profile, so a tab left open on a repository sent
   * three doomed requests per press on the next project. `unknown` passes for
   * the reason it passes there: it is the state between mounting and the
   * first status, and a press inside that window is about a repository the
   * panel simply has not been told about yet.
   */
  const refreshPanel = useCallback(() => {
    void refresh();
    if (repoState !== 'ready' && repoState !== 'unknown') return;
    for (const kind of REF_KINDS) {
      if (sections[kind]) void refreshRefs(kind);
    }
  }, [refresh, refreshRefs, repoState, sections]);

  const askThenStash = useCallback(async () => {
    const message = await prompt({ title: t('git.stash.messagePrompt') });
    if (message === null) return;
    // The message is OPTIONAL and the prompt says so, so a box left empty is
    // NOT a message: it goes as `null`, and git writes its own subject ("WIP
    // on main: <sha> <subject>") for the row instead. An empty STRING is a
    // different thing, and the route answers it with a 400.
    const named = message.trim();
    // Untracked files go with it. A stash that left a new graph file behind
    // would not free the tree for the checkout the stash was made for.
    await runStash(named === '' ? null : named, true);
  }, [runStash, t]);

  // One network operation at a time (R11), so while that lane is busy every
  // control that would join it is refused where it stands rather than a click
  // later. The local lane is not consulted: a commit during a fetch is allowed
  // on both sides. The reason is the sentence the busy bar is already showing,
  // which is what makes the refusal readable rather than merely grey.
  const netBusy = netOp !== null;
  const netBusyReason = netOp === null
    ? null
    : t('git.busy', { op: t(gitOpKey(netOp)) });

  /**
   * Why a remote row is refused, or null when it can be pressed.
   *
   * First match wins, weakest state first: a branch with no commits has
   * nothing to send anywhere, then a repository with no remote has nowhere to
   * send it, then a detached HEAD has no branch to send (a fetch still does),
   * and last a branch with no upstream has nothing to pull from or push to --
   * which is exactly what Publish is for, so Publish stays on.
   */
  const refusedBecause = (action: RemoteAction): string | null => {
    // Above every state reason: while the network lane is busy none of these
    // can start at all, and a row that offered the press would answer it with
    // the same fact as a toast a second later (R11).
    if (netBusyReason !== null) return netBusyReason;
    if (unborn) return t('git.unborn');
    if (noRemotes) return t('git.remote.empty');
    if (detached && action !== 'fetch') return t('git.detached');
    if (!hasUpstream && (action === 'pull' || action === 'push')) {
      return t('git.noUpstream');
    }
    return null;
  };

  /**
   * Whether the overflow menu carries a Publish row at all.
   *
   * It offers Publish where the branch line offers it, and nowhere else. Two
   * states where the line does not: a branch that already tracks an upstream
   * has nothing to publish (the line shows Sync), and a repository with
   * several remotes is answered by the picker the line's own button opens --
   * a row here could only send a publish naming no remote, which the server
   * refuses with a 400. Where the row is dropped, the control that replaces it
   * is already on screen.
   *
   * `publishable` is what makes that last clause true. Several remotes is only
   * the picker's business where the picker is DRAWN, and it is not drawn on an
   * unborn or a detached HEAD -- so without it those two states lost the row
   * as well, and with it the one thing on screen that said why.
   */
  const publishInMenu = !hasUpstream
    && !(publishable && remotes !== null && remotes.length > 1);

  const stashRefusedBecause = status?.merge_in_progress === true
    ? t('git.merge.banner')
    : clean
      ? t('git.empty.clean')
      : null;

  /** One overflow row for a git action, refused with its reason or live. */
  const gitRow = (
    id: string,
    label: string,
    reason: string | null,
    onSelect: () => void,
  ): ActionMenuItem => ({
    id,
    label,
    disabled: reason !== null,
    hint: reason ?? undefined,
    onSelect,
  });

  const items: ActionMenuItem[] = [
    // What git can be asked to do, above what the panel itself does. Only
    // where there IS a repository: every other state answers these with a
    // refusal, and a row greyed out for a reason no key describes says less
    // than no row at all.
    ...(ready
      ? [
        gitRow('fetch', t('git.action.fetch'), refusedBecause('fetch'), () => {
          void runFetch();
        }),
        gitRow('pull', t('git.action.pull'), refusedBecause('pull'), () => {
          // Fast-forward: an explicit merge is what the `diverged` refusal
          // offers next, so it is a decision rather than a default.
          void runPull('ff-only');
        }),
        gitRow('push', t('git.action.push'), refusedBecause('push'), () => {
          void runPush();
        }),
        ...(publishInMenu
          ? [
            gitRow('publish', t('git.action.publish'), refusedBecause('publish'), () => {
              void publishBranch();
            }),
          ]
          : []),
        gitRow('stash', t('git.action.stash'), stashRefusedBecause, () => {
          void askThenStash();
        }),
      ]
      : []),
    {
      id: 'hideLayout',
      label:
        hiddenCount > 0
          ? `${t('git.menu.hideLayout')} ${t('git.menu.hiddenCount', { count: hiddenCount })}`
          : t('git.menu.hideLayout'),
      checked: hideLayout,
      onSelect: () => setHideLayout(!hideLayout),
    },
    {
      id: 'identity',
      label: t('git.action.identity'),
      // Reading the config needs a repository: every other state answers
      // `GET /config` with its own refusal, so the row would open an empty
      // form above an error line. `repoState` moves only when the repository
      // itself does, never on a poll of an unchanged one.
      disabled: repoState !== 'ready',
      onSelect: () => openIdentityForm(),
    },
    {
      id: 'docs',
      label: t('git.action.docs'),
      onSelect: () => window.open(docsUrl(SCM_DOCS_PATH), '_blank', 'noopener,noreferrer'),
    },
  ];

  const branchText = status === null
    ? null
    : status.detached
      ? t('git.detached')
      : status.unborn
        ? t('git.unborn')
        : t('git.branch.label', { name: status.branch ?? '' });

  const branchIds = refSectionIds('branches');

  const toggleBranches = () => {
    const next = !branchesOpen;
    setSectionOpen('branches', next);
    if (!next) return;
    // On the next frame: the section is drawn by another component, and the
    // store's update has been applied but not necessarily rendered yet.
    // `scrollIntoView` is absent in jsdom, hence the optional call.
    requestAnimationFrame(() => {
      document.getElementById(branchIds.headingId)?.scrollIntoView?.({ block: 'nearest' });
    });
  };

  // The one refusal with a way out, and which way that is.
  const followUp = lastError === null
    ? null
    : followUpFor(lastError.code, lastError.op);
  const followUpPublish = followUp === 'publish' && publishable;
  // ONE control, never two: while the refusal is offering Publish, the branch
  // line does not offer it as well.
  const headerPublish = publishable && remotes !== null && !followUpPublish;
  const publishOffered = headerPublish || followUpPublish;

  // A branch with no commits has nothing to push and no upstream to lack, so
  // it gets no tracking half at all rather than "Not published" beside "No
  // commits yet", which says the same thing twice. "Not published" goes the
  // same way when the Publish button is on screen: the button says it, and
  // says what to do about it.
  const trackingText = status === null || status.unborn
    ? null
    : status.upstream_gone
      ? t('git.upstreamGone')
      : status.upstream === null
        ? (publishOffered ? null : t('git.noUpstream'))
        : t('git.aheadBehind', { ahead: status.ahead ?? 0, behind: status.behind ?? 0 });
  // The one tracking state that is a COUNT rather than a state. It is drawn
  // as the two numbers and read out as the sentence: translated, the clause
  // is twelve characters beside a branch name in a 156px column, and at
  // 250px it ellipsised both halves at once. The other three -- upstream
  // gone, not published, no commits -- stay as the words they are.
  const counted = status !== null
    && !status.unborn
    && !status.upstream_gone
    && status.upstream !== null;
  const trackingGlyphs = counted
    ? aheadBehindGlyphs(status?.ahead ?? 0, status?.behind ?? 0)
    : null;

  const stderr = lastError?.stderr ?? null;
  // The lanes are independent, so either one running is something happening.
  // The local one wins the label: it is the one the user pressed a button for.
  const runningOp = busyOp ?? netOp;

  /**
   * The Publish control, as a button or as the picker several remotes need.
   *
   * `aria-disabled` and a guarded handler, never the native attribute -- the
   * rule this panel already states on the commit button and on every menu row.
   * `netOp` is set synchronously, before the request goes out, so the control
   * the user just pressed is the one that turns refused on the very next
   * render: natively disabled it would stop being focusable there and the
   * browser would drop focus to `<body>`, seconds at a time, on every press.
   * The store's own busy toast is the backstop for a press that races the
   * render; the `title` is what makes the refusal readable before it.
   */
  const publishControl = (className: string) =>
    (remotes !== null && remotes.length > 1 ? (
      <ActionMenu
        label={t('git.action.publish')}
        items={remotes.map((entry) => ({
          id: entry.name,
          // The remote's own name, which is data and needs no translation.
          label: entry.name,
          onSelect: () => void runPublish(entry.name),
        }))}
        align="end"
        className={className}
        disabled={netBusy}
        disabledHint={netBusyReason ?? undefined}
      >
        {t('git.action.publish')}
      </ActionMenu>
    ) : (
      <button
        type="button"
        className={className}
        title={netBusyReason ?? t('git.action.publish')}
        aria-disabled={netBusy}
        onClick={() => {
          if (netBusy) return;
          void publishBranch();
        }}
      >
        {t('git.action.publish')}
      </button>
    ));

  return (
    <div className={`${shell.header} ${styles.header}`}>
      <div className={shell.headerRow}>
        <div
          className={shell.headerTitle}
          data-scm-focus={SCM_FOCUS.title}
          tabIndex={-1}
        >
          {t('sidebar.tab.git')}
        </div>
        <button
          type="button"
          className={shell.toolbarButton}
          onClick={refreshPanel}
          aria-label={t('sidebar.refresh')}
          title={t('sidebar.refresh')}
        >
          <RefreshIcon size={13} />
        </button>
        <ActionMenu
          label={t('git.action.more')}
          items={items}
          align="end"
          className={shell.toolbarButton}
        >
          <MoreHorizontalIcon size={13} />
        </ActionMenu>
      </div>
      {branchText !== null && (
        <div className={styles.branchRow}>
          {/*
            The name is the button that opens the branch list, which is the
            list it names. Styled as the text it replaced: the affordance is
            the section it expands, not a box drawn around a branch name.
          */}
          <button
            type="button"
            className={`${styles.branchName} ${styles.branchButton}`}
            aria-expanded={branchesOpen}
            aria-controls={branchIds.listId}
            title={branchText}
            onClick={toggleBranches}
          >
            {branchText}
          </button>
          {trackingText !== null && (
            /*
              The glyphs are what is on screen and the sentence is what is
              announced -- `aria-hidden` on the one, off-screen text for the
              other, and the sentence in the `title` for a pointer. A tracked
              branch that is level with its upstream draws nothing at all:
              two zeros say only that they are zero, and the sentence is
              still there for a reader, who has no glance to spend.
            */
            <span className={styles.tracking} title={trackingText}>
              {counted ? (
                <>
                  {trackingGlyphs !== null && (
                    <span aria-hidden="true">{trackingGlyphs}</span>
                  )}
                  <span className={styles.srOnly}>{trackingText}</span>
                </>
              ) : trackingText}
            </span>
          )}
          {ready && !unborn && !detached && hasUpstream && (
            /*
              See `publishControl`: `aria-disabled` and a guarded handler. This
              one is icon-only, so its `title` is its whole identity as well as
              where the reason goes -- and a natively disabled button opens no
              tooltip at all in Chrome. The name stays on `aria-label`, so it
              is still "Sync (pull, then push)" while it is refused.
            */
            <button
              type="button"
              className={styles.iconButton}
              aria-label={t('git.action.sync')}
              title={netBusyReason ?? t('git.action.sync')}
              aria-disabled={netBusy}
              onClick={() => {
                if (netBusy) return;
                void runSync();
              }}
            >
              <SyncIcon size={13} />
            </button>
          )}
          {headerPublish && publishControl(styles.publishButton)}
        </div>
      )}
      {runningOp !== null && (
        <div className={styles.busy}>
          <ProgressBar
            value={null}
            size="sm"
            label={t('git.busy', { op: t(gitOpKey(runningOp)) })}
          />
        </div>
      )}
      {(lastError !== null || loadError !== null) && (
        <div className={styles.error}>
          {/*
            ONE sentence, and the operation's is the one that wins. A stopped
            server fails the write and the poll behind it with the same words,
            and the panel showed both: "Could not read repository status:
            Failed to fetch" with "git failed: Failed to fetch" underneath it.
            `lastError` is the half that carries the hint and the Details
            toggle, so it is the half worth keeping -- and Dismiss still takes
            it down, after which the read's line is there again on the next
            poll that fails.

            `role="alert"` wraps the SENTENCE and nothing else. An alert
            re-announces whenever its subtree changes, so putting the Details
            toggle, the follow-up button and the `<pre>` inside it would read
            the whole refusal out again every time somebody opened or closed
            the stderr -- or every time the remote list came back and turned
            the follow-up into a picker.
          */}
          <div role="alert">
            {lastError !== null ? (
              <>
                <div className={styles.errorMessage}>{errorSentence(lastError, t)}</div>
                {lastError.hint !== null && (
                  <div className={styles.errorHint}>{lastError.hint}</div>
                )}
              </>
            ) : (
              loadError !== null && (
                <div className={styles.errorMessage}>
                  {t('git.error.loadFail', { error: loadError })}
                </div>
              )
            )}
          </div>
          {lastError !== null && (
            <>
              <div className={styles.errorActions}>
                {followUp === 'mergeRemote' && (
                  <button
                    type="button"
                    className={styles.linkButton}
                    onClick={() => void runPull('merge')}
                  >
                    {t('git.action.mergeRemote')}
                  </button>
                )}
                {followUpPublish && publishControl(styles.linkButton)}
                {stderr !== null && stderr !== '' && (
                  <button
                    type="button"
                    className={styles.linkButton}
                    aria-expanded={detailsOpen}
                    aria-controls={stderrId}
                    onClick={() => setDetailsOpen((was) => !was)}
                  >
                    {t('git.error.details')}
                  </button>
                )}
                <button
                  type="button"
                  className={styles.linkButton}
                  onClick={() => {
                    setDetailsOpen(false);
                    dismissError();
                  }}
                >
                  {t('git.error.dismiss')}
                </button>
              </div>
              {/* Mounted whenever there is one, and HIDDEN while it is
                  closed -- which is the state the toggle is in every time
                  an error line appears. An `aria-controls` pointing at an
                  element that is not in the document names nothing, and a
                  reader offering "go to the controlled element" has nowhere
                  to go. `RefSection` keeps its list mounted for the same
                  reason. The global reset carries
                  `[hidden] { display: none !important }`. */}
              {stderr !== null && stderr !== '' && (
                <pre
                  className={styles.errorStderr}
                  id={stderrId}
                  hidden={!detailsOpen}
                >
                  {stderr}
                </pre>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}
