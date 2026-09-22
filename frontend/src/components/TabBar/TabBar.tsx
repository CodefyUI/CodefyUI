import { useState, useCallback, useEffect, useRef } from 'react';
import { useTabStore, tabHasContent, tabNodeCount } from '../../store/tabStore';
import { isAnyModalOpen } from '../../store/modalState';
import { useI18n } from '../../i18n';
import { confirm } from '../../utils/dialog';
import styles from './TabBar.module.css';

/**
 * The keys a focused tab answers (#402). Each one stops at the tab, default
 * and all: on `document`, React Flow deletes the canvas selection on Delete
 * and the shortcut hook opens the selected node's details on Enter; on
 * `window`, React Flow arms canvas panning on Space; and Space and the arrows
 * would scroll. F2 starts the rename. Nothing else binds it today, but a key
 * the strip answers must not also reach a handler on `document`.
 */
const TAB_KEYS = new Set(['ArrowLeft', 'ArrowRight', 'Home', 'End', 'Enter', ' ', 'Delete', 'F2']);

/**
 * The tab the Tab key lands on: the active one, or the first when none is. A
 * plugin can open tabs without activating any, and a plugin's tab has no other
 * way in -- the Graphs panel raises tabs by file.
 */
function tabStopId(tabs: readonly { id: string }[], activeTabId: string): string | undefined {
  return tabs.some((tab) => tab.id === activeTabId) ? activeTabId : tabs[0]?.id;
}

/**
 * A mouse press on a tab or on its close button takes no focus (#402). Before
 * the strip took the keyboard a tab could not hold focus, so a press left focus
 * on the page body and the next Delete, Enter or Space went to the canvas --
 * Delete removed the selected nodes. A mouse user keeps exactly that; only the
 * keyboard puts focus on a tab.
 *
 * It has to be the press, since that is where the browser moves focus and a
 * click does not always follow one: the button can be released off the tab,
 * it can be the right button, and Safari and macOS Firefox hand a press on the
 * close button to the tab around it. Whatever had focus still loses it, as it
 * did to any press on the strip, so a rename box or a panel field commits on
 * blur. A press in the rename box itself is left alone: it places the caret.
 *
 * The strip has no drag to lose. A draggable tab would have to let its press
 * through: Firefox starts no drag from a press whose default was prevented.
 */
function refusePointerFocus(e: React.MouseEvent): void {
  // A real press counts clicks (1+); a screen reader's or a script's reports 0 and keeps its focus.
  if (e.detail === 0) return;
  if (e.target instanceof HTMLInputElement) return;
  e.preventDefault();
  (document.activeElement as HTMLElement | null)?.blur();
}

/**
 * The workspace tab strip.
 *
 * Keyboard (#402): the tabs form a tablist with a roving tabindex, like the
 * sidebar rail's, so the strip is a single Tab stop -- the active tab. Left
 * and Right move focus between tabs and wrap, Home/End jump to the ends,
 * Enter or Space switches to the focused tab, F2 renames it and Delete
 * closes it.
 */
export function TabBar() {
  const tabs = useTabStore((s) => s.tabs);
  const activeTabId = useTabStore((s) => s.activeTabId);
  const addTab = useTabStore((s) => s.addTab);
  const removeTab = useTabStore((s) => s.removeTab);
  const setActiveTab = useTabStore((s) => s.setActiveTab);
  const renameTab = useTabStore((s) => s.renameTab);
  const { t } = useI18n();

  const [editingId, setEditingId] = useState<string | null>(null);
  const [editingName, setEditingName] = useState('');
  // The rendered tabs by id, for the keys that move focus between them; and
  // the add button, which is all there is to focus once the last tab closes.
  const tabRefs = useRef(new Map<string, HTMLDivElement>());
  const addBtnRef = useRef<HTMLButtonElement>(null);
  // Whether F2 started the open rename (#402). Enter or Escape then hands
  // focus back to its tab; a rename a double-click started ends as it always
  // has, with focus on the page body.
  const renameByKeyboard = useRef(false);
  // The tab to focus once its rename box is gone -- not before: moving focus
  // while the box is still there blurs it, and its blur commits, so Escape
  // would save what it was meant to throw away.
  const refocusAfterRename = useRef<string | null>(null);

  const startRename = useCallback((id: string, currentName: string, byKeyboard = false) => {
    renameByKeyboard.current = byKeyboard;
    setEditingId(id);
    setEditingName(currentName);
  }, []);

  useEffect(() => {
    if (editingId !== null || refocusAfterRename.current === null) return;
    tabRefs.current.get(refocusAfterRename.current)?.focus();
    refocusAfterRename.current = null;
  }, [editingId]);

  const commitRename = useCallback(() => {
    if (editingId && editingName.trim()) {
      renameTab(editingId, editingName.trim());
    }
    setEditingId(null);
  }, [editingId, editingName, renameTab]);

  // The close button's handler, and Delete's (#402): one path, so a keyboard
  // close meets exactly the confirms a click does. Resolves to whether the
  // tab is gone.
  const handleClose = useCallback(
    async (e: React.SyntheticEvent, id: string): Promise<boolean> => {
      e.stopPropagation();
      const tab = tabs.find((t) => t.id === id);
      // Only reachable if the tab vanished between render and click.
      /* v8 ignore start */
      if (!tab) return false;
      /* v8 ignore stop */
      const hasContent = tabHasContent(tab);
      // The graph warning, reused by both branches below so a running tab with
      // a graph in it is still told what it is about to lose.
      const lossMessage = hasContent
        ? t('tabs.close.confirmMessage', { count: tabNodeCount(tab) })
        : undefined;
      if (tab.status === 'running') {
        // A running tab already asked before #331, and its question is the
        // stronger one (closing kills the run too). Chaining the content
        // confirm after it would mean two dialogs for one click, so the run
        // question just carries the graph warning as its body.
        const ok = await confirm({
          title: t('tabs.closeRunning'),
          message: lossMessage,
          variant: 'danger',
        });
        if (!ok) return false;
      } else if (hasContent) {
        // #331: one misclick used to discard a whole graph with no undo --
        // `removeTab` drops the tab's undo/redo stacks along with it, so there
        // was nothing left to undo from. An empty tab still closes silently:
        // asking about nothing is the noise that trains people to click
        // through the dialog that matters.
        const ok = await confirm({
          title: t('tabs.close.confirmTitle', { name: tab.name }),
          message: lossMessage,
          confirmText: t('tabs.close.confirmButton'),
          variant: 'danger',
        });
        if (!ok) return false;
      }
      removeTab(id);
      return true;
    },
    [tabs, removeTab, t]
  );

  // Where focus goes after a keyboard close: the tab that took the closed
  // one's place (the next, or the previous when it was last) -- the same tab
  // `removeTab` activates when the closed tab was the active one -- or the add
  // button when none is left. Left alone, focus would go down with the closed
  // tab to the page body. This runs once the close has resolved, after the
  // confirm dialog, if any, has handed focus back to the tab being removed.
  const focusAfterClose = useCallback((index: number) => {
    const open = useTabStore.getState().tabs;
    const next = open[Math.min(index, open.length - 1)];
    (next ? tabRefs.current.get(next.id) : addBtnRef.current)?.focus();
  }, []);

  // Manual activation: the arrows, Home and End move focus only, and Enter or
  // Space switches. The sidebar rail switches as focus moves, but its panels
  // are light; switching a workspace tab re-renders the whole canvas, and the
  // WAI-ARIA guidance is manual activation whenever showing the panel is not
  // instant.
  const handleTabKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLDivElement>, index: number) => {
      // A modal owns the keyboard even where it left focus behind (#475): the
      // shortcuts sheet takes none, and a Delete must not close a tab under it.
      if (isAnyModalOpen()) return;
      // Keys typed into the rename box bubble up through its tab, and they are
      // the box's: Delete deletes a character and Space types one. The close
      // button -- the one button in a tab -- is out of the Tab order, but a
      // screen reader or a script can still put focus on it, and its Delete
      // closes its tab as a Delete on the tab would. Left to bubble, it reached
      // React Flow on `document`, which deleted the canvas selection. Its other
      // keys stay its own: Enter and Space press it.
      const onTab = e.target === e.currentTarget;
      const onCloseButton = e.target instanceof HTMLButtonElement && e.key === 'Delete';
      if (!(onTab || onCloseButton) || !TAB_KEYS.has(e.key)) return;
      // Plain keys only. Alt+Left/Right are the browser's Back and Forward,
      // and Delete or Enter with Ctrl, Shift or Meta is another command. Shift
      // is refused with the rest; Shift+Tab is untouched, as Tab is never ours.
      if (e.altKey || e.ctrlKey || e.metaKey || e.shiftKey) return;
      e.preventDefault();
      e.stopPropagation();

      if (e.key === 'Delete') {
        void handleClose(e, tabs[index].id).then((closed) => {
          if (closed) focusAfterClose(index);
        });
        return;
      }
      if (e.key === 'Enter' || e.key === ' ') {
        setActiveTab(tabs[index].id);
        return;
      }
      if (e.key === 'F2') {
        // The keyboard's double-click, and exactly the same rename: a
        // read-only tab can be renamed either way.
        startRename(tabs[index].id, tabs[index].name, true);
        return;
      }
      const count = tabs.length;
      let next: number;
      if (e.key === 'ArrowRight') next = (index + 1) % count;
      else if (e.key === 'ArrowLeft') next = (index - 1 + count) % count;
      else if (e.key === 'Home') next = 0;
      else next = count - 1; // End
      tabRefs.current.get(tabs[next].id)?.focus();
    },
    [tabs, handleClose, focusAfterClose, setActiveTab, startRename],
  );

  const stopId = tabStopId(tabs, activeTabId);

  return (
    <div className={styles.bar}>
      {/* The tablist is the scrolling row, not the bar: a tablist may only own
          tabs, and the add button beside it is not one. */}
      <div className={styles.tabsScroll} role="tablist" aria-label={t('tabBar.aria')}>
        {tabs.map((tab, index) => {
          const isActive = tab.id === activeTabId;
          const isRunning = tab.status === 'running';
          const isEditing = editingId === tab.id;
          // A plugin-opened tab says so on hover rather than in the strip:
          // the tab is 180px wide and already carries a name, a running dot
          // and a close button.
          const sourceTitle = tab.source
            ? t('tabBar.sourceTitle', { plugin: tab.source.pluginId })
            : undefined;
          // The tab's accessible name (#402). Read off the content it would
          // include the close button's own name, so it is set instead -- and
          // then has to say what the tooltip and the badge show. `plugin`
          // goes in before `name` because `t()` fills the slots in order and
          // a name is the user's own text, which may itself hold "{plugin}".
          let label = tab.name;
          if (tab.source) {
            label = t('tabBar.source.aria', { plugin: tab.source.pluginId, name: label });
          }
          if (tab.readOnly) label = t('tabBar.readOnly.aria', { name: label });

          return (
            <div
              key={tab.id}
              ref={(el) => {
                if (el) tabRefs.current.set(tab.id, el);
                else tabRefs.current.delete(tab.id);
              }}
              id={`workspace-tab-${tab.id}`}
              role="tab"
              aria-selected={isActive}
              aria-label={label}
              tabIndex={tab.id === stopId ? 0 : -1}
              // Presses on the close button bubble up to here as well.
              onMouseDown={refusePointerFocus}
              onClick={() => setActiveTab(tab.id)}
              onDoubleClick={() => startRename(tab.id, tab.name)}
              onKeyDown={(e) => handleTabKeyDown(e, index)}
              className={styles.tab}
              title={sourceTitle}
              style={{
                background: isActive ? 'var(--surface-raised)' : 'transparent',
                borderBottom: isActive ? '2px solid var(--accent)' : '2px solid transparent',
                color: isActive ? 'var(--text-primary)' : 'var(--text-muted)',
                // Left as numeric literals: TabBar.test.tsx pins the exact
                // string ('600'/'400') and isn't in this migration's file
                // list to update.
                fontWeight: isActive ? 600 : 400,
              }}
            >
              {/* Running indicator. #FFC107 already equals --status-running
                  exactly (no contrast/colour bug here); left as a literal
                  because TabBar.test.tsx pins the hex substring in
                  boxShadow and the test file is outside this migration's
                  scope. */}
              {isRunning && (
                <span
                  className={styles.runningDot}
                  style={{
                    background: '#FFC107',
                    boxShadow: '0 0 4px #FFC107',
                  }}
                />
              )}

              {/* Tab name */}
              {isEditing ? (
                <input
                  autoFocus
                  value={editingName}
                  onChange={(e) => setEditingName(e.target.value)}
                  onBlur={commitRename}
                  onKeyDown={(e) => {
                    if ((e.key === 'Enter' || e.key === 'Escape') && renameByKeyboard.current) {
                      refocusAfterRename.current = tab.id;
                    }
                    if (e.key === 'Enter') commitRename();
                    if (e.key === 'Escape') setEditingId(null);
                  }}
                  onClick={(e) => e.stopPropagation()}
                  // Double-clicking a word selects it. Let through, it would
                  // reach the tab and start the rename over from the saved name.
                  onDoubleClick={(e) => e.stopPropagation()}
                  className={styles.editInput}
                />
              ) : (
                <span className={styles.tabName}>{tab.name}</span>
              )}

              {/* Read-only badge (#341). One word, after the name, so the
                  name is what gets the room and the badge is what gets
                  clipped last. `readOnly` means Save is refused and plugin
                  writes are refused -- it does not mean the tab is frozen,
                  so rename and close are untouched. */}
              {tab.readOnly && (
                <span className={styles.badge}>{t('tabBar.readOnly')}</span>
              )}

              {/* Close button. On every tab, including the last one: closing
                  it leaves the workspace with no tab open, which is the
                  welcome screen, not a broken state. Hiding it used to make a
                  single leftover tab permanent -- the one case where "I am
                  done with this" had no answer. The confirms in `handleClose`
                  are what keep an accidental click from costing a graph.
                  A real button, but out of the Tab order (#402): Delete
                  closes the focused tab, so a stop of its own on every tab
                  would only lengthen the way through the strip. It stays
                  inside the tab because a tablist may own only tabs: beside
                  the tab, it would be a child of the tablist that is not one.
                  A press on it takes no focus -- `refusePointerFocus`, on the
                  tab -- so a cancelled confirm hands focus back to the page
                  body.
                  A press with no click count -- a screen reader's, or Enter
                  or Space on the button -- leaves focus on the button, which
                  goes with the tab; so focus follows a keyboard close to the
                  neighbour instead of falling to the page body. */}
              <button
                type="button"
                tabIndex={-1}
                aria-label={t('tabBar.close.aria', { name: tab.name })}
                onClick={(e) => {
                  const pressedWithoutPointer = e.detail === 0;
                  void handleClose(e, tab.id).then((closed) => {
                    if (closed && pressedWithoutPointer) focusAfterClose(index);
                  });
                }}
                className={styles.closeBtn}
              >
                ×
              </button>
            </div>
          );
        })}
      </div>

      {/* Add tab button. Named with its tooltip's words (#402): named by its
          content it would be "+", and it is where focus goes once a keyboard
          close has taken the last tab. */}
      <button type="button"
        ref={addBtnRef}
        onClick={() => addTab()}
        title={t('tabs.add')}
        aria-label={t('tabs.add')}
        className={styles.addBtn}
      >
        +
      </button>
    </div>
  );
}
