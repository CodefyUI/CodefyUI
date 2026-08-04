import { useI18n, type TranslationKey } from '../../i18n';
import { useTabStore } from '../../store/tabStore';
import { useToastStore } from '../../store/toastStore';
import { isBypassable } from '../../utils';
import { prompt } from '../../utils/dialog';
import { checkCollapse, subgraphIdOf, type CollapseFailure } from '../../utils/subgraph';
import styles from './NodeContextMenu.module.css';

export interface ContextMenuPosition {
  nodeId: string;
  x: number;
  y: number;
}

interface MenuItem {
  label: string;
  action: () => void;
  color?: string;
  dividerAfter?: boolean;
}

interface NodeContextMenuProps {
  position: ContextMenuPosition;
  items: MenuItem[];
  onClose: () => void;
}

const NOTE_COLORS = [
  { label: 'Yellow', value: '#3d3d1a' },
  { label: 'Blue', value: '#1a2d3d' },
  { label: 'Green', value: '#1a3d1a' },
  { label: 'Red', value: '#3d1a1a' },
  { label: 'Purple', value: '#2d1a3d' },
  { label: 'Gray', value: '#2a2a2a' },
];

export function NodeContextMenu({ position, items, onClose }: NodeContextMenuProps) {
  return (
    <>
      {/* Backdrop */}
      <div
        className={styles.backdrop}
        onClick={onClose}
        onContextMenu={(e) => { e.preventDefault(); onClose(); }}
      />
      {/* Menu — left/top are dynamic from position */}
      <div
        className={styles.menu}
        style={{ left: position.x, top: position.y }}
      >
        {items.map((item, i) => (
          <div key={i}>
            <button type="button"
              onClick={() => { item.action(); onClose(); }}
              className={styles.menuItem}
              style={{ color: item.color ?? '#ccc' }}
            >
              {item.label}
            </button>
            {item.dividerAfter && <div className={styles.divider} />}
          </div>
        ))}
      </div>
    </>
  );
}

export function useNodeContextMenuItems(
  nodeId: string,
  callbacks: {
    onDelete: (id: string) => void;
    onRename: (id: string) => void;
    onDuplicate: (id: string) => void;
    onOpenDetails: (id: string) => void;
  },
) {
  const { t } = useI18n();
  const toggleNodeBypass = useTabStore((s) => s.toggleNodeBypass);
  const collapseSelection = useTabStore((s) => s.collapseSelectionToSubgraph);
  const expandInstanceNode = useTabStore((s) => s.expandSubgraphInstance);
  const enterSubgraph = useTabStore((s) => s.enterSubgraph);
  const addToast = useToastStore((s) => s.addToast);
  const node = useTabStore((s) => {
    const tab = s.tabs.find((tt) => tt.id === s.activeTabId);
    return tab?.nodes.find((n) => n.id === nodeId);
  });
  const selectedCount = useTabStore((s) => {
    const tab = s.tabs.find((tt) => tt.id === s.activeTabId);
    return tab ? tab.nodes.filter((n) => n.selected).length : 0;
  });
  const bypassable = isBypassable(node);
  const bypassed = node?.data.bypassed === true;
  const isInstance = subgraphIdOf(node?.data?.type) !== null;

  /**
   * Say WHY a collapse was refused, in terms the user can act on.
   *
   * The non-convex refusal names the nodes that have to join the selection,
   * and it names them by LABEL: node ids are `crypto.randomUUID()`s, so the
   * message used to read "...: 3f2b1a44-9c1b-4a2f-9b6e-2d0f1a4c8e11", which
   * is not something anyone can find on a canvas. The whole justification for
   * refusing (rather than collapsing into a block that draws a loop the graph
   * does not have) is that the user can fix it -- which needs a name.
   */
  const reportRefusal = (failure: CollapseFailure) => {
    if (failure.reason !== 'not-convex') {
      addToast(t(`subgraph.collapse.${failure.reason}` as TranslationKey), 'error');
      return;
    }
    const nodes = useTabStore.getState().getActiveTab().nodes;
    const labelById = new Map(nodes.map((n) => [n.id, n.data?.label]));
    addToast(
      t('subgraph.collapse.notConvex', {
        // Falls back to the id: a node with no label at all is still better
        // named by something than by nothing.
        nodes: failure.blockers.map((id) => labelById.get(id) || id).join(', '),
      }),
      'error',
    );
    // Put the blockers IN the selection, so acting on the message is one
    // retry rather than a hunt. Selection is not part of an undo snapshot,
    // so this changes nothing the user has to unwind.
    const blocking = new Set(failure.blockers);
    useTabStore.getState().setNodes(
      nodes.map((n) => (blocking.has(n.id) ? { ...n, selected: true } : n)),
    );
  };

  const runCollapse = async () => {
    const tab = useTabStore.getState().getActiveTab();
    // Refuse BEFORE asking for a name. `checkCollapse` is the same guard
    // chain `collapseSelection` runs, so the verdict cannot drift -- and
    // asking a user to name a block that is about to be refused anyway is a
    // worse experience than the refusal on its own. readOnly is checked here
    // rather than inside `checkCollapse` because it lives on the TAB, not on
    // the nodes the pure helper is given.
    if (tab.readOnly) {
      addToast(t('subgraph.collapse.read-only'), 'error');
      return;
    }
    const selectedIds = tab.nodes.filter((n) => n.selected).map((n) => n.id);
    const check = checkCollapse(tab.nodes, tab.edges, selectedIds);
    if (!check.ok) {
      reportRefusal(check);
      return;
    }

    // Named at collapse time: the breadcrumb's click-to-rename is only
    // discoverable once you are already inside the block, so every block
    // used to be called "Subgraph" until the user stumbled on it.
    const entered = await prompt({
      title: t('subgraph.collapse.namePrompt'),
      // The literal default, NOT a translated one: this becomes the block's
      // stored name, i.e. graph data that travels to other people and to the
      // server, so it must not depend on the author's UI language. It is the
      // same fallback `collapseSelection` applies when no name is given.
      defaultValue: 'Subgraph',
      placeholder: t('subgraph.rename.label'),
    });
    if (entered === null) return; // cancelled -- the graph is untouched

    // The dialog is awaited, so the graph could have changed underneath us
    // (another pane, a plugin, an undo). Re-running the real action is the
    // only way to be sure, and it re-checks everything anyway.
    const result = collapseSelection(entered.trim() || undefined);
    if (!result.ok) reportRefusal(result);
  };

  return [
    // Subgraph entries come first: they act on the SELECTION or open
    // something, so they sit above the per-node edit actions.
    ...(selectedCount >= 2
      ? [{
          label: t('contextMenu.collapseToSubgraph'),
          action: runCollapse,
          dividerAfter: true,
        }]
      : []),
    ...(isInstance
      ? [
          {
            label: t('contextMenu.enterSubgraph'),
            action: () => enterSubgraph(nodeId),
            dividerAfter: false,
          },
          {
            label: t('contextMenu.expandSubgraph'),
            action: () => expandInstanceNode(nodeId),
            dividerAfter: true,
          },
        ]
      : []),
    // First, and divided off from the edit actions: it is the only entry that
    // opens something rather than changing the graph.
    {
      label: t('contextMenu.openDetails'),
      action: () => callbacks.onOpenDetails(nodeId),
      dividerAfter: true,
    },
    { label: t('contextMenu.rename'), action: () => callbacks.onRename(nodeId), dividerAfter: false },
    { label: t('contextMenu.duplicate'), action: () => callbacks.onDuplicate(nodeId), dividerAfter: !bypassable },
    // Bypass (core#128) sits with the edit actions, not next to Delete: it
    // changes what runs, it does not remove anything. Absent entirely for the
    // node kinds bypass does not apply to (notes, Start, presets) rather than
    // shown greyed out — an entry that can never do anything is noise.
    ...(bypassable
      ? [{
          label: bypassed ? t('contextMenu.unbypass') : t('contextMenu.bypass'),
          action: () => toggleNodeBypass(nodeId),
          color: bypassed ? '#22d3ee' : undefined,
          dividerAfter: true,
        }]
      : []),
    { label: t('contextMenu.delete'), action: () => callbacks.onDelete(nodeId), color: '#F44336' },
  ];
}

export function useNoteContextMenuItems(
  nodeId: string,
  callbacks: {
    onDelete: (id: string) => void;
  },
) {
  const { t } = useI18n();
  const bindNoteToNearest = useTabStore((s) => s.bindNoteToNearest);
  const unbindNote = useTabStore((s) => s.unbindNote);
  const updateNoteData = useTabStore((s) => s.updateNoteData);

  const tab = useTabStore((s) => s.tabs.find((tab) => tab.id === s.activeTabId));
  const note = tab?.nodes.find((n) => n.id === nodeId);
  const isBound = !!note?.data.boundToNodeId;

  const colorItems: MenuItem[] = NOTE_COLORS.map((c) => ({
    label: `  ${c.label}`,
    action: () => updateNoteData(nodeId, { noteColor: c.value }),
    color: c.value === note?.data.noteColor ? '#fff' : '#999',
  }));

  return [
    // Bind / Unbind
    isBound
      ? { label: t('note.unbind'), action: () => unbindNote(nodeId), dividerAfter: false }
      : { label: t('note.bind'), action: () => bindNoteToNearest(nodeId), dividerAfter: false },
    // Color submenu (inline)
    { label: t('note.changeColor'), action: () => {}, color: '#888', dividerAfter: false },
    ...colorItems,
    { label: '', action: () => {}, dividerAfter: true },
    // Delete
    { label: t('contextMenu.delete'), action: () => callbacks.onDelete(nodeId), color: '#F44336' },
  ];
}
