import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import type { Node } from '@xyflow/react';
import type { NodeData } from '../../types';
import { useTabStore } from '../../store/tabStore';
import { useI18n } from '../../i18n';
import { CATEGORY_COLORS, STATUS_COLORS } from '../../styles/theme';
import { topologicalOrder } from '../../utils/topoOrder';
import { MathText } from '../shared/MathText';
import { NodeParamList } from '../shared/NodeParamList';
import {
  getNodeDetailTabs,
  type NodeDetailTabContext,
  type NodeDetailTabSpec,
} from './tabs';
import { TabErrorBoundary } from './TabErrorBoundary';
import styles from './NodeDetailModal.module.css';

/** Node kinds the detail modal has nothing useful to say about. */
const EXCLUDED_NODE_TYPES = new Set(['noteNode']);

/**
 * Calls a tab spec's `render` from inside its own component.
 *
 * This indirection is the whole reason the error boundary works. Calling
 * `spec.render(ctx)` directly in the modal's JSX would run third-party code
 * during the MODAL's render — above the boundary in the tree, where React
 * cannot catch it — so a spec that throws inline (rather than returning a
 * component that throws) would still unmount the app root.
 */
function TabBody({
  spec,
  ctx,
}: {
  spec: NodeDetailTabSpec;
  ctx: NodeDetailTabContext;
}) {
  return <>{spec.render(ctx)}</>;
}

/** True for elements that own the arrow keys (caret, value stepper, listbox). */
function isTextEntry(target: EventTarget | null): boolean {
  const el = target as HTMLElement | null;
  if (!el || typeof el.tagName !== 'string') return false;
  return (
    el.tagName === 'INPUT' ||
    el.tagName === 'TEXTAREA' ||
    el.tagName === 'SELECT' ||
    el.isContentEditable === true
  );
}

/**
 * The n8n-style Node Details View: one node, full screen, params on the left
 * and its data on the right.
 *
 * Mounted once at the app root and driven by `tab.nodeDetailNodeId`, so every
 * way in (double-click, the node context menu, Enter on a selection, the
 * modal's own prev/next arrows) is the same single store write.
 *
 * The `InspectorPanel` is untouched by this and stays the at-a-glance side
 * view; the two share their capture-reading code rather than competing.
 */
export function NodeDetailModal() {
  const nodeDetailNodeId = useTabStore(
    (s) => s.tabs.find((t) => t.id === s.activeTabId)?.nodeDetailNodeId ?? null,
  );
  if (nodeDetailNodeId === null) return null;
  return <NodeDetailModalBody nodeId={nodeDetailNodeId} />;
}

function NodeDetailModalBody({ nodeId }: { nodeId: string }) {
  const activeTab = useTabStore((s) => s.tabs.find((t) => t.id === s.activeTabId)!);
  const closeNodeDetail = useTabStore((s) => s.closeNodeDetail);
  const openNodeDetail = useTabStore((s) => s.openNodeDetail);
  const openPresetModal = useTabStore((s) => s.openPresetModal);
  const renameNode = useTabStore((s) => s.renameNode);
  const { t, tn } = useI18n();

  const nodes = activeTab.nodes;
  const edges = activeTab.edges;
  const node = nodes.find((n) => n.id === nodeId) as Node<NodeData> | undefined;

  // Seeded from the request rather than defaulted to 'inputs' and corrected by
  // the effect below: a deep-linked Stats tab that mounts one commit late
  // mounts the Inputs tab first, and the Inputs tab fetches.
  const [activeTabId, setActiveTabId] = useState(
    () => activeTab.nodeDetailTab ?? 'inputs',
  );
  const [draftName, setDraftName] = useState<string | null>(null);
  const surfaceRef = useRef<HTMLDivElement | null>(null);
  const nameInputRef = useRef<HTMLInputElement | null>(null);

  // Walk order for the arrows: what the engine would execute, minus the kinds
  // the modal cannot describe. Topological rather than selection order because
  // it reads as the data path — the sequence a lesson walks a class through.
  const navOrder = useMemo(() => {
    const navigable = nodes.filter((n) => !EXCLUDED_NODE_TYPES.has(n.type ?? ''));
    return topologicalOrder(navigable, edges);
  }, [nodes, edges]);

  const navIndex = navOrder.indexOf(nodeId);
  const prevId = navIndex > 0 ? navOrder[navIndex - 1] : null;
  const nextId =
    navIndex >= 0 && navIndex < navOrder.length - 1 ? navOrder[navIndex + 1] : null;

  const goTo = useCallback(
    (id: string | null) => {
      if (id) openNodeDetail(id);
    },
    [openNodeDetail],
  );

  // Leaving the name editor open across a node change would show the previous
  // node's draft over the new node's header.
  //
  // `requestedTab` is the deep link from `openNodeDetail(id, { tab })` (#129).
  // The effect keys on the request NONCE, not on the tab id: following "View
  // stats" from a second edge into the consumer already on screen changes
  // neither the node nor the requested tab, so an effect watching only those
  // would decide nothing had happened and strand the user on whatever tab
  // they had moved to. It still does not fight a manual tab switch, because
  // clicking a tab changes local state and leaves the store's request alone.
  const requestedTab = activeTab.nodeDetailTab;
  const requestNonce = activeTab.nodeDetailRequest;
  useEffect(() => {
    setDraftName(null);
    setActiveTabId(requestedTab ?? 'inputs');
    // `requestedTab` is read, not watched — the nonce already changes on every
    // open, and listing both would re-run twice for one request.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nodeId, requestNonce]);

  // Take focus so the key handling below has somewhere to sit and the page
  // behind the backdrop cannot be tabbed into blind — then hand it back to
  // whatever had it when the modal closes. Without the restore, a keyboard
  // user who opened the modal with Enter lands back at the top of the
  // document instead of on the node they were standing on.
  useEffect(() => {
    const previouslyFocused = document.activeElement as HTMLElement | null;
    const timer = setTimeout(() => surfaceRef.current?.focus(), 0);
    return () => {
      clearTimeout(timer);
      if (previouslyFocused && previouslyFocused.isConnected) {
        previouslyFocused.focus();
      }
    };
  }, []);

  // One window-level handler for the modal's keyboard contract. Deciding here
  // — rather than relying on a nested handler to stop propagation before the
  // event reaches window — is what keeps "Esc cancels the rename" and "Esc
  // closes the modal" from racing each other.
  const draftOpen = draftName !== null;
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        if (draftOpen) setDraftName(null);
        else closeNodeDetail();
        return;
      }
      if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
        // A number field's steppers and a text caret both own the arrows.
        if (isTextEntry(e.target)) return;
        e.preventDefault();
        goTo(e.key === 'ArrowLeft' ? prevId : nextId);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [draftOpen, closeNodeDetail, goTo, prevId, nextId]);

  useEffect(() => {
    if (draftName !== null) nameInputRef.current?.select();
  }, [draftName]);

  // The node was deleted (or the tab switched) while the modal was open.
  if (!node) return null;

  const def = node.data.definition;
  const nodeName = def?.node_name ?? node.data.type;
  const category = def?.category ?? 'Utility';
  const isPreset = Boolean(node.data.isPreset);
  const accent = isPreset ? '#D4A017' : (CATEGORY_COLORS[category] ?? '#607D8B');
  const status = node.data.executionStatus ?? 'idle';
  const statusColor = STATUS_COLORS[status];

  // Takes the value rather than reading `draftName`, so the committed text is
  // whatever the field actually holds at commit time.
  const commitName = (raw: string) => {
    const next = raw.trim();
    if (next && next !== node.data.label) renameNode(node.id, next);
    setDraftName(null);
  };

  const ctx: NodeDetailTabContext = {
    nodeId,
    node,
    runId: activeTab.lastRunId,
    nodes,
    edges,
    recordOutputs: activeTab.recordOutputs,
    outputSummaries: activeTab.outputSummaries,
    focusPort: activeTab.nodeDetailPort,
  };

  const tabs = getNodeDetailTabs(ctx);
  // A registered tab can disappear between renders (Steps before a run), so
  // resolve rather than trust the stored id.
  const current = tabs.find((tab) => tab.id === activeTabId) ?? tabs[0];

  return createPortal(
    <div
      className={styles.backdrop}
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) closeNodeDetail();
      }}
    >
      <div
        ref={surfaceRef}
        className={styles.modal}
        role="dialog"
        aria-modal="true"
        aria-label={`${t('nodeDetail.title')}: ${node.data.label}`}
        tabIndex={-1}
      >
        <div className={styles.header} style={{ borderBottomColor: accent }}>
          <div className={styles.nav}>
            <button
              type="button"
              className={styles.navBtn}
              onClick={() => goTo(prevId)}
              disabled={prevId === null}
              title={t('nodeDetail.prev')}
              aria-label={t('nodeDetail.prev')}
            >
              &lt;
            </button>
            <span className={styles.navCount}>
              {t('nodeDetail.position', {
                index: navIndex >= 0 ? navIndex + 1 : 0,
                total: navOrder.length,
              })}
            </span>
            <button
              type="button"
              className={styles.navBtn}
              onClick={() => goTo(nextId)}
              disabled={nextId === null}
              title={t('nodeDetail.next')}
              aria-label={t('nodeDetail.next')}
            >
              &gt;
            </button>
          </div>

          <span className={styles.icon} style={{ background: accent }} aria-hidden="true">
            {(nodeName || '?').slice(0, 1).toUpperCase()}
          </span>

          <div className={styles.identity}>
            {draftName === null ? (
              <button
                type="button"
                className={styles.nameBtn}
                onClick={() => setDraftName(node.data.label)}
                title={t('nodeDetail.rename')}
              >
                {node.data.label}
              </button>
            ) : (
              <input
                ref={nameInputRef}
                className={styles.nameInput}
                aria-label={t('nodeDetail.rename')}
                value={draftName}
                onChange={(e) => setDraftName(e.target.value)}
                onBlur={(e) => commitName(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    e.preventDefault();
                    commitName(e.currentTarget.value);
                  }
                }}
              />
            )}
            {draftName !== null && (
              <div className={styles.renameHint}>{t('nodeDetail.renameHint')}</div>
            )}
            <div className={styles.meta}>
              <span className={styles.metaType}>{nodeName}</span>
              <span className={styles.categoryChip} style={{ color: accent, borderColor: accent }}>
                {category}
              </span>
              {isPreset && (
                <span className={styles.presetChip} style={{ color: accent, borderColor: accent }}>
                  {t('preset.badge')}
                </span>
              )}
              {/* core#128: the canvas card is greyed out and struck through,
                  but the modal covers it — without this chip the params on
                  screen would read as though they still affect a run. */}
              {node.data.bypassed && (
                <span className={styles.bypassChip} title={t('node.bypassed.title')}>
                  {t('node.bypassed')}
                </span>
              )}
              <span className={styles.statusChip} style={{ color: statusColor, borderColor: statusColor }}>
                {t(`status.${status}` as const)}
              </span>
            </div>
          </div>

          <button
            type="button"
            className={styles.closeBtn}
            onClick={closeNodeDetail}
            title={t('nodeDetail.close')}
            aria-label={t('nodeDetail.close')}
          >
            &#215;
          </button>
        </div>

        <div className={styles.body}>
          <aside className={styles.paramColumn}>
            <div className={styles.columnTitle}>{t('nodeDetail.parameters')}</div>
            {isPreset ? (
              // A preset's params live on the nodes INSIDE it — its own
              // `definition.params` is synthesized empty, so the plain
              // no-params message would be a lie. Same branch ConfigPanel
              // takes, pointing at the same editor.
              <div className={styles.presetBlock}>
                <div className={styles.presetHint}>
                  {t('preset.nodeCount', {
                    count: node.data.presetDefinition?.nodes.length ?? 0,
                  })}
                </div>
                <button
                  type="button"
                  className={styles.presetConfigureBtn}
                  onClick={() => {
                    // One modal at a time. The preset editor sits at a lower
                    // z-index and has no Escape handler of its own, so
                    // stacking them would render it *behind* this modal and
                    // let Escape close the wrong surface.
                    closeNodeDetail();
                    openPresetModal(nodeId);
                  }}
                >
                  {t('preset.configure')}
                </button>
              </div>
            ) : def && def.params.length > 0 ? (
              <NodeParamList nodeId={nodeId} definition={def} params={node.data.params} />
            ) : (
              <div className={styles.noParams}>{t('nodeDetail.noParams')}</div>
            )}
            {def?.description && (
              // MathText, not raw text: node descriptions carry inline LaTeX
              // ($\alpha\,a + (1-\alpha)\,b$ on Lerp, for one), which the
              // config panel and the Docs tab both typeset.
              <MathText
                as="div"
                className={styles.paramFootnote}
                text={tn(nodeName, 'description', def.description)}
              />
            )}
          </aside>

          <section className={styles.dataColumn}>
            <div className={styles.tabStrip} role="tablist" aria-label={t('nodeDetail.title')}>
              {tabs.map((tab) => (
                <button
                  key={tab.id}
                  id={`node-detail-tab-${tab.id}`}
                  type="button"
                  role="tab"
                  aria-selected={current?.id === tab.id}
                  aria-controls="node-detail-tabpanel"
                  className={`${styles.tabBtn} ${current?.id === tab.id ? styles.tabActive : ''}`}
                  onClick={() => setActiveTabId(tab.id)}
                >
                  {t(tab.labelKey)}
                </button>
              ))}
            </div>
            <div
              id="node-detail-tabpanel"
              className={styles.tabPanel}
              role="tabpanel"
              aria-labelledby={current ? `node-detail-tab-${current.id}` : undefined}
            >
              {/* Registered tabs are third-party render code (#129, #131,
                  plugins). A throw here must cost the panel, not the editor. */}
              <TabErrorBoundary
                resetKey={`${nodeId}:${current?.id ?? ''}`}
                fallback={(error) => (
                  <div className={styles.emptyState}>
                    <div className={styles.emptyIcon}>!</div>
                    <div>{t('nodeDetail.tabError')}</div>
                    <div className={styles.emptyHint}>{error.message}</div>
                  </div>
                )}
              >
                {current && <TabBody spec={current} ctx={ctx} />}
              </TabErrorBoundary>
            </div>
          </section>
        </div>
      </div>
    </div>,
    document.body,
  );
}
