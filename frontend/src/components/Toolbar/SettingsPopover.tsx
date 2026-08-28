import { useEffect, useRef, useState } from 'react';
import { useTabStore } from '../../store/tabStore';
import { useUIStore } from '../../store/uiStore';
import { useToastStore } from '../../store/toastStore';
import { usePackStore } from '../../store/packStore';
import { useI18n } from '../../i18n';
import {
  resetWeights,
  fetchDevices,
  fetchCodexStatus,
  startCodexLogin,
  logoutCodex,
  type CodexAuthStatus,
  type DeviceInfo,
  type PackSummary,
} from '../../api/rest';
import { computeSegmentNodes } from '../../utils/segmentPath';
import { generateId } from '../../utils';
import { confirm } from '../../utils/dialog';
import { catalogKey } from '../PackCenter/packStatus';
import { HealthSection } from './HealthSection';
import { SettingsRow as Row } from './SettingsRow';
import styles from './SettingsPopover.module.css';

interface Props {
  open: boolean;
  onClose: () => void;
  triggerRef: React.RefObject<HTMLButtonElement | null>;
}

type PackStoreState = ReturnType<typeof usePackStore.getState>;

// Module-scope selectors, so each subscription compares the SAME function's
// output frame to frame. The job is narrowed to the one thing this row says
// about it — WHICH pack is being installed — because an install writes its
// log, its per-item bytes and its step list on every long-poll turn, and the
// settings panel must not re-render for any of that.
const selectPacks = (state: PackStoreState): PackSummary[] => state.packs;
const selectPacksLoaded = (state: PackStoreState): boolean => state.loaded;
const selectPacksUnsupported = (state: PackStoreState): boolean => state.unsupported;
const selectInstallingPackId = (state: PackStoreState): string | null =>
  state.job !== null && state.job.status === 'running' ? state.job.packId : null;

export function SettingsPopover({ open, onClose, triggerRef }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const { t } = useI18n();

  // Per-tab settings (Recording / Training)
  const activeTab = useTabStore((s) => s.tabs.find((tab) => tab.id === s.activeTabId)!);
  const recording = activeTab.recordOutputs ?? true;
  const verbose = activeTab.verboseMode ?? false;
  const persistent = activeTab.weightsPersistent ?? true;
  const backward = activeTab.backwardMode ?? false;
  const autoBackward = activeTab.autoBackward ?? false;
  const seed = activeTab.seed ?? null;
  const deterministic = activeTab.deterministic ?? false;
  const graphId = activeTab.graphId ?? '';
  const activeSegment = activeTab.activeSegment;

  const toggleRecord = useTabStore((s) => s.toggleRecord);
  const toggleVerbose = useTabStore((s) => s.toggleVerbose);
  const togglePersistWeights = useTabStore((s) => s.togglePersistWeights);
  const toggleBackward = useTabStore((s) => s.toggleBackward);
  const toggleAutoBackward = useTabStore((s) => s.toggleAutoBackward);
  const setSeed = useTabStore((s) => s.setSeed);
  const toggleDeterministic = useTabStore((s) => s.toggleDeterministic);
  const setActiveSegment = useTabStore((s) => s.setActiveSegment);
  const addSegmentGroup = useTabStore((s) => s.addSegmentGroup);
  const removeSegmentGroup = useTabStore((s) => s.removeSegmentGroup);

  // Global UI settings (Editor section)
  const gridSnapEnabled = useUIStore((s) => s.gridSnapEnabled);
  const tooltipsEnabled = useUIStore((s) => s.tooltipsEnabled);
  const beginnerMode = useUIStore((s) => s.beginnerMode);
  const edgeStyle = useUIStore((s) => s.edgeStyle);
  const toggleGridSnap = useUIStore((s) => s.toggleGridSnap);
  const toggleTooltips = useUIStore((s) => s.toggleTooltips);
  const toggleBeginnerMode = useUIStore((s) => s.toggleBeginnerMode);
  const setEdgeStyle = useUIStore((s) => s.setEdgeStyle);

  // Global execution device. Options come from the backend (CPU + any GPU
  // backend present); fall back to CPU-only if the fetch fails.
  const globalDevice = useUIStore((s) => s.globalDevice);
  const setGlobalDevice = useUIStore((s) => s.setGlobalDevice);
  const [devices, setDevices] = useState<DeviceInfo[]>([
    { value: 'cpu', label: 'CPU', detail: '', available: true },
  ]);
  useEffect(() => {
    let cancelled = false;
    fetchDevices()
      .then((r) => {
        if (!cancelled && r.devices.length > 0) setDevices(r.devices);
      })
      .catch(() => {
        /* keep the CPU-only fallback */
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // Optional packs. Everything here is READ from the pack store: an install
  // is a multi-gigabyte download that outlives this popover, so the panel
  // only ever shows what the store already knows.
  const openPackCenter = useUIStore((s) => s.openPackCenter);
  const packs = usePackStore(selectPacks);
  const packsLoaded = usePackStore(selectPacksLoaded);
  const packsUnsupported = usePackStore(selectPacksUnsupported);
  const installingPackId = usePackStore(selectInstallingPackId);

  // Gated on `open` rather than on mount, because this component is mounted
  // for the whole session: a bare mount effect would fire during boot, win
  // the race against the sidebar's `usePackCatalogBootstrap`, and cost that
  // read the two things only IT does — adopting an install that outlived the
  // last page load, and reporting a restart that finished while this page did
  // not exist. Guarded either way, so opening Settings never re-reads a
  // catalog that is already here.
  useEffect(() => {
    if (!open) return;
    const state = usePackStore.getState();
    if (!state.loaded && !state.loading) void state.refresh();
  }, [open]);

  const addToast = useToastStore((s) => s.addToast);

  const [codexStatus, setCodexStatus] = useState<CodexAuthStatus>({ status: 'logged_out' });
  const [codexBusy, setCodexBusy] = useState(false);
  useEffect(() => {
    let cancelled = false;
    fetchCodexStatus()
      .then((status) => {
        if (!cancelled) setCodexStatus(status);
      })
      .catch(() => {
        /* keep logged_out fallback */
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (codexStatus.status !== 'pending') return undefined;
    const id = window.setInterval(() => {
      fetchCodexStatus()
        .then((status) => setCodexStatus(status))
        .catch(() => undefined);
    }, 2000);
    return () => window.clearInterval(id);
  }, [codexStatus.status]);

  // Compare segment selection state
  const selected = activeTab.nodes.filter((n) => n.selected);
  const canCreateSegment = selected.length === 2;
  const canClearSegment = activeSegment !== null;

  // Close on outside click / Esc
  useEffect(() => {
    if (!open) return;
    const handleMouseDown = (e: MouseEvent) => {
      const target = e.target as Node;
      if (ref.current?.contains(target)) return;
      if (triggerRef.current?.contains(target)) return;
      onClose();
    };
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('mousedown', handleMouseDown);
    document.addEventListener('keydown', handleKey);
    return () => {
      document.removeEventListener('mousedown', handleMouseDown);
      document.removeEventListener('keydown', handleKey);
    };
  }, [open, onClose, triggerRef]);

  if (!open) return null;

  const handleResetWeights = async () => {
    // The reset button is disabled when !graphId, so graphId is always set here
    /* v8 ignore start */
    if (!graphId) return;
    /* v8 ignore stop */
    const ok = await confirm({
      title: t('toolbar.weights.resetAllConfirm'),
      confirmText: t('toolbar.weights.resetAll'),
      variant: 'danger',
    });
    if (!ok) return;
    try {
      const r = await resetWeights(graphId);
      addToast(`${t('toolbar.weights.resetAllOk')} (${r.evicted})`, 'success');
    } catch (e) {
      addToast(`${t('toolbar.weights.resetAll')}: ${(e as Error).message}`, 'error');
    }
  };

  const handleCodexRefresh = async () => {
    setCodexBusy(true);
    try {
      setCodexStatus(await fetchCodexStatus());
    } catch (e) {
      addToast(`${t('settings.codex.statusFailed')}: ${(e as Error).message}`, 'error');
    } finally {
      setCodexBusy(false);
    }
  };

  const handleCodexLogin = async () => {
    setCodexBusy(true);
    try {
      const { auth_url } = await startCodexLogin();
      window.open(auth_url, '_blank', 'noopener,noreferrer');
      setCodexStatus({ status: 'pending' });
      addToast(t('settings.codex.signInOpened'), 'success');
    } catch (e) {
      addToast(`${t('settings.codex.signInFailed')}: ${(e as Error).message}`, 'error');
    } finally {
      setCodexBusy(false);
    }
  };

  const handleCodexLogout = async () => {
    setCodexBusy(true);
    try {
      await logoutCodex();
      setCodexStatus({ status: 'logged_out' });
    } catch (e) {
      addToast(`${t('settings.codex.logoutFailed')}: ${(e as Error).message}`, 'error');
    } finally {
      setCodexBusy(false);
    }
  };
  const handleCompare = () => {
    if (canCreateSegment) {
      const [left, right] =
        selected[0].position.x <= selected[1].position.x
          ? [selected[0], selected[1]]
          : [selected[1], selected[0]];
      const seg = computeSegmentNodes(left.id, right.id, activeTab.nodes, activeTab.edges);
      if (seg.size === 0) {
        addToast(t('segment.noPath'), 'error');
        return;
      }
      const group = { id: generateId(), headNodeId: left.id, tailNodeId: right.id };
      addSegmentGroup(group);
      setActiveSegment(group);
      onClose();
      return;
    }
    // canClearSegment and activeSegment move together, so the false arm never runs
    /* v8 ignore start */
    if (canClearSegment && activeSegment) {
      /* v8 ignore stop */
      removeSegmentGroup(activeSegment.id);
      return;
    }
    // Reached only when the compare button is disabled (neither create nor clear possible)
    /* v8 ignore start */
    addToast(t('toolbar.compareSegment.needTwo'), 'warning');
    /* v8 ignore stop */
  };

  const codexDesc =
    codexStatus.status === 'logged_in'
      ? t('settings.codex.descLoggedIn', { email: codexStatus.email ?? 'ChatGPT' })
      : codexStatus.status === 'pending'
        ? t('settings.codex.descPending')
        : t('settings.codex.descLoggedOut');
  // What to call a pack: this build's own copy for the packs it ships, the
  // server's title for one it predates, and the bare id when the catalog in
  // hand does not list it at all (a job adopted from another tab can).
  const packName = (packId: string): string => {
    const key = catalogKey(packId, 'title');
    if (key !== null) return t(key);
    return packs.find((pack) => pack.id === packId)?.title ?? packId;
  };

  const packsDesc = packsUnsupported
    ? t('settings.packs.unsupported')
    : installingPackId !== null
      ? t('settings.packs.summaryInstalling', { pack: packName(installingPackId) })
      : packsLoaded
        ? t('settings.packs.summary', {
            installed: packs.filter((pack) => pack.status === 'installed').length,
            total: packs.length,
          })
        : // No catalog has arrived yet. "0 of 0 packs installed" would be a
          // wrong number rather than no number, so the row says what the
          // Package Center is for instead.
          t('settings.packs.desc');

  const compareLabel = canCreateSegment
    ? t('settings.compare.actionCreate')
    : canClearSegment
      ? t('settings.compare.actionClear')
      : t('settings.compare.actionDisabled');
  const compareDisabled = !canCreateSegment && !canClearSegment;

  return (
    <div ref={ref} className={styles.panel} role="dialog" aria-label={t('toolbar.settings')}>
      <div className={styles.head}>
        <h4>{t('toolbar.settings')}</h4>
        {/* Search input deferred until per-toggle filtering lands — a
            non-functional input at the top of the popover is more confusing
            than helpful. The i18n key `toolbar.settings.search` is kept for
            when it returns. */}
      </div>

      <div className={styles.body}>
        {/* ── Execution ──────────────────────────────────────────── */}
        <section className={styles.section}>
          <div className={styles.sectionTitle}>
            {t('toolbar.settings.section.execution')}
          </div>

          <Row
            name={t('settings.device.name')}
            desc={t('settings.device.desc')}
            ctrl={
              <select
                aria-label={t('settings.device.name')}
                className={styles.select}
                value={globalDevice}
                onChange={(e) => setGlobalDevice(e.target.value)}
              >
                {devices.map((d) => (
                  <option key={d.value} value={d.value}>
                    {d.detail ? `${d.label} — ${d.detail}` : d.label}
                  </option>
                ))}
              </select>
            }
          />
        </section>

        <section className={styles.section}>
          <div className={styles.sectionTitle}>
            {t('toolbar.settings.section.llm')}
          </div>

          <Row
            name={t('settings.codex.name')}
            desc={codexDesc}
            ctrl={
              <div className={styles.buttonGroup}>
                {codexStatus.status === 'logged_in' ? (
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleCodexLogout();
                    }}
                    className={styles.action}
                    disabled={codexBusy}
                  >
                    {t('settings.codex.actionSignOut')}
                  </button>
                ) : (
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleCodexLogin();
                    }}
                    className={styles.action}
                    disabled={codexBusy || codexStatus.status === 'pending'}
                  >
                    {t('settings.codex.actionSignIn')}
                  </button>
                )}
                {codexStatus.status !== 'logged_out' && (
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleCodexRefresh();
                    }}
                    className={styles.action}
                    disabled={codexBusy}
                  >
                    {t('settings.codex.actionRefresh')}
                  </button>
                )}
              </div>
            }
          />
        </section>

        {/* ── Optional packs ─────────────────────────────────────── */}
        <section className={styles.section}>
          <div className={styles.sectionTitle}>
            {t('toolbar.settings.section.packs')}
          </div>

          <Row
            name={t('settings.packs.name')}
            desc={packsDesc}
            ctrl={
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  onClose();
                  // No argument: this entry point opens the whole catalog,
                  // and passing the click event through would focus a "pack"
                  // named after a React synthetic event.
                  openPackCenter();
                }}
                className={styles.action}
              >
                {t('settings.packs.action')}
              </button>
            }
          />
        </section>

        {/* ── Recording & Inspection ─────────────────────────────── */}
        <section className={styles.section}>
          <div className={styles.sectionTitle}>
            {t('toolbar.settings.section.recording')}
          </div>

          <Row
            name={t('settings.record.name')}
            desc={t('settings.record.desc')}
            onClick={toggleRecord}
            ctrl={
              <button
                type="button"
                aria-label={t('settings.record.name')}
                aria-pressed={recording}
                className={`${styles.toggle} ${recording ? styles.on : ''}`}
                onClick={(e) => {
                  e.stopPropagation();
                  toggleRecord();
                }}
              />
            }
          />

          <Row
            name={t('settings.verbose.name')}
            desc={t('settings.verbose.desc')}
            onClick={toggleVerbose}
            ctrl={
              <button
                type="button"
                aria-label={t('settings.verbose.name')}
                aria-pressed={verbose}
                className={`${styles.toggle} ${verbose ? styles.on : ''}`}
                onClick={(e) => {
                  e.stopPropagation();
                  toggleVerbose();
                }}
              />
            }
          />

          <Row
            name={t('settings.compare.name')}
            desc={t('settings.compare.desc')}
            ctrl={
              <button
                type="button"
                disabled={compareDisabled}
                onClick={(e) => {
                  e.stopPropagation();
                  handleCompare();
                }}
                className={`${styles.action} ${canClearSegment && !canCreateSegment ? styles.accent : ''}`}
              >
                {compareLabel}
              </button>
            }
          />
        </section>

        {/* ── Training Behavior ──────────────────────────────────── */}
        <section className={styles.section}>
          <div className={styles.sectionTitle}>
            {t('toolbar.settings.section.training')}
          </div>

          <Row
            name={t('settings.persist.name')}
            desc={t('settings.persist.desc')}
            onClick={togglePersistWeights}
            ctrl={
              <button
                type="button"
                aria-label={t('settings.persist.name')}
                aria-pressed={persistent}
                className={`${styles.toggle} ${persistent ? styles.on : ''}`}
                onClick={(e) => {
                  e.stopPropagation();
                  togglePersistWeights();
                }}
              />
            }
          />

          <Row
            name={t('settings.resetWeights.name')}
            desc={t('settings.resetWeights.desc')}
            ctrl={
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  handleResetWeights();
                }}
                className={`${styles.action} ${styles.danger}`}
                disabled={!graphId}
              >
                {t('settings.resetWeights.action')}
              </button>
            }
          />

          <Row
            name={t('settings.gradients.name')}
            desc={t('settings.gradients.desc')}
            onClick={toggleBackward}
            ctrl={
              <button
                type="button"
                aria-label={t('settings.gradients.name')}
                aria-pressed={backward}
                className={`${styles.toggle} ${backward ? styles.on : ''}`}
                onClick={(e) => {
                  e.stopPropagation();
                  toggleBackward();
                }}
              />
            }
          />

          <Row
            name={t('settings.autoLoss.name')}
            desc={t('settings.autoLoss.desc')}
            disabled={!backward}
            onClick={backward ? toggleAutoBackward : undefined}
            ctrl={
              <button
                type="button"
                aria-label={t('settings.autoLoss.name')}
                aria-pressed={autoBackward}
                disabled={!backward}
                className={`${styles.toggle} ${autoBackward && backward ? styles.on : ''}`}
                onClick={(e) => {
                  e.stopPropagation();
                  // This control is disabled when !backward, so backward is always true here
                  /* v8 ignore start */
                  if (backward) toggleAutoBackward();
                  /* v8 ignore stop */
                }}
              />
            }
          />

          {/* Reproducibility (core#134). A blank field is "no seed", which
              is the historical behaviour; any number makes the run
              repeatable and, as a consequence, serial. */}
          <Row
            name={t('settings.seed.name')}
            desc={t('settings.seed.desc')}
            ctrl={
              <input
                type="number"
                className={styles.numberInput}
                aria-label={t('settings.seed.name')}
                placeholder={t('settings.seed.placeholder')}
                value={seed === null ? '' : seed}
                min={0}
                step={1}
                onClick={(e) => e.stopPropagation()}
                onChange={(e) => {
                  const raw = e.target.value.trim();
                  setSeed(raw === '' ? null : Number(raw));
                }}
              />
            }
          />

          <Row
            name={t('settings.deterministic.name')}
            desc={t('settings.deterministic.desc')}
            onClick={toggleDeterministic}
            ctrl={
              <button
                type="button"
                aria-label={t('settings.deterministic.name')}
                aria-pressed={deterministic}
                className={`${styles.toggle} ${deterministic ? styles.on : ''}`}
                onClick={(e) => {
                  e.stopPropagation();
                  toggleDeterministic();
                }}
              />
            }
          />
        </section>

        {/* ── Editor ─────────────────────────────────────────────── */}
        <section className={styles.section}>
          <div className={styles.sectionTitle}>
            {t('toolbar.settings.section.editor')}
          </div>

          <Row
            name={t('settings.gridSnap.name')}
            desc={t('settings.gridSnap.desc')}
            onClick={toggleGridSnap}
            ctrl={
              <button
                type="button"
                aria-label={t('settings.gridSnap.name')}
                aria-pressed={gridSnapEnabled}
                className={`${styles.toggle} ${gridSnapEnabled ? styles.on : ''}`}
                onClick={(e) => {
                  e.stopPropagation();
                  toggleGridSnap();
                }}
              />
            }
          />

          <Row
            name={t('settings.tooltips.name')}
            desc={t('settings.tooltips.desc')}
            onClick={toggleTooltips}
            ctrl={
              <button
                type="button"
                aria-label={t('settings.tooltips.name')}
                aria-pressed={tooltipsEnabled}
                className={`${styles.toggle} ${tooltipsEnabled ? styles.on : ''}`}
                onClick={(e) => {
                  e.stopPropagation();
                  toggleTooltips();
                }}
              />
            }
          />

          <Row
            name={t('settings.nodeMode.name')}
            desc={t('settings.nodeMode.desc')}
            ctrl={
              <div className={styles.seg} role="group" aria-label={t('settings.nodeMode.name')}>
                <button
                  type="button"
                  className={`${styles.segItem} ${beginnerMode ? styles.active : ''}`}
                  onClick={(e) => {
                    e.stopPropagation();
                    if (!beginnerMode) toggleBeginnerMode();
                  }}
                >
                  {t('settings.nodeMode.basic')}
                </button>
                <button
                  type="button"
                  className={`${styles.segItem} ${!beginnerMode ? styles.active : ''}`}
                  onClick={(e) => {
                    e.stopPropagation();
                    if (beginnerMode) toggleBeginnerMode();
                  }}
                >
                  {t('settings.nodeMode.all')}
                </button>
              </div>
            }
          />

          <Row
            name={t('settings.edgeStyle.name')}
            desc={t('settings.edgeStyle.desc')}
            ctrl={
              <div className={styles.seg} role="group" aria-label={t('settings.edgeStyle.name')}>
                <button
                  type="button"
                  className={`${styles.segItem} ${edgeStyle === 'circuit' ? styles.active : ''}`}
                  onClick={(e) => {
                    e.stopPropagation();
                    setEdgeStyle('circuit');
                  }}
                >
                  {t('settings.edgeStyle.circuit')}
                </button>
                <button
                  type="button"
                  className={`${styles.segItem} ${edgeStyle === 'curve' ? styles.active : ''}`}
                  onClick={(e) => {
                    e.stopPropagation();
                    setEdgeStyle('curve');
                  }}
                >
                  {t('settings.edgeStyle.curve')}
                </button>
              </div>
            }
          />
        </section>

        {/* Last, and read-only: everything above changes how a run behaves,
            this only reports what the server already is (#193 item 2). */}
        <HealthSection />
      </div>
    </div>
  );
}
