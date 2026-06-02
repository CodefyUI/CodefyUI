import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useTabStore } from '../../store/tabStore';
import { useI18n } from '../../i18n';
import { friendlyError } from '../../utils/errorMessages';
import { LossChart } from './LossChart';
import styles from './ResultsPanel.module.css';

const MIN_HEIGHT = 80;
const MAX_HEIGHT = 600;
const DEFAULT_HEIGHT = 200;

type PanelTab = 'log' | 'training';

function formatTimestamp(ts: number): string {
  const d = new Date(ts);
  const pad = (n: number) => String(n).padStart(2, '0');
  return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}.${String(d.getMilliseconds()).padStart(3, '0')}`;
}

const LOG_TYPE_COLORS = {
  info: '#2196F3',
  error: '#F44336',
  success: '#4CAF50',
} as const;

const LOG_TYPE_BG = {
  info: 'rgba(33,150,243,0.05)',
  error: 'rgba(244,67,54,0.08)',
  success: 'rgba(76,175,80,0.05)',
} as const;

export function ResultsPanel() {
  const activeTab = useTabStore((s) => s.tabs.find((t) => t.id === s.activeTabId)!);
  const logs = activeTab.logs;
  const clearLogs = useTabStore((s) => s.clearLogs);
  const bottomRef = useRef<HTMLDivElement>(null);
  const { t } = useI18n();

  const setSelectedNodeId = useTabStore((s) => s.setSelectedNodeId);

  const [collapsed, setCollapsed] = useState(false);
  const [panelHeight, setPanelHeight] = useState(DEFAULT_HEIGHT);
  const [panelTab, setPanelTab] = useState<PanelTab>('log');
  const [infoColWidth, setInfoColWidth] = useState(300);
  const [configHeight, setConfigHeight] = useState<number | null>(null);
  const [expandedIdx, setExpandedIdx] = useState<number | null>(null);
  const heightBeforeCollapse = useRef(DEFAULT_HEIGHT);
  const isDragging = useRef(false);
  const startY = useRef(0);
  const startHeight = useRef(0);
  const prevHadTraining = useRef(false);
  const colDragging = useRef(false);
  const rowDragging = useRef(false);

  // Parse training data from logs
  const trainingData = useMemo(() => {
    const epochs: { epoch: number; total: number; loss: number; ts: number }[] = [];
    let config: Record<string, any> | null = null;

    for (const entry of logs) {
      if (!entry.message.startsWith('__PROGRESS__:')) continue;
      try {
        const p = JSON.parse(entry.message.slice('__PROGRESS__:'.length));
        if (p.event === 'config' && p.config) {
          config = p.config;
        } else if (p.event === 'epoch') {
          epochs.push({
            epoch: p.epoch,
            total: p.total_epochs,
            loss: p.loss,
            ts: entry.timestamp,
          });
        }
      } catch { /* ignore */ }
    }
    return { epochs, config };
  }, [logs]);

  const hasTraining = trainingData.epochs.length > 0 || trainingData.config !== null;

  // Auto-switch to Training tab when first training data arrives
  useEffect(() => {
    if (hasTraining && !prevHadTraining.current) {
      setPanelTab('training');
    }
    prevHadTraining.current = hasTraining;
  }, [hasTraining]);

  // Only non-progress logs for the Log tab
  const filteredLogs = useMemo(
    () => logs.filter((e) => !e.message.startsWith('__PROGRESS__:')),
    [logs],
  );

  useEffect(() => {
    if (panelTab === 'log') {
      bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [filteredLogs, panelTab]);

  const handleResizeStart = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    isDragging.current = true;
    startY.current = e.clientY;
    startHeight.current = panelHeight;

    const onMouseMove = (ev: MouseEvent) => {
      // The move listener is attached/detached in lockstep with isDragging, so it is true here
      /* v8 ignore start */
      if (!isDragging.current) return;
      /* v8 ignore stop */
      const delta = startY.current - ev.clientY;
      const newHeight = Math.min(MAX_HEIGHT, Math.max(MIN_HEIGHT, startHeight.current + delta));
      setPanelHeight(newHeight);
      // The resize handle only renders when !collapsed, so a drag never starts collapsed
      /* v8 ignore start */
      if (collapsed) setCollapsed(false);
      /* v8 ignore stop */
    };

    const onMouseUp = () => {
      isDragging.current = false;
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mouseup', onMouseUp);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };

    document.body.style.cursor = 'ns-resize';
    document.body.style.userSelect = 'none';
    document.addEventListener('mousemove', onMouseMove);
    document.addEventListener('mouseup', onMouseUp);
  }, [panelHeight, collapsed]);

  const toggleCollapse = useCallback(() => {
    setCollapsed((prev) => {
      if (!prev) {
        heightBeforeCollapse.current = panelHeight;
      } else {
        setPanelHeight(heightBeforeCollapse.current);
      }
      return !prev;
    });
  }, [panelHeight]);

  const handleColDividerDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    colDragging.current = true;
    const startX = e.clientX;
    const startW = infoColWidth;

    const onMouseMove = (ev: MouseEvent) => {
      // The move listener is attached/detached in lockstep with colDragging, so it is true here
      /* v8 ignore start */
      if (!colDragging.current) return;
      /* v8 ignore stop */
      const delta = startX - ev.clientX;
      setInfoColWidth(Math.max(180, Math.min(600, startW + delta)));
    };

    const onMouseUp = () => {
      colDragging.current = false;
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mouseup', onMouseUp);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };

    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
    document.addEventListener('mousemove', onMouseMove);
    document.addEventListener('mouseup', onMouseUp);
  }, [infoColWidth]);

  const handleRowDividerDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    rowDragging.current = true;
    const startYPos = e.clientY;
    const startH = configHeight ?? e.currentTarget.parentElement?.querySelector('[data-config]')?.clientHeight ?? 100;

    const onMouseMove = (ev: MouseEvent) => {
      // The move listener is attached/detached in lockstep with rowDragging, so it is true here
      /* v8 ignore start */
      if (!rowDragging.current) return;
      /* v8 ignore stop */
      const delta = ev.clientY - startYPos;
      setConfigHeight(Math.max(40, startH + delta));
    };

    const onMouseUp = () => {
      rowDragging.current = false;
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mouseup', onMouseUp);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };

    document.body.style.cursor = 'ns-resize';
    document.body.style.userSelect = 'none';
    document.addEventListener('mousemove', onMouseMove);
    document.addEventListener('mouseup', onMouseUp);
  }, [configHeight]);

  return (
    <div
      className={styles.panel}
      style={{ height: collapsed ? undefined : panelHeight }}
    >
      {/* Resize handle */}
      {!collapsed && (
        <div className={styles.resizeHandle} onMouseDown={handleResizeStart} />
      )}

      {/* Panel header */}
      <div className={styles.header}>
        <div className={styles.headerLeft}>
          {/* Inner tabs */}
          <button type="button"
            className={`${styles.panelTabBtn} ${panelTab === 'log' ? styles.panelTabActive : ''}`}
            onClick={() => setPanelTab('log')}
          >
            {t('results.title')}
            {filteredLogs.length > 0 && (
              <span className={styles.countBadge}>{filteredLogs.length}</span>
            )}
          </button>
          <button type="button"
            className={`${styles.panelTabBtn} ${panelTab === 'training' ? styles.panelTabActive : ''} ${!hasTraining ? styles.panelTabDisabled : ''}`}
            onClick={() => hasTraining && setPanelTab('training')}
            disabled={!hasTraining}
          >
            {t('results.training')}
            {trainingData.epochs.length > 0 && (
              <span className={`${styles.countBadge} ${styles.countBadgeTraining}`}>
                {trainingData.epochs.length}
              </span>
            )}
          </button>
        </div>
        <div className={styles.headerRight}>
          <button type="button"
            onClick={clearLogs}
            disabled={logs.length === 0}
            className={`${styles.clearBtn} ${logs.length === 0 ? styles.clearBtnDisabled : styles.clearBtnEnabled}`}
          >
            {t('results.clear')}
          </button>
          <button type="button"
            onClick={toggleCollapse}
            className={styles.collapseBtn}
            title={collapsed ? t('results.expand') : t('results.collapse')}
            aria-label={collapsed ? t('results.expand') : t('results.collapse')}
          >
            {collapsed ? '▴' : '▾'}
          </button>
        </div>
      </div>

      {/* Tab content */}
      {!collapsed && panelTab === 'log' && (
        <div className={styles.logArea}>
          {filteredLogs.length === 0 ? (
            <div className={styles.emptyState}>{t('results.empty')}</div>
          ) : (
            filteredLogs.map((entry, i) => {
              const isError = entry.type === 'error';
              const isExpanded = expandedIdx === i;
              return (
                <div key={i}>
                  <div
                    className={`${styles.logEntry} ${isError ? styles.logEntryClickable : ''}`}
                    style={{
                      background: LOG_TYPE_BG[entry.type],
                      borderLeft: `2px solid ${LOG_TYPE_COLORS[entry.type]}`,
                    }}
                    onClick={isError ? () => setExpandedIdx(isExpanded ? null : i) : undefined}
                    title={isError ? t('results.clickToExpand') : undefined}
                  >
                    {isError && (
                      <span className={styles.expandIcon}>{isExpanded ? '\u25BE' : '\u25B8'}</span>
                    )}
                    <span className={styles.timestamp}>
                      {formatTimestamp(entry.timestamp)}
                    </span>
                    {entry.nodeId && (
                      <span
                        className={styles.nodeIdBadge}
                        style={{ cursor: 'pointer' }}
                        title={t('results.clickToHighlight')}
                        onClick={(e) => {
                          e.stopPropagation();
                          setSelectedNodeId(entry.nodeId!);
                        }}
                      >
                        {String(entry.nodeId).slice(0, 8)}
                      </span>
                    )}
                    {entry.message.startsWith('__IMAGE__:') ? (
                      <img
                        src={`data:image/png;base64,${entry.message.slice('__IMAGE__:'.length)}`}
                        alt="output"
                        className={styles.logImage}
                      />
                    ) : (
                      <span
                        className={styles.logMessage}
                        style={{ color: LOG_TYPE_COLORS[entry.type] }}
                      >
                        {entry.message}
                      </span>
                    )}
                  </div>
                  {isError && isExpanded && (
                    <div className={styles.expandedDetail}>
                      {friendlyError(entry.message)}
                    </div>
                  )}
                </div>
              );
            })
          )}
          <div ref={bottomRef} />
        </div>
      )}

      {!collapsed && panelTab === 'training' && (
        <div className={styles.logArea}>
          {!hasTraining ? (
            <div className={styles.emptyState}>{t('results.trainingEmpty')}</div>
          ) : (
            <div className={styles.trainingContent}>
              {/* Training summary stats */}
              {trainingData.epochs.length > 0 && (
                <div className={styles.trainingSummary}>
                  <div className={styles.summaryItem}>
                    <span className={styles.summaryLabel}>{t('results.epoch')}</span>
                    <span className={styles.summaryValue}>
                      {trainingData.epochs[trainingData.epochs.length - 1].epoch} / {trainingData.epochs[trainingData.epochs.length - 1].total}
                    </span>
                  </div>
                  <div className={styles.summaryItem}>
                    <span className={styles.summaryLabel}>{t('results.currentLoss')}</span>
                    <span className={styles.summaryValue}>
                      {trainingData.epochs[trainingData.epochs.length - 1].loss.toFixed(4)}
                    </span>
                  </div>
                  <div className={styles.summaryItem}>
                    <span className={styles.summaryLabel}>{t('results.bestLoss')}</span>
                    <span className={styles.summaryValue} style={{ color: '#4CAF50' }}>
                      {Math.min(...trainingData.epochs.map((e) => e.loss)).toFixed(4)}
                    </span>
                  </div>
                  <div className={styles.summaryProgress}>
                    <div
                      className={styles.progressBar}
                      style={{
                        width: `${(trainingData.epochs[trainingData.epochs.length - 1].epoch / trainingData.epochs[trainingData.epochs.length - 1].total) * 100}%`,
                      }}
                    />
                  </div>
                </div>
              )}
              {/* Two-column layout: left = chart, right = config */}
              <div className={styles.trainingColumns}>
                {/* Loss chart */}
                <div className={styles.trainingChartCol}>
                  <div className={styles.sectionHeader}>{t('results.lossCurve')}</div>
                  {trainingData.epochs.length > 0 ? (
                    <LossChart
                      losses={trainingData.epochs.map((e) => e.loss)}
                      height={Math.max(80, panelHeight - 90)}
                    />
                  ) : (
                    <div className={styles.emptyState}>{t('results.waitingEpoch')}</div>
                  )}
                </div>

                {/* Resizable column divider */}
                <div
                  className={styles.columnDivider}
                  onMouseDown={handleColDividerDown}
                />

                {/* Right column: config + epoch table */}
                <div className={styles.trainingInfoCol} style={{ width: infoColWidth }}>
                  {/* Training config */}
                  {trainingData.config && (
                    <div
                      className={styles.configSection}
                      data-config
                      style={configHeight != null ? { height: configHeight, flexShrink: 0 } : undefined}
                    >
                      <div className={styles.sectionHeader}>{t('results.trainingConfig')}</div>
                      <div className={styles.configGrid}>
                        {Object.entries(trainingData.config).map(([key, val]) => (
                          <div key={key} className={styles.configRow}>
                            <span className={styles.configKey}>{key}</span>
                            <span className={styles.configVal}>
                              {typeof val === 'number' && !Number.isInteger(val)
                                ? val.toFixed(6)
                                : String(val)}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Resizable row divider between config and epochs */}
                  {trainingData.config && trainingData.epochs.length > 0 && (
                    <div
                      className={styles.rowDivider}
                      onMouseDown={handleRowDividerDown}
                    />
                  )}

                  {/* Epoch table */}
                  {trainingData.epochs.length > 0 && (
                    <div className={styles.epochSection}>
                      {(() => {
                        const last = trainingData.epochs[trainingData.epochs.length - 1];
                        return (
                          <div className={styles.sectionHeader}>
                            {t('results.epochsHeader', { current: last.epoch, total: last.total })}
                          </div>
                        );
                      })()}
                      <div className={styles.epochTable}>
                        <div className={`${styles.epochRow} ${styles.epochRowHeader}`}>
                          <span>#</span>
                          <span>{t('results.col.loss')}</span>
                          <span>{t('results.col.delta')}</span>
                          <span>{t('results.col.time')}</span>
                        </div>
                        {trainingData.epochs.map((ep, i) => {
                          const prev = i > 0 ? trainingData.epochs[i - 1].loss : null;
                          const delta = prev !== null ? ep.loss - prev : null;
                          const elapsed = i > 0
                            ? ((ep.ts - trainingData.epochs[i - 1].ts) / 1000).toFixed(1) + 's'
                            : '-';
                          return (
                            <div key={i} className={styles.epochRow}>
                              <span>{ep.epoch}</span>
                              <span>{ep.loss.toFixed(4)}</span>
                              <span className={
                                delta === null ? '' : delta < 0 ? styles.deltaDown : styles.deltaUp
                              }>
                                {delta === null ? '-' : (delta < 0 ? '' : '+') + delta.toFixed(4)}
                              </span>
                              <span className={styles.epochTime}>{elapsed}</span>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
