import type { OutputData, TensorOutput } from '../../types';
import { TensorGridView } from './TensorGridView';
import type { PortMedia, PortMediaMap } from './portCaptures';
import { getPortColor } from '../../utils';
import styles from './InspectorPanel.module.css';

export interface PortTarget {
  nodeId: string;
  port: string;
  /** Optional label shown instead of the raw port (e.g. `→ NodeX.tensor` or `Src.y`) */
  displayName?: string;
  /** Port data type when known — drives the colored dot. */
  dataType?: string;
}

export interface PortFetchState {
  loading: boolean;
  error: string | null;
  data: OutputData | null;
}

export type FetchMap = Record<string, PortFetchState>;

export function keyOf(nodeId: string, port: string): string {
  return `${nodeId}::${port}`;
}

/**
 * One stacked group of ports (all inputs, or all outputs) with a title.
 * Shared by single-node Forward view and segment mode — the two must not
 * drift apart visually.
 */
export function PortGroup({
  kind,
  title,
  ports,
  fetches,
  emptyText,
  highlight,
  media,
}: {
  kind: 'input' | 'output';
  title: string;
  ports: PortTarget[];
  fetches: FetchMap;
  emptyText?: string;
  /** Per-cell heat for one specific port (the solo-output transform case). */
  highlight?: { portKey: string; fn: (i: number, j: number) => number };
  /**
   * What media ports produced last run, keyed by {@link keyOf} — from
   * `usePortMedia`. A port with an entry here renders the picture or the clip
   * INSTEAD of its captured value, because that value is not a readable one:
   * an image port's is a base64 PNG the capture endpoint truncates at 4000
   * chars, and a video port's is a reference dict that renders as a `repr`.
   */
  media?: PortMediaMap;
}) {
  return (
    <div className={styles.portGroup}>
      <div className={styles.portGroupTitle}>{title}</div>
      {ports.length === 0 ? (
        <div className={styles.portGroupEmpty}>{emptyText ?? '—'}</div>
      ) : (
        ports.map((p) => {
          const key = keyOf(p.nodeId, p.port);
          const state = fetches[key];
          const portMedia = media?.[key];
          return (
            <div key={key} className={styles.portBlock}>
              <div className={styles.portHeader}>
                {kind === 'input' ? '⟵ ' : '⟶ '}
                {p.dataType && (
                  <span
                    className={styles.portDot}
                    style={{ background: getPortColor(p.dataType) }}
                  />
                )}
                <span className={styles.portName}>{p.displayName ?? p.port}</span>
              </div>
              {state?.error && !portMedia && (
                <div className={styles.portError}>{state.error}</div>
              )}
              {portMedia ? (
                <PortMediaView media={portMedia} label={p.displayName ?? p.port} />
              ) : (
                <>
                  {state?.data && state.data.type === 'tensor' && (
                    <TensorGridView
                      tensor={state.data as TensorOutput}
                      highlight={
                        highlight && highlight.portKey === key ? highlight.fn : undefined
                      }
                    />
                  )}
                  {state?.data && state.data.type !== 'tensor' && (
                    <NonTensorView value={state.data} />
                  )}
                  {!state?.data && !state?.error && (
                    <div className={styles.diffMissing}>…</div>
                  )}
                </>
              )}
            </div>
          );
        })
      )}
    </div>
  );
}

/**
 * A media port's own output: the picture it drew, or the clip it wrote.
 *
 * Kept in step with the results panel's log rows deliberately — a gif is an
 * animated image and browsers refuse it as a `<video>` source, so it goes in
 * an `<img>` there and here alike.
 */
export function PortMediaView({ media, label }: { media: PortMedia; label: string }) {
  if (media.kind === 'image') {
    return (
      <img
        className={styles.portImage}
        src={`data:image/${media.image.format};base64,${media.image.data}`}
        alt={label}
      />
    );
  }
  const { video } = media;
  if (video.format === 'gif') {
    return <img className={styles.portVideo} src={video.url} alt={label} />;
  }
  return (
    <video
      className={styles.portVideo}
      src={video.url}
      aria-label={label}
      controls
      loop
      preload="metadata"
    />
  );
}

/**
 * The divider between the input and output groups: a thin flow line with an
 * arrow, plus the shape-transform chip when the transform is unambiguous.
 */
export function FlowDivider({ chip }: { chip?: string | null }) {
  return (
    <div className={styles.flowDivider}>
      <span className={styles.flowLine} />
      <span className={styles.flowArrow}>↓</span>
      {chip && <span className={styles.shapeChip}>{chip}</span>}
      <span className={styles.flowLine} />
    </div>
  );
}

export function NonTensorView({ value, label }: { value: OutputData; label?: string }) {
  const v = value as any;
  return (
    <div className={styles.tensorView}>
      {label && <div className={styles.tensorLabel}>{label}</div>}
      <div className={styles.tensorMeta}>
        <span className={styles.tensorDtype}>{value.type}</span>
      </div>
      <div className={styles.tensorScalar}>
        {value.type === 'scalar' && String(v.value)}
        {value.type === 'string' && v.value}
        {value.type === 'model' && (
          <div>
            {v.class ?? 'Module'} · params{' '}
            {typeof v.params === 'number' ? v.params.toLocaleString() : '?'}
          </div>
        )}
        {!['scalar', 'string', 'model'].includes(value.type) && (v.repr ?? value.type)}
      </div>
    </div>
  );
}
