import { useMemo, useState } from 'react';
import { createPortal } from 'react-dom';
import styles from './HeatmapPlot.module.css';

export type HeatmapColormap = 'viridis' | 'blues' | 'RdBu';

interface HeatmapPlotProps {
  /** 2D matrix of weights ([seq, seq]) or 3D for multi-head ([H, seq, seq]). */
  data: number[][] | number[][][];
  /** Token labels for the query axis (rows). */
  rowLabels?: string[];
  /** Token labels for the key axis (columns). Defaults to rowLabels for self-attention. */
  colLabels?: string[];
  /** Single-panel size for 2D, per-panel size for 3D grid. */
  panelWidth?: number;
  panelHeight?: number;
  colormap?: HeatmapColormap;
  /** When true, cells with value exactly 0 in the upper triangle get a striped overlay marking them as causal-masked. */
  detectCausalMask?: boolean;
  className?: string;
  /** When set, an "expand" button appears at the top-right and the wrapper becomes clickable for opening a larger modal view. */
  onExpand?: () => void;
  /**
   * When true, each row is colour-mapped relative to its own max value. Use
   * for attention weights (especially causal): later positions spread
   * softmax mass across many tokens, so absolute values are tiny and
   * everything looks dim under a global [0, 1] scale. Tooltips still show
   * the raw weight.
   */
  normalizePerRow?: boolean;
}

interface HoverCell {
  i: number;
  j: number;
  v: number;
  head?: number;
  screenX: number;
  screenY: number;
}

/**
 * Detect whether a 2D matrix has the strictly upper-triangular zero pattern
 * left by a causal mask. Used to render a diagonal-stripe overlay on
 * masked cells so they are visually distinguishable from "near zero" cells
 * where the model genuinely attended weakly.
 */
export function detectCausalPattern(matrix: number[][]): boolean {
  if (matrix.length === 0) return false;
  const n = matrix.length;
  if (matrix[0].length !== n) return false;
  // Strictly upper triangle: j > i. All entries must be exactly 0.
  // Lower triangle + diagonal: at least one entry must be > 0 (else this is
  // an empty / all-zero matrix, not a causal mask).
  let upperAllZero = true;
  let lowerHasNonzero = false;
  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < n; j += 1) {
      const v = matrix[i][j];
      if (j > i) {
        if (v !== 0) upperAllZero = false;
      } else if (v > 0) {
        lowerHasNonzero = true;
      }
    }
  }
  return upperAllZero && lowerHasNonzero;
}

/**
 * Map a normalised value t ∈ [0, 1] to an RGB triple for the chosen colormap.
 * These are coarse polynomial approximations — good enough for teaching, no
 * external dependency, ~10 LOC each.
 */
export function valueToColor(t: number, colormap: HeatmapColormap = 'viridis'): string {
  const c = Math.max(0, Math.min(1, t));
  if (colormap === 'blues') {
    // White → cyan/teal accent that matches the "Crafted dark" palette.
    const r = Math.round(255 - 235 * c);
    const g = Math.round(255 - 73 * c);
    const b = Math.round(255 - 67 * c);
    return `rgb(${r}, ${g}, ${b})`;
  }
  if (colormap === 'RdBu') {
    // Diverging around 0.5: blue (low) → white (mid) → red (high).
    if (c < 0.5) {
      const k = c * 2;
      const r = Math.round(50 + 205 * k);
      const g = Math.round(70 + 185 * k);
      const b = Math.round(150 + 105 * k);
      return `rgb(${r}, ${g}, ${b})`;
    }
    const k = (c - 0.5) * 2;
    const r = Math.round(255 - 50 * k);
    const g = Math.round(255 - 205 * k);
    const b = Math.round(255 - 205 * k);
    return `rgb(${r}, ${g}, ${b})`;
  }
  // viridis approximation — perceptually uniform, dark → bright.
  // Polynomial fit to the canonical viridis lookup table; close enough for
  // the eye, no external deps.
  const r = Math.round(68 + 188 * Math.pow(c, 1.5) - 0 * c);
  const g = Math.round(1 + 230 * c - 30 * Math.pow(c, 2));
  const b = Math.round(84 + 100 * Math.pow(1 - c, 1.5) - 50 * Math.pow(c, 2));
  return `rgb(${Math.max(0, Math.min(255, r))}, ${Math.max(0, Math.min(255, g))}, ${Math.max(0, Math.min(255, b))})`;
}

function isMatrix3D(data: number[][] | number[][][]): data is number[][][] {
  return Array.isArray(data) && data.length > 0 && Array.isArray(data[0]) && Array.isArray(data[0][0]);
}

interface SinglePanelProps {
  matrix: number[][];
  width: number;
  height: number;
  rowLabels?: string[];
  colLabels?: string[];
  colormap: HeatmapColormap;
  causalMasked: boolean;
  headIndex?: number;
  normalizePerRow: boolean;
  onHover: (cell: HoverCell | null) => void;
}

function SinglePanel({
  matrix,
  width,
  height,
  rowLabels,
  colLabels,
  colormap,
  causalMasked,
  headIndex,
  normalizePerRow,
  onHover,
}: SinglePanelProps) {
  const n = matrix.length;
  const m = matrix[0]?.length ?? 0;

  // Compute axis gutters from actual label content. Approx char width at
  // 9px monospace ≈ 6.5px. Column labels rotate -45°, so their vertical
  // projection is ``charW × len × sin(45°)`` plus padding for the font
  // ascender (otherwise the leading characters get clipped against the
  // SVG top edge — see https://github.com/treeleaves30760/CodefyUI/...).
  const charW = 6.5;
  const showLabels = n <= 16;
  const maxColLen = showLabels && colLabels
    ? Math.max(0, ...colLabels.slice(0, m).map((l) => String(l).length))
    : 0;
  const maxRowLen = showLabels && rowLabels
    ? Math.max(0, ...rowLabels.slice(0, n).map((l) => String(l).length))
    : 0;
  const topGutter = maxColLen > 0
    ? Math.max(36, Math.ceil(maxColLen * charW * 0.75) + 16)
    : 6;
  const leftGutter = maxRowLen > 0
    ? Math.max(36, Math.ceil(maxRowLen * charW) + 10)
    : 6;

  const innerW = Math.max(20, width - leftGutter);
  const innerH = Math.max(20, height - topGutter);
  const cellW = innerW / Math.max(1, m);
  const cellH = innerH / Math.max(1, n);

  // Pre-compute per-row max for normalisation. Rows with all-zero values
  // (e.g. fully masked) keep zero everywhere — divide-by-zero is suppressed
  // by treating the max as 1.
  const rowMaxes = normalizePerRow
    ? matrix.map((row) => {
        let max = 0;
        for (const v of row) if (v > max) max = v;
        return max > 0 ? max : 1;
      })
    : null;

  return (
    <svg
      width={width}
      height={height}
      className={styles.panel}
      // Allow rotated column labels to render past the nominal SVG bounds
      // when a tight gutter calculation under-estimates by a pixel or two.
      style={{ overflow: 'visible' }}
      data-causal-masked={causalMasked ? 'true' : 'false'}
      data-head-index={headIndex ?? -1}
    >
      <defs>
        <pattern
          id={`stripes-h${headIndex ?? 0}`}
          patternUnits="userSpaceOnUse"
          width={4}
          height={4}
          patternTransform="rotate(45)"
        >
          <line x1={0} y1={0} x2={0} y2={4} stroke="rgba(120,140,170,0.55)" strokeWidth={1} />
        </pattern>
      </defs>
      {/* Column (key) labels along the top.
       *
       * textAnchor="end" + rotate(+45) anchors the *end* of each label at the
       * top of its column and lets the text body extend up-and-to-the-left
       * into the gutter region. Earlier we tried rotate(-45) which sends the
       * text body down-and-to-the-left — visually that pushed the leading
       * characters underneath the heatmap cells (cells render after labels
       * in the DOM, so they hid the overlap). Reading direction now goes
       * upper-left → lower-right, which is the standard "\" convention for
       * column headers. */}
      {colLabels && showLabels && (
        <g>
          {colLabels.slice(0, m).map((label, j) => {
            const cx = leftGutter + j * cellW + cellW / 2;
            const cy = topGutter - 6;
            return (
              <text
                key={`c-${j}`}
                x={cx}
                y={cy}
                className={styles.axisLabel}
                textAnchor="end"
                transform={`rotate(45, ${cx}, ${cy})`}
              >
                {label}
              </text>
            );
          })}
        </g>
      )}
      {/* Row (query) labels along the left */}
      {rowLabels && showLabels && (
        <g>
          {rowLabels.slice(0, n).map((label, i) => (
            <text
              key={`r-${i}`}
              x={leftGutter - 6}
              y={topGutter + i * cellH + cellH / 2 + 3}
              className={styles.axisLabel}
              textAnchor="end"
            >
              {label}
            </text>
          ))}
        </g>
      )}
      {/* Cells */}
      <g transform={`translate(${leftGutter}, ${topGutter})`}>
        {matrix.map((row, i) =>
          row.map((v, j) => {
            const masked = causalMasked && j > i && v === 0;
            // For colouring: optionally normalise by row max so the relative
            // attention pattern is visible even when absolute weights are
            // tiny (later rows of a causal sequence). Tooltips still show v.
            const colorT = rowMaxes
              ? Math.max(0, Math.min(1, v / rowMaxes[i]))
              : Math.max(0, Math.min(1, v));
            return (
              <g key={`c-${i}-${j}`}>
                <rect
                  x={j * cellW}
                  y={i * cellH}
                  width={cellW}
                  height={cellH}
                  fill={valueToColor(colorT, colormap)}
                  stroke="rgba(0,0,0,0.15)"
                  strokeWidth={0.5}
                  data-i={i}
                  data-j={j}
                  data-masked={masked ? 'true' : 'false'}
                  data-color-t={colorT.toFixed(3)}
                  className={styles.cell}
                  onMouseEnter={(e) => {
                    onHover({
                      i,
                      j,
                      v,
                      head: headIndex,
                      screenX: e.clientX,
                      screenY: e.clientY,
                    });
                  }}
                  onMouseMove={(e) => {
                    onHover({
                      i,
                      j,
                      v,
                      head: headIndex,
                      screenX: e.clientX,
                      screenY: e.clientY,
                    });
                  }}
                  onMouseLeave={() => onHover(null)}
                />
                {masked && (
                  <rect
                    x={j * cellW}
                    y={i * cellH}
                    width={cellW}
                    height={cellH}
                    fill={`url(#stripes-h${headIndex ?? 0})`}
                    pointerEvents="none"
                  />
                )}
              </g>
            );
          }),
        )}
      </g>
      {headIndex !== undefined && (
        <text x={width - 4} y={12} textAnchor="end" className={styles.headLabel}>
          h{headIndex}
        </text>
      )}
    </svg>
  );
}

/**
 * Heatmap viz for attention weights. Accepts a 2D matrix (single head /
 * single panel) or a 3D tensor (multiple heads side-by-side).
 *
 * Cell colour ∝ weight, hover surfaces (token_i → token_j, weight). When
 * ``detectCausalMask`` is on we render a striped overlay on cells that
 * sit in the strictly-upper-triangle and have weight exactly 0, so users
 * can see "this position was forbidden from attending here" rather than
 * mistaking it for "the model attended weakly".
 */
export function HeatmapPlot({
  data,
  rowLabels,
  colLabels,
  panelWidth = 220,
  panelHeight = 220,
  colormap = 'viridis',
  detectCausalMask = true,
  className,
  onExpand,
  normalizePerRow = false,
}: HeatmapPlotProps) {
  const [hover, setHover] = useState<HoverCell | null>(null);

  const panels = useMemo(() => {
    if (isMatrix3D(data)) {
      return data.map((m, idx) => ({
        matrix: m,
        head: idx,
        causalMasked: detectCausalMask && detectCausalPattern(m),
      }));
    }
    return [
      {
        matrix: data,
        head: undefined,
        causalMasked: detectCausalMask && detectCausalPattern(data),
      },
    ];
  }, [data, detectCausalMask]);

  const effectiveCol = colLabels ?? rowLabels;

  if (panels.length === 0 || panels[0].matrix.length === 0) {
    return (
      <div className={`${styles.empty} ${className ?? ''}`}>
        <span>no data</span>
      </div>
    );
  }

  return (
    <div className={`${styles.wrapper} ${className ?? ''}`}>
      {onExpand && (
        <button
          type="button"
          className={styles.expandBtn}
          onClick={(e) => {
            e.stopPropagation();
            onExpand();
          }}
          title="Open larger view"
          aria-label="Expand heatmap"
        >
          ⤢
        </button>
      )}
      <div className={styles.grid}>
        {panels.map((p, idx) => (
          <SinglePanel
            key={idx}
            matrix={p.matrix}
            width={panelWidth}
            height={panelHeight}
            rowLabels={rowLabels}
            colLabels={effectiveCol}
            colormap={colormap}
            causalMasked={p.causalMasked}
            headIndex={p.head}
            normalizePerRow={normalizePerRow}
            onHover={setHover}
          />
        ))}
      </div>
      {hover &&
        createPortal(
          <div
            className={styles.tooltip}
            style={{ left: hover.screenX + 12, top: hover.screenY + 12 }}
          >
            <div className={styles.tooltipHeader}>
              {hover.head !== undefined && <span className={styles.tooltipHead}>head {hover.head} · </span>}
              w[{hover.i}, {hover.j}] = {hover.v.toFixed(3)}
            </div>
            {rowLabels && effectiveCol && (
              <div className={styles.tooltipPair}>
                <span>{rowLabels[hover.i] ?? `q${hover.i}`}</span>
                <span className={styles.tooltipArrow}>→</span>
                <span>{effectiveCol[hover.j] ?? `k${hover.j}`}</span>
              </div>
            )}
          </div>,
          document.body,
        )}
    </div>
  );
}
