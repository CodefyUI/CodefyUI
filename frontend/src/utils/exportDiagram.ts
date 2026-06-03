import dagre from '@dagrejs/dagre';
import type { Edge, Node } from '@xyflow/react';
import type { NodeData, PortDefinition } from '../types';
import { CATEGORY_COLORS } from '../styles/theme';
import { getPortColor, resolveDynamicOutputs } from './index';

/**
 * Render the active tab's graph as a standalone, self-contained SVG showing
 * the **architecture**: one card per node — its name on top, its input ports
 * on the left and output ports on the right (each labeled with the data type
 * that flows through it) — wired together by the directed connections. Tunable
 * parameter values and per-node visualizations are intentionally omitted; the
 * goal is a clean structural diagram a student can drop into slides or a report.
 *
 * Built straight from the node/edge data (not a DOM screenshot), so the output
 * is deterministic, dependency-free and crisp at any zoom. Each node card is
 * sized to fit its own content, and edges anchor at the exact port rows so the
 * data flow reads the same way it does on the canvas.
 */

// ── Card geometry ──
const TITLE_FS = 15; // node name font size
const TITLE_H = 34; // title band height
const PORT_FS = 11.5; // port label font size
const ROW_H = 20; // height of one port row
const PAD_V = 8; // vertical padding around the ports area
const PAD_H = 12; // horizontal padding inside the card
const DOT_R = 3.5; // port marker radius
const DOT_SPACE = 14; // horizontal room a port dot + gap occupies
const ARROW_SIZE = 6; // arrowhead marker size (small, dainty head)
const MIN_W = 160; // minimum card width (tall & slim cards)
const EMPTY_BODY = 12; // bottom padding for a node that has no ports
const PADDING = 48; // outer canvas padding around the whole diagram
const RANK_SEP = 90; // horizontal gap between layers (auto-layout)
const NODE_SEP = 32; // vertical gap between cards in the same layer (auto-layout)
const FONT_FAMILY =
  "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif";
// Labels longer than these are truncated with an ellipsis so cards stay tidy.
const MAX_LABEL_CHARS = 28;
const MAX_PORT_CHARS = 24;
const PRESET_COLOR = '#D4A017'; // reusable-subgraph accent (matches the toolbar)
const FALLBACK_NODE_COLOR = '#607D8B';
const START_COLOR = '#16a34a'; // Start/entry node accent (border, title, trigger edges)
const START_FILL = '#dcfce7'; // Start node light-green background

export type DiagramThemeName = 'light' | 'dark';

interface DiagramTheme {
  background: string;
  nodeFill: string;
  nodeText: string;
  edgeStroke: string;
}

export const DIAGRAM_THEMES: Record<DiagramThemeName, DiagramTheme> = {
  light: { background: '#ffffff', nodeFill: '#ffffff', nodeText: '#0f172a', edgeStroke: '#94a3b8' },
  dark: { background: '#0a0a0a', nodeFill: '#1e1e1e', nodeText: '#e5e7eb', edgeStroke: '#888888' },
};

export interface GraphToSvgOptions {
  /** Visual theme of the exported diagram. Defaults to `light` (document-friendly). */
  theme?: DiagramThemeName;
  /**
   * Color each edge by the data-type stroke it carries on the canvas (which
   * encodes what flows along it — educational). Defaults to `true`; set false
   * for a single neutral edge color.
   */
  preserveEdgeColors?: boolean;
  /**
   * `auto` (default) re-runs a clean left-to-right layout sized to the export
   * cards, so they never overlap regardless of how the canvas was arranged.
   * `preserve` keeps each node's canvas position.
   */
  layout?: 'auto' | 'preserve';
}

interface Entry {
  x: number;
  y: number;
  w: number;
  h: number;
  color: string;
  label: string;
  isStart: boolean;
  inputs: PortDefinition[];
  outputs: PortDefinition[];
}

/** Escape text for safe embedding in XML/SVG. */
function esc(s: string): string {
  return s.replace(
    /[&<>"']/g,
    (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&apos;' })[c]!,
  );
}

/** Round to 2 decimals to keep the markup compact. */
function r2(n: number): number {
  return Math.round(n * 100) / 100;
}

/** Rough text width estimate (no DOM) — good enough for card sizing. */
function estTextWidth(text: string, fontSize: number, bold = false): number {
  return text.length * fontSize * (bold ? 0.62 : 0.55);
}

/** A Start/entry node — drawn green and targeted at its top-left corner. */
function isStartNode(node: Node<NodeData>): boolean {
  return node.type === 'start' || node.data.type === 'Start';
}

/** Border/title color: Start green, else preset gold, else category color, else neutral. */
function nodeColor(node: Node<NodeData>): string {
  if (isStartNode(node)) return START_COLOR;
  if (node.data.isPreset) return PRESET_COLOR;
  const category = node.data.definition?.category;
  return (category && CATEGORY_COLORS[category]) || FALLBACK_NODE_COLOR;
}

/** Display title for a node, truncated with an ellipsis when overly long. */
function nodeLabel(node: Node<NodeData>): string {
  const raw = node.data.label || node.data.type || node.id;
  return raw.length > MAX_LABEL_CHARS ? `${raw.slice(0, MAX_LABEL_CHARS - 1)}…` : raw;
}

/**
 * Split a port into its display name and data type. The name is truncated (the
 * type is short and always kept whole) so the combined label stays tidy.
 */
function portParts(p: PortDefinition): { name: string; type: string } {
  const type = p.data_type;
  const maxName = Math.max(1, MAX_PORT_CHARS - type.length - 3); // 3 = " : "
  const name = p.name.length > maxName ? `${p.name.slice(0, maxName - 1)}…` : p.name;
  return { name, type };
}

/** Estimated rendered width of a port row: "name : " (normal) + TYPE (bold) + dot. */
function portWidth(p: PortDefinition): number {
  const { name, type } = portParts(p);
  return DOT_SPACE + estTextWidth(`${name} : `, PORT_FS) + estTextWidth(type, PORT_FS, true);
}

/** SVG markup for a port label: the name in normal weight, the TYPE in bold. */
function portTspans(p: PortDefinition): string {
  const { name, type } = portParts(p);
  return `<tspan>${esc(name)} : </tspan><tspan font-weight="700">${esc(type)}</tspan>`;
}

function edgeColor(edge: Edge, theme: DiagramTheme, preserve: boolean): string {
  const stroke = edge.style?.stroke;
  if (preserve && typeof stroke === 'string') return stroke;
  return theme.edgeStroke;
}

/** Stable, id-safe token derived from a color string (for <marker> ids). */
function colorId(color: string): string {
  return color.replace(/[^a-zA-Z0-9]/g, '');
}

/** Build a sized card descriptor for a node from its content. */
function buildEntry(node: Node<NodeData>): Entry {
  const def = node.data.definition;
  const isStart = isStartNode(node);
  // A Start node is a pure entry marker: just a green box with an outgoing
  // line, so its ports (the trigger) are not shown.
  const inputs = isStart ? [] : (def?.inputs ?? []);
  const outputs = isStart ? [] : resolveDynamicOutputs(def, node.data.params);
  const label = nodeLabel(node);
  // Inputs and outputs are stacked in separate rows (a single column), so the
  // card grows tall & slim rather than wide. Width fits the longest single
  // port label; height fits every input plus every output row.
  const totalRows = inputs.length + outputs.length;

  let labelW = 0;
  for (const p of [...inputs, ...outputs]) {
    labelW = Math.max(labelW, portWidth(p));
  }
  const w = Math.max(MIN_W, estTextWidth(label, TITLE_FS, true) + PAD_H * 2, labelW + PAD_H * 2);
  const h = TITLE_H + (totalRows > 0 ? PAD_V * 2 + totalRows * ROW_H : EMPTY_BODY);

  return {
    x: node.position.x,
    y: node.position.y,
    w,
    h,
    color: nodeColor(node),
    label,
    isStart,
    inputs,
    outputs,
  };
}

/** Y center of port row `idx` within a card. */
function portRowY(e: Entry, idx: number): number {
  return e.y + TITLE_H + PAD_V + (idx + 0.5) * ROW_H;
}

/**
 * Re-layout the cards left-to-right with dagre, sized to the export cards (not
 * the canvas nodes), so they spread out evenly and never overlap. Mutates each
 * entry's `x`/`y` in place.
 */
function layoutEntries(entries: Map<string, Entry>, edges: Edge[]): void {
  const g = new dagre.graphlib.Graph();
  g.setGraph({ rankdir: 'LR', nodesep: NODE_SEP, ranksep: RANK_SEP });
  g.setDefaultEdgeLabel(() => ({}));
  for (const [id, e] of entries) {
    g.setNode(id, { width: e.w, height: e.h });
  }
  for (const edge of edges) {
    if (entries.has(edge.source) && entries.has(edge.target)) {
      g.setEdge(edge.source, edge.target);
    }
  }
  dagre.layout(g);
  for (const [id, e] of entries) {
    const dn = g.node(id);
    // Dagre returns centers; convert to top-left.
    e.x = dn.x - dn.width / 2;
    e.y = dn.y - dn.height / 2;
  }
}

/**
 * Convert a graph to an SVG string. Note nodes are excluded, and edges are
 * drawn only when both endpoints are real (non-note) nodes.
 */
export function graphToSvg(
  nodes: Node<NodeData>[],
  edges: Edge[],
  options: GraphToSvgOptions = {},
): string {
  const theme = DIAGRAM_THEMES[options.theme ?? 'light'];
  const preserveEdgeColors = options.preserveEdgeColors ?? true;

  const entries = new Map<string, Entry>();
  for (const n of nodes) {
    if (n.type === 'noteNode') continue;
    entries.set(n.id, buildEntry(n));
  }

  // Spread the cards out cleanly unless the caller wants the canvas positions.
  if ((options.layout ?? 'auto') === 'auto' && entries.size > 0) {
    layoutEntries(entries, edges);
  }

  // Bounding box across all cards (empty graph → a zero-size origin box).
  let minX = 0;
  let minY = 0;
  let maxX = 0;
  let maxY = 0;
  if (entries.size > 0) {
    minX = Infinity;
    minY = Infinity;
    maxX = -Infinity;
    maxY = -Infinity;
    for (const e of entries.values()) {
      if (e.x < minX) minX = e.x;
      if (e.y < minY) minY = e.y;
      if (e.x + e.w > maxX) maxX = e.x + e.w;
      if (e.y + e.h > maxY) maxY = e.y + e.h;
    }
  }
  const width = r2(maxX - minX + PADDING * 2);
  const height = r2(maxY - minY + PADDING * 2);
  const offsetX = PADDING - minX;
  const offsetY = PADDING - minY;

  // ── Edges (drawn under the cards), anchored at the exact port rows ──
  const edgeParts: string[] = [];
  const markerColors = new Map<string, string>();
  for (const e of edges) {
    const s = entries.get(e.source);
    const t = entries.get(e.target);
    if (!s || !t) continue; // endpoint is a note or missing — skip
    // Trigger/control-flow edges (from a Start node) are not data: they point
    // at the target's top-left corner rather than an input port, and are drawn
    // in the Start green.
    const isTrigger = e.targetHandle === '__trigger' || e.sourceHandle === 'trigger';
    const color = isTrigger ? START_COLOR : edgeColor(e, theme, preserveEdgeColors);
    markerColors.set(colorId(color), color);

    // Outputs sit below the inputs, so an output's row is offset by the input count.
    const outIdx = e.sourceHandle ? s.outputs.findIndex((p) => p.name === e.sourceHandle) : -1;
    const sx = s.x + s.w;
    const sy = outIdx >= 0 ? portRowY(s, s.inputs.length + outIdx) : s.y + s.h / 2;

    let tx: number;
    let ty: number;
    if (isTrigger) {
      tx = t.x; // top-left corner
      ty = t.y;
    } else {
      const inIdx = e.targetHandle ? t.inputs.findIndex((p) => p.name === e.targetHandle) : -1;
      tx = t.x;
      ty = inIdx >= 0 ? portRowY(t, inIdx) : t.y + t.h / 2;
    }
    const dx = Math.max(40, Math.abs(tx - sx) / 2);
    const d = `M${r2(sx)},${r2(sy)} C${r2(sx + dx)},${r2(sy)} ${r2(tx - dx)},${r2(ty)} ${r2(tx)},${r2(ty)}`;
    edgeParts.push(
      `<path d="${d}" fill="none" stroke="${esc(color)}" stroke-width="2" ` +
        `marker-end="url(#arrow-${colorId(color)})" />`,
    );
  }

  // One arrowhead marker per distinct edge color so every arrow matches its line.
  const markers = Array.from(markerColors.entries())
    .map(
      ([id, color]) =>
        `<marker id="arrow-${id}" viewBox="0 0 10 10" refX="8" refY="5" ` +
        `markerWidth="${ARROW_SIZE}" markerHeight="${ARROW_SIZE}" orient="auto-start-reverse">` +
        `<path d="M0,0 L10,5 L0,10 z" fill="${esc(color)}" /></marker>`,
    )
    .join('');

  // ── Cards (drawn over the edges) ──
  const nodeParts: string[] = [];
  for (const e of entries.values()) {
    nodeParts.push(renderCard(e, theme));
  }

  return (
    `<?xml version="1.0" encoding="UTF-8"?>\n` +
    `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" ` +
    `viewBox="0 0 ${width} ${height}">` +
    `<defs>${markers}</defs>` +
    `<rect width="${width}" height="${height}" fill="${theme.background}" />` +
    `<g transform="translate(${r2(offsetX)},${r2(offsetY)})">` +
    edgeParts.join('') +
    nodeParts.join('') +
    `</g>` +
    `</svg>`
  );
}

/** A faint horizontal rule in the card's accent color, spanning its width. */
function cardRule(e: Entry, y: number): string {
  return (
    `<line x1="${r2(e.x + 8)}" y1="${r2(y)}" x2="${r2(e.x + e.w - 8)}" y2="${r2(y)}" ` +
    `stroke="${esc(e.color)}" stroke-opacity="0.35" stroke-width="1" />`
  );
}

/**
 * Render one node card: title on top, then inputs and outputs stacked in
 * separate rows (inputs above, outputs below). Start nodes get a light-green
 * fill so the entry point stands out.
 */
function renderCard(e: Entry, theme: DiagramTheme): string {
  const parts: string[] = ['<g>'];

  // Card surface (light green for Start) + accent-colored border.
  parts.push(
    `<rect x="${r2(e.x)}" y="${r2(e.y)}" width="${r2(e.w)}" height="${r2(e.h)}" ` +
      `rx="8" ry="8" fill="${e.isStart ? START_FILL : theme.nodeFill}" ` +
      `stroke="${esc(e.color)}" stroke-width="1.5" />`,
  );

  // Title — horizontally centered, larger, in the accent color. A node with
  // ports keeps its title in the top band; a port-less node (e.g. Start) centers
  // the title in the whole card so the text sits dead-centre.
  const hasPorts = e.inputs.length + e.outputs.length > 0;
  const titleY = e.y + (hasPorts ? TITLE_H / 2 : e.h / 2);
  parts.push(
    `<text x="${r2(e.x + e.w / 2)}" y="${r2(titleY)}" text-anchor="middle" ` +
      `dominant-baseline="central" font-family="${FONT_FAMILY}" font-size="${TITLE_FS}" ` +
      `font-weight="700" fill="${esc(e.color)}">${esc(e.label)}</text>`,
  );

  if (hasPorts) {
    parts.push(cardRule(e, e.y + TITLE_H)); // under the title

    // Inputs — upper rows, left-aligned with the dot on the left.
    e.inputs.forEach((p, i) => {
      const cy = portRowY(e, i);
      parts.push(
        `<circle cx="${r2(e.x + PAD_H)}" cy="${r2(cy)}" r="${DOT_R}" fill="${esc(getPortColor(p.data_type))}" />` +
          `<text x="${r2(e.x + PAD_H + DOT_SPACE - 4)}" y="${r2(cy)}" text-anchor="start" ` +
          `dominant-baseline="central" font-family="${FONT_FAMILY}" font-size="${PORT_FS}" ` +
          `fill="${theme.nodeText}">${portTspans(p)}</text>`,
      );
    });

    // Divider between the input layer and the output layer.
    if (e.inputs.length > 0 && e.outputs.length > 0) {
      parts.push(cardRule(e, e.y + TITLE_H + PAD_V + e.inputs.length * ROW_H));
    }

    // Outputs — lower rows, right-aligned with the dot on the right.
    e.outputs.forEach((p, j) => {
      const cy = portRowY(e, e.inputs.length + j);
      parts.push(
        `<circle cx="${r2(e.x + e.w - PAD_H)}" cy="${r2(cy)}" r="${DOT_R}" fill="${esc(getPortColor(p.data_type))}" />` +
          `<text x="${r2(e.x + e.w - PAD_H - DOT_SPACE + 4)}" y="${r2(cy)}" text-anchor="end" ` +
          `dominant-baseline="central" font-family="${FONT_FAMILY}" font-size="${PORT_FS}" ` +
          `fill="${theme.nodeText}">${portTspans(p)}</text>`,
      );
    });
  }

  parts.push('</g>');
  return parts.join('');
}

/** Pull the pixel dimensions out of an SVG produced by {@link graphToSvg}. */
function svgDimensions(svg: string): { width: number; height: number } {
  const w = svg.match(/<svg[^>]*\bwidth="([0-9.]+)"/);
  const h = svg.match(/<svg[^>]*\bheight="([0-9.]+)"/);
  return { width: w ? Number(w[1]) : 0, height: h ? Number(h[1]) : 0 };
}

/**
 * Rasterize an SVG diagram to a PNG `Blob` by drawing it onto a canvas. The
 * SVG carries its own background, so the PNG is opaque. `scale` (default 2)
 * super-samples for a crisp image. Browser-only — relies on `Image`/`<canvas>`.
 */
export function svgToPngBlob(svg: string, scale = 2): Promise<Blob> {
  return new Promise<Blob>((resolve, reject) => {
    const { width, height } = svgDimensions(svg);
    const url = URL.createObjectURL(new Blob([svg], { type: 'image/svg+xml' }));
    const img = new Image();

    img.onload = () => {
      try {
        const canvas = document.createElement('canvas');
        canvas.width = Math.max(1, Math.round(width * scale));
        canvas.height = Math.max(1, Math.round(height * scale));
        const ctx = canvas.getContext('2d');
        if (!ctx) {
          URL.revokeObjectURL(url);
          reject(new Error('Canvas 2D context is unavailable'));
          return;
        }
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        URL.revokeObjectURL(url);
        canvas.toBlob((blob) => {
          if (blob) resolve(blob);
          else reject(new Error('PNG encoding failed'));
        }, 'image/png');
      } catch (err) {
        URL.revokeObjectURL(url);
        reject(err as Error);
      }
    };
    img.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error('Failed to render the SVG for rasterization'));
    };
    img.src = url;
  });
}
