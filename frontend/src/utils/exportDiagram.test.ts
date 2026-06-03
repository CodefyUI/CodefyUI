import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import type { Edge, Node } from '@xyflow/react';
import type { NodeData, NodeDefinition, PortDefinition } from '../types';
import { graphToSvg, svgToPngBlob, DIAGRAM_THEMES } from './exportDiagram';

// ── Helpers ───────────────────────────────────────────────────────────

function port(name: string, data_type = 'TENSOR'): PortDefinition {
  return { name, data_type, description: '', optional: false };
}

function mkDef(over: Partial<NodeDefinition> = {}): NodeDefinition {
  return {
    node_name: 'X',
    category: 'CNN',
    description: '',
    inputs: [],
    outputs: [],
    params: [],
    ...over,
  };
}

function makeNode(
  id: string,
  data: Partial<NodeData> = {},
  over: Partial<Node<NodeData>> = {},
): Node<NodeData> {
  return {
    id,
    type: 'baseNode',
    position: { x: 0, y: 0 },
    data: { label: id, type: id, params: {}, ...data },
    ...over,
  };
}

function makeEdge(
  id: string,
  source: string,
  target: string,
  opts: { stroke?: string; sourceHandle?: string; targetHandle?: string } = {},
): Edge {
  return {
    id,
    source,
    target,
    ...(opts.sourceHandle ? { sourceHandle: opts.sourceHandle } : {}),
    ...(opts.targetHandle ? { targetHandle: opts.targetHandle } : {}),
    ...(opts.stroke ? { style: { stroke: opts.stroke, strokeWidth: 2 } } : {}),
  };
}

/** Count non-overlapping occurrences of `needle` in `hay`. */
function count(hay: string, needle: string): number {
  return hay.split(needle).length - 1;
}

/** Parse as XML and surface any well-formedness error element. */
function wellFormed(svg: string): boolean {
  const doc = new DOMParser().parseFromString(svg, 'application/xml');
  return doc.querySelector('parsererror') === null;
}

/** Extract every edge path's `d` attribute. */
function edgePaths(svg: string): string[] {
  return [...svg.matchAll(/<path d="([^"]+)" fill="none"/g)].map((m) => m[1]);
}

/** Width of the first node card (the rect carrying rx="8"). */
function firstCardWidth(svg: string): number {
  const m = svg.match(/<rect x="[^"]*" y="[^"]*" width="([0-9.]+)"[^>]*rx="8"/);
  return m ? Number(m[1]) : NaN;
}

/** A port label renders as `<tspan>name : </tspan><tspan font-weight="700">TYPE</tspan>`. */
function hasPortLabel(svg: string, name: string, type: string): boolean {
  return svg.includes(`>${name} : </tspan>`) && svg.includes(`<tspan font-weight="700">${type}</tspan>`);
}

/** Y coordinate of a node's centered title text, matched by its label. */
function titleY(svg: string, label: string): number {
  const m = svg.match(new RegExp(`<text x="[^"]*" y="([0-9.]+)"[^>]*font-weight="700"[^>]*>${label}<`));
  return m ? Number(m[1]) : NaN;
}

/** Centre-Y of every port dot in the SVG. */
function dotYs(svg: string): number[] {
  return [...svg.matchAll(/<circle cx="[^"]*" cy="([0-9.]+)"/g)].map((m) => Number(m[1]));
}

/** Parse each card's box (keyed by its title text) out of the rendered SVG. */
function parseCards(svg: string): Record<string, { x: number; y: number; w: number; h: number }> {
  const cards: Record<string, { x: number; y: number; w: number; h: number }> = {};
  for (const chunk of svg.split('<g>').slice(1)) {
    const rect = chunk.match(/<rect x="([0-9.-]+)" y="([0-9.-]+)" width="([0-9.]+)" height="([0-9.]+)"/);
    const title = chunk.match(/font-weight="700"[^>]*>([^<]+)</);
    if (rect && title) {
      cards[title[1]] = { x: +rect[1], y: +rect[2], w: +rect[3], h: +rect[4] };
    }
  }
  return cards;
}

// ── Tests ─────────────────────────────────────────────────────────────

describe('graphToSvg', () => {
  it('renders a well-formed SVG with one card per node and one path per edge', () => {
    const nodes = [
      makeNode('a', { label: 'Linear' }, { position: { x: 0, y: 0 } }),
      makeNode('b', { label: 'ReLU' }, { position: { x: 300, y: 0 } }),
    ];
    const svg = graphToSvg(nodes, [makeEdge('e1', 'a', 'b')]);

    expect(svg.startsWith('<?xml version="1.0" encoding="UTF-8"?>')).toBe(true);
    expect(svg).toContain('<svg xmlns="http://www.w3.org/2000/svg"');
    expect(svg).toContain('viewBox="0 0 ');
    expect(wellFormed(svg)).toBe(true);
    expect(count(svg, 'rx="8"')).toBe(2); // one card per node
    expect(count(svg, 'marker-end=')).toBe(1); // one edge
    expect(svg).toContain('>Linear<');
    expect(svg).toContain('>ReLU<');
    expect(count(svg, '<marker')).toBe(1);
    expect(svg).toContain('markerWidth="6"'); // small arrowhead
  });

  it('computes the bounding box across vertically stacked nodes', () => {
    // Two cards at the same x but different y: the second does not extend the
    // right edge (exercises the "not wider" branch of the bbox scan).
    const nodes = [
      makeNode('a', { label: 'Top' }, { position: { x: 0, y: 0 } }),
      makeNode('b', { label: 'Bottom' }, { position: { x: 0, y: 200 } }),
    ];
    const svg = graphToSvg(nodes, [], { layout: 'preserve' });
    // width = MIN_W(160) + 2*PADDING(48) = 256; height spans both rows.
    expect(svg).toContain('width="256" height="');
    expect(svg).toContain('>Top<');
    expect(svg).toContain('>Bottom<');
  });

  it('auto-layout (default) spreads overlapping nodes apart left-to-right', () => {
    // Both nodes sit at the same spot; auto-layout must separate them, and
    // edges to notes / missing nodes are ignored by the layout.
    const nodes = [
      makeNode('a', { definition: mkDef({ outputs: [port('o')] }) }, { position: { x: 0, y: 0 } }),
      makeNode('b', { definition: mkDef({ inputs: [port('i')] }) }, { position: { x: 0, y: 0 } }),
      makeNode('note', {}, { type: 'noteNode', position: { x: 0, y: 0 } }),
    ];
    const edges = [
      makeEdge('e', 'a', 'b', { sourceHandle: 'o', targetHandle: 'i' }),
      makeEdge('e2', 'a', 'note'), // target excluded from layout
      makeEdge('e3', 'ghost', 'b'), // source missing
      makeEdge('e4', 'a', 'ghost'), // target missing
    ];
    const cards = parseCards(graphToSvg(nodes, edges));
    // 'a' feeds 'b', so dagre puts 'a' in an earlier (left) rank — no overlap.
    expect(cards.a.x + cards.a.w).toBeLessThanOrEqual(cards.b.x);
  });

  it('renders input ports on the left and output ports on the right with their data types', () => {
    const def = mkDef({
      inputs: [port('x', 'TENSOR'), port('labels', 'TENSOR')],
      outputs: [port('out', 'MODEL')],
    });
    const svg = graphToSvg([makeNode('n', { label: 'Conv', definition: def })], []);

    expect(wellFormed(svg)).toBe(true);
    // Port labels show "name : TYPE" with the data type in bold.
    expect(hasPortLabel(svg, 'x', 'TENSOR')).toBe(true);
    expect(hasPortLabel(svg, 'labels', 'TENSOR')).toBe(true);
    expect(hasPortLabel(svg, 'out', 'MODEL')).toBe(true);
    // One colored dot per port (2 inputs + 1 output).
    expect(count(svg, '<circle')).toBe(3);
    expect(svg).toContain('<circle cx="12"'); // input dot at left padding
    expect(svg).toContain('fill="#4CAF50"'); // TENSOR
    expect(svg).toContain('fill="#2196F3"'); // MODEL
    // Inputs anchor start (left), outputs anchor end (right).
    expect(svg).toContain('text-anchor="start"');
    expect(svg).toContain('text-anchor="end"');
    // Title rule + input/output divider (both sides present → 2 rules).
    expect(count(svg, '<line')).toBe(2);
  });

  it('anchors an edge at the matching output and input port rows', () => {
    const srcDef = mkDef({ outputs: [port('a'), port('b')] });
    const dstDef = mkDef({ inputs: [port('p'), port('q')] });
    const nodes = [
      makeNode('s', { definition: srcDef }, { position: { x: 0, y: 0 } }),
      makeNode('t', { definition: dstDef }, { position: { x: 400, y: 0 } }),
    ];
    // Connect output index 1 (b) → input index 1 (q).
    const svg = graphToSvg(nodes, [makeEdge('e', 's', 't', { sourceHandle: 'b', targetHandle: 'q' })], {
      layout: 'preserve',
    });
    const [d] = edgePaths(svg);
    // Row 1 center = y(0) + TITLE_H(34) + PAD_V(8) + 1.5*ROW_H(20) = 72.
    expect(d).toContain(',72 C'); // start anchored at source row 1
    expect(d.endsWith(',72')).toBe(true); // end anchored at target row 1
  });

  it('falls back to the card center when an edge handle is absent or unmatched', () => {
    const srcDef = mkDef({ outputs: [port('only')] }); // 1 output → card center y = 35
    const dstDef = mkDef({ inputs: [port('only')] });
    const nodes = [
      makeNode('s', { definition: srcDef }, { position: { x: 0, y: 0 } }),
      makeNode('t', { definition: dstDef }, { position: { x: 400, y: 0 } }),
    ];
    // No sourceHandle, and a targetHandle that doesn't match any input.
    const svg = graphToSvg(nodes, [makeEdge('e', 's', 't', { targetHandle: 'nope' })], {
      layout: 'preserve',
    });
    const [d] = edgePaths(svg);
    // h = 34 + 8*2 + 1*20 = 70 → center = 35 (not the port-row value 52).
    expect(d).toContain(',35 C');
    expect(d.endsWith(',35')).toBe(true);
  });

  it('handles asymmetric port counts (more inputs than outputs)', () => {
    const def = mkDef({
      inputs: [port('i0'), port('i1'), port('i2')],
      outputs: [port('o0')],
    });
    const svg = graphToSvg([makeNode('n', { definition: def })], []);
    expect(count(svg, '<circle')).toBe(4); // 3 inputs + 1 output
    expect(hasPortLabel(svg, 'i2', 'TENSOR')).toBe(true);
    expect(hasPortLabel(svg, 'o0', 'TENSOR')).toBe(true);
  });

  it('handles asymmetric port counts (more outputs than inputs)', () => {
    const def = mkDef({
      inputs: [port('i0')],
      outputs: [port('o0'), port('o1'), port('o2')],
    });
    const svg = graphToSvg([makeNode('n', { definition: def })], []);
    expect(count(svg, '<circle')).toBe(4); // 1 input + 3 outputs
    expect(hasPortLabel(svg, 'o2', 'TENSOR')).toBe(true);
  });

  it('stacks inputs and outputs in separate rows (tall, single column)', () => {
    const def = mkDef({ inputs: [port('in1')], outputs: [port('out1')] });
    const svg = graphToSvg([makeNode('n', { definition: def })], [], { layout: 'preserve' });
    const card = parseCards(svg).n;
    // 1 input + 1 output stacked → 2 rows: h = TITLE_H(34) + PAD_V*2(16) + 2*ROW_H(40) = 90
    // (side-by-side would have been a single 70px-tall row).
    expect(card.h).toBe(90);
    // Title rule + input/output divider.
    expect(count(svg, '<line')).toBe(2);
  });

  it('centers the ports with equal top and bottom padding in their section', () => {
    const def = mkDef({ inputs: [port('a'), port('b')], outputs: [port('c')] });
    const svg = graphToSvg([makeNode('n', { definition: def })], [], { layout: 'preserve' });
    const card = parseCards(svg).n;
    const ys = dotYs(svg);
    const HALF_ROW = 10; // ROW_H / 2
    const SECTION_TOP = card.y + 34; // below the TITLE_H band
    const topPad = (Math.min(...ys) - HALF_ROW) - SECTION_TOP;
    const botPad = card.y + card.h - (Math.max(...ys) + HALF_ROW);
    expect(topPad).toBeCloseTo(botPad, 5); // equal whitespace above & below the ports
    expect(topPad).toBeGreaterThan(0);
  });

  it('places outputs in a lower layer than inputs (edge anchors below the inputs)', () => {
    const srcDef = mkDef({ inputs: [port('a')], outputs: [port('b')] });
    const dstDef = mkDef({ inputs: [port('c')] });
    const nodes = [
      makeNode('s', { definition: srcDef }, { position: { x: 0, y: 0 } }),
      makeNode('t', { definition: dstDef }, { position: { x: 400, y: 0 } }),
    ];
    const svg = graphToSvg(nodes, [makeEdge('e', 's', 't', { sourceHandle: 'b', targetHandle: 'c' })], {
      layout: 'preserve',
    });
    const [d] = edgePaths(svg);
    // output 'b' = row inputs.length(1)+0 = row 1 → y = 34+8+1.5*20 = 72 (below the input row).
    expect(d).toContain(',72 C');
    // input 'c' = row 0 → y = 34+8+0.5*20 = 52.
    expect(d.endsWith(',52')).toBe(true);
  });

  // ── Start node ───────────────────────────────────────────────────────

  it('renders a Start node (type "start") with a light-green fill and green accent', () => {
    const svg = graphToSvg([makeNode('s', { definition: mkDef({ category: 'Control' }) }, { type: 'start' })], []);
    expect(svg).toContain('fill="#dcfce7"'); // light-green background
    expect(svg).toContain('stroke="#16a34a"'); // green accent border
  });

  it('treats a node whose data.type is "Start" as a Start node', () => {
    const svg = graphToSvg([makeNode('s', { type: 'Start', definition: mkDef() })], []);
    expect(svg).toContain('fill="#dcfce7"');
  });

  it('does not render the Start trigger port — just the green box and title', () => {
    const startDef = mkDef({ category: 'Control', outputs: [port('trigger', 'TRIGGER')] });
    const svg = graphToSvg([makeNode('s', { label: 'Start', definition: startDef }, { type: 'start' })], []);
    expect(svg).toContain('>Start<');
    expect(count(svg, '<circle')).toBe(0); // no port dots
    expect(svg).not.toContain('trigger'); // trigger port not shown
    expect(svg).toContain('fill="#dcfce7"'); // still a green box
  });

  it('vertically centers the title of a port-less node (Start)', () => {
    const svg = graphToSvg(
      [makeNode('s', { label: 'Start', definition: mkDef({ category: 'Control' }) }, { type: 'start', position: { x: 0, y: 0 } })],
      [],
      { layout: 'preserve' },
    );
    const card = parseCards(svg).Start;
    // Title sits at the card's vertical centre, not the top title band.
    expect(titleY(svg, 'Start')).toBeCloseTo(card.y + card.h / 2, 5);
  });

  it('keeps a port-ful node title in the top band, not the card center', () => {
    const def = mkDef({ inputs: [port('a')], outputs: [port('b')] });
    const svg = graphToSvg([makeNode('Op', { label: 'Op', definition: def })], [], { layout: 'preserve' });
    const card = parseCards(svg).Op;
    expect(titleY(svg, 'Op')).toBeCloseTo(card.y + 34 / 2, 5); // TITLE_H / 2
  });

  it('points a trigger edge at the target top-left corner in green', () => {
    const startDef = mkDef({ category: 'Control', outputs: [port('trigger', 'TRIGGER')] });
    const dstDef = mkDef({ inputs: [port('x')] });
    const nodes = [
      makeNode('s', { definition: startDef }, { type: 'start', position: { x: 0, y: 0 } }),
      makeNode('t', { definition: dstDef }, { position: { x: 400, y: 50 } }),
    ];
    const svg = graphToSvg(nodes, [makeEdge('e', 's', 't', { sourceHandle: 'trigger', targetHandle: '__trigger' })], {
      layout: 'preserve',
    });
    const t = parseCards(svg).t;
    const [d] = edgePaths(svg);
    // Ends at the target's top-left corner (x, y) — not at the input port row.
    expect(d.endsWith(`${t.x},${t.y}`)).toBe(true);
    expect(svg).toContain('stroke="#16a34a"'); // trigger edges are green
  });

  it('treats an edge with sourceHandle "trigger" as a trigger even without __trigger', () => {
    const startDef = mkDef({ category: 'Control', outputs: [port('trigger', 'TRIGGER')] });
    const dstDef = mkDef({ inputs: [port('x')] });
    const nodes = [
      makeNode('s', { definition: startDef }, { type: 'start', position: { x: 0, y: 0 } }),
      makeNode('t', { definition: dstDef }, { position: { x: 400, y: 80 } }),
    ];
    // targetHandle is a real input name, not '__trigger' — still a trigger via the source.
    const svg = graphToSvg(nodes, [makeEdge('e', 's', 't', { sourceHandle: 'trigger', targetHandle: 'x' })], {
      layout: 'preserve',
    });
    const t = parseCards(svg).t;
    const [d] = edgePaths(svg);
    expect(d.endsWith(`${t.x},${t.y}`)).toBe(true); // corner, not the input row
  });

  it('expands Split dynamic outputs into chunk ports', () => {
    const def = mkDef({ node_name: 'Split', category: 'Tensor Operations', inputs: [port('tensor')] });
    const svg = graphToSvg([makeNode('n', { label: 'Split', definition: def, params: { chunks: 3 } })], []);
    expect(hasPortLabel(svg, 'chunk_0', 'TENSOR')).toBe(true);
    expect(hasPortLabel(svg, 'chunk_2', 'TENSOR')).toBe(true);
  });

  it('widens a card to fit long port labels', () => {
    const narrow = graphToSvg([makeNode('a', { definition: mkDef() })], []); // no ports → MIN_W
    const wide = graphToSvg(
      [makeNode('b', {
        definition: mkDef({
          inputs: [port('a_very_long_input_name', 'TENSOR')],
          outputs: [port('a_very_long_output_name', 'TENSOR')],
        }),
      })],
      [],
    );
    expect(firstCardWidth(narrow)).toBe(160); // MIN_W
    expect(firstCardWidth(wide)).toBeGreaterThan(160);
  });

  it('truncates an overly long port name but keeps the type whole and bold', () => {
    const def = mkDef({ inputs: [port('an_extremely_long_port_name_here', 'TENSOR')] });
    const svg = graphToSvg([makeNode('n', { definition: def })], []);
    expect(svg).toContain('…');
    expect(svg).not.toContain('an_extremely_long_port_name_here'); // full name truncated away
    expect(svg).toContain('<tspan font-weight="700">TENSOR</tspan>'); // type still whole & bold
  });

  it('renders the port data type in bold and the name in normal weight', () => {
    const def = mkDef({ inputs: [port('weights', 'MODEL')] });
    const svg = graphToSvg([makeNode('n', { definition: def })], []);
    expect(svg).toContain('<tspan>weights : </tspan><tspan font-weight="700">MODEL</tspan>');
  });

  it('renders a card with no ports (no separator, no dots)', () => {
    const svg = graphToSvg([makeNode('n', { label: 'NoPorts' })], []);
    expect(svg).toContain('>NoPorts<');
    expect(count(svg, '<circle')).toBe(0);
    expect(count(svg, '<line')).toBe(0);
  });

  it('excludes note nodes from the diagram', () => {
    const nodes = [
      makeNode('a', { label: 'Conv2d' }),
      makeNode('note1', { label: 'a helpful note' }, { type: 'noteNode' }),
    ];
    const svg = graphToSvg(nodes, []);
    expect(count(svg, 'rx="8"')).toBe(1);
    expect(svg).toContain('>Conv2d<');
    expect(svg).not.toContain('a helpful note');
  });

  it('skips edges whose endpoint is a note or a missing node', () => {
    const nodes = [makeNode('a'), makeNode('note1', {}, { type: 'noteNode' })];
    const edges = [
      makeEdge('e1', 'a', 'note1'), // target is a note → skipped
      makeEdge('e2', 'a', 'ghost'), // target missing → skipped
      makeEdge('e3', 'phantom', 'a'), // source missing → skipped
    ];
    const svg = graphToSvg(nodes, edges);
    expect(count(svg, 'marker-end=')).toBe(0);
    expect(count(svg, '<marker')).toBe(0);
  });

  it('produces a minimal background-only SVG when there are no drawable nodes', () => {
    const svg = graphToSvg([makeNode('note1', {}, { type: 'noteNode' })], []);
    expect(wellFormed(svg)).toBe(true);
    expect(svg).toContain('width="96" height="96"'); // 2 * PADDING
    expect(svg).not.toContain('rx="8"');
    expect(count(svg, 'marker-end=')).toBe(0);
  });

  it('colors edges by their data-type stroke by default', () => {
    const nodes = [makeNode('a'), makeNode('b', {}, { position: { x: 300, y: 0 } })];
    const svg = graphToSvg(nodes, [makeEdge('e1', 'a', 'b', { stroke: '#4CAF50' })]);
    expect(svg).toContain('stroke="#4CAF50"');
    expect(svg).toContain('id="arrow-4CAF50"');
    expect(svg).toContain('marker-end="url(#arrow-4CAF50)"');
  });

  it('uses the theme edge color when preserveEdgeColors is false', () => {
    const nodes = [makeNode('a'), makeNode('b', {}, { position: { x: 300, y: 0 } })];
    const svg = graphToSvg(nodes, [makeEdge('e1', 'a', 'b', { stroke: '#4CAF50' })], {
      preserveEdgeColors: false,
    });
    expect(svg).not.toContain('stroke="#4CAF50"');
    expect(svg).toContain(`stroke="${DIAGRAM_THEMES.light.edgeStroke}"`);
  });

  it('falls back to the theme edge color for an edge with no stroke', () => {
    const nodes = [makeNode('a'), makeNode('b', {}, { position: { x: 300, y: 0 } })];
    const svg = graphToSvg(nodes, [makeEdge('e1', 'a', 'b')]);
    expect(svg).toContain(`stroke="${DIAGRAM_THEMES.light.edgeStroke}"`);
  });

  it('deduplicates arrowhead markers across edges that share a color', () => {
    const nodes = [
      makeNode('a'),
      makeNode('b', {}, { position: { x: 300, y: 0 } }),
      makeNode('c', {}, { position: { x: 600, y: 0 } }),
    ];
    const edges = [
      makeEdge('e1', 'a', 'b', { stroke: '#4CAF50' }),
      makeEdge('e2', 'b', 'c', { stroke: '#4CAF50' }),
    ];
    const svg = graphToSvg(nodes, edges);
    expect(count(svg, 'marker-end=')).toBe(2);
    expect(count(svg, '<marker')).toBe(1);
  });

  it('honors the dark theme', () => {
    const def = mkDef({ inputs: [port('x')] }); // a port so node text color is exercised
    const svg = graphToSvg([makeNode('a', { definition: def })], [], { theme: 'dark' });
    expect(svg).toContain(`fill="${DIAGRAM_THEMES.dark.background}"`);
    expect(svg).toContain(`fill="${DIAGRAM_THEMES.dark.nodeFill}"`);
    expect(svg).toContain(`fill="${DIAGRAM_THEMES.dark.nodeText}"`);
  });

  it('defaults to the light theme', () => {
    const svg = graphToSvg([makeNode('a')], []);
    expect(svg).toContain(`fill="${DIAGRAM_THEMES.light.background}"`);
  });

  // ── Node accent color ────────────────────────────────────────────────

  it('colors a preset node with the preset gold accent', () => {
    const svg = graphToSvg([makeNode('p', { isPreset: true })], []);
    expect(svg).toContain('stroke="#D4A017"');
  });

  it('colors a node by its category', () => {
    const svg = graphToSvg([makeNode('c', { definition: mkDef({ category: 'CNN' }) })], []);
    expect(svg).toContain('stroke="#4CAF50"'); // CNN
  });

  it('falls back to the neutral color for an unknown category', () => {
    const svg = graphToSvg([makeNode('u', { definition: mkDef({ category: 'Nonexistent' }) })], []);
    expect(svg).toContain('stroke="#607D8B"');
  });

  it('falls back to the neutral color when a node has no definition', () => {
    const svg = graphToSvg([makeNode('n', { definition: undefined })], []);
    expect(svg).toContain('stroke="#607D8B"');
  });

  // ── Labels ───────────────────────────────────────────────────────────

  it('truncates an overly long title with an ellipsis', () => {
    const longLabel = 'ThisIsAnAbsurdlyLongNodeNameThatExceedsTheLimit';
    const svg = graphToSvg([makeNode('a', { label: longLabel })], []);
    expect(svg).toContain('…');
    expect(svg).not.toContain(longLabel);
    expect(svg).toContain(`>${longLabel.slice(0, 27)}…<`);
  });

  it('falls back from label to type to id', () => {
    const fromType = graphToSvg([makeNode('a', { label: '', type: 'AddOp' })], []);
    expect(fromType).toContain('>AddOp<');
    const fromId = graphToSvg([makeNode('the-id', { label: '', type: '' })], []);
    expect(fromId).toContain('>the-id<');
  });

  it('rounds coordinates to two decimals', () => {
    const svg = graphToSvg([makeNode('a', {}, { position: { x: 10.333333, y: 0 } })], [], {
      layout: 'preserve',
    });
    expect(svg).toContain('x="10.33"');
  });

  it('escapes XML-special characters in labels and ports', () => {
    const def = mkDef({ inputs: [port('<a> & "b"', 'TENSOR')] });
    const svg = graphToSvg([makeNode('a', { label: `<A> & "B" 'C'`, definition: def })], []);
    expect(svg).toContain('&lt;A&gt; &amp; &quot;B&quot; &apos;C&apos;');
    expect(svg).toContain('&lt;a&gt; &amp; &quot;b&quot;');
    expect(wellFormed(svg)).toBe(true);
  });

  it('exposes both light and dark theme presets', () => {
    expect(Object.keys(DIAGRAM_THEMES).sort()).toEqual(['dark', 'light']);
  });
});

// ── svgToPngBlob ──────────────────────────────────────────────────────

describe('svgToPngBlob', () => {
  const sampleSvg = graphToSvg([makeNode('a', { label: 'X' })], []);
  const svgWidth = Number(sampleSvg.match(/<svg[^>]*\bwidth="([0-9.]+)"/)![1]);

  let imgMode: 'load' | 'error';
  let canvasConfig: { ctx: unknown; toBlobResult: Blob | null };
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let lastCanvas: any;

  class FakeImage {
    onload: (() => void) | null = null;
    onerror: (() => void) | null = null;
    set src(_v: string) {
      Promise.resolve().then(() => {
        if (imgMode === 'error') this.onerror?.();
        else this.onload?.();
      });
    }
  }

  beforeEach(() => {
    imgMode = 'load';
    canvasConfig = { ctx: { drawImage: vi.fn() }, toBlobResult: new Blob(['png'], { type: 'image/png' }) };
    vi.stubGlobal('Image', FakeImage);
    vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:mock');
    vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {});
    const origCreate = document.createElement.bind(document);
    vi.spyOn(document, 'createElement').mockImplementation((tag: string) => {
      if (tag === 'canvas') {
        lastCanvas = {
          width: 0,
          height: 0,
          getContext: () => canvasConfig.ctx,
          toBlob: (cb: (b: Blob | null) => void) => cb(canvasConfig.toBlobResult),
        };
        return lastCanvas;
      }
      return origCreate(tag);
    });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it('rasterizes an SVG to a PNG blob and revokes the object URL', async () => {
    const blob = await svgToPngBlob(sampleSvg);
    expect(blob).toBeInstanceOf(Blob);
    expect(blob.type).toBe('image/png');
    expect(URL.createObjectURL).toHaveBeenCalled();
    expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:mock');
  });

  it('super-samples by the given scale factor', async () => {
    await svgToPngBlob(sampleSvg, 3);
    expect(lastCanvas.width).toBe(Math.round(svgWidth * 3));
  });

  it('clamps dimensionless SVGs to at least 1px', async () => {
    await svgToPngBlob('<svg></svg>');
    expect(lastCanvas.width).toBe(1);
    expect(lastCanvas.height).toBe(1);
  });

  it('rejects when the 2D context is unavailable', async () => {
    canvasConfig.ctx = null;
    await expect(svgToPngBlob(sampleSvg)).rejects.toThrow('Canvas 2D context is unavailable');
    expect(URL.revokeObjectURL).toHaveBeenCalled();
  });

  it('rejects when PNG encoding yields no blob', async () => {
    canvasConfig.toBlobResult = null;
    await expect(svgToPngBlob(sampleSvg)).rejects.toThrow('PNG encoding failed');
  });

  it('rejects (and cleans up) when drawing throws', async () => {
    canvasConfig.ctx = {
      drawImage: () => {
        throw new Error('draw boom');
      },
    };
    await expect(svgToPngBlob(sampleSvg)).rejects.toThrow('draw boom');
    expect(URL.revokeObjectURL).toHaveBeenCalled();
  });

  it('rejects when the image fails to load', async () => {
    imgMode = 'error';
    await expect(svgToPngBlob(sampleSvg)).rejects.toThrow('Failed to render the SVG');
    expect(URL.revokeObjectURL).toHaveBeenCalled();
  });
});
