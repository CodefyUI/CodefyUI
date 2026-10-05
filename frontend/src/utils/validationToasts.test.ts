import { describe, it, expect, beforeEach } from 'vitest';
import {
  canvasNodeFor,
  dismissValidationToasts,
  issueText,
  issuesFromErrors,
  nodeName,
  showValidationError,
  showValidationIssues,
} from './validationToasts';
import type { ValidationIssue } from '../api/rest';
import { useI18n } from '../i18n';
import { useTabStore } from '../store/tabStore';
import { useToastStore } from '../store/toastStore';
import { useUIStore } from '../store/uiStore';

// Canvas ids are UUIDs. Realistic ones, so the id-to-title fallback cannot
// pass by matching an ordinary word.
const ENC = '3f2a9c1e-0b7d-4c55-9e21-6a1f0c2d8b41';
const DEC = '8d0c6b2a-55e3-4f1e-a9b7-2c4d6e8f0a13';
const LOSS = 'c41e7a90-1f2b-4d3c-8e5f-7a6b5c4d3e2f';
const START = '0a9b8c7d-6e5f-4a3b-2c1d-0e9f8a7b6c5d';
const BLOCK = 'b7a1c2d3-e4f5-4a6b-8c7d-9e0f1a2b3c4d';
const CARD = 'e5f6a7b8-c9d0-4e1f-a2b3-c4d5e6f7a8b9';
const UNTITLED = '1c2d3e4f-5a6b-4c7d-8e9f-0a1b2c3d4e5f';
const GONE = 'deadbeef-0000-4000-8000-000000000000';

function node(id: string, label: string, type: string, x = 0, y = 0): any {
  return { id, type: 'baseNode', position: { x, y }, data: { label, type, params: {} } };
}

const NODES = [
  node(ENC, 'Encoder', 'Linear', 100, 200),
  node(DEC, 'Decoder', 'Linear', 400, 200),
  node(LOSS, 'Loss', 'Loss'),
  node(START, 'Start', 'Start'),
  node(BLOCK, 'Feature block', 'subgraph:blk'),
  node(CARD, 'Say hi card', 'preset:Say Hi'),
  node(UNTITLED, '', 'Dropout'),
];

function makeTab(id: string, overrides: Record<string, unknown> = {}): any {
  return {
    id,
    name: id,
    nodes: NODES.map((n) => ({ ...n, data: { ...n.data } })),
    edges: [],
    selectedNodeId: null,
    subgraphStack: [],
    ...overrides,
  };
}

function issue(
  code: string | null,
  nodeId: string | null,
  params: Record<string, unknown>,
  message = `backend sentence about ${nodeId}`,
): ValidationIssue {
  return { message, code, node_id: nodeId, params };
}

// Stable across `setState({ locale })`: it reads the locale when called.
const { t } = useI18n.getState();

const toasts = () => useToastStore.getState().toasts;

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  useToastStore.setState({ toasts: [] });
  useUIStore.setState({ layoutFitRequest: null });
  useTabStore.setState({ tabs: [makeTab('A'), makeTab('B')], activeTabId: 'A' });
  dismissValidationToasts();
});

describe('canvasNodeFor', () => {
  it('finds a node by its own id', () => {
    expect(canvasNodeFor(ENC, NODES)?.id).toBe(ENC);
  });

  it('finds the block a flattened inner id came from', () => {
    expect(canvasNodeFor(`${BLOCK}/inner-node`, NODES)?.id).toBe(BLOCK);
    expect(canvasNodeFor(`${BLOCK}/nested/deeper`, NODES)?.id).toBe(BLOCK);
  });

  it('finds the card a preset-internal id came from', () => {
    expect(canvasNodeFor(`${CARD}__a`, NODES)?.id).toBe(CARD);
    expect(canvasNodeFor(`${CARD}__inner__a`, NODES)?.id).toBe(CARD);
  });

  it('is null for an id nothing on the canvas stands for', () => {
    expect(canvasNodeFor(GONE, NODES)).toBeNull();
    expect(canvasNodeFor(null, NODES)).toBeNull();
    expect(canvasNodeFor(`${GONE}/x`, NODES)).toBeNull();
  });
});

describe('nodeName', () => {
  it('is the title the card shows, else its type, else its id', () => {
    expect(nodeName(NODES[0])).toBe('Encoder');
    expect(nodeName(node(UNTITLED, '', 'Dropout'))).toBe('Dropout');
    expect(nodeName(node(UNTITLED, '', ''))).toBe(UNTITLED);
  });
});

describe('issueText', () => {
  // One finding per code, the way the backend sends it.
  const BY_CODE: ValidationIssue[] = [
    issue('missing_input', ENC, { port: 'tensor', type: 'Linear' }),
    issue('missing_input', DEC, { port: 'tensor', type: 'Linear', cause: ENC }),
    issue('param_not_number', ENC, { param: 'p', value: 'abc' }),
    issue('param_below_min', ENC, { param: 'p', value: -0.5, min: 0 }),
    issue('param_above_max', ENC, { param: 'p', value: 1.5, max: 1 }),
    issue('unknown_node_type', ENC, { type: 'NoSuchNode' }),
    issue('unknown_preset', CARD, { preset: 'Gone Preset' }),
    issue('invalid_output_port', DEC, { port: 'nope', type: 'Linear', target: ENC }),
    issue('invalid_input_port', DEC, { port: 'nope', type: 'Linear', source: ENC }),
    issue('type_mismatch', DEC, {
      source: LOSS, source_port: 'loss_fn', source_type: 'LOSS_FN',
      port: 'tensor', target_type: 'TENSOR',
    }),
    issue('cycle', ENC, { path: [ENC, DEC, ENC] }),
    issue('preset_input_not_exposed', CARD, { port: 'nope', preset: 'Say Hi' }),
    issue('preset_output_not_exposed', CARD, { port: 'nope', preset: 'Say Hi' }),
    issue('preset_triggered_empty', CARD, { preset: 'Say Hi' }),
    issue('preset_triggered_all_fed', CARD, { preset: 'Say Hi' }),
    issue('trigger_source_missing', ENC, { source: GONE }),
    issue('trigger_target_missing', START, { target: GONE }),
    issue('no_entry_points', null, {}),
  ];

  it.each(['en', 'zh-TW'] as const)(
    'says every code in %s, naming nodes by title and leaving no slot unfilled',
    (locale) => {
      useI18n.setState({ locale });
      for (const finding of BY_CODE) {
        const text = issueText(finding, NODES, t);
        expect(text, finding.code!).not.toBe(finding.message);
        expect(text, finding.code!).not.toMatch(/\{[A-Za-z0-9_]+\}/);
        // No raw canvas id: the node is named by its title.
        for (const id of [ENC, DEC, LOSS, START, CARD]) {
          expect(text, finding.code!).not.toContain(id);
        }
      }
    },
  );

  it('names the node by its title in English', () => {
    expect(issueText(BY_CODE[0], NODES, t)).toBe('Encoder: input "tensor" is not connected');
    expect(issueText(BY_CODE[1], NODES, t)).toBe(
      'Decoder: input "tensor" gets no data because "Encoder" is bypassed',
    );
    expect(issueText(BY_CODE[9], NODES, t)).toBe(
      'Loss output "loss_fn" (LOSS_FN) cannot feed Decoder input "tensor" (TENSOR)',
    );
    expect(issueText(BY_CODE[10], NODES, t)).toBe(
      'The graph has a loop: Encoder -> Decoder -> Encoder',
    );
  });

  it('names the node by its title in Chinese', () => {
    useI18n.setState({ locale: 'zh-TW' });
    expect(issueText(BY_CODE[0], NODES, t)).toBe('「Encoder」的輸入「tensor」尚未連線');
    expect(issueText(BY_CODE[9], NODES, t)).toBe(
      '「Loss」的輸出「loss_fn」（LOSS_FN）不能接到「Decoder」的輸入「tensor」（TENSOR）',
    );
    expect(issueText(BY_CODE[16], NODES, t)).toBe(
      `從「Start」拉出的 trigger 連線接到不在圖中的節點「${GONE.slice(0, 8)}」，請從圖檔中移除這條線。`,
    );
  });

  it('quotes a value that is not a number, and says null for a cleared box', () => {
    expect(issueText(BY_CODE[2], NODES, t)).toBe('Encoder: "p" must be a number (got "abc")');
    expect(
      issueText(issue('param_not_number', ENC, { param: 'p', value: null }), NODES, t),
    ).toBe('Encoder: "p" must be a number (got null)');
    expect(issueText(BY_CODE[3], NODES, t)).toBe('Encoder: "p" is -0.5, below the minimum 0');
  });

  it('names a node inside a block or a card by the container the canvas shows', () => {
    expect(
      issueText(issue('missing_input', `${BLOCK}/lin`, { port: 'tensor', type: 'Linear' }), NODES, t),
    ).toBe('Feature block: input "tensor" is not connected');
    expect(
      issueText(issue('missing_input', `${CARD}__b`, { port: 'value', type: 'Print' }), NODES, t),
    ).toBe('Say hi card: input "value" is not connected');
  });

  it('names an untitled node by its type and a missing one by the start of its id', () => {
    expect(
      issueText(issue('param_above_max', UNTITLED, { param: 'p', value: 2, max: 1 }), NODES, t),
    ).toBe('Dropout: "p" is 2, above the maximum 1');
    expect(issueText(BY_CODE[15], NODES, t)).toBe(
      `A trigger into Encoder comes from node "${GONE.slice(0, 8)}", which is not in the graph. ` +
        'Remove that connection from the graph file.',
    );
  });

  it('falls back to the server sentence, ids swapped for titles, for a code it does not know', () => {
    const sentence = `Duplicate node id: the node '${ENC}' appears more than once`;
    expect(issueText(issue(null, null, {}, sentence), NODES, t)).toBe(
      "Duplicate node id: the node 'Encoder' appears more than once",
    );
    const fromTheFuture = `Edge targets input port 'x' which block 'B' does not expose (node ${BLOCK}/inner)`;
    expect(issueText(issue('some_newer_code', BLOCK, {}, fromTheFuture), NODES, t)).toBe(
      "Edge targets input port 'x' which block 'B' does not expose (node Feature block)",
    );
  });

  it('falls back to the server sentence when a param its words need is missing', () => {
    const sentence = `Missing required input 'tensor' on node ${ENC} (Linear) -- connect an output to this port`;
    expect(issueText(issue('missing_input', ENC, { type: 'Linear' }, sentence), NODES, t)).toBe(
      "Missing required input 'tensor' on node Encoder (Linear) -- connect an output to this port",
    );
  });

  it('swaps an id only where the sentence names a node, never a word of its prose', () => {
    // Hand-written and example files use word ids like these.
    const words = [
      node('input', 'Image source', 'ImageReader'),
      node('start', 'Begin', 'Start'),
      node('print', 'Show result', 'Print'),
    ];
    const say = (message: string) => issueText(issue(null, null, {}, message), words, t);

    expect(say("Missing required input 'tensor' on node print (Print) -- connect an output to this port"))
      .toBe("Missing required input 'tensor' on node Show result (Print) -- connect an output to this port");
    expect(say("Duplicate node id: the node 'start' appears more than once"))
      .toBe("Duplicate node id: the node 'Begin' appears more than once");
    expect(say("Node input is triggered, but preset 'P' has no node to start: it has no nodes"))
      .toBe("Node Image source is triggered, but preset 'P' has no node to start: it has no nodes");
  });
});

describe('issuesFromErrors', () => {
  it('reads an older server that sends sentences only', () => {
    expect(issuesFromErrors(['a', 'b'])).toEqual([
      { message: 'a', code: null, node_id: null, params: {} },
      { message: 'b', code: null, node_id: null, params: {} },
    ]);
  });
});

describe('showValidationIssues', () => {
  const FIVE = [
    issue('missing_input', ENC, { port: 'tensor', type: 'Linear' }),
    issue('missing_input', DEC, { port: 'tensor', type: 'Linear' }),
    issue('param_above_max', UNTITLED, { param: 'p', value: 2, max: 1 }),
    issue('unknown_preset', CARD, { preset: 'Gone Preset' }),
    issue('no_entry_points', null, {}, 'Graph has no entry points.'),
  ];

  it('raises three problems and one line for the rest', () => {
    showValidationIssues('A', FIVE);

    expect(toasts().map((toast) => toast.message)).toEqual([
      'Encoder: input "tensor" is not connected',
      'Decoder: input "tensor" is not connected',
      'Dropout: "p" is 2, above the maximum 1',
      '2 more not shown. Fix these and run again.',
    ]);
    expect(toasts().every((toast) => toast.type === 'error')).toBe(true);
    expect(toasts().slice(0, 3).every((toast) => toast.action?.label === 'Show')).toBe(true);
    expect(toasts()[3].action).toBeUndefined();
  });

  it('replaces the last set instead of stacking a second one', () => {
    showValidationIssues('A', FIVE);
    showValidationIssues('A', FIVE.slice(0, 1));

    expect(toasts().map((toast) => toast.message)).toEqual([
      'Encoder: input "tensor" is not connected',
    ]);
  });

  it('says one fact once', () => {
    showValidationIssues('A', [FIVE[0], { ...FIVE[0] }]);

    expect(toasts()).toHaveLength(1);
  });

  it('takes down only its own toasts', () => {
    useToastStore.getState().addToast('something else', 'error');
    showValidationIssues('A', FIVE);

    dismissValidationToasts();

    expect(toasts().map((toast) => toast.message)).toEqual(['something else']);
  });

  it("Show selects the node and brings it into view", () => {
    showValidationIssues('A', FIVE);

    toasts()[1].action!.onClick();

    const tab = useTabStore.getState().tabs.find((candidate) => candidate.id === 'A')!;
    expect(tab.selectedNodeId).toBe(DEC);
    expect(tab.nodes.filter((n) => n.selected).map((n) => n.id)).toEqual([DEC]);
    expect(useUIStore.getState().layoutFitRequest?.bounds).toMatchObject({ x: 400, y: 200 });
  });

  it('Show on a node inside a block frames the block', () => {
    showValidationIssues('A', [issue('missing_input', `${BLOCK}/lin`, { port: 'tensor', type: 'Linear' })]);

    toasts()[0].action!.onClick();

    expect(useTabStore.getState().tabs[0].selectedNodeId).toBe(BLOCK);
  });

  it('Show does nothing for a node deleted since', () => {
    showValidationIssues('A', FIVE);
    useTabStore.setState({
      tabs: [makeTab('A', { nodes: NODES.filter((n) => n.id !== DEC) }), makeTab('B')],
    });

    toasts()[1].action!.onClick();

    expect(useTabStore.getState().tabs[0].selectedNodeId).toBeNull();
    expect(useUIStore.getState().layoutFitRequest).toBeNull();
  });

  it('offers no Show for a problem with no node on the canvas', () => {
    showValidationIssues('A', [
      issue('no_entry_points', null, {}, 'Graph has no entry points.'),
      issue('trigger_target_missing', GONE, { target: 'nowhere' }),
    ]);

    expect(toasts()).toHaveLength(2);
    expect(toasts().some((toast) => toast.action)).toBe(false);
  });

  it('inside an open block names the node but offers no Show', () => {
    const top = makeTab('A').nodes;
    useTabStore.setState({
      tabs: [
        makeTab('A', {
          nodes: [node('inner-1', 'Inner', 'Print')],
          subgraphStack: [{ subgraphId: 'blk', nodes: top, edges: [] }],
        }),
        makeTab('B'),
      ],
    });

    showValidationIssues('A', FIVE.slice(0, 1));

    expect(toasts().map((toast) => toast.message)).toEqual([
      'Encoder: input "tensor" is not connected',
    ]);
    expect(toasts()[0].action).toBeUndefined();
  });

  it('raises nothing once the user has gone to another tab', () => {
    showValidationIssues('A', FIVE);
    useTabStore.setState({ activeTabId: 'B' });

    showValidationIssues('A', FIVE);

    expect(toasts()).toEqual([]);
  });

  it('raises nothing, and does not throw, when no tab is open', () => {
    useTabStore.setState({ tabs: [], activeTabId: '' });

    expect(() => showValidationIssues('A', FIVE)).not.toThrow();
    expect(toasts()).toEqual([]);
  });

  it('is in Chinese when the UI is', () => {
    useI18n.setState({ locale: 'zh-TW' });

    showValidationIssues('A', FIVE);

    expect(toasts()[0].message).toBe('「Encoder」的輸入「tensor」尚未連線');
    expect(toasts()[0].action?.label).toBe('顯示');
    expect(toasts()[3].message).toBe('還有 2 個問題未顯示，先修正這些問題再執行一次。');
  });
});

describe('showValidationError', () => {
  it('belongs to the same set: the next validation replaces it, dismissing removes it', () => {
    showValidationError('No entry points defined.');
    expect(toasts().map((toast) => toast.message)).toEqual(['No entry points defined.']);

    showValidationIssues('A', [issue('missing_input', ENC, { port: 'tensor', type: 'Linear' })]);
    expect(toasts()).toHaveLength(1);
    expect(toasts()[0].message).toBe('Encoder: input "tensor" is not connected');

    showValidationError('No entry points defined.');
    dismissValidationToasts();
    expect(toasts()).toEqual([]);
  });
});
