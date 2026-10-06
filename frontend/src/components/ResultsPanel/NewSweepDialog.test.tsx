import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import * as rest from '../../api/rest';
import { NewSweepDialog } from './NewSweepDialog';
import { useKeyboardShortcuts } from '../../hooks/useKeyboardShortcuts';
import { useI18n } from '../../i18n';
import { isAnyModalOpen } from '../../store/modalState';
import { useNodeDefStore } from '../../store/nodeDefStore';
import { _resetRunStoreForTesting, useRunStore } from '../../store/runStore';
import { useSweepStore, _resetSweepStoreForTesting } from '../../store/sweepStore';
import { useTabStore } from '../../store/tabStore';
import { subgraphIdOf } from '../../utils/subgraph';
import type { NodeDefinition } from '../../types';

vi.mock('../../api/rest', async (importOriginal) => {
  const actual = await importOriginal<typeof rest>();
  return { ...actual, createSweep: vi.fn(), getSweep: vi.fn(), getRunMetrics: vi.fn() };
});

const api = vi.mocked(rest);
// Some tests install a vi.fn() action with setState, which the store reset
// does not undo; the real one goes back before every test.
const realCreateSweep = useSweepStore.getState().createSweep;

const trainDefinition: NodeDefinition = {
  node_name: 'Train', category: 'Test', description: '', inputs: [], outputs: [],
  params: [
    { name: 'epochs', param_type: 'int', default: 2, description: '', options: [], min_value: 1, max_value: 10 },
    { name: 'rate', param_type: 'float', default: 0.1, description: '', options: [], min_value: 0, max_value: 1 },
    { name: 'label', param_type: 'string', default: 'training', description: '', options: [], min_value: null, max_value: null },
    { name: 'mode', param_type: 'select', default: 'fast', description: '', options: ['fast', 'safe'], min_value: null, max_value: null },
    { name: 'shuffle', param_type: 'bool', default: true, description: '', options: [], min_value: null, max_value: null },
    { name: 'api_key', param_type: 'secret', default: '', description: '', options: [], min_value: null, max_value: null },
  ],
};

function trainNode(id: string, label: string) {
  return {
    id, type: 'customNode', position: { x: 0, y: 0 },
    data: { type: 'Train', label, params: { epochs: 2, rate: 0.1, mode: 'fast', shuffle: true, api_key: 'sk-session' } },
  };
}

function setupGraph(nodes = [trainNode('train', 'Trainer')]) {
  useTabStore.getState().setNodes(nodes as never);
}

/** The editor for the Nth parameter, by its fieldset's legend. */
function editor(name: string) {
  return within(screen.getByRole('group', { name }));
}

function startButton() {
  return screen.getByRole('button', { name: /start sweep/i });
}

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  _resetSweepStoreForTesting();
  useSweepStore.setState({ createSweep: realCreateSweep });
  _resetRunStoreForTesting();
  useNodeDefStore.setState({ definitions: [trainDefinition] });
  useTabStore.setState({ tabs: [], activeTabId: '' });
  useTabStore.getState().addTab('test');
  setupGraph();
  api.createSweep.mockResolvedValue({
    sweep_id: 's1', state: 'running', method: 'grid', seed: null, seed_variants: false,
    objective: { metric: 'train_loss', direction: 'minimize' }, total_combinations: 2, params: [], variants: [],
  });
  api.getSweep.mockResolvedValue({
    sweep_id: 's1', name: null, state: 'running', method: 'grid', seed: null, seed_variants: false,
    objective: { metric: 'train_loss', direction: 'minimize' }, created_at: '2026-10-01T00:00:00Z',
    finished_at: null, error: null,
    counts: { queued: 2, running: 0, succeeded: 0, failed: 0, cancelled: 0, interrupted: 0, missing: 0 },
    params: [], variants: [], best: null,
  });
  api.getRunMetrics.mockResolvedValue({ run_id: 'r', names: [], metrics: [] });
});

afterEach(() => {
  act(() => {
    _resetSweepStoreForTesting();
    _resetRunStoreForTesting();
  });
  vi.clearAllMocks();
});

describe('NewSweepDialog', () => {
  it('owns focus, traps Tab, closes on Escape, and restores focus', () => {
    const opener = document.createElement('button');
    document.body.appendChild(opener);
    opener.focus();
    const onClose = vi.fn();
    const { unmount } = render(<NewSweepDialog onClose={onClose} />);

    const dialog = screen.getByRole('dialog', { name: /new sweep/i });
    expect(dialog).toContainElement(document.activeElement as HTMLElement);
    const focusable = within(dialog).getAllByRole('button');
    focusable[focusable.length - 1].focus();
    const tab = new KeyboardEvent('keydown', { key: 'Tab', bubbles: true, cancelable: true });
    dialog.dispatchEvent(tab);
    expect(tab.defaultPrevented).toBe(true);
    expect(dialog).toContainElement(document.activeElement as HTMLElement);
    fireEvent.keyDown(dialog, { key: 'Escape' });
    expect(onClose).toHaveBeenCalledTimes(1);

    unmount();
    expect(document.activeElement).toBe(opener);
    opener.remove();
  });

  it('keeps Escape and the Tab trap after a click between its controls', () => {
    // A click on a label or a gap focuses no control; the dialog itself takes
    // the focus then, out of the Tab order, instead of the page.
    const onClose = vi.fn();
    render(<NewSweepDialog onClose={onClose} />);
    const dialog = screen.getByRole('dialog');
    dialog.focus();
    expect(document.activeElement).toBe(dialog);
    expect(dialog).toHaveAttribute('tabindex', '-1');

    const back = new KeyboardEvent('keydown', { key: 'Tab', shiftKey: true, bubbles: true, cancelable: true });
    dialog.dispatchEvent(back);
    expect(back.defaultPrevented).toBe(true);
    expect(document.activeElement).toBe(screen.getByRole('button', { name: /start sweep/i }));

    dialog.focus();
    fireEvent.keyDown(dialog, { key: 'Escape' });
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('is a modal to the canvas shortcuts while it is open, and only then', () => {
    // Behind the dialog, Ctrl+Z used to undo the canvas and the changed graph
    // was then what Start sent.
    const originalUndo = useTabStore.getState().undo;
    const undo = vi.fn();
    useTabStore.setState({ undo });
    function Shortcuts() {
      useKeyboardShortcuts();
      return null;
    }
    try {
      render(<Shortcuts />);
      const { unmount } = render(<NewSweepDialog onClose={vi.fn()} />);

      expect(isAnyModalOpen()).toBe(true);
      fireEvent.keyDown(screen.getByRole('button', { name: /add parameter/i }), { key: 'z', ctrlKey: true });
      expect(undo).not.toHaveBeenCalled();

      unmount();
      expect(isAnyModalOpen()).toBe(false);
    } finally {
      useTabStore.setState({ undo: originalUndo });
    }
  });

  it('offers only eligible top-level params and previews exact grid variants', () => {
    render(<NewSweepDialog onClose={vi.fn()} />);
    const dialog = screen.getByRole('dialog');
    expect(within(dialog).getByRole('option', { name: /Trainer.*epochs/i })).toBeInTheDocument();
    expect(within(dialog).getByRole('option', { name: /Trainer.*mode/i })).toBeInTheDocument();
    expect(within(dialog).queryByRole('option', { name: /Trainer.*label/i })).toBeNull();
    expect(within(dialog).queryByRole('option', { name: /api_key/i })).toBeNull();

    fireEvent.change(within(dialog).getByLabelText('Values'), { target: { value: '2, 4' } });
    expect(within(dialog).getByText('2 variants')).toBeInTheDocument();
    fireEvent.change(within(dialog).getByLabelText('Values'), { target: { value: '2' } });
    expect(within(dialog).getByText('1 variant')).toBeInTheDocument();
  });

  it('offers the params of the graph Start sends while a block is open', () => {
    // Inside a block the canvas shows the block's nodes, but a sweep runs the
    // root graph, so a block node's address is a 400 from the server.
    setupGraph([trainNode('train', 'Trainer'), trainNode('inner1', 'Inner one'), trainNode('inner2', 'Inner two')]);
    const store = useTabStore.getState();
    store.setNodes(store.getActiveTab().nodes.map((node) => ({ ...node, selected: node.id !== 'train' })));
    expect(store.collapseSelectionToSubgraph('Block').ok).toBe(true);
    const instance = useTabStore.getState().getActiveTab().nodes.find((node) => subgraphIdOf(node.data.type))!;
    expect(useTabStore.getState().enterSubgraph(instance.id)).toBe(true);

    render(<NewSweepDialog onClose={vi.fn()} />);
    const dialog = screen.getByRole('dialog');
    expect(within(dialog).getByRole('option', { name: /Trainer.*epochs/i })).toBeInTheDocument();
    expect(within(dialog).queryByRole('option', { name: /Inner/i })).toBeNull();
  });

  it('builds multiple domains and previews their exact Cartesian product', async () => {
    const createSweep = vi.fn().mockResolvedValue(true);
    useSweepStore.setState({ createSweep });
    render(<NewSweepDialog onClose={vi.fn()} />);

    fireEvent.change(screen.getByLabelText('Values'), { target: { value: '2, 4' } });
    fireEvent.click(screen.getByRole('button', { name: /Add parameter/i }));
    fireEvent.change(screen.getByLabelText('Parameter 2'), { target: { value: 'train.rate' } });
    fireEvent.change(editor('Parameter 2').getByLabelText('Values'), { target: { value: '0.1, 0.2' } });

    expect(screen.getByText('4 variants')).toBeInTheDocument();
    await act(async () => {
      fireEvent.click(startButton());
    });
    expect(createSweep.mock.calls[0][0].sweep_spec.params).toEqual([
      { node_id: 'train', param: 'epochs', values: [2, 4] },
      { node_id: 'train', param: 'rate', values: [0.1, 0.2] },
    ]);
  });

  it('names each parameter once, by its group', () => {
    render(<NewSweepDialog onClose={vi.fn()} />);
    fireEvent.click(screen.getByRole('button', { name: /Add parameter/i }));
    const group = screen.getByRole('group', { name: 'Parameter 2' });
    expect(group.textContent!.match(/Parameter 2/g)).toHaveLength(1);
    expect(within(group).getByRole('combobox', { name: 'Parameter 2' })).toBeInTheDocument();
  });

  it('moves focus to a parameter that is still there when one is added or removed', () => {
    render(<NewSweepDialog onClose={vi.fn()} />);
    fireEvent.click(screen.getByRole('button', { name: /Add parameter/i }));
    expect(document.activeElement).toBe(screen.getByLabelText('Parameter 2'));

    const remove = screen.getByRole('button', { name: 'Remove parameter 2' });
    remove.focus();
    fireEvent.click(remove);
    // Escape and the Tab trap live on the dialog, so focus must stay inside.
    expect(document.activeElement).toBe(screen.getByLabelText('Parameter'));
  });

  it('goes back to a values list when the chosen param cannot take a range', () => {
    render(<NewSweepDialog onClose={vi.fn()} />);
    fireEvent.change(screen.getByLabelText('Domain'), { target: { value: 'range' } });
    expect(screen.getByLabelText('Count')).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText('Parameter'), { target: { value: 'train.shuffle' } });
    expect(screen.queryByLabelText('Domain')).toBeNull();
    fireEvent.change(screen.getByLabelText('Values'), { target: { value: 'true, false' } });
    expect(screen.getByText('2 variants')).toBeInTheDocument();
  });

  it('shows the seed it sends: for a random sweep, and for a seeded grid', async () => {
    const createSweep = vi.fn().mockResolvedValue(true);
    useSweepStore.setState({ createSweep });
    render(<NewSweepDialog onClose={vi.fn()} />);
    expect(screen.queryByLabelText('Seed')).toBeNull();

    fireEvent.click(screen.getByLabelText(/seed every variant/i));
    expect(screen.getByText(/run one at a time/i)).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText('Seed'), { target: { value: '41' } });
    fireEvent.change(screen.getByLabelText('Values'), { target: { value: '2, 4' } });
    await act(async () => {
      fireEvent.click(startButton());
    });
    expect(createSweep.mock.calls[0][0]).toMatchObject({
      sweep_spec: { method: 'grid', seed: 41, samples: null }, seed_variants: true,
    });
    expect(createSweep.mock.calls[0][0].options).not.toHaveProperty('seed');

    fireEvent.click(screen.getByLabelText(/seed every variant/i));
    expect(screen.queryByLabelText('Seed')).toBeNull();
    fireEvent.change(screen.getByLabelText(/method/i), { target: { value: 'random' } });
    expect(screen.getByLabelText('Samples')).toBeInTheDocument();
    expect(screen.getByLabelText('Seed')).toBeInTheDocument();
  });

  it('reveals the seed below the checkbox that asks for it', () => {
    // Above it, the field would push the checkbox down under the pointer.
    render(<NewSweepDialog onClose={vi.fn()} />);
    const checkbox = screen.getByLabelText(/seed every variant/i);
    fireEvent.click(checkbox);
    const seed = screen.getByLabelText('Seed');
    expect(checkbox.compareDocumentPosition(seed) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });

  it('refuses a seed the server cannot take', async () => {
    const createSweep = vi.fn().mockResolvedValue(true);
    useSweepStore.setState({ createSweep });
    render(<NewSweepDialog onClose={vi.fn()} />);
    fireEvent.change(screen.getByLabelText(/method/i), { target: { value: 'random' } });
    fireEvent.change(screen.getByLabelText('Values'), { target: { value: '2, 4' } });
    fireEvent.change(screen.getByLabelText('Seed'), { target: { value: '' } });
    await act(async () => {
      fireEvent.click(startButton());
    });
    expect(screen.getByRole('alert')).toHaveTextContent(/seed must be a whole number from 0 to 4294967295/i);
    expect(createSweep).not.toHaveBeenCalled();
  });

  it('sends the in-memory graph with its session key and the tab run settings, and keeps no copy', async () => {
    const tab = useTabStore.getState().getActiveTab();
    useTabStore.setState((state) => ({
      tabs: state.tabs.map((candidate) => candidate.id === tab.id
        ? { ...candidate, graphDevice: 'cuda:0', seed: 77, deterministic: true, recordOutputs: true }
        : candidate),
    }));
    const onClose = vi.fn();
    render(<NewSweepDialog onClose={onClose} />);

    fireEvent.change(screen.getByLabelText('Name'), { target: { value: 'Epoch search' } });
    fireEvent.change(screen.getByLabelText(/objective metric/i), { target: { value: 'val_loss' } });
    fireEvent.change(screen.getByLabelText('Values'), { target: { value: '2, 4' } });
    await act(async () => {
      fireEvent.click(startButton());
    });

    await waitFor(() => expect(onClose).toHaveBeenCalledTimes(1));
    const request = api.createSweep.mock.calls[0][0];
    expect(request).toMatchObject({
      name: 'Epoch search',
      objective: { metric: 'val_loss', direction: 'minimize' },
      sweep_spec: {
        method: 'grid', seed: null, samples: null,
        params: [{ node_id: 'train', param: 'epochs', values: [2, 4] }],
      },
      seed_variants: false,
      options: expect.objectContaining({ device: 'cuda:0', deterministic: true }),
    });
    // The tab's seed belongs to canvas runs: the server refuses options.seed.
    expect(request.options).not.toHaveProperty('seed');
    // Captured outputs are kept for only 20 runs, shared with the canvas, and
    // the server refuses record_outputs on a larger sweep.
    expect(request.options).not.toHaveProperty('record_outputs');
    // Each child needs the key to execute, so the request carries it; the
    // store, which outlives the request, holds nothing that does.
    expect((request.base_graph.nodes as Array<{ data: { params: Record<string, unknown> } }>)[0].data.params.api_key)
      .toBe('sk-session');
    expect(useSweepStore.getState().selectedSweepId).toBe('s1');
    expect(JSON.stringify(useSweepStore.getState())).not.toContain('sk-session');
    // The sweep's columns are later named from the graph it came from.
    expect(useSweepStore.getState().origins).toEqual({ s1: tab.id });
  });

  // #623: with the Name left blank, a sweep and every run it starts were
  // listed as "(unnamed)". They take the tab's name, as a canvas run does.
  async function startWithTabNamed(tabName: string, typedName?: string) {
    const tab = useTabStore.getState().getActiveTab();
    useTabStore.getState().renameTab(tab.id, tabName);
    const onClose = vi.fn();
    render(<NewSweepDialog onClose={onClose} />);
    if (typedName !== undefined) {
      fireEvent.change(screen.getByLabelText('Name'), { target: { value: typedName } });
    }
    fireEvent.change(screen.getByLabelText('Values'), { target: { value: '2, 4' } });
    await act(async () => {
      fireEvent.click(startButton());
    });
    await waitFor(() => expect(onClose).toHaveBeenCalledTimes(1));
    return api.createSweep.mock.calls[0][0].name;
  }

  it('names the sweep after its tab when the Name field is blank', async () => {
    expect(await startWithTabNamed('CF201 lab', '   ')).toBe('CF201 lab');
  });

  it('shows the tab name as the Name field placeholder', () => {
    const tab = useTabStore.getState().getActiveTab();
    useTabStore.getState().renameTab(tab.id, 'CF201 lab');
    render(<NewSweepDialog onClose={vi.fn()} />);
    expect(screen.getByLabelText('Name')).toHaveAttribute('placeholder', 'CF201 lab');
  });

  it('cuts a long tab name to the 64 characters the server takes, counting code points', async () => {
    // The sweep route refuses a longer name rather than cutting it, and a tab
    // name must never stop a sweep. Leading blanks do not use up the room.
    const astral = String.fromCodePoint(0x20000);
    const name = await startWithTabNamed(`  ${'x'.repeat(63)}${astral}${'y'.repeat(10)}`);
    // 64 code points, 65 UTF-16 units: the character outside the BMP is whole.
    expect(name).toBe(`${'x'.repeat(63)}${astral}`);
  });

  it('sends half of a surrogate pair in the tab name as "?", which the server can store', async () => {
    expect(await startWithTabNamed(`lab${String.fromCharCode(0xd800)}`)).toBe('lab?');
  });

  it('keeps a typed name as it is', async () => {
    expect(await startWithTabNamed('CF201 lab', 'Epoch search')).toBe('Epoch search');
  });

  it('takes no longer a typed name than the server does', () => {
    // A longer one was refused by the server, in its own English words.
    render(<NewSweepDialog onClose={vi.fn()} />);
    expect((screen.getByLabelText('Name') as HTMLInputElement).maxLength).toBe(64);
  });

  it('cuts a typed name past the limit as it cuts the tab name', async () => {
    // maxLength is not enforced while an input method composes, so 70 Chinese
    // characters can still reach the field; a change event sets them here.
    expect(await startWithTabNamed('CF201 lab', '學'.repeat(70))).toBe('學'.repeat(64));
  });

  it('sends no name, and shows none, for a tab whose name is not text', async () => {
    const tab = useTabStore.getState().getActiveTab();
    useTabStore.setState((state) => ({
      tabs: state.tabs.map((candidate) => candidate.id === tab.id
        ? { ...candidate, name: undefined as unknown as string }
        : candidate),
    }));
    const onClose = vi.fn();
    render(<NewSweepDialog onClose={onClose} />);
    expect(screen.getByLabelText('Name')).toHaveAttribute('placeholder', '');
    fireEvent.change(screen.getByLabelText('Values'), { target: { value: '2, 4' } });
    await act(async () => {
      fireEvent.click(startButton());
    });
    await waitFor(() => expect(onClose).toHaveBeenCalledTimes(1));
    expect(api.createSweep.mock.calls[0][0].name).toBeNull();
  });

  it('defaults the objective to the series the built-in training loop records', () => {
    useRunStore.setState({
      runs: [{ final_metrics: { eval_accuracy: 0.9, train_loss: 0.2 } }] as never,
    });
    render(<NewSweepDialog onClose={vi.fn()} />);
    const objective = screen.getByLabelText(/objective metric/i) as HTMLInputElement;
    expect(objective.value).toBe('train_loss');
    const suggestions = Array.from(document.getElementById(objective.getAttribute('list')!)!.querySelectorAll('option'))
      .map((option) => option.getAttribute('value'));
    expect(suggestions).toEqual(['eval_accuracy', 'train_loss', 'val_accuracy', 'val_loss']);
  });

  it('warns past the default server caps but leaves the decision to the server', async () => {
    const createSweep = vi.fn().mockResolvedValue(true);
    useSweepStore.setState({ createSweep });
    render(<NewSweepDialog onClose={vi.fn()} />);
    fireEvent.change(screen.getByLabelText('Parameter'), { target: { value: 'train.rate' } });
    fireEvent.change(screen.getByLabelText('Values'), {
      target: { value: Array.from({ length: 33 }, (_, index) => String(index / 32)).join(', ') },
    });

    expect(screen.getByText(/33 values; the server allows 32 per parameter unless CODEFYUI_MAX_SWEEP_DOMAIN/)).toBeInTheDocument();
    expect(screen.getByText(/33 variants; the server allows 32 unless CODEFYUI_MAX_SWEEP_RUNS is raised/)).toBeInTheDocument();
    // The count is in the warning; the plain preview line would say it twice.
    expect(screen.queryByText('33 variants')).toBeNull();
    expect(startButton()).toBeEnabled();
    await act(async () => {
      fireEvent.click(startButton());
    });
    expect(createSweep).toHaveBeenCalledTimes(1);

    fireEvent.change(screen.getByLabelText('Domain'), { target: { value: 'range' } });
    fireEvent.change(screen.getByLabelText('Count'), { target: { value: '100000' } });
    expect(screen.getByText(/Trainer · rate: 100000 values/)).toBeInTheDocument();
  });

  it('warns about a fifth parameter', () => {
    setupGraph([trainNode('train', 'Trainer'), trainNode('opt', 'Optimizer')]);
    render(<NewSweepDialog onClose={vi.fn()} />);
    const add = screen.getByRole('button', { name: /Add parameter/i });
    for (let count = 0; count < 4; count += 1) fireEvent.click(add);
    const values = ['2', '0.1', 'fast', 'true', '3'];
    values.forEach((value, index) => {
      fireEvent.change(editor(`Parameter ${index + 1}`).getByLabelText('Values'), { target: { value } });
    });
    expect(screen.getByText(/5 parameters; the server allows 4 unless CODEFYUI_MAX_SWEEP_PARAMS is raised/)).toBeInTheDocument();
  });

  it('shows local and server validation errors without closing', async () => {
    api.createSweep.mockRejectedValueOnce(new Error('server cap is 1'));
    const onClose = vi.fn();
    render(<NewSweepDialog onClose={onClose} />);

    fireEvent.change(screen.getByLabelText('Values'), { target: { value: 'not-an-int' } });
    await act(async () => {
      fireEvent.click(startButton());
    });
    expect(screen.getByRole('alert')).toHaveTextContent('"not-an-int" is not a finite number.');
    expect(api.createSweep).not.toHaveBeenCalled();

    fireEvent.change(screen.getByLabelText('Values'), { target: { value: '2, 4' } });
    await act(async () => {
      fireEvent.click(startButton());
    });
    expect(screen.getByRole('alert')).toHaveTextContent('server cap is 1');
    expect(onClose).not.toHaveBeenCalled();
  });

  it('words its validation in the active language', async () => {
    useI18n.setState({ locale: 'zh-TW' });
    render(<NewSweepDialog onClose={vi.fn()} />);
    fireEvent.change(screen.getByLabelText('值'), { target: { value: '2.5' } });
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: '開始掃描' }));
    });
    expect(screen.getByRole('alert')).toHaveTextContent('「2.5」必須是整數。');
  });
});
