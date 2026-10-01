import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { NewSweepDialog } from './NewSweepDialog';
import { useI18n } from '../../i18n';
import { useNodeDefStore } from '../../store/nodeDefStore';
import { useSweepStore, _resetSweepStoreForTesting } from '../../store/sweepStore';
import { useTabStore } from '../../store/tabStore';
import type { NodeDefinition } from '../../types';

const trainDefinition: NodeDefinition = {
  node_name: 'Train', category: 'Test', description: '', inputs: [], outputs: [],
  params: [
    { name: 'epochs', param_type: 'int', default: 2, description: '', options: [], min_value: 1, max_value: 10 },
    { name: 'rate', param_type: 'float', default: 0.1, description: '', options: [], min_value: 0, max_value: 1 },
    { name: 'mode', param_type: 'select', default: 'fast', description: '', options: ['fast', 'safe'], min_value: null, max_value: null },
    { name: 'api_key', param_type: 'secret', default: '', description: '', options: [], min_value: null, max_value: null },
  ],
};

function setupGraph() {
  const tabId = useTabStore.getState().activeTabId;
  useTabStore.getState().setNodes([{
    id: 'train', type: 'customNode', position: { x: 0, y: 0 },
    data: { type: 'Train', label: 'Trainer', params: { epochs: 2, rate: 0.1, mode: 'fast', api_key: 'sk-session' } },
  }] as never);
  return tabId;
}

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  _resetSweepStoreForTesting();
  useNodeDefStore.setState({ definitions: [trainDefinition] });
  useTabStore.setState({ tabs: [], activeTabId: '' });
  useTabStore.getState().addTab('test');
  setupGraph();
});

afterEach(() => {
  act(() => { _resetSweepStoreForTesting(); });
  vi.restoreAllMocks();
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

  it('offers only eligible top-level params and previews exact grid variants', () => {
    render(<NewSweepDialog onClose={vi.fn()} />);
    const dialog = screen.getByRole('dialog');
    expect(within(dialog).getByRole('option', { name: /Trainer.*epochs/i })).toBeInTheDocument();
    expect(within(dialog).queryByRole('option', { name: /api_key/i })).toBeNull();

    fireEvent.change(within(dialog).getByLabelText(/values/i), { target: { value: '2, 4' } });
    expect(within(dialog).getByText(/2 variants/i)).toBeInTheDocument();
  });

  it('builds multiple domains and previews their exact Cartesian product', async () => {
    const createSweep = vi.fn().mockResolvedValue(undefined);
    useSweepStore.setState({ createSweep });
    render(<NewSweepDialog onClose={vi.fn()} />);
    const dialog = screen.getByRole('dialog');

    fireEvent.change(within(dialog).getByLabelText('Values'), { target: { value: '2, 4' } });
    fireEvent.click(within(dialog).getByRole('button', { name: /Add parameter/i }));
    fireEvent.change(within(dialog).getByLabelText('Parameter 2'), { target: { value: 'train.rate' } });
    fireEvent.change(within(dialog).getByLabelText('Values 2'), { target: { value: '0.1, 0.2' } });

    expect(within(dialog).getByText(/4 variants/i)).toBeInTheDocument();
    await act(async () => {
      fireEvent.click(within(dialog).getByRole('button', { name: /start sweep/i }));
    });
    expect(createSweep.mock.calls[0][0].sweep_spec.params).toEqual([
      { node_id: 'train', param: 'epochs', values: [2, 4] },
      { node_id: 'train', param: 'rate', values: [0.1, 0.2] },
    ]);
  });

  it('reveals random seed controls and warns when execution seeds serialize variants', () => {
    render(<NewSweepDialog onClose={vi.fn()} />);
    const dialog = screen.getByRole('dialog');
    fireEvent.change(within(dialog).getByLabelText(/method/i), { target: { value: 'random' } });
    fireEvent.change(within(dialog).getByLabelText(/values/i), { target: { value: '2, 3, 4' } });
    fireEvent.change(within(dialog).getByLabelText(/samples/i), { target: { value: '2' } });
    fireEvent.click(within(dialog).getByLabelText(/seed every variant/i));

    expect(within(dialog).getByLabelText(/planner seed/i)).toBeInTheDocument();
    expect(within(dialog).getByText(/run one at a time/i)).toBeInTheDocument();
    expect(within(dialog).getByText(/2 variants/i)).toBeInTheDocument();
  });

  it('submits the scrubbed durable graph, selected domain and run settings', async () => {
    const createSweep = vi.fn().mockResolvedValue(undefined);
    useSweepStore.setState({ createSweep });
    const tab = useTabStore.getState().getActiveTab();
    useTabStore.setState((state) => ({
      tabs: state.tabs.map((candidate) => candidate.id === tab.id
        ? { ...candidate, graphDevice: 'cuda:0', seed: 77, deterministic: true }
        : candidate),
    }));
    const onClose = vi.fn();
    render(<NewSweepDialog onClose={onClose} />);
    const dialog = screen.getByRole('dialog');

    fireEvent.change(within(dialog).getByLabelText(/name/i), { target: { value: 'Epoch search' } });
    fireEvent.change(within(dialog).getByLabelText(/objective metric/i), { target: { value: 'val_loss' } });
    fireEvent.change(within(dialog).getByLabelText(/values/i), { target: { value: '2, 4' } });
    fireEvent.click(within(dialog).getByRole('button', { name: /start sweep/i }));

    await waitFor(() => expect(createSweep).toHaveBeenCalledTimes(1));
    const request = createSweep.mock.calls[0][0];
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
    expect((request.base_graph.nodes[0] as any).data.params.api_key).toBe('sk-session');
    // The request may carry the key in memory so every child can execute. The
    // dialog/store retain no request copy; RunService scrubs each durable row.
    expect(useSweepStore.getState()).not.toHaveProperty('baseGraph');
    expect(useSweepStore.getState()).not.toHaveProperty('request');
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('names the local default cap and blocks a known request over 32 variants', async () => {
    const createSweep = vi.fn().mockResolvedValue(undefined);
    useSweepStore.setState({ createSweep });
    render(<NewSweepDialog onClose={vi.fn()} />);
    const dialog = screen.getByRole('dialog');
    fireEvent.change(within(dialog).getByLabelText(/parameter/i), {
      target: { value: 'train.rate' },
    });
    fireEvent.change(within(dialog).getByLabelText(/values/i), {
      target: {
        value: Array.from(
          { length: 33 },
          (_, index) => String(index / 32),
        ).join(', '),
      },
    });

    const cap = within(dialog).getByRole('alert');
    expect(cap).toHaveTextContent(/33 variants exceed the local default limit of 32/i);
    expect(within(dialog).getByRole('button', { name: /start sweep/i })).toBeDisabled();
    fireEvent.submit(within(dialog).getByRole('button', { name: /start sweep/i }).closest('form')!);
    await act(async () => { await Promise.resolve(); });
    expect(createSweep).not.toHaveBeenCalled();
  });

  it('shows local and server validation errors without closing', async () => {
    const createSweep = vi.fn(async () => {
      useSweepStore.setState({ error: 'server cap is 1' });
    });
    useSweepStore.setState({ createSweep });
    const onClose = vi.fn();
    render(<NewSweepDialog onClose={onClose} />);
    const dialog = screen.getByRole('dialog');

    fireEvent.change(within(dialog).getByLabelText(/values/i), { target: { value: 'not-an-int' } });
    await act(async () => {
      fireEvent.click(within(dialog).getByRole('button', { name: /start sweep/i }));
    });
    expect(within(dialog).getByRole('alert')).toHaveTextContent(/finite number/i);
    expect(createSweep).not.toHaveBeenCalled();

    fireEvent.change(within(dialog).getByLabelText(/values/i), { target: { value: '2, 4' } });
    await act(async () => {
      fireEvent.click(within(dialog).getByRole('button', { name: /start sweep/i }));
    });
    expect(within(dialog).getByRole('alert')).toHaveTextContent('server cap is 1');
    expect(onClose).not.toHaveBeenCalled();
  });
});
