import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { act, render, screen, fireEvent, waitFor, within } from '@testing-library/react';
import { CustomNodeManager, CustomNodeManagerModal } from './CustomNodeManager';
import { DialogContainer } from '../shared/DialogContainer';
import { useI18n } from '../../i18n';
import { useDialogStore } from '../../store/dialogStore';
import * as rest from '../../api/rest';
import { useNodeDefStore } from '../../store/nodeDefStore';
import { useTabStore } from '../../store/tabStore';
import { useUIStore } from '../../store/uiStore';

// Mock the REST seam — the manager calls list/toggle/delete/upload.
vi.mock('../../api/rest', () => ({
  listCustomNodes: vi.fn(),
  toggleCustomNode: vi.fn(),
  uploadCustomNode: vi.fn(),
  deleteCustomNode: vi.fn(),
}));

const mockedRest = vi.mocked(rest);

const ORIGINAL_TABS = useTabStore.getState().tabs;
const ORIGINAL_ACTIVE = useTabStore.getState().activeTabId;

function customNode(overrides: Partial<rest.CustomNodeInfo> = {}): rest.CustomNodeInfo {
  return {
    filename: 'my_node.py',
    enabled: true,
    nodes: ['MyNode'],
    ...overrides,
  };
}

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  useDialogStore.setState({ active: null, resolve: null });
  useUIStore.setState({ customNodeManagerOpen: false });
  // Stub the store reload() so toggling/deleting/uploading doesn't hit fetch.
  vi.spyOn(useNodeDefStore.getState(), 'reload').mockResolvedValue(undefined);
  // Sensible defaults; individual tests override as needed.
  mockedRest.listCustomNodes.mockResolvedValue([]);
  mockedRest.toggleCustomNode.mockResolvedValue({});
  mockedRest.deleteCustomNode.mockResolvedValue({});
  mockedRest.uploadCustomNode.mockResolvedValue({});
});

afterEach(() => {
  vi.restoreAllMocks();
  vi.clearAllMocks();
  useTabStore.setState({ tabs: ORIGINAL_TABS, activeTabId: ORIGINAL_ACTIVE });
});

describe('CustomNodeManager', () => {
  it('shows a loading message while fetching, then the empty state', async () => {
    // Defer the resolution so we can observe the loading message.
    let resolveList: (v: rest.CustomNodeInfo[]) => void = () => {};
    mockedRest.listCustomNodes.mockReturnValue(
      new Promise((res) => {
        resolveList = res;
      }),
    );
    render(<CustomNodeManager onClose={vi.fn()} />);
    expect(screen.getByText('Loading...')).toBeTruthy();
    resolveList([]);
    // Upload .py is the only control in this modal, so its empty state names
    // it. The sidebar's section keeps the bare `customNodes.empty`: its own
    // action is the Manage... button that opens this modal.
    expect(
      await screen.findByText('No custom nodes yet. Upload a .py file to add one.'),
    ).toBeTruthy();
  });

  it('renders the list of custom nodes with their node names', async () => {
    mockedRest.listCustomNodes.mockResolvedValue([
      customNode({ filename: 'a.py', enabled: true, nodes: ['Alpha', 'Beta'] }),
      customNode({ filename: 'b.py', enabled: false, nodes: [] }),
    ]);
    render(<CustomNodeManager onClose={vi.fn()} />);
    expect(await screen.findByText('a.py')).toBeTruthy();
    expect(screen.getByText('Alpha, Beta')).toBeTruthy();
    expect(screen.getByText('b.py')).toBeTruthy();
    // a.py is enabled, b.py disabled — toggle button labels reflect that.
    expect(screen.getByText('Enabled')).toBeTruthy();
    expect(screen.getByText('Disabled')).toBeTruthy();
    // b.py has no node names → no joined-names span.
    const bRow = screen.getByText('b.py').closest('div')!.parentElement!;
    expect(within(bRow).queryByText(/,/)).toBeNull();
  });

  it('toggling a node calls toggleCustomNode, refetches and reloads', async () => {
    mockedRest.listCustomNodes
      .mockResolvedValueOnce([customNode({ filename: 'a.py', enabled: true })])
      .mockResolvedValueOnce([customNode({ filename: 'a.py', enabled: false })]);
    render(<CustomNodeManager onClose={vi.fn()} />);
    fireEvent.click(await screen.findByText('Enabled'));
    await waitFor(() => {
      expect(mockedRest.toggleCustomNode).toHaveBeenCalledWith('a.py');
    });
    expect(useNodeDefStore.getState().reload).toHaveBeenCalled();
    // List re-fetched: now Disabled.
    expect(await screen.findByText('Disabled')).toBeTruthy();
  });

  it('surfaces an error when toggle fails', async () => {
    mockedRest.listCustomNodes.mockResolvedValue([customNode({ filename: 'a.py' })]);
    mockedRest.toggleCustomNode.mockRejectedValue(new Error('toggle boom'));
    render(<CustomNodeManager onClose={vi.fn()} />);
    fireEvent.click(await screen.findByText('Enabled'));
    expect(await screen.findByText('toggle boom')).toBeTruthy();
  });

  it('surfaces an error when the initial list fails', async () => {
    mockedRest.listCustomNodes.mockRejectedValue(new Error('list boom'));
    render(<CustomNodeManager onClose={vi.fn()} />);
    expect(await screen.findByText('list boom')).toBeTruthy();
  });

  it('deleting a node asks for confirmation; confirming deletes + reloads', async () => {
    mockedRest.listCustomNodes
      .mockResolvedValueOnce([customNode({ filename: 'a.py' })])
      .mockResolvedValueOnce([]);
    render(<CustomNodeManager onClose={vi.fn()} />);
    fireEvent.click(await screen.findByText('Delete'));
    // confirm() opened a dialog on the store.
    await waitFor(() => {
      expect(useDialogStore.getState().active).not.toBeNull();
    });
    expect(useDialogStore.getState().active?.title).toBe(
      'Delete "a.py"? This cannot be undone.',
    );
    // Approve.
    useDialogStore.getState().close(true);
    await waitFor(() => {
      expect(mockedRest.deleteCustomNode).toHaveBeenCalledWith('a.py');
    });
    expect(useNodeDefStore.getState().reload).toHaveBeenCalled();
  });

  it('passes a translated confirmText for the delete confirmation, not a raw literal (#160)', async () => {
    useI18n.setState({ locale: 'zh-TW' });
    mockedRest.listCustomNodes.mockResolvedValue([customNode({ filename: 'a.py' })]);
    render(<CustomNodeManager onClose={vi.fn()} />);
    // The row's own delete button is already translated -- find it by its
    // zh-TW text, the same string the confirm dialog's button should use.
    fireEvent.click(await screen.findByText('刪除'));
    await waitFor(() => {
      expect(useDialogStore.getState().active).not.toBeNull();
    });
    expect(useDialogStore.getState().active?.confirmText).toBe('刪除');
  });

  it('deleting a node does nothing when the confirmation is cancelled', async () => {
    mockedRest.listCustomNodes.mockResolvedValue([customNode({ filename: 'a.py' })]);
    render(<CustomNodeManager onClose={vi.fn()} />);
    fireEvent.click(await screen.findByText('Delete'));
    await waitFor(() => {
      expect(useDialogStore.getState().active).not.toBeNull();
    });
    useDialogStore.getState().close(false);
    await waitFor(() => {
      expect(useDialogStore.getState().active).toBeNull();
    });
    expect(mockedRest.deleteCustomNode).not.toHaveBeenCalled();
  });

  it('surfaces an error when delete fails', async () => {
    mockedRest.listCustomNodes.mockResolvedValue([customNode({ filename: 'a.py' })]);
    mockedRest.deleteCustomNode.mockRejectedValue(new Error('delete boom'));
    render(<CustomNodeManager onClose={vi.fn()} />);
    fireEvent.click(await screen.findByText('Delete'));
    await waitFor(() => {
      expect(useDialogStore.getState().active).not.toBeNull();
    });
    useDialogStore.getState().close(true);
    expect(await screen.findByText('delete boom')).toBeTruthy();
  });

  it('uploading a .py file calls uploadCustomNode and refetches', async () => {
    mockedRest.listCustomNodes
      .mockResolvedValueOnce([])
      .mockResolvedValueOnce([customNode({ filename: 'uploaded.py' })]);
    const { container } = render(<CustomNodeManager onClose={vi.fn()} />);
    await screen.findByText('No custom nodes yet. Upload a .py file to add one.');
    const fileInput = container.querySelector(
      'input[type="file"]',
    ) as HTMLInputElement;
    const file = new File(['print(1)'], 'uploaded.py', { type: 'text/x-python' });
    fireEvent.change(fileInput, { target: { files: [file] } });
    await waitFor(() => {
      expect(mockedRest.uploadCustomNode).toHaveBeenCalledWith(file);
    });
    expect(useNodeDefStore.getState().reload).toHaveBeenCalled();
    expect(await screen.findByText('uploaded.py')).toBeTruthy();
    // Input value reset after upload.
    expect(fileInput.value).toBe('');
  });

  it('upload is a no-op when no file is selected', async () => {
    mockedRest.listCustomNodes.mockResolvedValue([]);
    const { container } = render(<CustomNodeManager onClose={vi.fn()} />);
    await screen.findByText('No custom nodes yet. Upload a .py file to add one.');
    const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement;
    fireEvent.change(fileInput, { target: { files: [] } });
    expect(mockedRest.uploadCustomNode).not.toHaveBeenCalled();
  });

  it('surfaces an error when upload fails', async () => {
    mockedRest.listCustomNodes.mockResolvedValue([]);
    mockedRest.uploadCustomNode.mockRejectedValue(new Error('upload boom'));
    const { container } = render(<CustomNodeManager onClose={vi.fn()} />);
    await screen.findByText('No custom nodes yet. Upload a .py file to add one.');
    const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement;
    const file = new File(['x'], 'bad.py');
    fireEvent.change(fileInput, { target: { files: [file] } });
    expect(await screen.findByText('upload boom')).toBeTruthy();
  });

  it('the upload button proxies the click to the hidden file input', async () => {
    mockedRest.listCustomNodes.mockResolvedValue([]);
    const { container } = render(<CustomNodeManager onClose={vi.fn()} />);
    await screen.findByText('No custom nodes yet. Upload a .py file to add one.');
    const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement;
    const clickSpy = vi.spyOn(fileInput, 'click').mockImplementation(() => {});
    fireEvent.click(screen.getByText('Upload .py'));
    expect(clickSpy).toHaveBeenCalled();
  });

  it('clicking the overlay (but not the modal body) calls onClose', async () => {
    mockedRest.listCustomNodes.mockResolvedValue([]);
    const onClose = vi.fn();
    const { container } = render(<CustomNodeManager onClose={onClose} />);
    await screen.findByText('No custom nodes yet. Upload a .py file to add one.');
    // Overlay is the root element.
    const overlay = container.firstChild as HTMLElement;
    fireEvent.click(overlay);
    expect(onClose).toHaveBeenCalledTimes(1);

    // Clicking inside the modal stops propagation → onClose not called again.
    onClose.mockClear();
    const modalTitle = screen.getByText('Custom Node Manager');
    fireEvent.click(modalTitle);
    expect(onClose).not.toHaveBeenCalled();
  });

  it('the header close button calls onClose', async () => {
    mockedRest.listCustomNodes.mockResolvedValue([]);
    const onClose = vi.fn();
    render(<CustomNodeManager onClose={onClose} />);
    await screen.findByText('No custom nodes yet. Upload a .py file to add one.');
    fireEvent.click(screen.getByRole('button', { name: 'Close custom node manager' }));
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});

/** A promise the test settles by hand, for a request that is still out. */
function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((res) => {
    resolve = res;
  });
  return { promise, resolve };
}

/** The row that lists `filename`: the name's span sits in the info column. */
function rowOf(filename: string): HTMLElement {
  return screen.getByText(filename).closest('div')!.parentElement!;
}

function toggleIn(filename: string): HTMLElement {
  return within(rowOf(filename)).getByRole('button', { name: /^(Enabled|Disabled)$/ });
}

function deleteIn(filename: string): HTMLElement {
  return within(rowOf(filename)).getByRole('button', { name: 'Delete' });
}

/** Press a row's Delete from the keyboard and say yes to the confirm. */
async function deleteByKeyboard(filename: string) {
  const button = deleteIn(filename);
  // A keyboard press: the button holds focus when it is activated.
  button.focus();
  fireEvent.click(button);
  await waitFor(() => {
    expect(useDialogStore.getState().active).not.toBeNull();
  });
  act(() => useDialogStore.getState().close(true));
}

// Keyboard focus through an action (#504). A re-read used to swap every row
// for "Loading...", and a row was keyed by its file name, which Enable and
// Disable change -- either way the button just pressed left the page, and
// focus fell to the page body.
describe('CustomNodeManager focus and re-reads', () => {
  afterEach(() => {
    // These cases queue answers with the `...Once` helpers. `clearAllMocks`
    // keeps a queue a failing case left unread, and the next case would get it.
    mockedRest.listCustomNodes.mockReset();
    mockedRest.toggleCustomNode.mockReset();
    mockedRest.deleteCustomNode.mockReset();
  });

  it('keeps the rows up while it re-reads, and focus on the toggle that was pressed', async () => {
    const reread = deferred<rest.CustomNodeInfo[]>();
    mockedRest.listCustomNodes
      .mockResolvedValueOnce([
        customNode({ filename: 'a.py', enabled: true }),
        customNode({ filename: 'b.py', enabled: true }),
      ])
      .mockReturnValueOnce(reread.promise);
    render(<CustomNodeManager onClose={vi.fn()} />);
    await screen.findByText('a.py');
    const toggle = toggleIn('a.py');
    toggle.focus();
    fireEvent.click(toggle);

    await waitFor(() => expect(mockedRest.listCustomNodes).toHaveBeenCalledTimes(2));
    expect(screen.queryByText('Loading...')).toBeNull();
    expect(screen.getByText('b.py')).toBeTruthy();
    expect(document.activeElement).toBe(toggle);

    // The toggle renamed the file, and it is still the same row -- the same
    // button, with focus still on it.
    await act(async () => {
      reread.resolve([
        customNode({ filename: 'a.py.disabled', enabled: false }),
        customNode({ filename: 'b.py', enabled: true }),
      ]);
    });
    expect(screen.getByText('a.py.disabled')).toBeTruthy();
    expect(toggle.isConnected).toBe(true);
    expect(toggle.textContent).toBe('Disabled');
    expect(document.activeElement).toBe(toggle);
  });

  it('keeps focus on the toggle when a file is enabled again', async () => {
    mockedRest.listCustomNodes
      .mockResolvedValueOnce([customNode({ filename: 'a.py.disabled', enabled: false })])
      .mockResolvedValueOnce([customNode({ filename: 'a.py', enabled: true })]);
    render(<CustomNodeManager onClose={vi.fn()} />);
    await screen.findByText('a.py.disabled');
    const toggle = toggleIn('a.py.disabled');
    toggle.focus();
    fireEvent.click(toggle);

    expect(await screen.findByText('a.py')).toBeTruthy();
    expect(mockedRest.toggleCustomNode).toHaveBeenCalledWith('a.py.disabled');
    expect(toggle.textContent).toBe('Enabled');
    expect(document.activeElement).toBe(toggle);
  });

  it('lists a file and its disabled copy side by side, each once', async () => {
    // Uploading x.py again after disabling it leaves both on disk. Two rows
    // cannot share one key, or React would drop or merge one of them.
    const warn = vi.spyOn(console, 'error').mockImplementation(() => {});
    mockedRest.listCustomNodes.mockResolvedValue([
      customNode({ filename: 'x.py', enabled: true }),
      customNode({ filename: 'x.py.disabled', enabled: false }),
    ]);
    render(<CustomNodeManager onClose={vi.fn()} />);
    await screen.findByText('x.py.disabled');
    expect(screen.getAllByText(/^x\.py(\.disabled)?$/)).toHaveLength(2);
    const duplicateKeys = warn.mock.calls.filter((args) => String(args[0]).includes('same key'));
    expect(duplicateKeys).toEqual([]);
  });

  it('ignores a second press on a row whose toggle is still out', async () => {
    // Its row still shows the old file name until the re-read lands, and a
    // second toggle of a name the server has just renamed away is a 404.
    const toggled = deferred<unknown>();
    mockedRest.toggleCustomNode.mockReturnValueOnce(toggled.promise);
    mockedRest.listCustomNodes
      .mockResolvedValueOnce([customNode({ filename: 'a.py', enabled: true })])
      .mockResolvedValueOnce([customNode({ filename: 'a.py.disabled', enabled: false })]);
    render(<CustomNodeManager onClose={vi.fn()} />);
    await screen.findByText('a.py');
    const toggle = toggleIn('a.py');
    fireEvent.click(toggle);
    fireEvent.click(toggle);
    expect(mockedRest.toggleCustomNode).toHaveBeenCalledTimes(1);

    await act(async () => {
      toggled.resolve({ filename: 'a.py.disabled', enabled: false });
    });
    await screen.findByText('a.py.disabled');
    expect(mockedRest.toggleCustomNode).toHaveBeenCalledTimes(1);
    expect(screen.queryByText(/Toggle failed/)).toBeNull();
  });

  it('after a Delete, puts focus on the next row\'s Delete', async () => {
    mockedRest.listCustomNodes
      .mockResolvedValueOnce([
        customNode({ filename: 'a.py' }),
        customNode({ filename: 'b.py' }),
        customNode({ filename: 'c.py' }),
      ])
      .mockResolvedValueOnce([customNode({ filename: 'a.py' }), customNode({ filename: 'c.py' })]);
    render(
      <>
        <CustomNodeManager onClose={vi.fn()} />
        <DialogContainer />
      </>,
    );
    await screen.findByText('b.py');
    await deleteByKeyboard('b.py');

    await waitFor(() => expect(screen.queryByText('b.py')).toBeNull());
    expect(mockedRest.deleteCustomNode).toHaveBeenCalledWith('b.py');
    await waitFor(() => expect(document.activeElement).toBe(deleteIn('c.py')));
  });

  it('after deleting the last row, puts focus on the previous row\'s Delete', async () => {
    mockedRest.listCustomNodes
      .mockResolvedValueOnce([customNode({ filename: 'a.py' }), customNode({ filename: 'b.py' })])
      .mockResolvedValueOnce([customNode({ filename: 'a.py' })]);
    render(
      <>
        <CustomNodeManager onClose={vi.fn()} />
        <DialogContainer />
      </>,
    );
    await screen.findByText('b.py');
    await deleteByKeyboard('b.py');

    await waitFor(() => expect(screen.queryByText('b.py')).toBeNull());
    await waitFor(() => expect(document.activeElement).toBe(deleteIn('a.py')));
  });

  it('after deleting the only row, puts focus on Upload .py', async () => {
    mockedRest.listCustomNodes
      .mockResolvedValueOnce([customNode({ filename: 'a.py' })])
      .mockResolvedValueOnce([]);
    render(
      <>
        <CustomNodeManager onClose={vi.fn()} />
        <DialogContainer />
      </>,
    );
    await screen.findByText('a.py');
    await deleteByKeyboard('a.py');

    await screen.findByText('No custom nodes yet. Upload a .py file to add one.');
    await waitFor(() =>
      expect(document.activeElement).toBe(screen.getByRole('button', { name: 'Upload .py' })),
    );
  });

  it('leaves focus alone when a Delete fails and the row stays', async () => {
    mockedRest.listCustomNodes.mockResolvedValue([
      customNode({ filename: 'a.py' }),
      customNode({ filename: 'b.py' }),
    ]);
    mockedRest.deleteCustomNode.mockRejectedValue(new Error('delete boom'));
    render(
      <>
        <CustomNodeManager onClose={vi.fn()} />
        <DialogContainer />
      </>,
    );
    await screen.findByText('a.py');
    await deleteByKeyboard('a.py');

    expect(await screen.findByText('delete boom')).toBeTruthy();
    expect(document.activeElement).toBe(deleteIn('a.py'));
    // And the row is free again: its Delete asks once more.
    fireEvent.click(deleteIn('a.py'));
    expect(useDialogStore.getState().active).not.toBeNull();
    await act(async () => useDialogStore.getState().close(false));
  });

  it('applies only the newest list when two re-reads cross (#506)', async () => {
    const older = deferred<rest.CustomNodeInfo[]>();
    const newer = deferred<rest.CustomNodeInfo[]>();
    mockedRest.listCustomNodes
      .mockResolvedValueOnce([
        customNode({ filename: 'a.py', enabled: true }),
        customNode({ filename: 'b.py', enabled: true }),
      ])
      .mockReturnValueOnce(older.promise)
      .mockReturnValueOnce(newer.promise);
    render(<CustomNodeManager onClose={vi.fn()} />);
    await screen.findByText('a.py');

    fireEvent.click(toggleIn('a.py'));
    await waitFor(() => expect(mockedRest.listCustomNodes).toHaveBeenCalledTimes(2));
    fireEvent.click(toggleIn('b.py'));
    await waitFor(() => expect(mockedRest.listCustomNodes).toHaveBeenCalledTimes(3));

    // The answer to the second read lands first; the first read's, which
    // predates the second toggle, lands after it and must not win.
    await act(async () => {
      newer.resolve([
        customNode({ filename: 'a.py.disabled', enabled: false }),
        customNode({ filename: 'b.py.disabled', enabled: false }),
      ]);
    });
    await act(async () => {
      older.resolve([
        customNode({ filename: 'a.py.disabled', enabled: false }),
        customNode({ filename: 'b.py', enabled: true }),
      ]);
    });
    expect(screen.getByText('b.py.disabled')).toBeTruthy();
    expect(screen.queryByText('b.py')).toBeNull();
  });

  it('keeps a failed action\'s message up when a re-read begun before it lands', async () => {
    const rereadA = deferred<rest.CustomNodeInfo[]>();
    mockedRest.listCustomNodes
      .mockResolvedValueOnce([
        customNode({ filename: 'a.py', enabled: true }),
        customNode({ filename: 'b.py', enabled: true }),
      ])
      .mockReturnValueOnce(rereadA.promise)
      .mockResolvedValueOnce([
        customNode({ filename: 'a.py', enabled: true }),
        customNode({ filename: 'b.py', enabled: true }),
      ]);
    mockedRest.toggleCustomNode
      .mockResolvedValueOnce({})
      .mockRejectedValueOnce(new Error('toggle b boom'));
    render(<CustomNodeManager onClose={vi.fn()} />);
    await screen.findByText('a.py');

    fireEvent.click(toggleIn('a.py'));
    await waitFor(() => expect(mockedRest.listCustomNodes).toHaveBeenCalledTimes(2));
    fireEvent.click(toggleIn('b.py'));
    expect(await screen.findByText('toggle b boom')).toBeTruthy();

    // a.py's re-read was out before b.py's toggle failed and knows nothing of
    // it: b.py still says Enabled, so the message is the only word on it.
    await act(async () => {
      rereadA.resolve([
        customNode({ filename: 'a.py.disabled', enabled: false }),
        customNode({ filename: 'b.py', enabled: true }),
      ]);
    });
    expect(screen.getByText('a.py.disabled')).toBeTruthy();
    expect(screen.getByText('toggle b boom')).toBeTruthy();

    // A read begun after the failure does take it down.
    fireEvent.click(toggleIn('a.py.disabled'));
    await waitFor(() => expect(screen.queryByText('toggle b boom')).toBeNull());
  });

  it('frees a row whose toggle failed: a second press sends a second request', async () => {
    mockedRest.listCustomNodes
      .mockResolvedValueOnce([customNode({ filename: 'a.py', enabled: true })])
      .mockResolvedValueOnce([customNode({ filename: 'a.py.disabled', enabled: false })]);
    mockedRest.toggleCustomNode.mockRejectedValueOnce(new Error('toggle boom'));
    render(<CustomNodeManager onClose={vi.fn()} />);
    await screen.findByText('a.py');
    fireEvent.click(toggleIn('a.py'));
    expect(await screen.findByText('toggle boom')).toBeTruthy();

    fireEvent.click(toggleIn('a.py'));
    await screen.findByText('a.py.disabled');
    expect(mockedRest.toggleCustomNode).toHaveBeenCalledTimes(2);
    expect(mockedRest.toggleCustomNode).toHaveBeenLastCalledWith('a.py');
  });

  it('frees a toggled row once its re-read lands, before the palette reload ends', async () => {
    // The reload is a full rediscovery on the server. Holding the row through
    // it made a second Enter -- an undo -- do nothing, silently.
    const paletteReload = deferred<void>();
    vi.mocked(useNodeDefStore.getState().reload).mockReturnValueOnce(paletteReload.promise);
    mockedRest.listCustomNodes
      .mockResolvedValueOnce([customNode({ filename: 'a.py', enabled: true })])
      .mockResolvedValueOnce([customNode({ filename: 'a.py.disabled', enabled: false })])
      .mockResolvedValueOnce([customNode({ filename: 'a.py', enabled: true })]);
    render(<CustomNodeManager onClose={vi.fn()} />);
    await screen.findByText('a.py');
    fireEvent.click(toggleIn('a.py'));
    await screen.findByText('a.py.disabled');
    await waitFor(() => expect(useNodeDefStore.getState().reload).toHaveBeenCalledTimes(1));

    // The reload is still out, and the second press goes by the new name.
    fireEvent.click(toggleIn('a.py.disabled'));
    await waitFor(() => expect(mockedRest.toggleCustomNode).toHaveBeenCalledTimes(2));
    expect(mockedRest.toggleCustomNode).toHaveBeenLastCalledWith('a.py.disabled');
    await screen.findByText('a.py');
    await act(async () => paletteReload.resolve());
  });

  it('holds a toggled row until its new name is drawn, not only until the answer arrives', async () => {
    const reread = deferred<rest.CustomNodeInfo[]>();
    mockedRest.listCustomNodes
      .mockResolvedValueOnce([customNode({ filename: 'a.py', enabled: true })])
      .mockReturnValueOnce(reread.promise);
    render(<CustomNodeManager onClose={vi.fn()} />);
    await screen.findByText('a.py');
    const toggle = toggleIn('a.py');
    fireEvent.click(toggle);
    await waitFor(() => expect(mockedRest.listCustomNodes).toHaveBeenCalledTimes(2));

    // Outside act, as in a browser: React draws the answer in a task of its
    // own, after the promise callbacks that receive it. A press in between --
    // a held Enter repeating as the re-read lands -- reaches the button as it
    // is still drawn, carrying the old file name.
    const env = globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean };
    const actEnvironment = env.IS_REACT_ACT_ENVIRONMENT;
    env.IS_REACT_ACT_ENVIRONMENT = false;
    try {
      reread.resolve([customNode({ filename: 'a.py.disabled', enabled: false })]);
      for (let i = 0; i < 5; i++) await Promise.resolve();
      // Not drawn yet, or this would test nothing.
      expect(toggle.textContent).toBe('Enabled');
      toggle.dispatchEvent(new MouseEvent('click', { bubbles: true }));
      expect(mockedRest.toggleCustomNode).toHaveBeenCalledTimes(1);
    } finally {
      env.IS_REACT_ACT_ENVIRONMENT = actEnvironment;
    }

    // Once drawn, the next press goes by the new name.
    await screen.findByText('a.py.disabled');
    fireEvent.click(toggle);
    await waitFor(() => expect(mockedRest.toggleCustomNode).toHaveBeenCalledTimes(2));
    expect(mockedRest.toggleCustomNode).toHaveBeenLastCalledWith('a.py.disabled');
  });

  it('frees a toggled row when its re-read fails, rather than holding it for good', async () => {
    // No list is coming to be drawn, so nothing else would ever free it: a
    // press would do nothing, silently, until the manager was reopened.
    mockedRest.listCustomNodes
      .mockResolvedValueOnce([customNode({ filename: 'a.py', enabled: true })])
      .mockRejectedValueOnce(new Error('list boom'))
      .mockResolvedValueOnce([customNode({ filename: 'a.py', enabled: true })]);
    render(<CustomNodeManager onClose={vi.fn()} />);
    await screen.findByText('a.py');
    fireEvent.click(toggleIn('a.py'));
    expect(await screen.findByText('list boom')).toBeTruthy();

    fireEvent.click(toggleIn('a.py'));
    await waitFor(() => expect(mockedRest.toggleCustomNode).toHaveBeenCalledTimes(2));
  });

  it('frees a row whose Delete was cancelled: Delete asks again, and its toggle works', async () => {
    mockedRest.listCustomNodes
      .mockResolvedValueOnce([customNode({ filename: 'a.py', enabled: true })])
      .mockResolvedValueOnce([customNode({ filename: 'a.py.disabled', enabled: false })]);
    render(<CustomNodeManager onClose={vi.fn()} />);
    await screen.findByText('a.py');

    fireEvent.click(deleteIn('a.py'));
    expect(useDialogStore.getState().active).not.toBeNull();
    await act(async () => useDialogStore.getState().close(false));

    fireEvent.click(deleteIn('a.py'));
    expect(useDialogStore.getState().active).not.toBeNull();
    await act(async () => useDialogStore.getState().close(false));

    fireEvent.click(toggleIn('a.py'));
    await screen.findByText('a.py.disabled');
    expect(mockedRest.toggleCustomNode).toHaveBeenCalledWith('a.py');
    expect(mockedRest.deleteCustomNode).not.toHaveBeenCalled();
  });

  it('leaves focus where the user moved it while a Delete\'s re-read was out', async () => {
    const reread = deferred<rest.CustomNodeInfo[]>();
    mockedRest.listCustomNodes
      .mockResolvedValueOnce([
        customNode({ filename: 'a.py' }),
        customNode({ filename: 'b.py' }),
        customNode({ filename: 'c.py' }),
      ])
      .mockReturnValueOnce(reread.promise);
    render(
      <>
        <CustomNodeManager onClose={vi.fn()} />
        <DialogContainer />
      </>,
    );
    await screen.findByText('b.py');
    await deleteByKeyboard('b.py');
    await waitFor(() => expect(mockedRest.listCustomNodes).toHaveBeenCalledTimes(2));

    const elsewhere = toggleIn('a.py');
    elsewhere.focus();
    await act(async () => {
      reread.resolve([customNode({ filename: 'a.py' }), customNode({ filename: 'c.py' })]);
    });
    expect(screen.queryByText('b.py')).toBeNull();
    expect(document.activeElement).toBe(elsewhere);
  });

  it('ignores a second Delete on a row whose Delete is already asking', async () => {
    mockedRest.listCustomNodes.mockResolvedValueOnce([customNode({ filename: 'a.py' })]);
    render(<CustomNodeManager onClose={vi.fn()} />);
    await screen.findByText('a.py');
    fireEvent.click(deleteIn('a.py'));
    const asking = useDialogStore.getState().active;
    expect(asking).not.toBeNull();

    fireEvent.click(deleteIn('a.py'));
    // Still the first confirm: a second one would have replaced it.
    expect(useDialogStore.getState().active).toBe(asking);
    await act(async () => useDialogStore.getState().close(false));
  });

  it('ignores Delete on a row whose toggle is still out, and asks once it has landed', async () => {
    const toggled = deferred<unknown>();
    mockedRest.toggleCustomNode.mockReturnValueOnce(toggled.promise);
    mockedRest.listCustomNodes
      .mockResolvedValueOnce([customNode({ filename: 'a.py', enabled: true })])
      .mockResolvedValueOnce([customNode({ filename: 'a.py.disabled', enabled: false })]);
    render(<CustomNodeManager onClose={vi.fn()} />);
    await screen.findByText('a.py');
    fireEvent.click(toggleIn('a.py'));
    fireEvent.click(deleteIn('a.py'));
    expect(useDialogStore.getState().active).toBeNull();

    await act(async () => {
      toggled.resolve({ filename: 'a.py.disabled', enabled: false });
    });
    await screen.findByText('a.py.disabled');
    fireEvent.click(deleteIn('a.py.disabled'));
    expect(useDialogStore.getState().active?.title).toBe(
      'Delete "a.py.disabled"? This cannot be undone.',
    );
    await act(async () => useDialogStore.getState().close(false));
  });

  it('ignores the row\'s toggle while its Delete is asking', async () => {
    mockedRest.listCustomNodes.mockResolvedValueOnce([customNode({ filename: 'a.py' })]);
    render(<CustomNodeManager onClose={vi.fn()} />);
    await screen.findByText('a.py');
    fireEvent.click(deleteIn('a.py'));
    fireEvent.click(toggleIn('a.py'));
    expect(mockedRest.toggleCustomNode).not.toHaveBeenCalled();
    await act(async () => useDialogStore.getState().close(false));
  });
});

// The one manager both of its buttons open, mounted at the app root and
// driven by `uiStore.customNodeManagerOpen` (the Custom Nodes manager issue).
describe('CustomNodeManagerModal', () => {
  it('renders nothing, and reads nothing, while the flag is down', () => {
    const { container } = render(<CustomNodeManagerModal />);
    expect(container.firstChild).toBeNull();
    expect(mockedRest.listCustomNodes).not.toHaveBeenCalled();
  });

  it('renders the manager while the flag is up, even with a tab store that holds no tabs', async () => {
    // Since #472 `tabs: []` is a real state, and a root-mounted component
    // that reads the active tab with `!` crashes in it. That the manager is
    // mounted where it outlives the graph toolbar is App.test.tsx's to prove.
    useTabStore.setState({ tabs: [], activeTabId: null } as any);
    useUIStore.setState({ customNodeManagerOpen: true });
    render(<CustomNodeManagerModal />);
    expect(screen.getByRole('heading', { name: 'Custom Node Manager' })).toBeTruthy();
    await screen.findByText('No custom nodes yet. Upload a .py file to add one.');
  });

  // Keyboard reach (the Custom Nodes manager issue). Mounted at the app root,
  // the manager comes after the whole editor in the page, so a manager that
  // took no focus left a keyboard user to Tab through the sidebar, every node
  // and edge on the canvas and the panels to reach it.
  it('is a dialog named by its title, and takes focus as it opens', async () => {
    render(
      <>
        <button type="button">opener</button>
        <CustomNodeManagerModal />
      </>,
    );
    screen.getByRole('button', { name: 'opener' }).focus();
    act(() => useUIStore.getState().openCustomNodeManager());

    const dialog = screen.getByRole('dialog', { name: 'Custom Node Manager' });
    expect(dialog.getAttribute('aria-modal')).toBe('true');
    expect(document.activeElement).toBe(dialog);
    await screen.findByText('No custom nodes yet. Upload a .py file to add one.');
  });

  it('gives focus back to what had it when it closes', async () => {
    render(
      <>
        <button type="button">opener</button>
        <CustomNodeManagerModal />
      </>,
    );
    const opener = screen.getByRole('button', { name: 'opener' });
    opener.focus();
    act(() => useUIStore.getState().openCustomNodeManager());
    await screen.findByText('No custom nodes yet. Upload a .py file to add one.');
    // Where focus went, or its coming back would prove nothing.
    expect(document.activeElement).toBe(screen.getByRole('dialog'));

    fireEvent.click(screen.getByRole('button', { name: 'Close custom node manager' }));
    expect(document.activeElement).toBe(opener);
  });

  it('closes on Escape', async () => {
    useUIStore.setState({ customNodeManagerOpen: true });
    const { container } = render(<CustomNodeManagerModal />);
    await screen.findByText('No custom nodes yet. Upload a .py file to add one.');

    fireEvent.keyDown(document.activeElement!, { key: 'Escape' });
    expect(useUIStore.getState().customNodeManagerOpen).toBe(false);
    expect(container.firstChild).toBeNull();
  });

  it('leaves Escape to the confirm dialog open over it', async () => {
    // Deleting a file asks first, and the confirm renders above the manager
    // with an Escape of its own. One press closes one window: the one on top.
    mockedRest.listCustomNodes.mockResolvedValue([customNode({ filename: 'a.py' })]);
    useUIStore.setState({ customNodeManagerOpen: true });
    render(
      <>
        <CustomNodeManagerModal />
        <DialogContainer />
      </>,
    );
    fireEvent.click(await screen.findByText('Delete'));
    await waitFor(() => {
      expect(useDialogStore.getState().active).not.toBeNull();
    });

    fireEvent.keyDown(document.activeElement!, { key: 'Escape' });
    await waitFor(() => {
      expect(useDialogStore.getState().active).toBeNull();
    });
    expect(useUIStore.getState().customNodeManagerOpen).toBe(true);
    expect(mockedRest.deleteCustomNode).not.toHaveBeenCalled();

    // The next press is the manager's.
    fireEvent.keyDown(document.activeElement!, { key: 'Escape' });
    expect(useUIStore.getState().customNodeManagerOpen).toBe(false);
  });

  it('its close button lowers the flag, and the manager goes', async () => {
    useUIStore.setState({ customNodeManagerOpen: true });
    const { container } = render(<CustomNodeManagerModal />);
    await screen.findByText('No custom nodes yet. Upload a .py file to add one.');

    fireEvent.click(screen.getByRole('button', { name: 'Close custom node manager' }));
    expect(useUIStore.getState().customNodeManagerOpen).toBe(false);
    expect(container.firstChild).toBeNull();
  });

  it('a click on the scrim lowers the flag too', async () => {
    useUIStore.setState({ customNodeManagerOpen: true });
    const { container } = render(<CustomNodeManagerModal />);
    await screen.findByText('No custom nodes yet. Upload a .py file to add one.');

    fireEvent.click(container.firstChild as HTMLElement);
    expect(useUIStore.getState().customNodeManagerOpen).toBe(false);
    expect(container.firstChild).toBeNull();
  });
});
