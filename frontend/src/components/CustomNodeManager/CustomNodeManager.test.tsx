import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react';
import { CustomNodeManager } from './CustomNodeManager';
import { useI18n } from '../../i18n';
import { useDialogStore } from '../../store/dialogStore';
import * as rest from '../../api/rest';
import { useNodeDefStore } from '../../store/nodeDefStore';

// Mock the REST seam — the manager calls list/toggle/delete/upload.
vi.mock('../../api/rest', () => ({
  listCustomNodes: vi.fn(),
  toggleCustomNode: vi.fn(),
  uploadCustomNode: vi.fn(),
  deleteCustomNode: vi.fn(),
}));

const mockedRest = vi.mocked(rest);

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
    expect(
      await screen.findByText('No custom nodes. Upload a .py file to get started.'),
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
    await screen.findByText('No custom nodes. Upload a .py file to get started.');
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
    await screen.findByText('No custom nodes. Upload a .py file to get started.');
    const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement;
    fireEvent.change(fileInput, { target: { files: [] } });
    expect(mockedRest.uploadCustomNode).not.toHaveBeenCalled();
  });

  it('surfaces an error when upload fails', async () => {
    mockedRest.listCustomNodes.mockResolvedValue([]);
    mockedRest.uploadCustomNode.mockRejectedValue(new Error('upload boom'));
    const { container } = render(<CustomNodeManager onClose={vi.fn()} />);
    await screen.findByText('No custom nodes. Upload a .py file to get started.');
    const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement;
    const file = new File(['x'], 'bad.py');
    fireEvent.change(fileInput, { target: { files: [file] } });
    expect(await screen.findByText('upload boom')).toBeTruthy();
  });

  it('the upload button proxies the click to the hidden file input', async () => {
    mockedRest.listCustomNodes.mockResolvedValue([]);
    const { container } = render(<CustomNodeManager onClose={vi.fn()} />);
    await screen.findByText('No custom nodes. Upload a .py file to get started.');
    const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement;
    const clickSpy = vi.spyOn(fileInput, 'click').mockImplementation(() => {});
    fireEvent.click(screen.getByText('Upload .py'));
    expect(clickSpy).toHaveBeenCalled();
  });

  it('clicking the overlay (but not the modal body) calls onClose', async () => {
    mockedRest.listCustomNodes.mockResolvedValue([]);
    const onClose = vi.fn();
    const { container } = render(<CustomNodeManager onClose={onClose} />);
    await screen.findByText('No custom nodes. Upload a .py file to get started.');
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

  it('the header close (x) button calls onClose', async () => {
    mockedRest.listCustomNodes.mockResolvedValue([]);
    const onClose = vi.fn();
    render(<CustomNodeManager onClose={onClose} />);
    await screen.findByText('No custom nodes. Upload a .py file to get started.');
    fireEvent.click(screen.getByText('x'));
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});
