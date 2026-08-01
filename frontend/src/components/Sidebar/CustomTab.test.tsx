import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { CustomTab } from './CustomTab';
import { useI18n } from '../../i18n';
import * as rest from '../../api/rest';
import type { CustomNodeInfo, PluginSummary } from '../../api/rest';

vi.mock('../../api/rest', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../api/rest')>();
  return { ...actual, listCustomNodes: vi.fn(), listPlugins: vi.fn() };
});

// The manager modal is the existing CustomNodeManager; this tab only owns
// opening and closing it, so a stub keeps its own fetches out of these tests.
vi.mock('../CustomNodeManager/CustomNodeManager', () => ({
  CustomNodeManager: ({ onClose }: { onClose: () => void }) => (
    <div data-testid="custom-node-manager">
      <button type="button" onClick={onClose}>close manager</button>
    </div>
  ),
}));

const mockedRest = vi.mocked(rest);

function customNode(overrides: Partial<CustomNodeInfo> = {}): CustomNodeInfo {
  return { filename: 'my_node.py', enabled: true, nodes: ['MyNode'], ...overrides };
}

function plugin(overrides: Partial<PluginSummary> = {}): PluginSummary {
  return {
    id: 'c1',
    name: 'Chapter 1',
    version: '1.0.0',
    description: 'Intro nodes',
    enabled: true,
    nodes: ['EduAdd', 'EduMul'],
    ...overrides,
  };
}

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  mockedRest.listCustomNodes.mockReset();
  mockedRest.listPlugins.mockReset();
  mockedRest.listCustomNodes.mockResolvedValue([]);
  mockedRest.listPlugins.mockResolvedValue([]);
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('CustomTab', () => {
  it('shows the loading state, then both section headers', async () => {
    render(<CustomTab />);
    expect(screen.getByText('Loading...')).toBeTruthy();
    await waitFor(() => expect(screen.queryByText('Loading...')).toBeNull());
    expect(screen.getByText('Custom & Plugins')).toBeTruthy();
    expect(screen.getByText('Custom Nodes')).toBeTruthy();
    expect(screen.getByText('Plugins')).toBeTruthy();
  });

  it('shows an empty state per section, with the install hint for plugins', async () => {
    render(<CustomTab />);
    await screen.findByText('No custom nodes yet');
    expect(screen.getByText('No plugins installed')).toBeTruthy();
    expect(screen.getByText('Install packs with the cdui plugin CLI')).toBeTruthy();
  });

  it('lists custom node files with their node names and enabled chip', async () => {
    mockedRest.listCustomNodes.mockResolvedValue([
      customNode({ filename: 'a.py', nodes: ['Alpha', 'Beta'] }),
      customNode({ filename: 'b.py', enabled: false, nodes: [] }),
    ]);
    render(<CustomTab />);
    await screen.findByText('a.py');
    expect(screen.getByText('Alpha, Beta')).toBeTruthy();
    expect(screen.getByText('Enabled')).toBeTruthy();
    expect(screen.getByText('Disabled')).toBeTruthy();
    // A file with no parsed nodes renders no node-name line.
    expect(screen.getByText('b.py')).toBeTruthy();
  });

  it('lists plugin packs with version, description and node count', async () => {
    mockedRest.listPlugins.mockResolvedValue([plugin()]);
    render(<CustomTab />);
    await screen.findByText('Chapter 1');
    expect(screen.getByText('v1.0.0')).toBeTruthy();
    expect(screen.getByText('Intro nodes')).toBeTruthy();
    expect(screen.getByText('2 nodes')).toBeTruthy();
    expect(screen.getByText('Enabled')).toBeTruthy();
  });

  it('greys out a disabled plugin and omits absent optional fields', async () => {
    mockedRest.listPlugins.mockResolvedValue([
      plugin({ id: 'c2', name: 'Chapter 2', version: '', description: '', enabled: false, nodes: [] }),
    ]);
    const { container } = render(<CustomTab />);
    await screen.findByText('Chapter 2');
    expect(screen.getByText('Disabled')).toBeTruthy();
    expect(container.querySelector('[data-disabled="true"]')).toBeTruthy();
    expect(container.querySelectorAll('[class*="rowVersion"]')).toHaveLength(0);
    expect(container.querySelectorAll('[class*="rowDesc"]')).toHaveLength(0);
    expect(container.querySelectorAll('[class*="rowMeta"]')).toHaveLength(0);
  });

  it('shows the section counts', async () => {
    mockedRest.listCustomNodes.mockResolvedValue([customNode({ filename: 'a.py' })]);
    mockedRest.listPlugins.mockResolvedValue([plugin(), plugin({ id: 'c2', name: 'Chapter 2' })]);
    const { container } = render(<CustomTab />);
    await screen.findByText('a.py');
    const counts = Array.from(container.querySelectorAll('[class*="sectionCount"]')).map(
      (el) => el.textContent,
    );
    expect(counts).toEqual(['1', '2']);
  });

  it('opens the custom node manager and re-fetches when it closes', async () => {
    render(<CustomTab />);
    await screen.findByText('No custom nodes yet');
    expect(mockedRest.listCustomNodes).toHaveBeenCalledTimes(1);

    fireEvent.click(screen.getByText('Manage...'));
    expect(screen.getByTestId('custom-node-manager')).toBeTruthy();

    mockedRest.listCustomNodes.mockResolvedValue([customNode({ filename: 'uploaded.py' })]);
    fireEvent.click(screen.getByText('close manager'));
    expect(screen.queryByTestId('custom-node-manager')).toBeNull();
    await screen.findByText('uploaded.py');
  });

  it('shows the error state and retries on click', async () => {
    mockedRest.listPlugins.mockRejectedValueOnce(new Error('backend gone'));
    render(<CustomTab />);
    await screen.findByText('Failed to load: backend gone');

    mockedRest.listPlugins.mockResolvedValue([plugin({ name: 'Recovered' })]);
    fireEvent.click(screen.getByText('Retry'));
    await screen.findByText('Recovered');
  });

  it('re-fetches from the refresh button', async () => {
    render(<CustomTab />);
    await screen.findByText('No plugins installed');
    mockedRest.listPlugins.mockResolvedValue([plugin({ name: 'Just installed' })]);
    fireEvent.click(screen.getByRole('button', { name: 'Refresh' }));
    await screen.findByText('Just installed');
  });
});
