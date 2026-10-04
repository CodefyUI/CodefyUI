import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { act, render, screen } from '@testing-library/react';
import type { Node } from '@xyflow/react';
import { useI18n } from '../../i18n';
import { autosaveScope, useTabStore } from '../../store/tabStore';
import type { NodeData } from '../../types';
import { AppErrorBoundary } from './AppErrorBoundary';
import { buildWorkspaceBackup, readAutosavedTabs } from './workspaceBackup';

/**
 * The boundary stops autosave when it catches (#555). A file of its own: the
 * suspension lasts for the life of the page, which in a test file is its
 * module registry, and any other crash test would already have set it.
 */

const TITLE = 'The page stopped because of an error.';
const store = () => useTabStore.getState();
const saved = () => localStorage.getItem(autosaveScope());

/** Stands in for any editor part with a render bug for one value. */
function Canvas() {
  const tab = useTabStore((s) => s.tabs.find((t) => t.id === s.activeTabId));
  if (tab?.nodes.some((n) => n.data.params?.poison)) {
    throw new TypeError('cannot render the poisoned node');
  }
  return <p>canvas ok</p>;
}

function flowNode(id: string): Node<NodeData> {
  return { id, type: 'baseNode', position: { x: 0, y: 0 }, data: { label: id, type: 'Linear', params: {} } };
}

beforeEach(() => {
  // Autosave is a 250 ms debounce on setTimeout; jsdom has no IndexedDB, so
  // a save lands in localStorage, synchronously, when the clock moves.
  vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
  localStorage.clear();
  useI18n.setState({ locale: 'en' });
  const consoleError = console.error;
  vi.spyOn(console, 'error').mockImplementation((...args: unknown[]) => {
    const caughtReport = args.some(
      (arg) => typeof arg === 'string' && arg.startsWith('React will try to recreate'),
    );
    if (!caughtReport) consoleError(...args);
  });
});

afterEach(() => {
  vi.clearAllTimers();
  vi.useRealTimers();
  vi.restoreAllMocks();
  localStorage.clear();
});

describe('AppErrorBoundary and autosave', () => {
  it('stops autosave when it catches, so the last good save is what a reload and the backup get', async () => {
    useTabStore.setState({ tabs: [], activeTabId: '' });
    const id = store().createTab({ title: 'Exam', activate: true });
    store().loadGraphDocumentInto(id, { nodes: [flowNode('n1')], edges: [], boundFile: null });
    vi.advanceTimersByTime(250);
    const good = saved();
    // Autosave works until the crash.
    expect(good).toContain('"Exam"');

    render(
      <AppErrorBoundary>
        <Canvas />
      </AppErrorBoundary>,
    );
    expect(screen.getByText('canvas ok')).toBeInTheDocument();

    // The change that breaks rendering arms a save, as every change does.
    act(() => {
      useTabStore.setState((s) => ({
        tabs: s.tabs.map((t) =>
          t.id === id
            ? { ...t, nodes: t.nodes.map((n) => ({ ...n, data: { ...n.data, params: { poison: true } } })) }
            : t,
        ),
      }));
    });
    expect(screen.getByRole('heading', { name: TITLE })).toBeInTheDocument();
    // A change after the catch, as a run's status stream would make.
    act(() => store().renameTab(id, 'After the crash'));
    act(() => {
      vi.advanceTimersByTime(1000);
    });

    expect(saved()).toBe(good);
    const backup = buildWorkspaceBackup(await readAutosavedTabs(), new Date());
    expect(backup.file!.tabs.map((tab) => tab.title)).toEqual(['Exam']);
    expect(JSON.stringify(backup.file)).not.toContain('poison');
  });
});
