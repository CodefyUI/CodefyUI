import { describe, it, expect, beforeEach, afterEach, vi, type MockInstance } from 'vitest';
import { useEffect } from 'react';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { useI18n } from '../../i18n';
import { parseWorkspaceFile } from '../../utils/workspaceFile';
import { AppErrorBoundary } from './AppErrorBoundary';

// Only `main.tsx` reaches these, in the one case that loads it: the real App
// would start the whole editor, and the token bootstrap would call a server.
vi.mock('../../App', () => ({
  default: () => {
    throw new Error('App failed to render');
  },
}));
vi.mock('../../api/_auth', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../../api/_auth')>()),
  getSessionToken: vi.fn(async () => 'token'),
}));

const BASE_SCOPE = 'codefyui-tabs';
const TITLE = 'The page stopped because of an error.';
const HINT =
  'Download a backup of your autosaved tabs first. If Reload brings this page back, ' +
  'start with an empty workspace, choose New blank graph, and open the backup with ' +
  'Import... in the Graphs panel.';

function Thrower({ error }: { error: unknown }): never {
  throw error;
}

function crash(error: unknown = new TypeError('nodes is undefined')) {
  return render(
    <AppErrorBoundary>
      <Thrower error={error} />
    </AppErrorBoundary>,
  );
}

/** Autosaved tabs, as an older build wrote them, in the localStorage tier. */
function seedAutosave(tabs: unknown[], activeTabId: string): string {
  const blob = JSON.stringify({ activeTabId, tabs });
  localStorage.setItem(BASE_SCOPE, blob);
  return blob;
}

function savedTab(id: string, name: string, over: Record<string, unknown> = {}) {
  return {
    id,
    name,
    nodes: [
      {
        id: `${id}-n1`,
        type: 'baseNode',
        position: { x: 0, y: 0 },
        data: { label: 'Linear', type: 'Linear', params: { in_features: 4 } },
      },
    ],
    edges: [],
    ...over,
  };
}

/** jsdom's Blob has no .text(); read it the way the import flow does. */
function readBlob(blob: Blob): Promise<string> {
  return new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(reader.error);
    reader.readAsText(blob);
  });
}

const button = (name: string) => screen.getByRole('button', { name });
const status = () => screen.getByRole('status');

let originalLocation: Location;
let reload: ReturnType<typeof vi.fn>;
let clickSpy: MockInstance<() => void>;
let confirmSpy: MockInstance<(message?: string) => boolean>;
let realT: ReturnType<typeof useI18n.getState>['t'];

beforeEach(() => {
  localStorage.clear();
  realT = useI18n.getState().t;
  useI18n.setState({ locale: 'en' });
  // React reports each error a boundary catches through console.error. Only
  // that report is dropped: anything else, an act() warning above all, still
  // reaches the console and the act-warnings gate.
  const consoleError = console.error;
  vi.spyOn(console, 'error').mockImplementation((...args: unknown[]) => {
    const caughtReport = args.some(
      (arg) => typeof arg === 'string' && arg.startsWith('React will try to recreate'),
    );
    if (!caughtReport) consoleError(...args);
  });
  originalLocation = window.location;
  reload = vi.fn();
  Object.defineProperty(window, 'location', {
    value: { ...originalLocation, reload },
    configurable: true,
  });
  vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:backup');
  vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {});
  clickSpy = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});
  confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(true);
});

afterEach(() => {
  Object.defineProperty(window, 'location', {
    value: originalLocation,
    configurable: true,
  });
  useI18n.setState({ locale: 'en', t: realT });
  vi.restoreAllMocks();
  localStorage.clear();
});

describe('AppErrorBoundary', () => {
  it('renders its children when nothing throws', () => {
    render(
      <AppErrorBoundary>
        <p>Editor</p>
      </AppErrorBoundary>,
    );
    expect(screen.getByText('Editor')).toBeInTheDocument();
    expect(screen.queryByRole('heading')).toBeNull();
  });

  it('says what failed, shows the message once, and keeps the stack behind Details', () => {
    const error = new TypeError('nodes is undefined');
    error.stack = 'TypeError: nodes is undefined\n    at FlowCanvas (canvas.tsx:12:3)';
    crash(error);

    const heading = screen.getByRole('heading', { name: TITLE });
    // The page under the screen is gone, and focus went with it.
    expect(heading).toHaveFocus();
    expect(screen.getByText('TypeError: nodes is undefined')).toBeInTheDocument();
    expect(screen.getByText(HINT)).toBeInTheDocument();

    const details = screen.getByText('Details').closest('details')!;
    expect(details.open).toBe(false);
    expect(details).toHaveTextContent('at FlowCanvas (canvas.tsx:12:3)');
    // The stack's first line repeats the message shown above; it is dropped.
    expect(details).not.toHaveTextContent('nodes is undefined');
    // React's component stack, which names the component that threw.
    expect(details).toHaveTextContent('Thrower');
  });

  it('shows the error that started it, and lists what followed in Details', () => {
    // A sibling whose effect cleanup throws while React takes the broken tree
    // down: a second error, reported after the first.
    function Leaky() {
      useEffect(
        () => () => {
          throw new Error('cleanup failed');
        },
        [],
      );
      return null;
    }
    function Bomb({ go }: { go: boolean }) {
      if (go) throw new Error('render failed');
      return <p>idle</p>;
    }
    const tree = (go: boolean) => (
      <AppErrorBoundary>
        <Leaky />
        <Bomb go={go} />
      </AppErrorBoundary>
    );
    const { rerender } = render(tree(false));
    rerender(tree(true));

    expect(screen.getByText('Error: render failed')).toBeInTheDocument();
    const details = screen.getByText('Details').closest('details')!;
    expect(details).toHaveTextContent('Later errors:');
    expect(details).toHaveTextContent('Error: cleanup failed');
  });

  it('still says what failed when the thrown value is not an Error', () => {
    crash('a plain string');
    expect(screen.getByRole('heading', { name: TITLE })).toBeInTheDocument();
    expect(screen.getByText('a plain string')).toBeInTheDocument();
  });

  it('reloads the page', () => {
    crash();
    fireEvent.click(button('Reload'));
    expect(reload).toHaveBeenCalledTimes(1);
  });

  it('downloads the autosaved workspace as a file Import accepts', async () => {
    seedAutosave([savedTab('p1', 'Lab 3')], 'p1');
    crash();

    const download = button('Download backup');
    download.focus();
    fireEvent.click(download);
    // Marked busy without `disabled`, which would drop focus to <body>; a
    // second click while it works starts nothing.
    expect(download).toHaveAttribute('aria-disabled', 'true');
    expect(download).not.toBeDisabled();
    expect(download).toHaveFocus();
    fireEvent.click(download);
    await waitFor(() => expect(download).toHaveAttribute('aria-disabled', 'false'));

    expect(URL.createObjectURL).toHaveBeenCalledTimes(1);
    const anchor = clickSpy.mock.contexts[0] as HTMLAnchorElement;
    expect(anchor.download).toMatch(/^workspace-\d{4}-\d{2}-\d{2}\.cduiworkspace$/);
    expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:backup');

    const blob = vi.mocked(URL.createObjectURL).mock.calls[0][0] as Blob;
    const parsed = parseWorkspaceFile(JSON.parse(await readBlob(blob)));
    if (!parsed.ok) throw new Error(`refused: ${parsed.reason}`);
    expect(parsed.workspace.tabs.map((tab) => tab.title)).toEqual(['Lab 3']);
    expect(parsed.workspace.tabs[0].graph).toMatchObject({
      nodes: [{ id: 'p1-n1', type: 'Linear', data: { params: { in_features: 4 } } }],
    });
    expect(reload).not.toHaveBeenCalled();
  });

  it('says so when there is nothing to back up', async () => {
    crash();
    fireEvent.click(button('Download backup'));
    expect(await screen.findByText('Nothing to back up.')).toBeInTheDocument();
    expect(URL.createObjectURL).not.toHaveBeenCalled();
  });

  it('names the tabs it had to leave out', async () => {
    seedAutosave(
      [savedTab('ro', 'Newer format', { readOnly: true }), null, savedTab('ok', 'Kept')],
      'ok',
    );
    crash();
    fireEvent.click(button('Download backup'));

    expect(await screen.findByText('1 read-only tab(s) were left out.')).toBeInTheDocument();
    expect(screen.getByText('1 tab(s) could not be read and were left out.')).toBeInTheDocument();
    expect(URL.createObjectURL).toHaveBeenCalledTimes(1);
  });

  it('says why when the autosaved workspace cannot be read', async () => {
    localStorage.setItem(BASE_SCOPE, '{not json');
    crash();
    fireEvent.click(button('Download backup'));

    expect(
      await screen.findByText(/^Could not read the autosaved workspace: SyntaxError: /),
    ).toBeInTheDocument();
    expect(URL.createObjectURL).not.toHaveBeenCalled();
  });

  it('tells a failure after the read apart from a read failure', async () => {
    seedAutosave([savedTab('p1', 'Lab 3')], 'p1');
    vi.mocked(URL.createObjectURL).mockImplementation(() => {
      throw new Error('no blob URLs here');
    });
    crash();
    fireEvent.click(button('Download backup'));

    expect(
      await screen.findByText('Could not make the backup: Error: no blob URLs here'),
    ).toBeInTheDocument();
  });

  it('keeps its status line mounted, so a screen reader hears what lands in it', async () => {
    crash();
    const line = status();
    expect(line).toBeEmptyDOMElement();
    fireEvent.click(button('Download backup'));
    await waitFor(() => expect(line).toHaveTextContent('Nothing to back up.'));
    expect(status()).toBe(line);
  });

  it('falls back to English when the translations themselves throw', () => {
    useI18n.setState({
      locale: 'zh-TW',
      t: () => {
        throw new Error('i18n is broken');
      },
    });
    crash();

    expect(screen.getByRole('heading', { name: TITLE })).toBeInTheDocument();
    expect(screen.getByText(HINT)).toBeInTheDocument();
    expect(button('Download backup')).toBeInTheDocument();
    expect(button('Reload')).toBeInTheDocument();
    expect(button('Start with an empty workspace')).toBeInTheDocument();
  });

  it('speaks the UI language when the translations work', () => {
    useI18n.setState({ locale: 'zh-TW' });
    crash();

    expect(screen.getByRole('heading', { name: '頁面因為錯誤而停止運作。' })).toBeInTheDocument();
    expect(
      screen.getByText(
        '請先下載自動儲存分頁的備份。如果重新載入後仍出現這個頁面，請以空白工作區重新開始，按「新增空白圖表」，再到「圖表」面板按「匯入...」開啟備份。',
      ),
    ).toBeInTheDocument();
    expect(button('下載備份')).toBeInTheDocument();
    expect(button('重新載入')).toBeInTheDocument();
    expect(button('以空白工作區重新開始')).toBeInTheDocument();
  });
});

describe('Start with an empty workspace', () => {
  const EMPTY = { activeTabId: '', tabs: [] };

  it('asks first when no backup was downloaded, and changes nothing on No', () => {
    const blob = seedAutosave([savedTab('p1', 'Lab 3')], 'p1');
    confirmSpy.mockReturnValue(false);
    crash();

    fireEvent.click(button('Start with an empty workspace'));

    expect(confirmSpy).toHaveBeenCalledWith(
      'No backup was downloaded. Start with an empty workspace anyway?',
    );
    expect(localStorage.getItem(BASE_SCOPE)).toBe(blob);
    expect(reload).not.toHaveBeenCalled();
  });

  it('on Yes, keeps a copy aside, empties the workspace and reloads', async () => {
    const blob = seedAutosave([savedTab('p1', 'Lab 3')], 'p1');
    crash();

    fireEvent.click(button('Start with an empty workspace'));

    await waitFor(() => expect(reload).toHaveBeenCalledTimes(1));
    expect(JSON.parse(localStorage.getItem(BASE_SCOPE)!)).toEqual(EMPTY);
    const aside = Object.keys(localStorage).filter((key) => key.startsWith(`${BASE_SCOPE}-aside::`));
    expect(aside.map((key) => localStorage.getItem(key))).toEqual([blob]);
  });

  it('does not ask again once the backup is downloaded', async () => {
    seedAutosave([savedTab('p1', 'Lab 3')], 'p1');
    crash();
    fireEvent.click(button('Download backup'));
    await waitFor(() => expect(URL.createObjectURL).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(button('Download backup')).toHaveAttribute('aria-disabled', 'false'));

    fireEvent.click(button('Start with an empty workspace'));

    await waitFor(() => expect(reload).toHaveBeenCalledTimes(1));
    expect(confirmSpy).not.toHaveBeenCalled();
  });

  // Over half of jsdom's 5,000,000-unit localStorage quota: no room for a copy.
  const BIG_NOTE = {
    id: 'big',
    type: 'noteNode',
    position: { x: 0, y: 0 },
    data: { label: 'Note', type: 'note', params: {}, noteKind: 'text', noteContent: 'x'.repeat(2_600_000) },
  };

  it('asks for a backup first, and changes nothing, when no copy of the tabs fits', async () => {
    const blob = seedAutosave([savedTab('p1', 'Lab 3', { nodes: [BIG_NOTE] })], 'p1');
    crash();

    fireEvent.click(button('Start with an empty workspace'));

    expect(
      await screen.findByText(
        'A copy of the current tabs could not be kept, so nothing was changed. Download a backup, then try again.',
      ),
    ).toBeInTheDocument();
    expect(localStorage.getItem(BASE_SCOPE)).toBe(blob);
    expect(reload).not.toHaveBeenCalled();
  });

  it('does not count a backup that left tabs out as a copy', async () => {
    const blob = seedAutosave(
      [
        savedTab('p1', 'Lab 3', { nodes: [BIG_NOTE] }),
        savedTab('ro', 'Reference', { readOnly: true }),
      ],
      'p1',
    );
    crash();
    fireEvent.click(button('Download backup'));
    expect(await screen.findByText('1 read-only tab(s) were left out.')).toBeInTheDocument();

    fireEvent.click(button('Start with an empty workspace'));

    expect(
      await screen.findByText(
        'A copy of the current tabs could not be kept, and the backup leaves some tabs out, so nothing was changed.',
      ),
    ).toBeInTheDocument();
    expect(localStorage.getItem(BASE_SCOPE)).toBe(blob);
    expect(reload).not.toHaveBeenCalled();
  });

  it('goes on without a copy once the backup is downloaded', async () => {
    seedAutosave([savedTab('p1', 'Lab 3', { nodes: [BIG_NOTE] })], 'p1');
    crash();
    fireEvent.click(button('Download backup'));
    await waitFor(() => expect(URL.createObjectURL).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(button('Download backup')).toHaveAttribute('aria-disabled', 'false'));

    fireEvent.click(button('Start with an empty workspace'));

    await waitFor(() => expect(reload).toHaveBeenCalledTimes(1));
    expect(JSON.parse(localStorage.getItem(BASE_SCOPE)!)).toEqual(EMPTY);
  });

  it('says so, and does not reload, when the saved tabs did not change', async () => {
    const blob = seedAutosave([savedTab('p1', 'Lab 3')], 'p1');
    const setItem = Storage.prototype.setItem;
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(function (this: Storage, key, value) {
      if (key === BASE_SCOPE) throw new DOMException('denied', 'SecurityError');
      setItem.call(this, key, value);
    });
    crash();

    fireEvent.click(button('Start with an empty workspace'));

    expect(
      await screen.findByText(
        'The saved tabs did not change, so the page was not reloaded. Another browser tab may be editing this workspace.',
      ),
    ).toBeInTheDocument();
    expect(localStorage.getItem(BASE_SCOPE)).toBe(blob);
    expect(reload).not.toHaveBeenCalled();
  });

  it('says why, and does not reload, when storage cannot be reached at all', async () => {
    vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => {
      throw new DOMException('denied', 'SecurityError');
    });
    crash();

    fireEvent.click(button('Start with an empty workspace'));

    expect(
      await screen.findByText('Could not start an empty workspace: SecurityError: denied'),
    ).toBeInTheDocument();
    expect(reload).not.toHaveBeenCalled();
  });
});

describe('main.tsx', () => {
  it('wraps the app, so a render error leaves the recovery screen instead of a blank page', async () => {
    const root = document.createElement('div');
    root.id = 'root';
    document.body.appendChild(root);
    try {
      await act(async () => {
        await import('../../main');
      });
      expect(root).toHaveTextContent(TITLE);
      expect(root).toHaveTextContent('Error: App failed to render');
    } finally {
      root.remove();
    }
  });
});
