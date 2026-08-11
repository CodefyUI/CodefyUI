import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { HealthSection } from './HealthSection';
import { useI18n } from '../../i18n';
import { useToastStore } from '../../store/toastStore';
import { fetchHealth } from '../../api/rest';

vi.mock('../../api/rest', () => ({
  fetchHealth: vi.fn(),
}));

const mockedFetchHealth = vi.mocked(fetchHealth);

/** A full /api/health body, shaped exactly like backend/app/main.py health()
 *  (three stores, each with its OWN key set — see CacheUsage). */
function healthBody(overrides: Record<string, unknown> = {}) {
  return {
    status: 'ok',
    version: '2.2.0',
    nodes_loaded: 137,
    presets_loaded: 12,
    caches: {
      execution_cache: {
        instances: 1,
        entries: 4,
        bytes: 2 * 1024 * 1024,
        max_bytes_each: 512 * 1024 * 1024,
      },
      run_output_store: { runs: 2, max_runs: 20, bytes: 0, max_bytes: 256 * 1024 * 1024 },
      node_state_store: { modules: 3, max_modules: 64, bytes: 1024 ** 3, max_bytes: 2 * 1024 ** 3 },
    },
    project: null,
    ...overrides,
  } as never;
}

describe('HealthSection', () => {
  beforeEach(() => {
    useI18n.setState({ locale: 'en' });
    useToastStore.setState({ toasts: [] });
    mockedFetchHealth.mockReset();
    mockedFetchHealth.mockResolvedValue(healthBody());
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('reads the server once on mount and shows the version and registry counts', async () => {
    render(<HealthSection />);
    // Mounting IS "on open": SettingsPopover renders nothing while closed, so
    // this component only exists while the popover is on screen.
    expect(mockedFetchHealth).toHaveBeenCalledTimes(1);
    expect(await screen.findByText('2.2.0')).toBeInTheDocument();
    expect(screen.getByText('137')).toBeInTheDocument();
    expect(screen.getByText('12')).toBeInTheDocument();
  });

  it('lists every cache store with human-readable bytes against its budget', async () => {
    render(<HealthSection />);
    expect(await screen.findByText('Node outputs (per editor connection)')).toBeInTheDocument();
    // The execution cache reports max_bytes_each, the other two max_bytes;
    // both have to resolve to the same "used of budget" line.
    expect(screen.getByText('2.0 MB of 512.0 MB')).toBeInTheDocument();
    expect(screen.getByText('Recorded run outputs')).toBeInTheDocument();
    expect(screen.getByText('0 B of 256.0 MB')).toBeInTheDocument();
    expect(screen.getByText('Layer weights kept between runs')).toBeInTheDocument();
    expect(screen.getByText('1.0 GB of 2.0 GB')).toBeInTheDocument();
  });

  it('explains what a cache is, once, without naming a delete button that does not exist yet', async () => {
    render(<HealthSection />);
    const hint = await screen.findByText(/hold results the server already computed/);
    // The honest version of "it is only derived data": the weight store holds
    // TRAINED weights, so clearing it costs training time, not a recompute.
    expect(hint).toHaveTextContent(/for the weight cache that means training time/);
  });

  it('shows a store with no reported budget as a bare size', async () => {
    mockedFetchHealth.mockResolvedValue(
      healthBody({ caches: { run_output_store: { runs: 1, bytes: 4096 } } }),
    );
    render(<HealthSection />);
    expect(await screen.findByText('4.0 KB')).toBeInTheDocument();
  });

  it('lists an unknown store under its raw name rather than dropping it', async () => {
    // The payload can grow a store this build has never heard of (a fourth
    // cache, a plugin's own); it must still be counted on screen.
    mockedFetchHealth.mockResolvedValue(
      healthBody({ caches: { asset_cache: { bytes: 1536, max_bytes: 1024 * 1024 } } }),
    );
    render(<HealthSection />);
    expect(await screen.findByText('asset_cache')).toBeInTheDocument();
    expect(screen.getByText('1.5 KB of 1.0 MB')).toBeInTheDocument();
  });

  it('says so plainly when the server reports no caches at all', async () => {
    // Reachable: the backend omits a store that is not running.
    mockedFetchHealth.mockResolvedValue(healthBody({ caches: {} }));
    render(<HealthSection />);
    expect(await screen.findByText('No caches are running yet.')).toBeInTheDocument();
  });

  it('renders a placeholder rather than "null" when the server sends no version', async () => {
    mockedFetchHealth.mockResolvedValue(healthBody({ version: null }));
    render(<HealthSection />);
    expect(await screen.findByText('unknown')).toBeInTheDocument();
  });

  // ── failure path ──────────────────────────────────────────────────

  it('reports an unreachable server inline, not as a toast', async () => {
    mockedFetchHealth.mockRejectedValue(new Error('offline'));
    render(<HealthSection />);
    expect(
      await screen.findByText('Could not read the server status. Press Refresh to try again.'),
    ).toBeInTheDocument();
    // A toast would outlive the popover the user opened to read this.
    expect(useToastStore.getState().toasts).toHaveLength(0);
    // Nothing invented in place of the numbers.
    expect(screen.queryByText('Version')).not.toBeInTheDocument();
  });

  it('keeps the numbers it already had when a REFRESH fails', async () => {
    render(<HealthSection />);
    expect(await screen.findByText('137')).toBeInTheDocument();

    mockedFetchHealth.mockRejectedValueOnce(new Error('offline'));
    fireEvent.click(screen.getByRole('button', { name: 'Refresh server status' }));

    await screen.findByText('Could not read the server status. Press Refresh to try again.');
    // Stale counts plus the warning beat a blank panel: the reader can see
    // WHICH half is untrustworthy.
    expect(screen.getByText('137')).toBeInTheDocument();
  });

  // ── refresh ───────────────────────────────────────────────────────

  it('refetches on Refresh and renders the new numbers', async () => {
    render(<HealthSection />);
    expect(await screen.findByText('137')).toBeInTheDocument();

    mockedFetchHealth.mockResolvedValue(
      healthBody({
        nodes_loaded: 140,
        caches: { execution_cache: { bytes: 3 * 1024 * 1024, max_bytes_each: 512 * 1024 * 1024 } },
      }),
    );
    fireEvent.click(screen.getByRole('button', { name: 'Refresh server status' }));

    await waitFor(() => expect(mockedFetchHealth).toHaveBeenCalledTimes(2));
    expect(await screen.findByText('140')).toBeInTheDocument();
    expect(screen.getByText('3.0 MB of 512.0 MB')).toBeInTheDocument();
  });

  it('clears a previous failure once a refresh succeeds', async () => {
    mockedFetchHealth.mockRejectedValueOnce(new Error('offline'));
    render(<HealthSection />);
    await screen.findByText('Could not read the server status. Press Refresh to try again.');

    fireEvent.click(screen.getByRole('button', { name: 'Refresh server status' }));

    expect(await screen.findByText('137')).toBeInTheDocument();
    expect(
      screen.queryByText('Could not read the server status. Press Refresh to try again.'),
    ).not.toBeInTheDocument();
  });

  it('disables Refresh while a read is in flight', async () => {
    let release: (() => void) | undefined;
    mockedFetchHealth.mockImplementationOnce(
      () => new Promise((resolve) => { release = () => resolve(healthBody()); }),
    );
    render(<HealthSection />);
    expect(screen.getByRole('button', { name: 'Refresh server status' })).toBeDisabled();
    expect(screen.getByText('Reading the server…')).toBeInTheDocument();

    release!();
    await waitFor(() => expect(screen.getByRole('button', { name: 'Refresh server status' })).not.toBeDisabled());
  });

  it('does not write state after the popover closes mid-read', async () => {
    let release: ((body: unknown) => void) | undefined;
    mockedFetchHealth.mockImplementationOnce(
      () => new Promise((resolve) => { release = resolve as (body: unknown) => void; }),
    );
    const { unmount } = render(<HealthSection />);
    unmount();
    // Resolving after unmount must be inert; a throw here would surface as an
    // unhandled rejection in the run.
    release!(healthBody());
    await Promise.resolve();
    expect(screen.queryByText('137')).not.toBeInTheDocument();
  });

  // ── zh-TW ─────────────────────────────────────────────────────────

  it('renders the zh-TW strings, including the cache caption', async () => {
    useI18n.setState({ locale: 'zh-TW' });
    render(<HealthSection />);
    expect(await screen.findByText('這台伺服器載入了什麼')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: '重新讀取伺服器狀態' })).toBeInTheDocument();
    expect(screen.getByText('版本')).toBeInTheDocument();
    expect(screen.getByText('節點輸出（每個編輯器連線一份）')).toBeInTheDocument();
    // The unit itself stays Latin: "2.0 MB" is how the budget is configured.
    expect(screen.getByText('2.0 MB / 上限 512.0 MB')).toBeInTheDocument();
    expect(screen.getByText(/這些存放伺服器已經算過的結果/)).toBeInTheDocument();
  });
});
