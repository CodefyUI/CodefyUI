import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, render, screen } from '@testing-library/react';
import * as rest from '../api/rest';
import { useProjectStore } from '../store/projectStore';
import {
  DRIFT_CHECK_INTERVAL_MS,
  _resetDriftCheckForTesting,
  checkProjectDrift,
  useProjectDriftCheck,
} from './useProjectDriftCheck';

vi.mock('../api/rest', async (importOriginal) => {
  const actual = await importOriginal<typeof rest>();
  return { ...actual, fetchHealth: vi.fn() };
});

const health = vi.mocked(rest.fetchHealth);

function answer(project: string | null) {
  health.mockResolvedValue({ project } as Awaited<ReturnType<typeof rest.fetchHealth>>);
}

function Probe() {
  useProjectDriftCheck();
  const stale = useProjectStore((s) => s.stale);
  return <span data-testid="stale">{String(stale)}</span>;
}

const loadOverview = vi.fn(async () => null);
const realLoadOverview = useProjectStore.getState().loadOverview;

beforeEach(() => {
  _resetDriftCheckForTesting();
  useProjectStore.setState({ projectDir: '/p/alpha', projectName: 'alpha', loaded: true,
    stale: false, loadOverview });
});

afterEach(() => {
  vi.clearAllMocks();
  useProjectStore.setState({ loadOverview: realLoadOverview });
});

describe('the project drift check', () => {
  it('marks the window stale on focus when the server opened another project', async () => {
    answer('/p/beta');
    render(<Probe />);
    expect(screen.getByTestId('stale').textContent).toBe('false');
    await act(async () => {
      window.dispatchEvent(new Event('focus'));
    });
    expect(health).toHaveBeenCalledTimes(1);
    expect(screen.getByTestId('stale').textContent).toBe('true');
  });

  it('checks when the tab becomes visible, and stays quiet while the project is the same', async () => {
    answer('/p/alpha');
    render(<Probe />);
    await act(async () => {
      document.dispatchEvent(new Event('visibilitychange'));
    });
    expect(health).toHaveBeenCalledTimes(1);
    expect(screen.getByTestId('stale').textContent).toBe('false');
  });

  it('reads the server at most once per interval', async () => {
    answer('/p/alpha');
    await checkProjectDrift(1_000);
    await checkProjectDrift(1_000 + DRIFT_CHECK_INTERVAL_MS - 1);
    expect(health).toHaveBeenCalledTimes(1);
    await checkProjectDrift(1_000 + DRIFT_CHECK_INTERVAL_MS);
    expect(health).toHaveBeenCalledTimes(2);
  });

  it('treats flat mode as a project of its own', async () => {
    useProjectStore.setState({ projectDir: null, projectName: null });
    answer('/p/alpha');
    await checkProjectDrift(10_000);
    expect(useProjectStore.getState().stale).toBe(true);
  });

  it('says nothing when the server cannot be reached, or before the page knows its project', async () => {
    health.mockRejectedValue(new Error('down'));
    await checkProjectDrift(20_000);
    expect(useProjectStore.getState().stale).toBe(false);
    useProjectStore.setState({ loaded: false });
    answer('/p/beta');
    await checkProjectDrift(40_000);
    expect(health).toHaveBeenCalledTimes(1);
    expect(useProjectStore.getState().stale).toBe(false);
  });

  it('re-reads the overview when the project is the same, so a pulled mods/ is noticed', async () => {
    answer('/p/alpha');
    await checkProjectDrift(0);
    expect(loadOverview).toHaveBeenCalledTimes(1);
    answer('/p/beta');
    await checkProjectDrift(DRIFT_CHECK_INTERVAL_MS * 2);
    expect(loadOverview).toHaveBeenCalledTimes(1);   // another project: stale, not re-read
  });
});
