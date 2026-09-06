import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, fireEvent, within } from '@testing-library/react';
import type { PluginCatalogEntry } from '../../api/rest';
import { emptyPluginJob, type PluginJob } from '../../store/pluginStore';
import { useI18n } from '../../i18n';
import { PluginActivityPane } from './PluginActivityPane';
import { jobOverallPercent } from './pluginStatus';

function entry(over: Partial<PluginCatalogEntry> & { id: string }): PluginCatalogEntry {
  return {
    name: over.id,
    description: '',
    kind: 'builtin',
    official: true,
    status: 'available',
    source_kind: null,
    source: over.id,
    repo: null,
    ref: null,
    sha: null,
    url: null,
    homepage: '',
    version: null,
    installed_at: null,
    enabled: false,
    chapters: [],
    lessons: [],
    tags: [],
    nodes: [],
    node_count: 0,
    capabilities: [],
    trusted_modules: [],
    python_deps: {},
    has_frontend: false,
    consent_required: false,
    frontend_entry: null,
    job: null,
    ...over,
  };
}

/** A third-party plugin off GitHub, installed and pinned to a tag. */
const demo = entry({
  id: 'demo',
  name: 'Demo plugin',
  kind: 'github',
  official: false,
  status: 'installed',
  source_kind: 'github_url',
  repo: 'owner/demo',
  ref: 'v1.2.0',
  version: '1.2.0',
});

/** A built-in teaching pack: a catalog NAME is what installs it. */
const stats = entry({ id: 'stats', name: 'Stats nodes', source_kind: 'builtin' });

/** A folder somebody linked from disk. `cdui plugin install` cannot take it. */
const linked = entry({
  id: 'lab',
  name: 'Lab nodes',
  kind: 'external',
  official: false,
  status: 'installed',
  source_kind: 'local',
  source: 'D:/work/lab-nodes',
});

function job(over: Partial<PluginJob> = {}): PluginJob {
  return { ...emptyPluginJob('j1', 'demo'), ...over };
}

/** A job that has got as far as its second step, with nothing downloaded. */
function downloading(over: Partial<PluginJob> = {}): PluginJob {
  return job({
    steps: [
      { step: 'resolve', label: 'Resolving owner/demo', state: 'done' },
      { step: 'download', label: 'Downloading owner/demo@4f0a1c9', state: 'running' },
    ],
    ...over,
  });
}

/** Fresh mocks for every handler the pane is allowed to call. */
function makeHandlers() {
  return { onCancel: vi.fn(), onDismiss: vi.fn(), onRefresh: vi.fn() };
}

let handlers: ReturnType<typeof makeHandlers>;

/**
 * Mount the pane over one job. Fresh `vi.fn()` handlers every time, so no
 * case can read another's calls.
 */
function paint(over: {
  job?: PluginJob | null;
  entry?: PluginCatalogEntry | undefined;
  cancelling?: boolean;
} = {}) {
  handlers = makeHandlers();
  return render(
    <PluginActivityPane
      job={over.job ?? null}
      entry={over.entry}
      cancelling={over.cancelling ?? false}
      {...handlers}
    />,
  );
}

/** The result banner, the only `status` element the pane renders. */
const banner = () => screen.getByRole('status');

const progressBar = () => screen.getByRole('progressbar', { name: 'Install progress' });

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
});

// ── Idle ────────────────────────────────────────────────────────────────────

describe('PluginActivityPane — idle', () => {
  it('says nothing is installing, and offers no controls', () => {
    paint();

    expect(screen.getByText('Nothing is installing right now.')).toBeInTheDocument();
    expect(
      screen.getByText('Downloads keep going if you close this window.'),
    ).toBeInTheDocument();
    expect(screen.queryByRole('progressbar')).toBeNull();
    expect(screen.queryByRole('log')).toBeNull();
    expect(screen.queryAllByRole('button')).toEqual([]);
  });
});

// ── A job in flight ─────────────────────────────────────────────────────────

describe('PluginActivityPane — a running job', () => {
  it('names the plugin it is installing, the step it is on and how far it has got', () => {
    paint({ job: downloading(), entry: demo });

    expect(screen.getByText('Installing Demo plugin')).toBeInTheDocument();
    // The step id is what is translated; the server's English label is only
    // the fallback for an id this build has never heard of.
    expect(screen.getByText('Step 2: Downloading')).toBeInTheDocument();
    expect(screen.getByText('Overall progress')).toBeInTheDocument();
    expect(progressBar()).toHaveAttribute('aria-valuenow', '12.5');
    // Nothing has ended, so there is no banner to say otherwise.
    expect(screen.queryByRole('status')).toBeNull();
  });

  it('says it is UPDATING when that is what the job is', () => {
    paint({ job: downloading({ kind: 'update' }), entry: demo });

    expect(screen.getByText('Updating Demo plugin')).toBeInTheDocument();
    expect(screen.queryByText('Installing Demo plugin')).toBeNull();
  });

  it('names a plugin the catalog has no row for by its id', () => {
    // A job adopted from another tab, or a plugin installed from a URL this
    // registry does not list: the pane still has a name to print.
    paint({ job: downloading({ pluginId: 'owner-demo' }), entry: undefined });

    expect(screen.getByText('Installing owner-demo')).toBeInTheDocument();
  });

  it('shows the transcript, and says so while it is still empty', () => {
    paint({ job: downloading(), entry: demo });
    expect(
      within(screen.getByRole('log', { name: 'Install log' })).getByText(
        'Waiting for the first message...',
      ),
    ).toBeInTheDocument();
  });

  it('prints the server log verbatim, in the server language', () => {
    paint({
      job: downloading({
        log: [
          { seq: 1, ts: null, kind: 'step', text: 'Resolving owner/demo' },
          { seq: 2, ts: null, kind: 'log', text: 'Collecting model2vec>=0.8.0' },
        ],
      }),
      entry: demo,
    });

    const log = within(screen.getByRole('log', { name: 'Install log' }));
    expect(log.getByText('Resolving owner/demo')).toBeInTheDocument();
    expect(log.getByText('Collecting model2vec>=0.8.0')).toBeInTheDocument();
  });

  it('is indeterminate until the first step is announced', () => {
    // 0 % about a job that is clearly working is a wrong number rather than
    // no number: ARIA spells "unknown" as a missing `aria-valuenow`.
    paint({ job: job(), entry: demo });

    expect(progressBar()).not.toHaveAttribute('aria-valuenow');
    expect(screen.queryByText(/^Step /)).toBeNull();
  });

  it('hands Cancel to the store', () => {
    paint({ job: downloading(), entry: demo });

    fireEvent.click(screen.getByRole('button', { name: 'Cancel install' }));
    expect(handlers.onCancel).toHaveBeenCalledTimes(1);
    // Nothing to dismiss while it runs.
    expect(screen.queryByRole('button', { name: 'Dismiss' })).toBeNull();
  });

  it('says it is cancelling, and takes no second press', () => {
    paint({ job: downloading(), entry: demo, cancelling: true });

    const button = screen.getByRole('button', { name: 'Cancelling...' });
    expect(button).toBeDisabled();
    fireEvent.click(button);
    expect(handlers.onCancel).not.toHaveBeenCalled();
  });
});

// ── The overall bar ─────────────────────────────────────────────────────────

describe('PluginActivityPane — the overall bar', () => {
  it('lets the tarball bytes refine the step that is downloading', () => {
    // A quarter of the download, which is one step of eight: a quarter of
    // that step, on top of the one that is done.
    const bytes = downloading({
      items: { tarball: { bytesDone: 300, bytesTotal: 1200, percent: 25 } },
    });
    expect(jobOverallPercent(bytes, demo)).toBe(15.625);

    paint({ job: bytes, entry: demo });
    expect(progressBar()).toHaveAttribute('aria-valuenow', '15.625');
  });

  it('stops reading full through the steps after the download', () => {
    // The bytes are ONE step's worth. Counted as the whole job, a finished
    // download parked the bar at 100 % through `deps` -- minutes of uv --
    // and everything after it.
    const unpacking = job({
      steps: [
        { step: 'resolve', label: 'Resolving owner/demo', state: 'done' },
        { step: 'download', label: 'Downloading', state: 'done' },
        { step: 'extract', label: 'Unpacking demo', state: 'done' },
        { step: 'verify', label: 'Scanning demo for unsafe code', state: 'done' },
        { step: 'deps', label: 'Installing packages', state: 'running' },
      ],
      items: { tarball: { bytesDone: 1000, bytesTotal: 1000, percent: 100 } },
    });

    expect(jobOverallPercent(unpacking, demo)).toBe(50);
  });

  it('counts finished steps out of the eight a GitHub install has', () => {
    // Out of EIGHT and not out of the steps announced so far: the server
    // announces a step as it starts, so "one of the two we know about" would
    // read as half a bar during the second step of eight.
    const steps = job({
      steps: [
        { step: 'resolve', label: 'Resolving owner/demo', state: 'done' },
        { step: 'download', label: 'Downloading', state: 'done' },
        { step: 'extract', label: 'Unpacking demo', state: 'done' },
        { step: 'verify', label: 'Scanning demo for unsafe code', state: 'running' },
      ],
    });
    expect(jobOverallPercent(steps, demo)).toBe(37.5);

    paint({ job: steps, entry: demo });
    expect(progressBar()).toHaveAttribute('aria-valuenow', '37.5');
    expect(screen.getByText('Step 4: Checking the code')).toBeInTheDocument();
  });

  it('lets a built-in pack reach the end of its own four steps', () => {
    // A built-in has nothing to download: `resolve [deps] lock reload`. Out
    // of eight it topped out at half a bar and was replaced by the banner
    // there, which reads as an install that stopped halfway.
    const builtin = job({
      pluginId: 'stats',
      steps: [
        { step: 'resolve', label: 'Resolving stats', state: 'done' },
        { step: 'deps', label: 'Installing packages', state: 'done' },
        { step: 'lock', label: 'Recording the install', state: 'done' },
        { step: 'reload', label: 'Loading the nodes', state: 'done' },
      ],
    });

    expect(jobOverallPercent(builtin, stats)).toBe(100);
  });

  it('falls back to the steps when the download never learned its size', () => {
    // `codeload` generates a tarball on demand and often sends no
    // Content-Length at all, which is a null total rather than a zero one.
    expect(
      jobOverallPercent(
        downloading({
          items: { tarball: { bytesDone: 4096, bytesTotal: null, percent: null } },
        }),
        demo,
      ),
    ).toBe(12.5);
  });

  it('counts no more than one step of bytes, whatever the server says', () => {
    expect(
      jobOverallPercent(
        downloading({
          items: { tarball: { bytesDone: 1500, bytesTotal: 1000, percent: 100 } },
        }),
        demo,
      ),
    ).toBe(25);
  });

  it('counts a row the catalog has not got as the longer install', () => {
    // A first install from a typed repository has no row at all, and eight
    // steps is both the likely truth and the understated guess.
    expect(jobOverallPercent(downloading(), undefined)).toBe(12.5);
  });

  it('has no answer at all for no job and no steps', () => {
    expect(jobOverallPercent(null, demo)).toBeNull();
    expect(jobOverallPercent(job(), demo)).toBeNull();
  });
});

// ── How a job ended ─────────────────────────────────────────────────────────

describe('PluginActivityPane — how a job ended', () => {
  it('reports an install that finished, and titles it by its plugin', () => {
    paint({ job: job({ status: 'done' }), entry: demo });

    expect(banner()).toHaveAttribute('data-tone', 'success');
    expect(within(banner()).getByText('Installed Demo plugin.')).toBeInTheDocument();
    // "Installing X" over "Installed X." would be the panel saying both.
    expect(screen.queryByText('Installing Demo plugin')).toBeNull();
    expect(screen.getByText('Demo plugin')).toBeInTheDocument();
    // The bar belongs to a job that is still going.
    expect(screen.queryByRole('progressbar')).toBeNull();
  });

  it('reports an update that finished', () => {
    paint({ job: job({ status: 'done', kind: 'update' }), entry: demo });
    expect(within(banner()).getByText('Updated Demo plugin.')).toBeInTheDocument();
  });

  it('reports a failure with the server hint, and the same install in a terminal', () => {
    paint({
      job: job({
        status: 'failed',
        error: { message: 'HTTP 404', hint: 'Check the repository name.' },
      }),
      entry: demo,
    });

    expect(banner()).toHaveAttribute('data-tone', 'error');
    expect(within(banner()).getByText('Install failed: HTTP 404')).toBeInTheDocument();
    expect(within(banner()).getByText('Check the repository name.')).toBeInTheDocument();
    expect(within(banner()).getByText('Or install from a terminal:')).toBeInTheDocument();
    // The repository and the ref this row is pinned to, so what the user
    // pastes reproduces THIS row rather than whatever the branch holds now.
    expect(
      within(banner()).getByText('cdui plugin install owner/demo@v1.2.0'),
    ).toBeInTheDocument();
  });

  it('offers a built-in pack its catalog name', () => {
    paint({
      job: job({ status: 'failed', pluginId: 'stats', error: { message: 'boom', hint: null } }),
      entry: stats,
    });
    expect(within(banner()).getByText('cdui plugin install stats')).toBeInTheDocument();
  });

  it('offers a linked folder no command at all', () => {
    // `cdui plugin install` takes a name or a repository. A linked row's
    // source is a path on this machine, so the command would be one the CLI
    // refuses; a folder is put back with `cdui plugin link`.
    paint({
      job: job({ status: 'failed', pluginId: 'lab', error: { message: 'boom', hint: null } }),
      entry: linked,
    });

    expect(within(banner()).getByText('Install failed: boom')).toBeInTheDocument();
    expect(screen.queryByText('Or install from a terminal:')).toBeNull();
    expect(screen.queryByText(/^cdui plugin/)).toBeNull();
  });

  it('offers no command for a job whose row the catalog has not got', () => {
    paint({
      job: job({ status: 'failed', error: { message: 'boom', hint: null } }),
      entry: undefined,
    });

    expect(within(banner()).getByText('Install failed: boom')).toBeInTheDocument();
    expect(screen.queryByText('Or install from a terminal:')).toBeNull();
  });

  it('says an UPDATE failed in the words an update gets', () => {
    paint({
      job: job({ status: 'failed', kind: 'update', error: { message: 'HTTP 500', hint: null } }),
      entry: demo,
    });
    expect(within(banner()).getByText('Update failed: HTTP 500')).toBeInTheDocument();
  });

  it('reports an install the user stopped', () => {
    paint({ job: job({ status: 'cancelled' }), entry: demo });

    expect(banner()).toHaveAttribute('data-tone', 'neutral');
    expect(within(banner()).getByText('Install cancelled.')).toBeInTheDocument();
  });

  it('says nothing was installed, and prints the command the server cannot run', () => {
    // The shape the backend really sends: a plugin reaches this status only
    // when its Python packages would replace one the running interpreter has
    // loaded, which the resolver refuses BEFORE anything is written -- so the
    // command is the install to run with the server stopped, and the panel
    // install has to be repeated afterwards.
    paint({
      job: job({
        status: 'needs_restart',
        restartCommand: 'uv pip install "torch==2.4.0"',
      }),
      entry: demo,
    });

    expect(banner()).toHaveAttribute('data-tone', 'warning');
    expect(
      within(banner()).getByText(
        "The install stopped before changing anything: Demo plugin's Python packages "
        + 'would replace one the server has loaded. With the server stopped, run this, '
        + 'then install again:',
      ),
    ).toBeInTheDocument();
    expect(within(banner()).getByText('uv pip install "torch==2.4.0"')).toBeInTheDocument();
  });

  it('leaves the banner alone when a stopped install carried no command', () => {
    paint({ job: job({ status: 'needs_restart', restartCommand: null }), entry: demo });

    expect(banner()).toHaveAttribute('data-tone', 'warning');
    expect(within(banner()).getByText(/^The install stopped/)).toBeInTheDocument();
    // `CommandBlock` is the only thing in the pane with a copy button.
    expect(screen.queryByRole('button', { name: 'Copy command' })).toBeNull();
  });

  it('offers to re-read the catalog when the follower lost contact', () => {
    paint({ job: job({ status: 'lost' }), entry: demo });

    expect(banner()).toHaveAttribute('data-tone', 'warning');
    expect(
      within(banner()).getByText(
        'Lost contact with the server. Refresh to check the plugin status.',
      ),
    ).toBeInTheDocument();

    // "Refresh", not the header icon's "Refresh plugin status": the sentence
    // above it already says what refreshing is for, and the dialog would
    // otherwise offer two controls with one name.
    fireEvent.click(screen.getByRole('button', { name: 'Refresh' }));
    expect(handlers.onRefresh).toHaveBeenCalledTimes(1);

    // A job nobody can report is still a job to put away.
    fireEvent.click(screen.getByRole('button', { name: 'Dismiss' }));
    expect(handlers.onDismiss).toHaveBeenCalledTimes(1);
  });

  it('hands Dismiss to the store, and offers no cancel', () => {
    paint({ job: job({ status: 'done' }), entry: demo });

    expect(screen.queryByRole('button', { name: 'Cancel install' })).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: 'Dismiss' }));
    expect(handlers.onDismiss).toHaveBeenCalledTimes(1);
  });

  it('says how a job ended exactly once', () => {
    // The banner is a live region in its own right (`role="status"`), so a
    // copy of its sentence in the announcer above it was one fact said
    // twice: read out twice, and printed twice in the pane -- once in the
    // banner and once in the status line over the log.
    paint({ job: job({ status: 'done' }), entry: demo });

    expect(screen.getAllByText('Installed Demo plugin.')).toHaveLength(1);
    expect(within(banner()).getByText('Installed Demo plugin.')).toBeInTheDocument();
  });

  it('hands the ending to the banner and goes quiet', () => {
    // The announcer carries the running commentary the banner cannot give,
    // and stops where the banner starts. It stays MOUNTED and empty rather
    // than unmounting: a live region that appears with its text already in
    // it is not reliably announced, so the next job's first step has to land
    // in a region that was already there.
    const { container, rerender } = paint({ job: downloading(), entry: demo });
    const live = () => container.querySelector('[aria-atomic="true"]');
    expect(live()).toHaveTextContent('Step 2: Downloading 13%');

    rerender(
      <PluginActivityPane
        job={job({ status: 'cancelled' })}
        entry={demo}
        cancelling={false}
        {...handlers}
      />,
    );
    expect(live()).not.toBeNull();
    expect(live()?.textContent).toBe('');
    expect(screen.getAllByText('Install cancelled.')).toHaveLength(1);
  });

  it('announces the job by its id while the catalog has no row for it', () => {
    // Before the first step event there is no step sentence and no
    // percentage, so the headline is all the region has -- and a job adopted
    // from another tab is named by its plugin id until the catalog answers.
    const { container } = paint({
      job: job({ pluginId: 'owner-demo' }), entry: undefined,
    });

    expect(container.querySelector('[aria-atomic="true"]')).toHaveTextContent(
      'Installing owner-demo',
    );
  });
});
