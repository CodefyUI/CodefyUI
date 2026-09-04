import { describe, it, expect, beforeEach } from 'vitest';
import type { PluginCatalogEntry, PluginStatus } from '../../api/rest';
import { useI18n } from '../../i18n';
import {
  capabilityKey,
  cliInstallCommand,
  depSpec,
  httpUrl,
  isAvailableStatus,
  isInstalledStatus,
  manifestAuthor,
  matchesFilter,
  originLabel,
  provenancePin,
  parseGitHubSource,
  statusKey,
  statusTone,
  stepLabel,
} from './pluginStatus';

/**
 * The Plugin Center's pure rules.
 *
 * Every case here is a value that arrives from a SERVER: a status, a step id,
 * a capability name, a row with half its fields null. What is pinned is what
 * this build does with one it has never seen, because that is the answer the
 * panel cannot be tested for once it is on screen.
 */

function entry(
  partial: Partial<PluginCatalogEntry> & { id: string },
): PluginCatalogEntry {
  return {
    name: partial.id,
    description: '',
    kind: 'github',
    official: false,
    status: 'available',
    source_kind: null,
    source: '',
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
    ...partial,
  };
}

const t = useI18n.getState().t;

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
});

// ── status ───────────────────────────────────────────────────────────────

describe('statusTone', () => {
  it('greens what is here and blues what is arriving', () => {
    expect(statusTone('installed')).toBe('success');
    expect(statusTone('installing')).toBe('info');
  });

  it('warns only about the status that is actually wrong', () => {
    // The registry has the plugin and its directory is gone. Everything else
    // below is a state the user chose, and dressing a deliberate switch-off
    // as a warning would say something went wrong.
    expect(statusTone('missing_files')).toBe('warning');
    expect(statusTone('disabled')).toBe('neutral');
    expect(statusTone('removed')).toBe('neutral');
    expect(statusTone('available')).toBe('neutral');
  });

  it('reads a status this build has never heard of as neutral', () => {
    expect(statusTone('quarantined' as PluginStatus)).toBe('neutral');
  });
});

describe('statusKey', () => {
  it('labels every status the catalog can send', () => {
    expect(t(statusKey('installed'))).toBe('Installed');
    expect(t(statusKey('disabled'))).toBe('Disabled');
    expect(t(statusKey('available'))).toBe('Not installed');
    expect(t(statusKey('installing'))).toBe('Installing');
    expect(t(statusKey('removed'))).toBe('Removed');
    expect(t(statusKey('missing_files'))).toBe('Files missing');
  });

  it('falls back to "not installed" for a status off a newer server', () => {
    // The safe reading: it offers an Install button rather than claiming
    // something is in place. `undefined` would render as the raw key.
    expect(t(statusKey('quarantined' as PluginStatus))).toBe('Not installed');
  });

  it('translates with the locale, not with the call site', () => {
    useI18n.setState({ locale: 'zh-TW' });
    expect(t(statusKey('missing_files'))).toBe('檔案遺失');
  });
});

// ── the filter ───────────────────────────────────────────────────────────

describe('isAvailableStatus', () => {
  it('calls a plugin available when it is not here and could be', () => {
    expect(isAvailableStatus('available')).toBe(true);
    // A tombstone is a plugin that is not here and can be put back.
    expect(isAvailableStatus('removed')).toBe(true);
    expect(isAvailableStatus('installed')).toBe(false);
    expect(isAvailableStatus('disabled')).toBe(false);
  });

  it('reads a row the lockfile has as installed, however wrong it is', () => {
    // The two statuses that are neither "here" nor "not here": one is being
    // written and one lost its directory. Both are IN the lockfile, so both
    // belong to the Installed half -- under Available they would offer to
    // install what is already being installed.
    expect(isAvailableStatus('installing')).toBe(false);
    expect(isAvailableStatus('missing_files')).toBe(false);
  });

  it('puts a status this build has never heard of on the installed side', () => {
    // Written as the narrow half and negated for the other, so no filter can
    // make a row disappear from both tabs.
    expect(isAvailableStatus('quarantined' as PluginStatus)).toBe(false);
  });
});

describe('isInstalledStatus', () => {
  it('calls a plugin installed when its files are here, switched on or not', () => {
    expect(isInstalledStatus('installed')).toBe(true);
    expect(isInstalledStatus('disabled')).toBe(true);
    expect(isInstalledStatus('available')).toBe(false);
    expect(isInstalledStatus('removed')).toBe(false);
  });

  it('is NOT the filter negated: two statuses are in neither count', () => {
    // The sidebar list and the settings summary answer "what have I got" and
    // "what can I get". A download in flight is neither, and a lockfile entry
    // with no directory is neither -- while the FILTER has to put both
    // somewhere, so that no row vanishes from both of its tabs.
    for (const status of ['installing', 'missing_files'] as PluginStatus[]) {
      expect(isInstalledStatus(status)).toBe(false);
      expect(isAvailableStatus(status)).toBe(false);
      expect(isInstalledStatus(status)).not.toBe(!isAvailableStatus(status));
    }
  });

  it('leaves a status this build has never heard of out of both counts', () => {
    expect(isInstalledStatus('quarantined' as PluginStatus)).toBe(false);
    expect(isAvailableStatus('quarantined' as PluginStatus)).toBe(false);
  });
});

describe('matchesFilter', () => {
  it('lets everything past the All tab', () => {
    expect(matchesFilter('all', 'available')).toBe(true);
    expect(matchesFilter('all', 'installed')).toBe(true);
  });

  it('splits the catalog into two halves with nothing in neither', () => {
    const statuses: PluginStatus[] = [
      'installed', 'disabled', 'available', 'removed', 'installing', 'missing_files',
    ];
    for (const status of statuses) {
      expect(matchesFilter('installed', status)).toBe(!matchesFilter('available', status));
    }
  });
});

// ── steps ────────────────────────────────────────────────────────────────

describe('stepLabel', () => {
  it('has a sentence of its own for seven of the eight steps', () => {
    // The eighth is `deps`, which shares the Package Center's pip sentence
    // and is pinned in the case below rather than counted here.
    //
    // The server's own label is a log line ("Scanning demo for unsafe code");
    // it is passed in here to prove the id wins over it.
    expect(stepLabel(t, 'resolve', 'Resolving owner/demo')).toBe('Resolving the source');
    expect(stepLabel(t, 'download', 'Downloading owner/demo@abc')).toBe('Downloading');
    expect(stepLabel(t, 'extract', 'Unpacking demo')).toBe('Unpacking');
    expect(stepLabel(t, 'verify', 'Scanning demo')).toBe('Checking the code');
    expect(stepLabel(t, 'stage', 'Installing demo')).toBe('Copying files');
    expect(stepLabel(t, 'lock', 'Recording demo')).toBe('Recording the install');
    expect(stepLabel(t, 'reload', 'Loading demo')).toBe('Loading the nodes');
  });

  it('says pip the way the Package Center says it', () => {
    // One sentence for one job: `deps` has no key of its own on purpose.
    expect(stepLabel(t, 'deps', 'Installing requests')).toBe(
      'Installing Python packages',
    );
  });

  it('reads the kind half of an id that carries an item', () => {
    // No plugin step is shaped this way today; the job protocol allows it,
    // and a newer backend's `download:tarball` still has to find its
    // sentence rather than fall through to a log line.
    expect(stepLabel(t, 'download:tarball', 'Downloading the tarball')).toBe(
      'Downloading',
    );
  });

  it('falls back to the server label, and to the id when there is none', () => {
    expect(stepLabel(t, 'quarantine', 'Checking the quarantine')).toBe(
      'Checking the quarantine',
    );
    expect(stepLabel(t, 'quarantine', '')).toBe('quarantine');
  });
});

// ── capabilities ─────────────────────────────────────────────────────────

describe('capabilityKey', () => {
  /** The line a card would print for *id*. Throws rather than falling back. */
  function line(id: string): string {
    const key = capabilityKey(id);
    if (key === null) throw new Error(`no capability line for "${id}"`);
    return t(key);
  }

  it('says what granting each of the three costs', () => {
    expect(line('network')).toBe(
      'network: reach any host, and write what it downloads to disk',
    );
    expect(line('filesystem')).toBe(
      'filesystem: use the file libraries (pathlib, shutil, zip/tar, sqlite3)',
    );
    expect(line('process-env')).toContain('the whole os module');
    // The hyphenated id keeps its wire spelling; only the KEY is camelCase.
    expect(capabilityKey('process-env')).toBe('pluginCenter.cap.processEnv');
  });

  it('answers null for an id this build has no line for', () => {
    // The card prints the raw id then: an unknown capability must still be
    // visible on a consent screen, not silently dropped.
    expect(capabilityKey('gpu')).toBeNull();
    expect(capabilityKey('')).toBeNull();
  });
});

// ── origin and the CLI line ──────────────────────────────────────────────

describe('originLabel', () => {
  it('names the origins that are not obvious from the row', () => {
    expect(originLabel(entry({ id: 'edu', kind: 'builtin', official: true })))
      .toBe('pluginCenter.origin.builtin');
    expect(originLabel(entry({ id: 'demo', official: true })))
      .toBe('pluginCenter.origin.official');
    expect(originLabel(entry({ id: 'demo', kind: 'external', source_kind: 'local' })))
      .toBe('pluginCenter.origin.local');
  });

  it('says nothing about a plain third-party repository', () => {
    // The card prints owner/repo beside this; a "GitHub" chip over a GitHub
    // link is the same fact twice.
    expect(originLabel(entry({ id: 'demo', repo: 'owner/demo' }))).toBeNull();
  });

  it('prefers the linked folder to who published it', () => {
    // What is actually being loaded is the folder on disk, whoever wrote it.
    expect(originLabel(entry({ id: 'demo', official: true, source_kind: 'local' })))
      .toBe('pluginCenter.origin.local');
  });
});

describe('provenancePin', () => {
  it('names the ref and the seven characters of the commit', () => {
    expect(provenancePin('main', 'abcdef1234567890', '1.2.0')).toBe('main @ abcdef1');
  });

  it('says the commit alone for a default-branch install', () => {
    // '' is the server saying "the default branch", not a missing value: a
    // bare `@` would read as a pin that lost half of itself.
    expect(provenancePin('', 'abcdef1234567890', '1.2.0')).toBe('abcdef1');
    expect(provenancePin(null, 'abcdef1234567890', null)).toBe('abcdef1');
  });

  it('drops a ref that only repeats the version the header prints', () => {
    // A card's header says `v1.2.0`. `v1.2.0 @ abcdef1` under it is one
    // release wearing two rows -- and the tag and the bare number are the
    // same release named two ways.
    expect(provenancePin('v1.2.0', 'abcdef1234567890', '1.2.0')).toBe('abcdef1');
    expect(provenancePin('1.2.0', 'abcdef1234567890', '1.2.0')).toBe('abcdef1');
    // A ref that says something else keeps its half.
    expect(provenancePin('v1.3.0', 'abcdef1234567890', '1.2.0')).toBe('v1.3.0 @ abcdef1');
  });

  it('has nothing to pin without a commit', () => {
    expect(provenancePin('main', null, '1.2.0')).toBeNull();
    // A lockfile entry for a built-in records `''` rather than null, and
    // `''.slice(0, 7)` is an empty phrase on the meta line.
    expect(provenancePin('main', '', '1.2.0')).toBeNull();
  });
});

describe('depSpec', () => {
  it('uses an operator-led constraint exactly as the manifest wrote it', () => {
    expect(depSpec('model2vec', '>=0.8.0')).toBe('model2vec>=0.8.0');
    expect(depSpec('numpy', '~=1.24')).toBe('numpy~=1.24');
  });

  it('pins a bare version, the way the installer would', () => {
    // `torch = "2.1.0"` in a manifest means that version, and `uv pip install
    // torch2.1.0` is not a spec. `_build_dep_spec` adds the `==`; so does this.
    expect(depSpec('torch', '2.1.0')).toBe('torch==2.1.0');
  });

  it('leaves a name alone when nothing is constrained', () => {
    expect(depSpec('requests', '')).toBe('requests');
  });
});

describe('manifestAuthor', () => {
  it('reads the list the scaffolded manifest writes', () => {
    expect(manifestAuthor({ plugin: { authors: ['Ada', 'Grace'] } })).toBe('Ada, Grace');
  });

  it('takes the singular a hand-written manifest as often has', () => {
    expect(manifestAuthor({ plugin: { author: '  Ada  ' } })).toBe('Ada');
  });

  it('says nothing rather than something it cannot read', () => {
    // A manifest is hand-written, and an inspection echoes it whole and
    // unvalidated: `[object Object]` on a consent screen is worse than no
    // line at all.
    expect(manifestAuthor({})).toBeNull();
    expect(manifestAuthor({ plugin: {} })).toBeNull();
    expect(manifestAuthor({ plugin: { authors: [] } })).toBeNull();
    expect(manifestAuthor({ plugin: { authors: [{ name: 'Ada' }] } })).toBeNull();
    expect(manifestAuthor({ plugin: { author: '   ' } })).toBeNull();
    expect(manifestAuthor({ plugin: 'Ada' })).toBeNull();
  });
});

describe('httpUrl', () => {
  it('passes a web address through', () => {
    expect(httpUrl('https://example.com/demo')).toBe('https://example.com/demo');
    expect(httpUrl('  http://example.com  ')).toBe('http://example.com');
  });

  it('refuses a scheme that would run rather than navigate', () => {
    // The homepage comes out of a manifest at a source NOBODY has installed
    // yet, and the review card is the one screen that asks the user to trust
    // it: a `javascript:` href there runs inside the editor on a click.
    expect(httpUrl('javascript:alert(1)')).toBeNull();
    expect(httpUrl('data:text/html,<script>alert(1)</script>')).toBeNull();
    expect(httpUrl('file:///etc/passwd')).toBeNull();
    expect(httpUrl('example.com')).toBeNull();
    expect(httpUrl('')).toBeNull();
  });
});

describe('cliInstallCommand', () => {
  it('installs a built-in pack by name', () => {
    expect(cliInstallCommand(entry({ id: 'edu', kind: 'builtin', source: 'edu' })))
      .toBe('cdui plugin install edu');
  });

  it('pins the ref a github row was installed at', () => {
    expect(cliInstallCommand(entry({
      id: 'demo', repo: 'owner/demo', ref: 'v1.2', source: 'owner/demo',
    }))).toBe('cdui plugin install owner/demo@v1.2');
  });

  it('leaves the ref off a default-branch install', () => {
    // '' is the server's way of saying "the default branch", not a missing
    // value: `owner/demo@` would be a command that does not run.
    expect(cliInstallCommand(entry({ id: 'demo', repo: 'owner/demo', ref: '' })))
      .toBe('cdui plugin install owner/demo');
  });

  it('falls back to the source, and then to the id', () => {
    expect(cliInstallCommand(entry({ id: 'demo', source: 'owner/demo' })))
      .toBe('cdui plugin install owner/demo');
    expect(cliInstallCommand(entry({ id: 'demo' }))).toBe('cdui plugin install demo');
  });
});

// ── the re-export ────────────────────────────────────────────────────────

describe('parseGitHubSource', () => {
  it('is the store parser, so the form and the store agree on a source', () => {
    expect(parseGitHubSource('owner/demo@v1')).toEqual({
      kind: 'github', owner: 'owner', repo: 'demo', ref: 'v1',
    });
    expect(parseGitHubSource('not a repo!')).toBeNull();
  });
});
