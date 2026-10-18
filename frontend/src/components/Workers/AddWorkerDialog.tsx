import { useEffect, useId, useRef, useState, type KeyboardEvent } from 'react';
import { createPortal } from 'react-dom';
import {
  agentCommand,
  addSshWorker,
  createEnrollment,
  deleteEnrollment,
  fetchEnrollments,
  fetchSshWorkers,
  type CreatedEnrollment,
  type EnrollmentList,
  type NewSshWorker,
  type SshWorkerList,
} from '../../api/workers';
import { useI18n } from '../../i18n';
import { useToastStore } from '../../store/toastStore';
import { friendlyError } from '../../utils/errorMessages';
import { trapTab } from '../shared/focusTrap';
import styles from './WorkersPanel.module.css';

/**
 * How the dialog tells the user to add a worker. `remote` (#780): a
 * one-time enrollment token and the one-line command that uses it, for an
 * agent on another machine. `ssh` (#784): this server starts the agent on
 * `user@host` itself and the agent connects back through `ssh -R`.
 * `same-host`: `cdui agent start` beside the server, which needs no token.
 * The prop picks the tab it opens on.
 */
export type AddWorkerMode = 'remote' | 'ssh' | 'same-host';

/** Where `install.sh` puts the launcher; the server's own default. */
export const DEFAULT_REMOTE_CDUI = '~/CodefyUI/cdui';

export const AGENT_START_COMMAND = 'cdui agent start';
export const AGENT_STOP_COMMAND = 'cdui agent stop';

/** The enrollment lifetimes offered, in seconds; 24 h is the default. */
export const TTL_CHOICES = [
  { seconds: 3600, key: 'workers.add.remote.ttl.1h' },
  { seconds: 86400, key: 'workers.add.remote.ttl.24h' },
  { seconds: 604800, key: 'workers.add.remote.ttl.7d' },
] as const;
const DEFAULT_TTL_S = 86400;
const MAX_USES = 100;

/**
 * A worker name, as the server checks it (`WORKER_NAME_PATTERN`). This and
 * the reserved names are pinned to the server's by
 * `backend/tests/test_api_workers.py`.
 */
const WORKER_NAME = /^[a-z0-9][a-z0-9-]{0,62}$/;
/** Names the server keeps for itself (`RESERVED_WORKER_NAMES`). */
const RESERVED_NAMES: ReadonlySet<string> = new Set(['local', 'any', 'enroll', 'enrollments']);

/**
 * Why *typed* cannot be a worker name, as a message key, or null when it
 * can. The server lower-cases the name, so case is not a fault (#840).
 */
export function workerNameProblem(typed: string): 'workers.add.ssh.badName' | 'workers.add.ssh.reservedName' | null {
  const name = typed.trim().toLowerCase();
  if (name === '') return null;
  if (!WORKER_NAME.test(name)) return 'workers.add.ssh.badName';
  return RESERVED_NAMES.has(name) ? 'workers.add.ssh.reservedName' : null;
}

interface AddWorkerDialogProps {
  mode: AddWorkerMode;
  /** Re-key this enrolled worker: the token is bound to it (#780). */
  rekey?: string | null;
  onClose: () => void;
  /** Called after an SSH worker was added, before the dialog closes. */
  onAdded?: () => void;
}

/** Whether *url* names this machine only (127.x, localhost, ::1). */
export function isLoopbackUrl(url: string): boolean {
  let host: string;
  try {
    host = new URL(url).hostname;
  } catch {
    return false;
  }
  return host === 'localhost' || host === '[::1]' || host.startsWith('127.');
}

/** Whether *url* is an `https://` address. */
export function isHttpsUrl(url: string): boolean {
  return /^https:\/\//i.test(url);
}

/**
 * How the agent will verify this server (#786). Over `https`: the
 * certificate fingerprint when the server can read it, else the `openssl`
 * command that computes it on the TLS proxy's host. Over plain `http` to
 * another machine: a warning that the token and the secrets in every run
 * travel unencrypted.
 */
function TrustNote({ address, list }: { address: string; list: EnrollmentList | null }) {
  const { t } = useI18n();
  if (isHttpsUrl(address)) {
    const fingerprint = list?.tls_fingerprint ?? null;
    if (fingerprint) {
      return (
        <p role="note">
          {t('workers.add.remote.tls.fingerprint')} <code className={styles.fingerprint}>{fingerprint}</code>
        </p>
      );
    }
    return (
      <div role="note">
        <p>{t('workers.add.remote.tls.compute')}</p>
        {list?.fingerprint_command && <CommandLine command={list.fingerprint_command} />}
      </div>
    );
  }
  if (address === '' || isLoopbackUrl(address)) return null;
  return <p className={styles.warn} role="note">{t('workers.add.remote.plainHttp')}</p>;
}

function errorText(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

function CommandLine({ command }: { command: string }) {
  const { t } = useI18n();
  const copy = async () => {
    const { addToast } = useToastStore.getState();
    try {
      await navigator.clipboard.writeText(command);
      addToast(t('workers.add.copied'), 'success');
    } catch {
      addToast(t('workers.add.copyFailed'), 'error');
    }
  };
  return (
    <div className={styles.commandLine}>
      <code className={styles.command}>{command}</code>
      <button type="button" className={styles.btn} onClick={() => void copy()}>
        {t('workers.add.copy')}
      </button>
    </div>
  );
}

function SameHostInstructions() {
  const { t } = useI18n();
  return (
    <>
      <p>{t('workers.add.sameHost.intro')}</p>
      <p>{t('workers.add.sameHost.start')}</p>
      <CommandLine command={AGENT_START_COMMAND} />
      <p>{t('workers.add.sameHost.what')}</p>
      <h3 className={styles.subTitle}>{t('workers.add.sameHost.options')}</h3>
      <ul className={styles.options}>
        <li><code>--name &lt;name&gt;</code> {t('workers.add.sameHost.optName')}</li>
        <li><code>--devices cuda:0,cuda:1</code> {t('workers.add.sameHost.optDevices')}</li>
        <li><code>--foreground</code> {t('workers.add.sameHost.optForeground')}</li>
      </ul>
      <p>{t('workers.add.sameHost.stop')}</p>
      <CommandLine command={AGENT_STOP_COMMAND} />
    </>
  );
}

/**
 * Another machine (#780): pick a lifetime and a number of uses, generate an
 * enrollment token, copy the command. The server address in the command is
 * the one this page uses, editable, with the server's other addresses
 * offered when this page is on a loopback address another machine cannot
 * dial. The tokens already issued are listed (never their secrets) with a
 * revoke button each.
 */
/** A list the dialog read this long ago (ms) is not read again on mount. */
export const ENROLLMENT_LIST_FRESH_MS = 5000;

function RemoteInstructions({ rekey, initialList = null, initialAt = 0, onList, onUseSsh }: {
  rekey: string | null;
  /** The last list the dialog saw, so the form starts filled. */
  initialList?: EnrollmentList | null;
  /** When (Date.now()) *initialList* was read: an older one is read again on mount. */
  initialAt?: number;
  /** Every list this form reads (after a generate or a revoke too), for the dialog to keep. */
  onList?: (list: EnrollmentList) => void;
  /** Switch the dialog to Over SSH; offered when the server is on loopback. */
  onUseSsh?: () => void;
}) {
  const { t, locale } = useI18n();
  const [list, setList] = useState<EnrollmentList | null>(initialList);
  const [error, setError] = useState<string | null>(null);
  const [url, setUrl] = useState<string | null>(initialList?.control_url ?? null);
  const [ttl, setTtl] = useState(DEFAULT_TTL_S);
  const [uses, setUses] = useState(1);
  const [created, setCreated] = useState<CreatedEnrollment | null>(null);
  // The server-built command for the generated token and the address it
  // dials (#786): `created.command` at first, asked again when the address
  // field changes, so the flags are decided by the server's one builder.
  const [shown, setShown] = useState<{ url: string; command: string } | null>(null);
  const [busy, setBusy] = useState(false);

  const load = async () => {
    try {
      const next = await fetchEnrollments();
      setList(next);
      onList?.(next);
      setUrl((current) => current ?? next.control_url);
      setError(null);
    } catch (err) {
      setError(errorText(err));
    }
  };

  useEffect(() => {
    // On mount, unless the dialog's list was just read (its probe on open):
    // after a switch back from Over SSH the seed may predate a token
    // generated or revoked since (#840).
    if (initialList !== null && Date.now() - initialAt < ENROLLMENT_LIST_FRESH_MS) return;
    void load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const generate = async () => {
    if (busy) return;
    setBusy(true);
    try {
      const options = rekey ? { ttl_s: ttl, uses, worker: rekey } : { ttl_s: ttl, uses };
      const next = await createEnrollment(address ? { ...options, control_url: address } : options);
      setCreated(next);
      setShown({ url: next.control_url, command: next.command });
      await load();
    } catch (err) {
      setError(errorText(err));
    } finally {
      setBusy(false);
    }
  };

  const revoke = async (id: string) => {
    try {
      await deleteEnrollment(id);
      if (created?.id === id) {
        setCreated(null);
        setShown(null);
      }
      await load();
    } catch (err) {
      setError(errorText(err));
    }
  };

  const address = (url ?? '').trim().replace(/\/+$/, '');
  // The address the command uses: the typed one, else the server's own.
  const target = address || created?.control_url || list?.control_url || '';
  const others = list?.other_machine_urls ?? [];

  useEffect(() => {
    if (created === null || shown?.url === target || !/^https?:\/\/\S+$/i.test(target)) return;
    let live = true;
    agentCommand({ control_url: target, token: created.token, worker: created.worker })
      .then((command) => {
        if (live) setShown({ url: target, command });
      })
      .catch((err: unknown) => {
        if (live) setError(errorText(err));
      });
    return () => {
      live = false;
    };
  }, [created, shown, target]);
  return (
    <>
      <p>{rekey ? t('workers.add.rekey.intro', { name: rekey }) : t('workers.add.remote.intro')}</p>
      {error !== null && <div className={styles.error} role="alert">{friendlyError(error)}</div>}
      {list?.listening_on_loopback && (
        <div className={styles.warn} role="note">
          <p>{t('workers.add.remote.loopbackServer')}</p>
          {onUseSsh && (
            <button type="button" className={`${styles.btn} ${styles.useSsh}`} onClick={onUseSsh}>
              {t('workers.add.remote.useSsh')}
            </button>
          )}
        </div>
      )}
      <label className={styles.field}>
        {t('workers.add.remote.url')}
        <input
          className={`${styles.input} ${styles.inputUrl}`}
          value={url ?? ''}
          spellCheck={false}
          onChange={(event) => setUrl(event.target.value)}
        />
        <span className={styles.muted}>{t('workers.add.remote.urlHint')}</span>
      </label>
      {isLoopbackUrl(address) && others.length > 0 && (
        <>
          <p className={styles.warn}>{t('workers.add.remote.loopbackUrl', { url: address })}</p>
          <div className={styles.suggestions}>
            {others.map((other) => (
              <button key={other} type="button" className={styles.btn} onClick={() => setUrl(other)}>
                {other}
              </button>
            ))}
          </div>
        </>
      )}
      {isLoopbackUrl(address) && others.length === 0 && (
        <p className={styles.warn}>{t('workers.add.remote.loopbackUrlNone', { url: address })}</p>
      )}
      <TrustNote address={target} list={list} />
      <div className={styles.fieldRow}>
        <label className={styles.field}>
          {t('workers.add.remote.ttl')}
          <select className={styles.input} value={ttl} onChange={(event) => setTtl(Number(event.target.value))}>
            {TTL_CHOICES.map((choice) => (
              <option key={choice.seconds} value={choice.seconds}>{t(choice.key)}</option>
            ))}
          </select>
        </label>
        <label className={styles.field}>
          {t('workers.add.remote.uses')}
          <input
            className={`${styles.input} ${styles.inputUses}`}
            type="number"
            min={1}
            max={MAX_USES}
            value={uses}
            onChange={(event) => setUses(Math.min(MAX_USES, Math.max(1, Math.trunc(Number(event.target.value)) || 1)))}
          />
        </label>
        <button
          type="button"
          className={`${styles.btn} ${styles.accent}`}
          aria-disabled={busy}
          onClick={() => void generate()}
        >
          {t('workers.add.remote.generate')}
        </button>
      </div>
      {created !== null && shown?.url === target && (
        <>
          <p>{t('workers.add.remote.command')}</p>
          <CommandLine command={shown.command} />
          <p className={styles.muted}>{t('workers.add.remote.restart')}</p>
        </>
      )}
      <h3 className={styles.subTitle}>{t('workers.add.remote.tokens')}</h3>
      {list !== null && list.enrollments.length === 0 && (
        <p className={styles.muted}>{t('workers.add.remote.tokens.none')}</p>
      )}
      {list !== null && list.enrollments.length > 0 && (
        <ul className={styles.tokens}>
          {list.enrollments.map((entry) => (
            <li key={entry.id} className={styles.tokenRow}>
              <code>{entry.id}</code>
              {entry.label && <span>{entry.label}</span>}
              {entry.worker && (
                <span className={styles.fact}>{t('workers.add.remote.token.rekeys', { name: entry.worker })}</span>
              )}
              <span className={`${styles.chip} ${styles[`chip_${entry.state}`]}`}>
                {t(`workers.add.remote.state.${entry.state}`)}
              </span>
              <span className={styles.fact}>
                {t('workers.add.remote.token.detail', {
                  when: new Date(entry.expires_at).toLocaleString(locale),
                  uses: entry.uses_left,
                })}
              </span>
              <span className={styles.spacer} />
              <button
                type="button"
                className={styles.btn}
                aria-label={`${t(entry.state === 'pending' ? 'workers.add.remote.revoke' : 'workers.add.remote.delete')} ${entry.id}`}
                onClick={() => void revoke(entry.id)}
              >
                {t(entry.state === 'pending' ? 'workers.add.remote.revoke' : 'workers.add.remote.delete')}
              </button>
            </li>
          ))}
        </ul>
      )}
    </>
  );
}

/**
 * Connect over SSH (#784): the server runs `ssh -R ... user@host cdui agent
 * start` itself, so the worker needs no route to this server and neither
 * end opens a port. The fields are what ssh is given; the tunnel's state
 * shows on the worker's card in the Workers panel.
 */
function SshInstructions({ onAdded, onClose }: { onAdded?: () => void; onClose: () => void }) {
  const { t } = useI18n();
  const [info, setInfo] = useState<SshWorkerList | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [target, setTarget] = useState('');
  const [port, setPort] = useState('22');
  const [identity, setIdentity] = useState('');
  const [remoteCdui, setRemoteCdui] = useState('');
  const [name, setName] = useState('');
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    fetchSshWorkers().then(setInfo, (err) => setError(errorText(err)));
  }, []);

  const submit = async () => {
    if (busy) return;
    if (!target.trim()) {
      setError(t('workers.add.ssh.needTarget'));
      return;
    }
    const problem = workerNameProblem(name);
    if (problem !== null) {
      setError(t(problem, { name: name.trim() }));
      return;
    }
    const body: NewSshWorker = { target: target.trim(), port: Number(port) || 22 };
    if (identity.trim()) body.identity = identity.trim();
    if (remoteCdui.trim()) body.remote_cdui = remoteCdui.trim();
    if (name.trim()) body.name = name.trim();
    setBusy(true);
    try {
      const entry = await addSshWorker(body);
      useToastStore.getState().addToast(t('workers.add.ssh.added', { name: entry.id }), 'success');
      onAdded?.();
      onClose();
    } catch (err) {
      setError(errorText(err));
      setBusy(false);
    }
  };

  const defaultCdui = info?.default_remote_cdui ?? DEFAULT_REMOTE_CDUI;
  return (
    <form
      onSubmit={(event) => {
        event.preventDefault();
        void submit();
      }}
    >
      <p>{t('workers.add.ssh.intro')}</p>
      {info !== null && !info.ssh_available && (
        <p className={styles.warn} role="note">{t('workers.add.ssh.noSsh')}</p>
      )}
      {error !== null && <div className={styles.error} role="alert">{friendlyError(error)}</div>}
      <label className={styles.field}>
        {t('workers.add.ssh.target')}
        <input
          className={`${styles.input} ${styles.inputUrl}`}
          value={target}
          placeholder="alice@gpu-box"
          spellCheck={false}
          autoCapitalize="off"
          onChange={(event) => setTarget(event.target.value)}
        />
        <span className={styles.muted}>{t('workers.add.ssh.targetHint')}</span>
      </label>
      <div className={styles.fieldRow}>
        <label className={styles.field}>
          {t('workers.add.ssh.port')}
          <input
            className={`${styles.input} ${styles.inputUses}`}
            type="number"
            min={1}
            max={65535}
            value={port}
            onChange={(event) => setPort(event.target.value)}
          />
        </label>
        <label className={styles.field}>
          {t('workers.add.ssh.name')}
          <input
            className={styles.input}
            value={name}
            placeholder={t('workers.add.ssh.namePlaceholder')}
            spellCheck={false}
            onChange={(event) => setName(event.target.value)}
          />
        </label>
      </div>
      <label className={styles.field}>
        {t('workers.add.ssh.identity')}
        <input
          className={`${styles.input} ${styles.inputUrl}`}
          value={identity}
          placeholder="~/.ssh/id_ed25519"
          spellCheck={false}
          onChange={(event) => setIdentity(event.target.value)}
        />
        <span className={styles.muted}>{t('workers.add.ssh.identityHint')}</span>
      </label>
      <label className={styles.field}>
        {t('workers.add.ssh.remoteCdui')}
        <input
          className={`${styles.input} ${styles.inputUrl}`}
          value={remoteCdui}
          placeholder={defaultCdui}
          spellCheck={false}
          onChange={(event) => setRemoteCdui(event.target.value)}
        />
        <span className={styles.muted}>{t('workers.add.ssh.remoteCduiHint', { path: defaultCdui })}</span>
      </label>
      <div className={styles.fieldRow}>
        <button type="submit" className={`${styles.btn} ${styles.accent}`} aria-disabled={busy}>
          {t('workers.add.ssh.add')}
        </button>
      </div>
      <p className={styles.muted}>{t('workers.add.ssh.how')}</p>
      <p className={styles.muted}>{t('workers.add.ssh.security')}</p>
    </form>
  );
}

/** The mode switch's buttons, in order. */
const MODES = [
  ['remote', 'workers.add.mode.remote'],
  ['ssh', 'workers.add.mode.ssh'],
  ['same-host', 'workers.add.mode.sameHost'],
] as const;

/**
 * The Workers panel's "Add worker" dialog, over the panel.
 *
 * A re-key (#780) is the remote form alone, under its own title (#840):
 * a token bound to one worker has nothing to do with the other modes.
 */
export function AddWorkerDialog({ mode, rekey = null, onClose, onAdded }: AddWorkerDialogProps) {
  const { t } = useI18n();
  const titleId = useId();
  const dialogRef = useRef<HTMLDivElement>(null);
  const closeRef = useRef<HTMLButtonElement>(null);
  // Another machine cannot reach a server on loopback (#840): the dialog
  // reads the enrollment facts first and then opens on Over SSH, which
  // works from there. null while that read is in flight.
  const probing = mode === 'remote' && rekey === null;
  const [current, setCurrent] = useState<AddWorkerMode | null>(probing ? null : mode);
  const [seed, setSeed] = useState<{ list: EnrollmentList; at: number } | null>(null);
  const keepList = (list: EnrollmentList) => setSeed({ list, at: Date.now() });

  useEffect(() => {
    if (!probing) return;
    let live = true;
    fetchEnrollments().then(
      (list) => {
        if (!live) return;
        keepList(list);
        setCurrent((now) => now ?? (list.listening_on_loopback ? 'ssh' : 'remote'));
      },
      // The remote form reads it again and shows the error.
      () => {
        if (live) setCurrent((now) => now ?? 'remote');
      },
    );
    return () => {
      live = false;
    };
  }, [probing]);

  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null;
    closeRef.current?.focus();
    return () => {
      if (previous?.isConnected) previous.focus();
    };
  }, []);

  const handleKeys = (event: KeyboardEvent<HTMLDivElement>) => {
    // React events bubble through the portal to the Workers panel, whose own
    // Escape would close it too.
    event.stopPropagation();
    if (event.key === 'Escape') {
      event.preventDefault();
      onClose();
      return;
    }
    trapTab(event, dialogRef.current);
  };

  return createPortal(
    <div
      className={`${styles.backdrop} ${styles.backdropTop}`}
      onMouseDown={(event) => {
        event.stopPropagation();
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <div
        ref={dialogRef}
        className={`${styles.dialog} ${styles.dialogNarrow}`}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        tabIndex={-1}
        onKeyDown={handleKeys}
        data-mode={current}
      >
        <header className={styles.header}>
          <h2 id={titleId}>{rekey ? t('workers.rekey.dialogTitle', { name: rekey }) : t('workers.add.title')}</h2>
          <button
            ref={closeRef}
            type="button"
            className={styles.close}
            onClick={onClose}
            aria-label={t('workers.add.close')}
          >
            &#215;
          </button>
        </header>
        <div className={`${styles.body} ${styles.prose}`}>
          {rekey !== null ? <RemoteInstructions rekey={rekey} /> : (
            <>
              <div className={styles.modeSwitch} role="group" aria-label={t('workers.add.mode')}>
                {MODES.map(([value, key]) => (
                  <button
                    key={value}
                    type="button"
                    className={styles.btn}
                    aria-pressed={current === value}
                    onClick={() => setCurrent(value)}
                  >
                    {t(key)}
                  </button>
                ))}
              </div>
              {current === null && <p className={styles.muted}>{t('workers.add.probing')}</p>}
              {current === 'remote' && (
                <RemoteInstructions
                  rekey={null}
                  initialList={seed?.list ?? null}
                  initialAt={seed?.at}
                  onList={keepList}
                  onUseSsh={() => setCurrent('ssh')}
                />
              )}
              {current === 'ssh' && <SshInstructions onAdded={onAdded} onClose={onClose} />}
              {current === 'same-host' && <SameHostInstructions />}
            </>
          )}
        </div>
      </div>
    </div>,
    document.body,
  );
}
