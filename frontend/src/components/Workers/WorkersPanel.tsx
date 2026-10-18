import { Suspense, useEffect, useId, useRef, useState, type KeyboardEvent, type ReactNode } from 'react';
import { createPortal } from 'react-dom';
import type {
  DevicePolicy,
  SshWorker,
  TunnelState,
  WorkerDevice,
  WorkerDeviceAdmission,
  WorkerInfo,
  WorkerNodeCatalog,
  WorkerOneShot,
  WorkerRejection,
  WorkerStatus,
} from '../../api/workers';
import { useI18n, type TranslationKey } from '../../i18n';
import { useRunStore } from '../../store/runStore';
import { useWorkerStore } from '../../store/workerStore';
import { friendlyError } from '../../utils/errorMessages';
import { formatBytes } from '../../utils/formatBytes';
import { formatGiB } from '../../utils/formatGiB';
import { relativeTime } from '../SourceControl/scm';
import { trapTab } from '../shared/focusTrap';
import { lazyRetry } from '../shared/lazyRetry';
import styles from './WorkersPanel.module.css';

/** Hex characters of the environment digest shown; the tooltip has it all. */
export const FINGERPRINT_SHORT_LENGTH = 12;
/** Characters of a run id shown in a link; the tooltip has it all. */
const RUN_ID_SHORT_LENGTH = 8;

/**
 * The statuses this build has a word and a colour for. A newer server may
 * send another; it is shown as sent, in a neutral chip.
 */
const KNOWN_STATUSES: ReadonlySet<string> = new Set<WorkerStatus>([
  'online', 'degraded', 'disconnected', 'draining', 'offline',
]);

function RunLinks({ runIds }: { runIds: string[] }) {
  const { t } = useI18n();
  const revealRuns = useWorkerStore((s) => s.revealRuns);
  if (runIds.length === 0) return <span className={styles.muted}>{t('workers.running.none')}</span>;
  return (
    <span className={styles.runLinks}>
      {runIds.map((id) => (
        <button
          key={id}
          type="button"
          className={styles.linkBtn}
          title={t('workers.runLink', { id })}
          onClick={() => {
            revealRuns();
            void useRunStore.getState().select(id);
          }}
        >
          {id.slice(0, RUN_ID_SHORT_LENGTH)}
        </button>
      ))}
    </span>
  );
}

/** The device kinds a policy can be set for (#788). */
const SHAREABLE_KINDS: ReadonlySet<string> = new Set(['cuda', 'mps']);

/** Percent of *total* that *part* is, clamped to 0..100. */
function share(part: number, total: number): number {
  return Math.max(0, Math.min(100, (part / total) * 100));
}

/**
 * A shared card's memory (#788): in use on the card now (any process,
 * other users' jobs included), promised to runs that have not started yet,
 * and what a new budget may take, with the budgets in words beside it.
 */
function AdmissionBar({ admission }: { admission: WorkerDeviceAdmission }) {
  const { t } = useI18n();
  const { total_bytes: total, free_bytes: free, headroom_bytes: headroom } = admission;
  const context = admission.context_bytes ?? 0;
  // The largest budget that fits: a run also holds its CUDA context.
  const fits = headroom === null ? null : Math.max(headroom - context, 0);
  const line = t('workers.admission.line', {
    headroom: fits === null ? '-' : formatGiB(fits),
    running: formatGiB(admission.running_bytes),
    reserved: formatGiB(admission.reserved_bytes),
  });
  return (
    <div className={styles.admission}>
      {total !== null && total > 0 && (
        <div className={styles.bar} role="img" aria-label={line}>
          <span
            className={styles.barUsed}
            style={{ width: `${share(free === null ? 0 : total - free, total)}%` }}
            title={free === null ? undefined : t('workers.admission.used', { amount: formatGiB(total - free) })}
          />
          <span
            className={styles.barReserved}
            style={{ width: `${share(admission.reserved_bytes, total)}%` }}
            title={t('workers.admission.reserved', { amount: formatGiB(admission.reserved_bytes) })}
          />
          <span
            className={styles.barHeadroom}
            style={{ width: `${share(headroom ?? 0, total)}%` }}
            title={t('workers.admission.headroom', { amount: formatGiB(fits ?? 0) })}
          />
        </div>
      )}
      <span className={styles.fact}>{line}</span>
      {context > 0 && (
        <span className={styles.fact}>{t('workers.admission.context', { amount: formatGiB(context) })}</span>
      )}
      {admission.runs.length > 0 && <RunLinks runIds={admission.runs} />}
    </div>
  );
}

function DeviceRow({ device, workerId, enforces = true }: {
  device: WorkerDevice;
  workerId: string;
  /** This worker's agent enforces memory budgets (#788); `false` from a server that says it cannot. */
  enforces?: boolean;
}) {
  const { t } = useI18n();
  const busy = useWorkerStore((s) => Boolean(s.busy[workerId]));
  const setPolicy = useWorkerStore((s) => s.setPolicy);
  const memory = device.memory;
  const hasMemory = memory != null && memory.free_bytes !== null && memory.total_bytes !== null;
  const shareable = device.policy !== undefined && SHAREABLE_KINDS.has(device.kind);
  const admission = device.admission ?? null;
  const peers = device.same_card ?? [];
  const peerName = (peer: { worker: string; device: string }) =>
    peer.worker === 'local'
      ? t('workers.device.sameCard.local', { device: peer.device })
      : `${peer.worker} ${peer.device}`;
  const list = (names: string[]) => names.reduce((all, name) => t('workers.device.sameCard.more', { list: all, peer: name }));
  // Other online workers offering the card with another policy: the card
  // admits by count, the stricter policy (#842).
  const otherPolicy = device.policy === undefined ? [] : peers.filter(
    (peer) => peer.worker !== 'local' && peer.online !== false && peer.policy != null && peer.policy !== device.policy,
  );
  const byCount = admission === null || !admission.by_memory;
  // Another online offerer whose agent cannot enforce budgets makes the
  // card admit by count (#842): name it rather than this worker's agent.
  const peersWithout = peers.filter(
    (peer) => peer.worker !== 'local' && peer.online !== false && peer.memory_budget === false,
  );
  // The CPU and Apple GPU of one machine are not matched by a card UUID.
  const machineWide = device.kind === 'cpu' || device.kind === 'mps';
  const limitedBy = device.slots?.limited_by ?? null;
  return (
    <li className={styles.device}>
      <span className={styles.deviceName}>
        <code>{device.value}</code>
        {device.name && device.name !== device.value && (
          <span className={styles.muted}> {device.name}</span>
        )}
      </span>
      {device.slots && byCount && (
        limitedBy === null
          ? (
            <span className={styles.fact}>
              {t('workers.device.slots', { taken: device.slots.taken, limit: device.slots.limit })}
            </span>
          )
          : (
            <span className={styles.fact} title={t('workers.device.slotsLimitedBy.title')}>
              {t('workers.device.slotsLimitedBy', {
                taken: device.slots.taken,
                limit: device.slots.limit,
                peer: peerName(limitedBy),
              })}
            </span>
          )
      )}
      {hasMemory && (
        <span className={styles.fact} title={memory.source}>
          {t('workers.device.memory', {
            free: formatBytes(memory.free_bytes as number),
            total: formatBytes(memory.total_bytes as number),
          })}
        </span>
      )}
      {shareable && (
        // Never disabled: a disabled control drops the focus to the page,
        // where the dialog's Escape and Tab trap cannot hear it.
        <select
          className={styles.policy}
          value={device.policy}
          aria-label={t('workers.policy.aria', { device: device.value })}
          aria-disabled={busy}
          title={t('workers.policy.title')}
          onChange={(event) => {
            if (!busy) void setPolicy(workerId, device.value, event.target.value as DevicePolicy);
          }}
        >
          <option value="exclusive">{t('workers.policy.exclusive')}</option>
          <option value="shared">{t('workers.policy.shared')}</option>
        </select>
      )}
      {peers.length > 0 && (
        <span
          className={styles.sameCard}
          title={t(machineWide ? 'workers.device.sameMachine.title' : 'workers.device.sameCard.title')}
        >
          {t(machineWide ? 'workers.device.sameMachine' : 'workers.device.sameCard', {
            peers: list(peers.map(peerName)),
          })}
        </span>
      )}
      {otherPolicy.length > 0 && (
        <span className={`${styles.warn} ${styles.admission}`}>
          {t(device.policy === 'shared' ? 'workers.device.policyDiffers' : 'workers.device.policyDiffersHere', {
            peers: list(otherPolicy.map(peerName)),
          })}
        </span>
      )}
      {device.running && <RunLinks runIds={device.running} />}
      {admission !== null && (
        admission.by_memory
          ? <AdmissionBar admission={admission} />
          : otherPolicy.length === 0 && (
            <span className={`${styles.warn} ${styles.admission}`}>
              {!enforces || peersWithout.length === 0
                ? t('workers.admission.notEnforced')
                : t('workers.admission.peerNotEnforced', { peers: list(peersWithout.map(peerName)) })}
            </span>
          )
      )}
    </li>
  );
}

function VersionLine({ version, serverVersion }: { version: string; serverVersion: string | null }) {
  const { t } = useI18n();
  let check = null;
  if (serverVersion !== null) {
    check = version === serverVersion
      ? <span className={styles.ok}>{t('workers.version.matches')}</span>
      : <span className={styles.warn}>{t('workers.version.differs', { server: serverVersion })}</span>;
  }
  return (
    <span>
      {version}
      {check && <> - {check}</>}
    </span>
  );
}

/**
 * How the worker's node types compare with this server's (#781). A
 * mismatch lists the types on each side: a run naming the worker may use
 * only the types it has, and `any` never picks it.
 */
function CatalogLine({ catalog }: { catalog: WorkerNodeCatalog | undefined }) {
  const { t } = useI18n();
  if (catalog === undefined || catalog.status === 'unknown') {
    return <span className={styles.warn}>{t('workers.catalog.unknown')}</span>;
  }
  if (catalog.status === 'match') {
    return <span className={styles.ok}>{t('workers.catalog.matches')}</span>;
  }
  return (
    <span className={styles.catalog}>
      <span className={styles.warn}>{t('workers.catalog.differs')}</span>
      {catalog.missing.length > 0 && (
        <span className={styles.typeList}>
          {t('workers.catalog.missing', { count: catalog.missing.length })}{' '}
          <code>{catalog.missing.join(', ')}</code>
        </span>
      )}
      {catalog.extra.length > 0 && (
        <span className={styles.typeList}>
          {t('workers.catalog.extra', { count: catalog.extra.length })}{' '}
          <code>{catalog.extra.join(', ')}</code>
        </span>
      )}
    </span>
  );
}

/** A refused handshake: why, when, and from where (#781). */
function RejectionLine({ rejection, now }: { rejection: WorkerRejection; now: number }) {
  const { t, locale } = useI18n();
  const when = relativeTime(rejection.at, locale, now) || '-';
  return (
    <span className={styles.warn}>
      {rejection.reason}
      <span className={styles.muted}>
        {' '}({rejection.peer ? t('workers.rejected.from', { when, peer: rejection.peer }) : when})
      </span>
    </span>
  );
}

/** The tunnel states this build has a word for; another is shown as sent. */
const TUNNEL_STATES: ReadonlySet<string> = new Set<TunnelState>(['connecting', 'up', 'failed', 'stopped']);

/**
 * An SSH worker's tunnel (#784): its state, where it goes, Enable /
 * Disable, and while it is not up, the last lines ssh or the remote agent
 * wrote (the reason it failed: a refused key, an unknown host, no cdui).
 */
function TunnelLine({ entry }: { entry: SshWorker }) {
  const { t } = useI18n();
  const busy = useWorkerStore((s) => Boolean(s.busy[entry.id]));
  const setSshEnabled = useWorkerStore((s) => s.setSshEnabled);
  const where = entry.port === 22 ? entry.target : `${entry.target}:${entry.port}`;
  const showError = entry.state !== 'up' && entry.last_error;
  const known = TUNNEL_STATES.has(entry.state);
  return (
    <span className={styles.tunnel}>
      <span className={`${styles.chip} ${known ? styles[`chip_${entry.state}`] : styles.chip_neutral}`}>
        {known ? t(`workers.ssh.state.${entry.state}`) : entry.state}
      </span>
      <code title={entry.identity ?? undefined}>{where}</code>
      {entry.remote_port !== null && (
        <span className={styles.fact}>{t('workers.ssh.remotePort', { port: entry.remote_port })}</span>
      )}
      {entry.remote_pid != null && (
        <span className={styles.fact}>
          {t(entry.reattached ? 'workers.ssh.remotePidReattached' : 'workers.ssh.remotePid', { pid: entry.remote_pid })}
        </span>
      )}
      <button
        type="button"
        className={styles.btn}
        aria-disabled={busy}
        title={t(entry.enabled ? 'workers.ssh.disable.title' : 'workers.ssh.enable.title')}
        onClick={() => {
          if (!busy) void setSshEnabled(entry.id, !entry.enabled);
        }}
      >
        {t(entry.enabled ? 'workers.ssh.disable' : 'workers.ssh.enable')}
      </button>
      {entry.enabled && entry.gave_up && (
        <button
          type="button"
          className={styles.btn}
          aria-disabled={busy}
          title={t('workers.ssh.retry.title')}
          onClick={() => {
            if (!busy) void setSshEnabled(entry.id, true);
          }}
        >
          {t('workers.ssh.retry')}
        </button>
      )}
      {showError && (
        <pre className={styles.tunnelError} role="status">
          {entry.last_error}
        </pre>
      )}
      {entry.tail.length > 0 && (
        <details className={styles.tunnelTail}>
          <summary>{t('workers.ssh.output')}</summary>
          <pre>{entry.tail.join('\n')}</pre>
        </details>
      )}
    </span>
  );
}

/** An SSH worker whose agent has not connected yet (no worker entry). */
function SshOnlyCard({ entry }: { entry: SshWorker }) {
  const { t } = useI18n();
  const busy = useWorkerStore((s) => Boolean(s.busy[entry.id]));
  const deleteSsh = useWorkerStore((s) => s.deleteSsh);
  return (
    <article className={styles.card} aria-label={entry.id}>
      <header className={styles.cardHeader}>
        <h3 className={styles.workerName}>{entry.id}</h3>
        <span className={`${styles.chip} ${styles.chip_offline}`}>{t('workers.ssh.notYet')}</span>
        <span className={styles.spacer} />
        <button
          type="button"
          className={`${styles.btn} ${styles.danger}`}
          aria-disabled={busy}
          title={t('workers.ssh.delete.title')}
          onClick={() => {
            if (!busy) void deleteSsh(entry.id);
          }}
        >
          {t('workers.ssh.delete')}
        </button>
      </header>
      <dl className={styles.facts}>
        <dt>{t('workers.ssh.tunnel')}</dt>
        <dd><TunnelLine entry={entry} /></dd>
      </dl>
    </article>
  );
}

/** Statuses whose worker has no live connection: removing one ends nothing. */
const AWAY_STATUSES: ReadonlySet<string> = new Set<WorkerStatus>(['disconnected', 'offline']);

/**
 * A one-shot worker's run (#790): the run it took, else what it waits for,
 * or that it left without one.
 */
function OneShotLine({ oneShot, away }: { oneShot: WorkerOneShot; away: boolean }) {
  const { t } = useI18n();
  if (oneShot.took !== null) return <RunLinks runIds={[oneShot.took]} />;
  if (away) return <span className={styles.muted}>{t('workers.oneShot.none')}</span>;
  if (oneShot.run_id === null) return <span className={styles.muted}>{t('workers.oneShot.waiting')}</span>;
  return (
    <span className={styles.muted} title={oneShot.run_id}>
      {t('workers.oneShot.waitingFor', { id: oneShot.run_id.slice(0, RUN_ID_SHORT_LENGTH) })}
    </span>
  );
}

/** Instances of a shared credential (#845) with an agent connected first. */
function byActivity(instances: readonly WorkerInfo[]): WorkerInfo[] {
  return [...instances].sort((a, b) => Number(AWAY_STATUSES.has(a.status)) - Number(AWAY_STATUSES.has(b.status)));
}

/** An instance that left (#845): no agent connected, nothing in flight. */
function hasLeft(worker: WorkerInfo): boolean {
  return AWAY_STATUSES.has(worker.status) && worker.running.length === 0;
}

/**
 * One line for an instance that left (#845): which one, how its run went,
 * when it was last heard from. A SLURM array leaves many; a full card
 * each would bury the live ones.
 */
function LeftInstanceRow({ worker, now, onRemoved }: {
  worker: WorkerInfo;
  now: number;
  onRemoved: () => void;
}) {
  const { t, locale } = useI18n();
  const busy = useWorkerStore((s) => Boolean(s.busy[worker.id]));
  const remove = useWorkerStore((s) => s.remove);
  return (
    <li className={styles.leftInstance} aria-label={worker.name}>
      <span className={styles.workerName} title={worker.id}>{worker.instance ?? worker.id}</span>
      <span className={`${styles.chip} ${styles.chip_offline}`}>{t('workers.instance.left')}</span>
      {worker.one_shot && <OneShotLine oneShot={worker.one_shot} away />}
      <span className={styles.muted} title={t('workers.heartbeat')}>
        {worker.last_heartbeat_at === null
          ? t('workers.heartbeat.never')
          : relativeTime(worker.last_heartbeat_at, locale, now) || '-'}
      </span>
      <span className={styles.spacer} />
      <button
        type="button"
        className={`${styles.btn} ${styles.danger}`}
        aria-disabled={busy}
        title={t('workers.instance.remove.title')}
        onClick={() => {
          if (busy) return;
          void remove(worker.id).then(() => {
            if (document.activeElement === document.body) onRemoved();
          });
        }}
      >
        {t('workers.remove')}
      </button>
    </li>
  );
}

/**
 * The instances of one credential (#845): how many, how many are connected
 * and running, then a card for each. Open by default; a large array can
 * be folded away.
 */
function InstanceList({ instances, children }: { instances: readonly WorkerInfo[]; children: ReactNode }) {
  const { t } = useI18n();
  const connected = instances.filter((w) => !AWAY_STATUSES.has(w.status)).length;
  const running = instances.filter((w) => w.running.length > 0).length;
  return (
    <details className={styles.instances} open>
      <summary>{t('workers.instances.summary', { count: instances.length, connected, running })}</summary>
      <div className={styles.instanceList}>{children}</div>
    </details>
  );
}

/**
 * Instances whose credential has no card of its own (#845): same-host
 * agents on the session token, whose name nobody enrolled.
 */
function InstanceGroup({ name, instances, children }: {
  name: string;
  instances: readonly WorkerInfo[];
  children: ReactNode;
}) {
  const { t } = useI18n();
  return (
    <article className={styles.card} aria-label={name}>
      <header className={styles.cardHeader}>
        <h3 className={styles.workerName}>{name}</h3>
        <span className={`${styles.chip} ${styles.chip_neutral}`} title={t('workers.instances.title')}>
          {t('workers.instances')}
        </span>
      </header>
      <InstanceList instances={instances}>{children}</InstanceList>
    </article>
  );
}

function WorkerCard({ worker, rejection, ssh, serverVersion, now, onRemoved, onRekey, instances = [], children }: {
  worker: WorkerInfo;
  /**
   * The instances of this worker's credential (#845), shown in the card
   * as *children*; empty for a worker no instance shares.
   */
  instances?: readonly WorkerInfo[];
  children?: ReactNode;
  /** Its name's last refused handshake, if the server remembers one (#781). */
  rejection: WorkerRejection | undefined;
  /** Its SSH tunnel, when this server starts it over ssh (#784). */
  ssh?: SshWorker;
  serverVersion: string | null;
  now: number;
  /** Called after this card's worker was removed, to keep the focus in the dialog. */
  onRemoved: () => void;
  /** Open the Add worker dialog for a token bound to this worker (#780). */
  onRekey: (workerId: string) => void;
}) {
  const { t, locale } = useI18n();
  const busy = useWorkerStore((s) => Boolean(s.busy[worker.id]));
  const setDraining = useWorkerStore((s) => s.setDraining);
  const remove = useWorkerStore((s) => s.remove);
  const [confirming, setConfirming] = useState(false);
  const actionRef = useRef<HTMLButtonElement>(null);
  const mounts = Object.entries(worker.mounts);
  // Without per-device run ids from the server, the worker's runs are
  // listed once, under the worker. With them, a run no device lists is
  // still listed there: it is never hidden.
  const perDeviceRuns = worker.devices.some((d) => Array.isArray(d.running));
  const onDevices = new Set(worker.devices.flatMap((d) => d.running ?? []));
  const workerRuns = worker.running.filter((id) => !onDevices.has(id));
  const digest = worker.fingerprint_digest;
  const known = KNOWN_STATUSES.has(worker.status);
  /** One of a shared credential's instances (#845), listed in its card. */
  const instance = worker.instance_of ? worker.instance ?? worker.id : null;
  /**
   * The credential of instances whose own agent is not connected (#845):
   * the card is their header, and its own last facts would only mislead.
   */
  const shared = instances.length > 0 && AWAY_STATUSES.has(worker.status);
  const anyConnected = !AWAY_STATUSES.has(worker.status)
    || instances.some((w) => !AWAY_STATUSES.has(w.status));
  // A credential with instances says what revoking it does, asked or not.
  let removeTitle: TranslationKey = 'workers.remove.title';
  if (instances.length > 0) removeTitle = 'workers.instances.remove.title';
  else if (confirming) removeTitle = 'workers.remove.confirmTitle';

  // One button that changes its word, never disabled: a disabled or swapped
  // button drops the focus to the page, where the dialog's Escape and Tab
  // trap cannot hear it. A press while one is in flight does nothing.
  const toggleDrain = async () => {
    if (busy) return;
    await setDraining(worker.id, !worker.draining);
    const button = actionRef.current;
    if (button !== null && document.activeElement === document.body) button.focus();
  };

  // Removing a connected worker disconnects it and interrupts its runs, so
  // that takes a second press; one that is away goes at once.
  const removeWorker = async () => {
    if (busy) return;
    if (anyConnected && !confirming) {
      setConfirming(true);
      return;
    }
    await remove(worker.id);
    if (document.activeElement === document.body) onRemoved();
  };

  return (
    <article className={instance === null ? styles.card : `${styles.card} ${styles.nested}`} aria-label={worker.name}>
      <header className={styles.cardHeader}>
        {instance === null
          ? <h3 className={styles.workerName}>{worker.name}</h3>
          : <h4 className={styles.workerName} title={worker.id}>{instance}</h4>}
        {shared ? (
          <span className={`${styles.chip} ${styles.chip_neutral}`} title={t('workers.instances.title')}>
            {t('workers.instances.credential')}
          </span>
        ) : (
          <span className={`${styles.chip} ${known ? styles[`chip_${worker.status}`] : styles.chip_neutral}`}>
            {known ? t(`workers.status.${worker.status}`) : worker.status}
          </span>
        )}
        {worker.plain_http && (
          <span className={`${styles.chip} ${styles.chip_degraded}`} title={t('workers.plainHttp.title')}>
            {t('workers.plainHttp')}
          </span>
        )}
        {worker.one_shot && (
          <span className={`${styles.chip} ${styles.chip_neutral}`} title={t('workers.oneShot.title')}>
            {t('workers.oneShot')}
          </span>
        )}
        <span className={styles.spacer} />
        {/* A one-shot worker takes one run and leaves: nothing to drain (#790). */}
        {!worker.one_shot && (
          <button
            ref={actionRef}
            type="button"
            className={styles.btn}
            aria-disabled={busy}
            title={t(worker.draining ? 'workers.undrain.title' : 'workers.drain.title')}
            onClick={() => void toggleDrain()}
          >
            {t(worker.draining ? 'workers.undrain' : 'workers.drain')}
          </button>
        )}
        {/* An instance has no credential of its own to re-key (#845). */}
        {instance === null && (
          <button
            type="button"
            className={styles.btn}
            title={t('workers.rekey.title')}
            onClick={() => onRekey(worker.id)}
          >
            {t('workers.rekey')}
          </button>
        )}
        {/*
          A live instance (#845) is not removed on its own: its agent would
          only reconnect. It goes once it left (its line), or with its
          credential.
        */}
        {instance === null && (
          <button
            type="button"
            className={`${styles.btn} ${styles.danger}`}
            aria-disabled={busy}
            title={t(removeTitle)}
            onClick={() => void removeWorker()}
            onBlur={() => setConfirming(false)}
          >
            {t(confirming ? 'workers.remove.confirm' : 'workers.remove')}
          </button>
        )}
      </header>
      {shared ? rejection && (
        <dl className={styles.facts}>
          <dt>{t('workers.rejected')}</dt>
          <dd><RejectionLine rejection={rejection} now={now} /></dd>
        </dl>
      ) : (
      <dl className={styles.facts}>
        {ssh && (
          <>
            <dt>{t('workers.ssh.tunnel')}</dt>
            <dd><TunnelLine entry={ssh} /></dd>
          </>
        )}
        <dt>{t('workers.version')}</dt>
        <dd><VersionLine version={worker.version} serverVersion={serverVersion} /></dd>
        <dt>{t('workers.fingerprint')}</dt>
        <dd>
          {digest
            ? <code title={digest}>{digest.slice(0, FINGERPRINT_SHORT_LENGTH)}</code>
            : <span className={styles.muted}>{t('workers.fingerprint.none')}</span>}
        </dd>
        <dt>{t('workers.catalog')}</dt>
        <dd><CatalogLine catalog={worker.node_catalog} /></dd>
        {rejection && (
          <>
            <dt>{t('workers.rejected')}</dt>
            <dd><RejectionLine rejection={rejection} now={now} /></dd>
          </>
        )}
        {worker.one_shot && (
          <>
            <dt>{t('workers.oneShot.run')}</dt>
            <dd><OneShotLine oneShot={worker.one_shot} away={AWAY_STATUSES.has(worker.status)} /></dd>
          </>
        )}
        <dt>{t('workers.mounts')}</dt>
        <dd>
          {mounts.length === 0
            ? <span className={styles.muted}>{t('workers.mounts.none')}</span>
            : mounts.map(([label, path]) => (
                <span key={label} className={styles.mount} title={path}>{label}</span>
              ))}
        </dd>
        <dt>{t('workers.heartbeat')}</dt>
        <dd>
          {worker.last_heartbeat_at === null
            ? <span className={styles.muted}>{t('workers.heartbeat.never')}</span>
            : relativeTime(worker.last_heartbeat_at, locale, now) || '-'}
        </dd>
        {(!perDeviceRuns || workerRuns.length > 0) && (
          <>
            <dt>{t('workers.running')}</dt>
            <dd><RunLinks runIds={workerRuns} /></dd>
          </>
        )}
      </dl>
      )}
      {!shared && (
        <ul className={styles.devices}>
          {worker.devices.map((device) => (
            <DeviceRow key={device.value} device={device} workerId={worker.id} enforces={worker.memory_budget !== false} />
          ))}
        </ul>
      )}
      {instances.length > 0 && <InstanceList instances={instances}>{children}</InstanceList>}
    </article>
  );
}

/**
 * The Workers panel (#776): every worker agent the server knows, polled
 * every 5 s while open, with Drain / Undrain and the Add worker dialog.
 * Opened from Settings -> Execution and from the Runs panel header; mounted
 * once in `App`.
 */
/** The Add worker dialog (#776), loaded the first time it opens. */
const AddWorkerDialog = lazyRetry(
  () => import('./AddWorkerDialog').then((module) => module.AddWorkerDialog),
  (props) => props.onClose(),
);

export function WorkersPanel() {
  const open = useWorkerStore((s) => s.panelOpen);
  if (!open) return null;
  return <WorkersDialog />;
}

function WorkersDialog() {
  const { t } = useI18n();
  const titleId = useId();
  const workers = useWorkerStore((s) => s.workers);
  const rejections = useWorkerStore((s) => s.rejections);
  const ssh = useWorkerStore((s) => s.ssh);
  const sshError = useWorkerStore((s) => s.sshError);
  const loaded = useWorkerStore((s) => s.loaded);
  const error = useWorkerStore((s) => s.error);
  const serverVersion = useWorkerStore((s) => s.serverVersion);
  const closePanel = useWorkerStore((s) => s.closePanel);
  const refresh = useWorkerStore((s) => s.refresh);
  /** The Add worker dialog: closed (null), open, or open to re-key a worker. */
  const [adding, setAdding] = useState<{ rekey: string | null } | null>(null);
  const dialogRef = useRef<HTMLDivElement>(null);
  const closeRef = useRef<HTMLButtonElement>(null);

  useEffect(() => useWorkerStore.getState().watch(), []);

  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null;
    closeRef.current?.focus();
    return () => {
      if (previous?.isConnected) previous.focus();
    };
  }, []);

  const handleKeys = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key === 'Escape') {
      event.preventDefault();
      closePanel();
      return;
    }
    trapTab(event, dialogRef.current);
  };

  const now = Date.now();
  // A refusal of another agent's attempt to take a connected worker's name
  // is not that worker's own: it is listed with the refused attempts.
  const own = rejections.filter((r) => !r.live_owner);
  const rejectionOf = new Map(own.map((r) => [r.name, r]));
  // A refused name with no card is an agent that never appeared: listed on
  // its own, so the user sees why (#781).
  const names = new Set(workers.map((w) => w.name));
  const refusedOnly = rejections.filter((r) => r.live_owner || !names.has(r.name));
  const sshById = new Map(ssh.map((entry) => [entry.id, entry]));
  const sshOnly = ssh.filter((entry) => !workers.some((worker) => worker.id === entry.id));
  let content;
  if (!loaded) {
    content = <p className={styles.muted}>{t('workers.loading')}</p>;
  } else if (workers.length === 0 && refusedOnly.length === 0 && sshOnly.length === 0 && error === null) {
    content = <p className={styles.muted}>{t('workers.empty')}</p>;
  } else {
    // A shared credential's instances (#845) are listed in its card, or
    // under a header of their own when the credential has none.
    const instancesOf = new Map<string, WorkerInfo[]>();
    for (const worker of workers) {
      if (!worker.instance_of) continue;
      const group = instancesOf.get(worker.instance_of);
      if (group) group.push(worker);
      else instancesOf.set(worker.instance_of, [worker]);
    }
    const card = (worker: WorkerInfo, children?: ReactNode) => (
      <WorkerCard
        key={worker.id}
        worker={worker}
        rejection={rejectionOf.get(worker.name)}
        ssh={sshById.get(worker.id)}
        serverVersion={serverVersion}
        now={now}
        onRemoved={() => closeRef.current?.focus()}
        onRekey={(workerId) => setAdding({ rekey: workerId })}
        instances={worker.instance_of ? [] : instancesOf.get(worker.id)}
      >
        {children}
      </WorkerCard>
    );
    // A card for each live instance, one line for each that left.
    const instanceCards = (group: readonly WorkerInfo[] | undefined) => {
      if (!group) return undefined;
      const left = group.filter(hasLeft);
      return (
        <>
          {byActivity(group.filter((w) => !hasLeft(w))).map((w) => card(w))}
          {left.length > 0 && (
            <ul className={styles.leftInstances} aria-label={t('workers.instances.left', { count: left.length })}>
              {left.map((w) => (
                <LeftInstanceRow key={w.id} worker={w} now={now} onRemoved={() => closeRef.current?.focus()} />
              ))}
            </ul>
          )}
        </>
      );
    };
    const tops = workers.filter((w) => !w.instance_of);
    const topIds = new Set(tops.map((w) => w.id));
    content = (
      <>
        {tops.map((worker) => card(worker, instanceCards(instancesOf.get(worker.id))))}
        {[...instancesOf.entries()].filter(([name]) => !topIds.has(name)).map(([name, group]) => (
          <InstanceGroup key={`instances:${name}`} name={name} instances={group}>
            {instanceCards(group)}
          </InstanceGroup>
        ))}
        {sshOnly.map((entry) => <SshOnlyCard key={`ssh:${entry.id}`} entry={entry} />)}
      </>
    );
  }

  return createPortal(
    <div
      className={styles.backdrop}
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) closePanel();
      }}
    >
      <div
        ref={dialogRef}
        className={styles.dialog}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        tabIndex={-1}
        onKeyDown={handleKeys}
      >
        <header className={styles.header}>
          <h2 id={titleId}>{t('workers.title')}</h2>
          <button type="button" className={styles.btn} onClick={() => void refresh()}>
            {t('workers.refresh')}
          </button>
          <button
            type="button"
            className={`${styles.btn} ${styles.accent}`}
            onClick={() => setAdding({ rekey: null })}
          >
            {t('workers.add')}
          </button>
          <button
            ref={closeRef}
            type="button"
            className={styles.close}
            onClick={closePanel}
            aria-label={t('workers.close')}
          >
            &#215;
          </button>
        </header>
        <div className={styles.body}>
          {error !== null && <div className={styles.error} role="alert">{friendlyError(error)}</div>}
          {sshError !== null && (
            <div className={styles.error} role="alert">
              {t('workers.ssh.readFailed', { error: friendlyError(sshError) })}
            </div>
          )}
          {content}
          {loaded && refusedOnly.length > 0 && (
            <section className={styles.rejections} aria-label={t('workers.rejections.title')}>
              <h3 className={styles.workerName}>{t('workers.rejections.title')}</h3>
              <p className={styles.muted}>{t('workers.rejections.intro')}</p>
              <dl className={styles.facts}>
                {refusedOnly.map((rejection) => (
                  <div key={rejection.name} className={styles.rejection}>
                    <dt>{rejection.name}</dt>
                    <dd><RejectionLine rejection={rejection} now={now} /></dd>
                  </div>
                ))}
              </dl>
            </section>
          )}
        </div>
      </div>
      {adding !== null && (
        <Suspense fallback={null}>
          <AddWorkerDialog
            mode="remote"
            rekey={adding.rekey}
            onClose={() => setAdding(null)}
            onAdded={() => void refresh()}
          />
        </Suspense>
      )}
    </div>,
    document.body,
  );
}
