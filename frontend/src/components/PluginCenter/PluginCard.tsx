import { Fragment, type ReactNode } from 'react';
import type { PluginCatalogEntry } from '../../api/rest';
import type { PluginJob } from '../../store/pluginStore';
import { useI18n } from '../../i18n';
import { Pill } from '../shared/Pill';
import { originLabel, statusKey, statusTone } from './pluginStatus';
import styles from '../PackCenter/PackCenterModal.module.css';

/**
 * What separates two facts on one meta line.
 *
 * The node docs' own separator (`NodeDetailModal/DocsTab.tsx`), so a meta line
 * reads the same wherever the editor prints one. Punctuation, not information:
 * it is `aria-hidden`, and each fact keeps its own element so a reader — and a
 * query — gets one phrase at a time rather than a run-on.
 */
const FACT_SEPARATOR = '  ·  ';

interface Fact {
  key: string;
  node: ReactNode;
}

/** One line of small facts, or nothing at all when there are none. */
function MetaLine({ facts }: { facts: Fact[] }) {
  if (facts.length === 0) return null;
  return (
    <div className={styles.cardMeta}>
      {facts.map((fact, index) => (
        <Fragment key={fact.key}>
          {index > 0 && <span aria-hidden="true">{FACT_SEPARATOR}</span>}
          {fact.node}
        </Fragment>
      ))}
    </div>
  );
}

/**
 * One `name>=version` the way the installer would write it.
 *
 * Mirrors `backend/app/core/plugins/deps.py: _build_dep_spec`: a constraint
 * that starts with an operator is used as written, a bare version is pinned.
 * What the card prints is then what `uv pip install` would be given, rather
 * than a prettier string that means something slightly different.
 */
function depSpec(name: string, constraint: string): string {
  if (constraint === '') return name;
  return /^[<>=~!]/.test(constraint) ? `${name}${constraint}` : `${name}==${constraint}`;
}

export interface PluginCardProps {
  entry: PluginCatalogEntry;
  /** The job in flight anywhere, or null. May be another plugin's. */
  job: PluginJob | null;
  /** This plugin has a request in flight — an enable, an uninstall, an install. */
  busy: boolean;
  highlighted: boolean;
  /** False when the server refuses installs from this browser (remote). */
  canInstall: boolean;
  onInstall: () => void;
  onUpdate: () => void;
  onUninstall: () => void;
  onSetEnabled: (enabled: boolean) => void;
}

/**
 * One plugin: what it is, where it came from, what it brings, and the buttons
 * that change that.
 *
 * A pure view — every button hands straight back to the store, which owns the
 * confirm dialog, the job and the three-step refresh that follows a change.
 * Nothing here is scratch state, which is why this card has none.
 */
export function PluginCard({
  entry,
  job,
  busy,
  highlighted,
  canInstall,
  onInstall,
  onUpdate,
  onUninstall,
  onSetEnabled,
}: PluginCardProps) {
  const { t } = useI18n();

  const running = job !== null && job.pluginId === entry.id && job.status === 'running';
  // A job the catalog has not caught up with still says so on the row: the
  // server calls any row carrying a job `installing`, and the store sets it
  // the moment a 202 lands, but a job adopted from another tab arrives first.
  const status = running ? 'installing' : entry.status;
  // Every button this row has, off for as long as something is happening to
  // it. One flag rather than a check per button: five ways to leave a row
  // half-live are five ways to send a second request into the first one.
  const locked = busy || running;

  // A linked folder is a directory on this machine that the CLI pointed at.
  // Nothing here installs, updates or removes one — `cdui plugin link` owns
  // it — so the row offers the one thing the server CAN do: switch it off.
  const local = entry.source_kind === 'local';
  // Said in `title` only. The footer already carries this sentence once; a
  // copy beside every button it disables would be the same fact three times
  // down one card.
  const remoteReason = canInstall ? undefined : t('packs.remoteDisabled');

  const showInstall =
    !local && (status === 'available' || status === 'removed' || status === 'missing_files');
  const showEnable = status === 'disabled';
  const showDisable = status === 'installed';
  // Only a plugin fetched from a repository has anything to update FROM: a
  // built-in ships with the server, and a linked folder is already whatever
  // the folder says.
  const showUpdate = status === 'installed' && entry.source_kind === 'github_url';
  // `missing_files` is in the lockfile with its directory gone, so removing
  // the record is a real answer to it. `removed` is the opposite — already a
  // tombstone, with nothing left to remove.
  const showUninstall =
    !local && (status === 'installed' || status === 'disabled' || status === 'missing_files');
  const hasActions = showInstall || showEnable || showDisable || showUpdate || showUninstall;

  const origin = originLabel(entry);
  const repo = entry.repo ?? '';
  const sha7 = entry.sha === null ? null : entry.sha.slice(0, 7);
  // `ref` is `''` for a default-branch install — an answer, not a miss — so a
  // row pinned to no branch says the commit alone rather than a bare `@`.
  const pin = sha7 === null ? null : entry.ref ? `${entry.ref} @ ${sha7}` : sha7;

  const provenance: Fact[] = [];
  if (origin !== null) provenance.push({ key: 'origin', node: <span>{t(origin)}</span> });
  if (repo !== '') {
    provenance.push({
      key: 'repo',
      node:
        entry.url === null || entry.url === '' ? (
          <span>{repo}</span>
        ) : (
          <a
            className={styles.linkBtn}
            href={entry.url}
            target="_blank"
            rel="noreferrer"
          >
            {repo}
          </a>
        ),
    });
  }
  if (pin !== null) provenance.push({ key: 'pin', node: <span>{pin}</span> });

  // Chapters when the row has them, lesson ids when it has only those: both
  // answer "which part of the book is this", and a plugin that declares the
  // finer list has still answered it. One line, under the one label.
  const taught = entry.chapters.length > 0 ? entry.chapters : entry.lessons;
  const deps = Object.entries(entry.python_deps).map(([name, spec]) => depSpec(name, spec));

  const contents: Fact[] = [];
  if (taught.length > 0) {
    contents.push({
      key: 'chapters',
      node: <span>{t('pluginCenter.chapters', { chapters: taught.join(', ') })}</span>,
    });
  }
  // The count, not the list: a pack of a dozen node types would be a paragraph
  // here, and the palette is where they are read one by one.
  if (entry.node_count > 0) {
    contents.push({
      key: 'nodes',
      node: <span>{t('empty.nodeCount', { count: entry.node_count })}</span>,
    });
  }
  if (deps.length > 0) {
    contents.push({
      key: 'deps',
      node: <span>{t('packs.pip', { specs: deps.join(', ') })}</span>,
    });
  }

  return (
    <div
      className={`${styles.card} ${highlighted ? styles.cardHighlighted : ''}`}
      data-status={status}
      data-plugin-id={entry.id}
    >
      <div className={styles.cardHeader}>
        <span className={styles.cardTitle}>{entry.name || entry.id}</span>
        <Pill tone={statusTone(status)} pulse={status === 'installing'}>
          {t(statusKey(status))}
        </Pill>
        {/* The right-hand metadata slot the pack card puts a download size in:
            same place, same monospace, and a version is the one number a
            plugin row has. */}
        {entry.version !== null && entry.version !== '' && (
          <span className={styles.cardSize}>v{entry.version}</span>
        )}
      </div>

      {entry.description !== '' && <p className={styles.cardDesc}>{entry.description}</p>}

      <MetaLine facts={provenance} />
      <MetaLine facts={contents} />

      {hasActions && (
        <div className={styles.cardActions}>
          {showInstall && (
            <button
              type="button"
              className={styles.primaryBtn}
              disabled={locked || !canInstall}
              title={remoteReason}
              onClick={onInstall}
            >
              {t('pluginCenter.install')}
            </button>
          )}
          {showEnable && (
            <button
              type="button"
              className={styles.primaryBtn}
              disabled={locked}
              onClick={() => onSetEnabled(true)}
            >
              {t('pluginCenter.enable')}
            </button>
          )}
          {showDisable && (
            <button
              type="button"
              className={styles.secondaryBtn}
              disabled={locked}
              onClick={() => onSetEnabled(false)}
            >
              {t('pluginCenter.disable')}
            </button>
          )}
          {showUpdate && (
            <button
              type="button"
              className={styles.secondaryBtn}
              disabled={locked || !canInstall}
              title={remoteReason}
              onClick={onUpdate}
            >
              {t('pluginCenter.update')}
            </button>
          )}
          {showUninstall && (
            <button
              type="button"
              className={styles.secondaryBtn}
              disabled={locked || !canInstall}
              title={remoteReason}
              onClick={onUninstall}
            >
              {t('pluginCenter.uninstall')}
            </button>
          )}
        </div>
      )}
    </div>
  );
}
