import { useEffect, useId, useRef, useState, type ReactNode } from 'react';
import type { InspectionState } from '../../store/pluginStore';
import { useI18n, type TranslationKey } from '../../i18n';
import {
  capabilityKey,
  depSpec,
  httpUrl,
  manifestAuthor,
  provenancePin,
} from './pluginStatus';
import packStyles from '../PackCenter/PackCenterModal.module.css';
import styles from './PluginCenterModal.module.css';

/**
 * The consent screen: what was found at a source, and what installing it
 * would cost.
 *
 * An inspection is not an install. The server has read a manifest at one
 * exact commit and is holding it under an `inspection_id`; this card is what
 * turns that into a yes about THAT version rather than about whatever the
 * branch holds a minute later. Nothing is sent until every box the manifest
 * asks for is ticked.
 *
 * Rendered only while the review is `ready`. A built-in pack that asks for
 * nothing never gets here at all -- the store installs it straight from the
 * inspection, because a dialog whose only answer is yes is not a question.
 */

/** The review, once the server has answered. */
type ReadyInspection = Extract<InspectionState, { phase: 'ready' }>;

interface Fact {
  key: string;
  node: ReactNode;
}

export interface PluginReviewCardProps {
  inspection: ReadyInspection;
  /**
   * The node types the catalog says are registered under this plugin id.
   *
   * An inspection cannot say what a source WOULD register: naming a plugin's
   * nodes means importing it, and nothing on this path runs a line of the
   * code under review. So this is what is here now, from a different source
   * of truth -- and the card prints it only for a fresh install, where the
   * two cannot disagree. On an update, "what is registered today" is the
   * version being REPLACED, and this is the one screen whose whole job is
   * saying what is about to arrive: a plugin that adds a node in the version
   * under review would have its old set stated as fact.
   */
  nodes: string[];
  /** This plugin has a request in flight. */
  busy: boolean;
  /** False when the server refuses installs from this browser (remote). */
  canInstall: boolean;
  onInstall: (opts: {
    acceptCapabilities: boolean;
    trustAuthor: boolean;
    force?: boolean;
  }) => void;
  onCancel: () => void;
}

export function PluginReviewCard({
  inspection, nodes, busy, canInstall, onInstall, onCancel,
}: PluginReviewCardProps) {
  const { t } = useI18n();
  const { data, error } = inspection;

  const [granted, setGranted] = useState(false);
  const [trusted, setTrusted] = useState(false);
  const titleId = useId();
  const cardRef = useRef<HTMLElement | null>(null);

  // A review can be raised by the Install button on a row far down the list,
  // and this card renders at the TOP of it. Without this, clicking Install on
  // the eighth plugin looks like nothing happened at all.
  //
  // Once per review: the panel keys this component on the inspection id, so a
  // second Review remounts rather than re-runs.
  useEffect(() => {
    cardRef.current?.scrollIntoView({ block: 'nearest' });
  }, []);

  const needsGrant = data.capabilities.length > 0;
  const needsTrust = data.allowed_modules.length > 0;
  // Every question this manifest asks has an answer. The button is dead
  // until then, rather than sending an install the server would refuse.
  const answered = (!needsGrant || granted) && (!needsTrust || trusted);

  // A 409 the server sends as an OFFER: the plugin is already here, and the
  // review it refused is still spendable. The card keeps everything it was
  // showing and changes what the button does.
  const reinstall = error !== null && error.code === 'already_installed';
  const actionKey: TranslationKey = reinstall
    ? 'pluginCenter.reinstall'
    : inspection.kind === 'update'
      ? 'pluginCenter.update'
      : 'pluginCenter.install';

  const author = manifestAuthor(data.manifest);
  const homepage = httpUrl(data.homepage);
  const deps = Object.entries(data.python_deps).map(([name, spec]) => depSpec(name, spec));
  // The version is in the header, so a ref that only repeats it is dropped.
  const pin = provenancePin(data.ref, data.sha, data.version);

  const facts: Fact[] = [];
  if (author !== null) {
    facts.push({ key: 'author', node: t('pluginCenter.review.author', { author }) });
  }
  // Only for a fresh install: see the prop's docblock. On an update this
  // would state the outgoing version's nodes as a fact about the incoming one.
  if (inspection.kind === 'install' && nodes.length > 0) {
    facts.push({
      key: 'nodes', node: t('pluginCenter.review.nodes', { nodes: nodes.join(', ') }),
    });
  }
  if (deps.length > 0) {
    facts.push({ key: 'deps', node: t('packs.pip', { specs: deps.join(', ') }) });
  }
  if (pin !== null) facts.push({ key: 'pin', node: pin });
  if (homepage !== null) {
    facts.push({
      key: 'homepage',
      node: (
        <a className={packStyles.linkBtn} href={homepage} target="_blank" rel="noreferrer">
          {t('pluginCenter.homepage')}
        </a>
      ),
    });
  }

  return (
    <section
      ref={cardRef}
      // The accent ring the panel puts on a row somebody asked for: this is
      // the one card on screen that is waiting for an answer.
      className={`${packStyles.card} ${packStyles.cardHighlighted}`}
      // Labelled BY the eyebrow rather than with a copy of it: an `aria-label`
      // saying what the next line already says is the region announced twice.
      aria-labelledby={titleId}
      // Which plugin this review is about, for a caller that has to find it:
      // `forPluginId` never reaches this component, so on a card raised by a
      // row's button this is the only thing tying the two together.
      data-review-for={data.plugin_id}
    >
      <div id={titleId} className={packStyles.cardMeta}>
        {t('pluginCenter.review.title')}
      </div>

      <div className={packStyles.cardHeader}>
        <span className={packStyles.cardTitle}>{data.name || data.plugin_id}</span>
        {data.version !== '' && (
          <span className={packStyles.cardSize}>v{data.version}</span>
        )}
      </div>

      {data.description !== '' && (
        <p className={packStyles.cardDesc}>{data.description}</p>
      )}

      {facts.length > 0 && (
        <ul className={packStyles.facts}>
          {facts.map((fact) => <li key={fact.key}>{fact.node}</li>)}
        </ul>
      )}

      {needsGrant && (
        <>
          <div className={packStyles.note}>{t('pluginCenter.review.capabilities')}</div>
          <ul className={packStyles.facts}>
            {data.capabilities.map((id) => {
              // An id from a newer server has no line here, and a consent
              // screen must not silently drop what it is consenting to.
              const key = capabilityKey(id);
              return <li key={id}>{key === null ? id : t(key)}</li>;
            })}
          </ul>
          <div className={packStyles.caption}>{t('pluginCenter.review.capNote')}</div>
          <label className={styles.consentCheck}>
            <input
              type="checkbox"
              className={packStyles.itemCheck}
              checked={granted}
              disabled={busy}
              onChange={(event) => setGranted(event.target.checked)}
            />
            {t('pluginCenter.review.grant')}
          </label>
        </>
      )}

      {needsTrust && (
        <label className={styles.consentCheck}>
          <input
            type="checkbox"
            className={packStyles.itemCheck}
            checked={trusted}
            disabled={busy}
            onChange={(event) => setTrusted(event.target.checked)}
          />
          {t('pluginCenter.review.trust', { modules: data.allowed_modules.join(', ') })}
        </label>
      )}

      {/* Browser code is a different trust decision from anything the import
          gate covers: it runs in this editor, with everything the editor has. */}
      {data.has_frontend && (
        <div className={packStyles.resultBanner} data-tone="warning">
          {t('pluginCenter.review.frontend')}
        </div>
      )}

      <div className={packStyles.cardActions}>
        <button
          type="button"
          className={packStyles.primaryBtn}
          disabled={!answered || busy || !canInstall}
          title={canInstall ? undefined : t('packs.remoteDisabled')}
          onClick={() => onInstall({
            // Nothing declared is nothing to withhold, and the store's own
            // auto-install path sends the (empty) list too: this way every
            // install body is built the same way.
            acceptCapabilities: needsGrant ? granted : true,
            trustAuthor: trusted,
            ...(reinstall ? { force: true } : {}),
          })}
        >
          {t(actionKey)}
        </button>
        <button
          type="button"
          className={packStyles.secondaryBtn}
          disabled={busy}
          onClick={onCancel}
        >
          {t('dialog.cancel')}
        </button>
      </div>
    </section>
  );
}
