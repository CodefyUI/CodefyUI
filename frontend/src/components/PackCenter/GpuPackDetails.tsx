import { useCallback, useState } from 'react';
import type { LaunchMode, PackGpuInfo, PackSummary } from '../../api/rest';
import { useI18n } from '../../i18n';
import { confirm } from '../../utils/dialog';
import { CommandBlock } from './CommandBlock';
import styles from './PackCenterModal.module.css';

export interface GpuPackDetailsProps {
  pack: PackSummary;
  gpu: PackGpuInfo | null;
  launchMode: LaunchMode;
  /** False when the server refuses installs from this browser (remote). */
  canInstall: boolean;
  /** This pack already has an install request in flight. */
  busy: boolean;
  /**
   * Whether the server can install this pack and restart itself — its own
   * `restart_available`, straight off the catalog.
   *
   * The one condition, and deliberately not `launchMode` as well: the server
   * asks MORE than the launch mode does (its launcher still on disk, its kill
   * switch off) before it says yes, so a second guess here could only ever
   * disagree with the process that actually has to come back.
   */
  restartAvailable: boolean;
  onInstall: (variant: string) => void;
}

/**
 * The body of the GPU PyTorch card.
 *
 * Every other pack downloads files next to a running server. This one swaps
 * the torch wheel out from under the interpreter that is executing the
 * request, which no process can do to itself: a helper outside the server
 * does the swap while the server is down.
 *
 * So the card has two shapes. When the server says it can arrange that, the
 * button starts it — with the command still printed underneath, because a
 * user who would rather watch it happen in a terminal loses nothing by being
 * offered both. When it cannot, the command IS the card, and the note above
 * it says why.
 */
export function GpuPackDetails({
  pack,
  gpu,
  launchMode,
  canInstall,
  busy,
  restartAvailable,
  onInstall,
}: GpuPackDetailsProps) {
  const { t } = useI18n();
  const variants = gpu?.variants ?? [];
  const [variant, setVariant] = useState<string>(
    () => gpu?.recommended_variant ?? variants[0] ?? '',
  );

  // The pack's own command is the specific one (it names the variant the
  // server picked); the GPU-wide one is the generic fallback.
  const command = pack.install_command ?? gpu?.install_command ?? null;

  const install = useCallback(async () => {
    const ok = await confirm({
      title: t('packs.gpu.installRestart'),
      message: t('packs.gpu.restartConfirm', { variant }),
      confirmText: t('packs.gpu.installRestart'),
      variant: 'danger',
    });
    if (!ok) return;
    onInstall(variant);
  }, [onInstall, t, variant]);

  return (
    <>
      <ul className={styles.facts}>
        <li>
          {gpu?.detected_label
            ? t('packs.gpu.detected', { label: gpu.detected_label })
            : t('packs.gpu.none')}
        </li>
        {/* `installed_variant: null` means "cannot tell which wheel is here",
            which is not the same claim as "none" — so the line is omitted
            rather than filled with a guess. Same for the recommendation. */}
        {gpu?.installed_variant && (
          <li>{t('packs.gpu.installed', { variant: gpu.installed_variant })}</li>
        )}
        {gpu?.recommended_variant && (
          <li>{t('packs.gpu.recommended', { variant: gpu.recommended_variant })}</li>
        )}
      </ul>

      {variants.length > 1 && (
        <div className={styles.gpuVariant}>
          <select
            className={styles.select}
            aria-label={t('packs.gpu.variant')}
            value={variant}
            onChange={(e) => setVariant(e.target.value)}
          >
            {variants.map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>
        </div>
      )}

      {restartAvailable ? (
        <>
          <div className={styles.note}>{t('packs.gpu.restartNote')}</div>
          <div className={styles.cardActions}>
            <button
              type="button"
              className={styles.primaryBtn}
              disabled={!canInstall || busy}
              title={canInstall ? undefined : t('packs.remoteDisabled')}
              onClick={() => void install()}
            >
              {t('packs.gpu.installRestart')}
            </button>
          </div>
          {/* Underneath, not instead of: the button is the shorter path and
              the command is the same install by hand, so showing both costs
              a reader nothing and gives a terminal user their way through.
              No `noCommand` sentence in this branch — a card whose button
              works has nothing to apologise for. */}
          {command !== null && (
            <>
              <div className={styles.note}>{t('packs.manualCommand')}</div>
              <CommandBlock command={command} />
            </>
          )}
        </>
      ) : (
        <>
          <div className={styles.note}>
            {/* `dev` only. An `unknown` launch mode means no catalog has
                answered yet (or a server too old to say), and telling a user
                they started CodefyUI a particular way when we do not know it
                is worse than the neutral sentence — which is true either
                way: run the command with the server stopped. */}
            {launchMode === 'dev' ? t('packs.gpu.devMode') : t('packs.gpu.notYet')}
          </div>
          {command === null ? (
            <div className={styles.note}>{t('packs.gpu.noCommand')}</div>
          ) : (
            <CommandBlock command={command} />
          )}
        </>
      )}
    </>
  );
}
