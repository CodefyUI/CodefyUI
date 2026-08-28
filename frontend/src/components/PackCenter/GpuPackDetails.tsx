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
   * Whether the server can install this pack and restart itself.
   *
   * FALSE for the whole of this PR: the backend answers a restart-mode install
   * with 409 and the command to run, and PR 5 is what turns it on. Wiring the
   * button now — behind a flag, with tests on both sides of it — is what keeps
   * PR 5 to a one-line change instead of a redesign.
   */
  restartAvailable: boolean;
  onInstall: (variant: string) => void;
}

/**
 * The body of the GPU PyTorch card.
 *
 * Every other pack downloads files next to a running server. This one swaps
 * the torch wheel out from under the interpreter that is executing this
 * request, which no process can do to itself — so the card's real job is to
 * hand the user a command they can run, and to say WHY it cannot just do it.
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

  // A `cdui dev` server has no supervisor to relaunch it, so even once PR 5
  // lands there is nothing to restart — that mode always gets the command.
  const canRestartHere = restartAvailable && launchMode === 'start';

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

      {canRestartHere ? (
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
        </>
      ) : (
        <>
          <div className={styles.note}>
            {launchMode === 'start' ? t('packs.gpu.notYet') : t('packs.gpu.devMode')}
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
