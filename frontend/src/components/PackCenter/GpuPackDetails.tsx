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
   * The server's whole say in it, and deliberately not `launchMode` as well:
   * the server asks MORE than the launch mode does (its launcher still on
   * disk, its kill switch off) before it says yes, so a second guess here
   * could only ever disagree with the process that actually has to come back.
   * What the card does with a yes still depends on there being a build worth
   * installing — see `canSwitchInApp` and `alreadyRecommended` below.
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
 * So the card has three shapes. When this machine already runs the build it
 * should and has no other to switch to — an Apple Silicon Mac, whose
 * acceleration ships in the wheel that is already installed — there is
 * nothing to swap and the card says so in one sentence. When a swap is both
 * possible and worth making, the button starts it — with the command still
 * there underneath, folded, because a user who would rather watch it happen
 * in a terminal loses nothing by being offered both. Otherwise the command IS
 * the card: open, never behind a disclosure, under the note saying why the
 * app cannot do it — a note left off where that would be untrue, on a server
 * that could swap the wheel and a machine that has no reason to.
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
  // `variants` is what THIS platform can install, so an empty list means
  // there is no build to switch to at all — the answer on every Mac, where
  // acceleration ships in the default wheel. A null `gpu` is a server that
  // said nothing about the GPU, which is no offer either: there would be no
  // build to name in the request.
  const canSwitchInApp = restartAvailable && variants.length > 0;
  // This machine already runs what it should. `installed_variant` is null
  // when the wheel cannot be read, and two unknowns are not a match.
  const alreadyRecommended = !!gpu?.installed_variant
    && gpu.installed_variant === gpu.recommended_variant;
  // The one shape with nothing to offer AND nothing to apologise for: the
  // acceleration is in the wheel that is already here and no other build
  // exists for this machine. An empty offer alone is not that claim — an
  // Intel Mac, or an arm Mac too old for MPS, also has nothing to switch to
  // and has no acceleration installed, so it keeps the command below rather
  // than being told it is already optimal while its pill reads "Not
  // installed".
  const alreadyOptimal = alreadyRecommended && variants.length === 0;
  // The seed has to be a build the server will accept. Seeding from
  // `recommended_variant` put `mps` in the select on every Mac — an option
  // that did exist back when the server offered every variant it knows, so it
  // rendered as selected and looked right; what it produced was a 400, MPS
  // acceleration having no index to reinstall from. `variants` now lists only
  // what this platform can install, so the offer is the thing to seed from
  // and the recommendation is taken only when it is on it.
  const [variant, setVariant] = useState<string>(() =>
    gpu?.recommended_variant && variants.includes(gpu.recommended_variant)
      ? gpu.recommended_variant
      : variants[0] ?? '',
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
            rather than filled with a guess. Same for the recommendation.

            One line, not two: on the machine this was written for both read
            `cu128`, and "Installed build: cu128" over "Recommended build:
            cu128" is one fact wearing two rows. The recommendation earns its
            words only when it disagrees with what is here. */}
        {gpu?.installed_variant && (
          <li>
            {t('packs.gpu.installed', { variant: gpu.installed_variant })}
            {gpu.recommended_variant
              && gpu.recommended_variant !== gpu.installed_variant && (
              <>
                {' · '}
                {t('packs.gpu.recommended', { variant: gpu.recommended_variant })}
              </>
            )}
          </li>
        )}
        {!gpu?.installed_variant && gpu?.recommended_variant && (
          <li>{t('packs.gpu.recommended', { variant: gpu.recommended_variant })}</li>
        )}
      </ul>

      {alreadyOptimal ? (
        /* No picker, no button and no command: every one of them would be an
           offer to install something that is already here. The facts above
           still stand — "Detected GPU: Apple Silicon (MPS)" over "Installed
           build: mps" is the reason this sentence is true. */
        <div className={styles.note}>{t('packs.gpu.alreadyOptimal')}</div>
      ) : canSwitchInApp && !alreadyRecommended ? (
        <>
          {/* Pick and go, on one row. The select had a row of its own above a
              note above the button — three stacked rows for one decision. */}
          <div className={styles.cardActions}>
            {variants.length > 1 && (
              <select
                className={styles.select}
                aria-label={t('packs.gpu.variant')}
                title={t('packs.gpu.variant')}
                value={variant}
                onChange={(e) => setVariant(e.target.value)}
              >
                {variants.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </select>
            )}
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
          {/* Below the button, as its caption rather than above it as a note:
              that the server restarts is the button's own label and the
              confirm dialog's question, and this is the one thing neither of
              them says. */}
          <div className={styles.caption}>{t('packs.gpu.restartNote')}</div>
          {/* Underneath, not instead of: the button is the shorter path and
              the command is the same install by hand, so a terminal user
              keeps their way through — folded, because next to a button that
              works it is a choice rather than an instruction. No `noCommand`
              sentence in this branch: a card whose button works has nothing
              to apologise for. */}
          {command !== null && (
            <details className={styles.manual}>
              <summary>{t('packs.manualCommand')}</summary>
              <CommandBlock command={command} />
            </details>
          )}
        </>
      ) : (
        <>
          {/* Why there is no button — said only where it is true. A card
              reaches this shape for two different reasons: the app cannot do
              the swap here (no server restart, or no build this platform can
              load), which is what these sentences describe, or it can and the
              machine has no reason to, where "not available yet" would claim
              a limit the app does not have. The facts above are that card's
              own explanation: "Installed build: cu128" with no recommendation
              beside it is a machine already running the right wheel. */}
          {!canSwitchInApp && (
            <div className={styles.note}>
              {/* `dev` only. An `unknown` launch mode means no catalog has
                  answered yet (or a server too old to say), and telling a user
                  they started CodefyUI a particular way when we do not know it
                  is worse than the neutral sentence — which is true either
                  way: run the command with the server stopped. */}
              {launchMode === 'dev' ? t('packs.gpu.devMode') : t('packs.gpu.notYet')}
            </div>
          )}
          {/* The command stays even where nothing needs installing: a
              deliberate downgrade is nobody's default, and this line is the
              only way left to reach one. */}
          {command !== null && <CommandBlock command={command} />}
          {/* `noCommand` belongs to the card whose ONLY way through is a
              command it cannot print. A machine that needs no install has
              nothing missing, so the apology would be about a command it
              never had to run. */}
          {command === null && !canSwitchInApp && (
            <div className={styles.note}>{t('packs.gpu.noCommand')}</div>
          )}
        </>
      )}
    </>
  );
}
