import { useCallback, useEffect, useId, useLayoutEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { useI18n } from '../../i18n';
import { useUIStore } from '../../store/uiStore';
import { availableSteps, guideTarget, type GuideStep } from './guideSteps';
import styles from './GuideTour.module.css';

/** Space between the framed element and the frame, and the frame and the card. */
const PAD = 6;
const GAP = 12;
const EDGE = 12;

interface Box {
  top: number;
  left: number;
  width: number;
  height: number;
}

function frameOf(el: HTMLElement): Box {
  const r = el.getBoundingClientRect();
  return { top: r.top - PAD, left: r.left - PAD, width: r.width + PAD * 2, height: r.height + PAD * 2 };
}

/**
 * Where the card goes: below the frame, else above, else beside it, kept
 * inside the window. Centred when there is no frame.
 */
function cardPosition(frame: Box | null, card: { width: number; height: number }) {
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  if (!frame) return { top: (vh - card.height) / 2, left: (vw - card.width) / 2 };
  const clampLeft = (x: number) => Math.max(EDGE, Math.min(x, vw - EDGE - card.width));
  const clampTop = (y: number) => Math.max(EDGE, Math.min(y, vh - EDGE - card.height));
  const centredX = clampLeft(frame.left + frame.width / 2 - card.width / 2);
  if (frame.top + frame.height + GAP + card.height <= vh - EDGE) {
    return { top: frame.top + frame.height + GAP, left: centredX };
  }
  if (frame.top - GAP - card.height >= EDGE) {
    return { top: frame.top - GAP - card.height, left: centredX };
  }
  const centredY = clampTop(frame.top + frame.height / 2 - card.height / 2);
  if (frame.left + frame.width + GAP + card.width <= vw - EDGE) {
    return { top: centredY, left: frame.left + frame.width + GAP };
  }
  return { top: centredY, left: clampLeft(frame.left - GAP - card.width) };
}

/**
 * The getting-started tour: one area of the screen at a time, framed, with a
 * short card saying what it is for, and Previous / Next to move between them.
 *
 * Keyboard: → or Enter for next, ← for previous, Escape to leave. The page
 * behind is dimmed and does not take clicks while the tour is open.
 */
export function GuideTour() {
  const open = useUIStore((s) => s.guideOpen);
  if (!open) return null;
  return <GuideTourBody />;
}

function GuideTourBody() {
  const { t } = useI18n();
  const close = useUIStore((s) => s.closeGuide);
  const titleId = useId();
  // Decided once, when the tour opens: what is on screen now. Never empty:
  // the welcome stop has no target.
  const [steps] = useState<GuideStep[]>(() => availableSteps());
  const [index, setIndex] = useState(0);
  const [frame, setFrame] = useState<Box | null>(null);
  const [cardPos, setCardPos] = useState<{ top: number; left: number } | null>(null);
  const cardRef = useRef<HTMLDivElement>(null);
  const nextRef = useRef<HTMLButtonElement>(null);
  const step = steps[index];
  const last = index === steps.length - 1;

  const layout = useCallback(() => {
    const el = guideTarget(step);
    const nextFrame = el ? frameOf(el) : null;
    setFrame(nextFrame);
    // Mounted: this runs from a layout effect and from listeners that the
    // body removes when it unmounts.
    const card = cardRef.current!.getBoundingClientRect();
    setCardPos(cardPosition(nextFrame, { width: card.width, height: card.height }));
  }, [step]);

  useLayoutEffect(() => {
    const el = guideTarget(step);
    el?.scrollIntoView?.({ block: 'nearest', inline: 'nearest' });
    layout();
  }, [layout, step]);

  useEffect(() => {
    nextRef.current?.focus();
  }, [index]);

  useEffect(() => {
    window.addEventListener('resize', layout);
    window.addEventListener('scroll', layout, true);
    return () => {
      window.removeEventListener('resize', layout);
      window.removeEventListener('scroll', layout, true);
    };
  }, [layout]);

  const next = useCallback(() => {
    if (last) close();
    else setIndex((i) => i + 1);
  }, [close, last]);
  const previous = useCallback(() => setIndex((i) => Math.max(0, i - 1)), []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        e.stopPropagation();
        close();
      } else if (e.key === 'ArrowRight') {
        e.preventDefault();
        e.stopPropagation();
        next();
      } else if (e.key === 'ArrowLeft') {
        e.preventDefault();
        e.stopPropagation();
        previous();
      }
    };
    window.addEventListener('keydown', onKey, true);
    return () => window.removeEventListener('keydown', onKey, true);
  }, [close, next, previous]);

  return createPortal(
    <div className={styles.root}>
      {/* Takes the clicks meant for the page behind. */}
      <div className={styles.blocker} />
      {frame ? (
        <div
          className={styles.frame}
          style={{ top: frame.top, left: frame.left, width: frame.width, height: frame.height }}
        />
      ) : (
        <div className={styles.scrim} />
      )}
      <div
        ref={cardRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        className={styles.card}
        style={cardPos ?? { visibility: 'hidden', top: 0, left: 0 }}
      >
        <div className={styles.head}>
          <span className={styles.count}>
            {t('guide.count', { n: index + 1, total: steps.length })}
          </span>
          <button
            type="button"
            className={styles.close}
            onClick={close}
            aria-label={t('guide.skip')}
            title={t('guide.skip')}
          >
            ×
          </button>
        </div>
        <h3 id={titleId} className={styles.title}>
          {t(step.title)}
        </h3>
        {t(step.body)
          .split('\n')
          .map((line, i) => (
            <p key={i} className={styles.body}>
              {line}
            </p>
          ))}
        <div className={styles.progress} aria-hidden="true">
          {steps.map((s, i) => (
            <span key={s.id} className={`${styles.dot} ${i === index ? styles.dotOn : ''}`} />
          ))}
        </div>
        <div className={styles.actions}>
          <button
            type="button"
            className={styles.secondary}
            onClick={previous}
            disabled={index === 0}
          >
            {t('guide.previous')}
          </button>
          <button type="button" ref={nextRef} className={styles.primary} onClick={next}>
            {last ? t('guide.done') : t('guide.next')}
          </button>
        </div>
      </div>
    </div>,
    document.body,
  );
}
