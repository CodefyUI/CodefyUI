import type { TranslationKey } from '../../i18n';

/**
 * The stops of the getting-started tour, in order.
 *
 * `target` names a `data-tour` attribute on the element to frame. A stop
 * whose element is not on screen is left out when the tour opens (the node
 * panel appears only with a node selected; the canvas only with a tab open),
 * so the tour never frames empty space. A stop with no target is a centred
 * card.
 */
export interface GuideStep {
  id: string;
  target?: string;
  title: TranslationKey;
  body: TranslationKey;
}

export const GUIDE_STEPS: readonly GuideStep[] = [
  { id: 'welcome', title: 'guide.welcome.title', body: 'guide.welcome.body' },
  { id: 'palette', target: 'palette', title: 'guide.palette.title', body: 'guide.palette.body' },
  { id: 'rail', target: 'rail', title: 'guide.rail.title', body: 'guide.rail.body' },
  { id: 'canvas', target: 'canvas', title: 'guide.canvas.title', body: 'guide.canvas.body' },
  { id: 'node', target: 'node-panel', title: 'guide.node.title', body: 'guide.node.body' },
  { id: 'run', target: 'run', title: 'guide.run.title', body: 'guide.run.body' },
  { id: 'device', target: 'device', title: 'guide.device.title', body: 'guide.device.body' },
  { id: 'templates', target: 'templates', title: 'guide.templates.title', body: 'guide.templates.body' },
  { id: 'results', target: 'results', title: 'guide.results.title', body: 'guide.results.body' },
  { id: 'settings', target: 'settings', title: 'guide.settings.title', body: 'guide.settings.body' },
  { id: 'help', target: 'help', title: 'guide.help.title', body: 'guide.help.body' },
];

/** The element a stop frames, or null when it is not on screen. */
export function guideTarget(step: GuideStep): HTMLElement | null {
  if (!step.target) return null;
  const el = document.querySelector<HTMLElement>(`[data-tour="${step.target}"]`);
  if (!el) return null;
  const rect = el.getBoundingClientRect();
  return rect.width > 0 && rect.height > 0 ? el : null;
}

/** The stops this screen can show: those without a target, and those whose target is there. */
export function availableSteps(steps: readonly GuideStep[] = GUIDE_STEPS): GuideStep[] {
  return steps.filter((step) => !step.target || guideTarget(step) !== null);
}
