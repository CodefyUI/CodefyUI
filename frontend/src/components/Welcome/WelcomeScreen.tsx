import { useCallback } from 'react';
import { useI18n } from '../../i18n';
import { useTabStore } from '../../store/tabStore';
import { useUIStore } from '../../store/uiStore';
import { openExampleInNewTab } from '../../utils/openExample';
import type { ExampleSummary } from '../../api/rest';
import { ExampleBrowser } from '../shared/ExampleBrowser';
import styles from './WelcomeScreen.module.css';

/**
 * What the workspace shows when no tab is open at all.
 *
 * Reachable since the last tab became closable: `removeTab` now empties the
 * list rather than refusing, and `App` renders this instead of an editor with
 * nothing to edit. It is deliberately NOT the empty-canvas overlay in a
 * different frame -- that overlay floats over a real canvas with a palette
 * beside it and offers only "pick an example", which is the wrong offer when
 * the user has no graph to pick one INTO. Here the two ways in are stated
 * side by side: start blank on the left, start from something on the right.
 */
export function WelcomeScreen() {
  const { t } = useI18n();
  const addTab = useTabStore((s) => s.addTab);

  // The one difference from the empty-canvas overlay: there is no tab to open
  // an example into, so one is made first. `openExampleInNewTab` also holds
  // the tab back until the fetch succeeds, which matters more here -- a
  // failed load would otherwise leave the user staring at a blank canvas with
  // only a toast to say what happened to the screen they were just on.
  const handlePick = useCallback(
    (example: ExampleSummary) => void openExampleInNewTab(example.path),
    [],
  );

  return (
    <div className={styles.screen}>
      <div className={styles.welcome}>
        {/* The product name, not a translated string: it is the same word in
            every locale, and a key for it would only invite one of them to
            drift. */}
        <div className={styles.brand}>CodefyUI</div>
        <p className={styles.tagline}>{t('welcome.tagline')}</p>
        <div className={styles.actions}>
          <button
            type="button"
            className={styles.primaryButton}
            onClick={() => addTab()}
          >
            {t('welcome.newGraph')}
          </button>
          <button
            type="button"
            className={styles.secondaryButton}
            onClick={() => useUIStore.getState().openTemplateGallery()}
            title={t('gallery.open.title')}
          >
            {t('gallery.browse')}
          </button>
        </div>
      </div>

      {/* Scrolls on its own, so the welcome column stays put no matter how
          many packs are installed. */}
      <div className={styles.examples}>
        <h2 className={styles.examplesTitle}>{t('welcome.examples')}</h2>
        <ExampleBrowser onPick={handlePick} />
      </div>
    </div>
  );
}
