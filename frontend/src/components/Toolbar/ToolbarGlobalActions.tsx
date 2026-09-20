import { useState, useRef } from 'react';
import { useI18n, SUPPORTED_LOCALES } from '../../i18n';
import { useUIStore } from '../../store/uiStore';
import { SettingsPopover } from './SettingsPopover';
import { PluginToolbarButtons } from './PluginToolbarButtons';
import { FontSizeMenu } from './FontSizeMenu';
import styles from './Toolbar.module.css';

export interface ToolbarGlobalActionsProps {
  /**
   * Whether the plugin-registered buttons (#132) lead the group.
   *
   * True on the editor toolbar. False on the welcome bar, where there is no
   * active tab: a plugin's button is written against the graph in front of
   * the user, and the ones that read `getActiveTab()` would find nothing
   * there. A button that cannot work is worse than one that is not shown.
   */
  plugins?: boolean;
}

/**
 * Settings, Help, font size and language — the part of the toolbar that
 * belongs to the app rather than to a graph.
 *
 * Its own component because the welcome screen needs exactly this and nothing
 * else: the toolbar around it reads `activeTab` from the first line of its
 * body down, and there is no active tab to read while the welcome screen is
 * up. Splitting it is what lets a user with no tab open still change the
 * language, which is the setting a first-run user is most likely to want.
 */
export function ToolbarGlobalActions({ plugins = true }: ToolbarGlobalActionsProps) {
  const { t, locale, setLocale } = useI18n();
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [fontSizeMenuOpen, setFontSizeMenuOpen] = useState(false);
  const [langMenuOpen, setLangMenuOpen] = useState(false);
  const settingsTriggerRef = useRef<HTMLButtonElement>(null);
  const fontSizeTriggerRef = useRef<HTMLButtonElement>(null);
  const langTriggerRef = useRef<HTMLDivElement>(null);

  return (
    <div className={`${styles.cluster} ${styles.right}`}>
      {/* Plugin buttons (#132) lead the right-hand group so an installed
          plugin never pushes Settings or Help off the row. Renders nothing
          at all — no element, no gap — when no plugin registered one. */}
      {plugins && <PluginToolbarButtons />}

      {/* Settings ⚙ */}
      <div className={styles.menuWrapper}>
        <button type="button"
          ref={settingsTriggerRef}
          onClick={() => setSettingsOpen((v) => !v)}
          title={t('toolbar.settings.title')}
          className={`${styles.iconBtn} ${settingsOpen ? styles.active : ''}`}
          aria-label={t('toolbar.settings')}
          aria-expanded={settingsOpen}
        >
          ⚙
        </button>
        <SettingsPopover
          open={settingsOpen}
          onClose={() => setSettingsOpen(false)}
          triggerRef={settingsTriggerRef}
        />
      </div>

      {/* Help ? — opens shortcuts modal */}
      <button type="button"
        onClick={() => useUIStore.getState().toggleShortcutsModal()}
        className={styles.iconBtn}
        title={t('shortcuts.title')}
        aria-label={t('shortcuts.title')}
      >
        ?
      </button>

      {/* Font size Aa */}
      <div className={styles.menuWrapper}>
        <button type="button"
          ref={fontSizeTriggerRef}
          onClick={() => setFontSizeMenuOpen((v) => !v)}
          className={`${styles.dropdown} ${styles.dropdownNoCaret} ${fontSizeMenuOpen ? styles.open : ''}`}
          title={t('toolbar.fontSize.title')}
          aria-label={t('toolbar.fontSize.title')}
          aria-expanded={fontSizeMenuOpen}
        >
          Aa
        </button>
        <FontSizeMenu
          open={fontSizeMenuOpen}
          onClose={() => setFontSizeMenuOpen(false)}
          triggerRef={fontSizeTriggerRef}
        />
      </div>

      {/* Language */}
      <div ref={langTriggerRef} className={styles.menuWrapper}>
        <button type="button"
          onClick={() => setLangMenuOpen((v) => !v)}
          className={`${styles.dropdown} ${langMenuOpen ? styles.open : ''}`}
          aria-label={t('toolbar.language.aria')}
          aria-expanded={langMenuOpen}
        >
          {SUPPORTED_LOCALES.find((l) => l.code === locale)?.label ?? locale}
        </button>
        {langMenuOpen && (
          <>
            <div className={styles.overlay} onClick={() => setLangMenuOpen(false)} />
            <div className={`${styles.menuPanel} ${styles.menuPanelRight}`}>
              {SUPPORTED_LOCALES.map((l) => (
                <button type="button"
                  key={l.code}
                  onClick={() => { setLocale(l.code); setLangMenuOpen(false); }}
                  className={`${styles.langOption} ${l.code === locale ? styles.activeOption : ''}`}
                >
                  <span>{l.nativeName}</span>
                  {l.code === locale && <span className={styles.langOptionCheck}>✓</span>}
                </button>
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
