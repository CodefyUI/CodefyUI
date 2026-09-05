import { create } from 'zustand';
import en from './locales/en';
import zhTW from './locales/zh-TW';
import zhTWNodes from './nodeLocales/zh-TW';
import type { TranslationKey } from './locales/en';
import type { NodeTranslations } from './nodeLocales/types';

export type Locale = 'en' | 'zh-TW';

const messages: Record<Locale, Record<TranslationKey, string>> = {
  en,
  'zh-TW': zhTW,
};

const nodeMessages: Partial<Record<Locale, NodeTranslations>> = {
  'zh-TW': zhTWNodes,
};

// All supported locales — add new ones here
export const SUPPORTED_LOCALES: { code: Locale; label: string; nativeName: string }[] = [
  { code: 'en', label: 'EN', nativeName: 'English' },
  { code: 'zh-TW', label: '中', nativeName: '繁體中文' },
];

const STORAGE_KEY = 'codefyui-locale';

function getInitialLocale(): Locale {
  const stored = localStorage.getItem(STORAGE_KEY);
  if (stored && stored in messages) return stored as Locale;
  const browserLang = navigator.language;
  if (browserLang.startsWith('zh')) return 'zh-TW';
  return 'en';
}

interface I18nState {
  locale: Locale;
  setLocale: (locale: Locale) => void;
  t: (key: TranslationKey, vars?: Record<string, string | number>) => string;
  /** Translate node field. Falls back to `fallback` (the English backend text) if no translation exists. */
  tn: (nodeName: string, field: 'description' | `param.${string}`, fallback: string) => string;
}

export const useI18n = create<I18nState>((set, get) => ({
  locale: getInitialLocale(),

  setLocale: (locale: Locale) => {
    localStorage.setItem(STORAGE_KEY, locale);
    set({ locale });
  },

  t: (key: TranslationKey, vars?: Record<string, string | number>) => {
    const { locale } = get();
    let text = messages[locale]?.[key] ?? messages.en[key] ?? key;
    if (vars) {
      for (const [k, v] of Object.entries(vars)) {
        // A FUNCTION replacement, so the value goes in verbatim. Passed as a
        // string it is a replacement PATTERN: `$&`, `` $` ``, `$'` and `$$`
        // are substitutions there, so a value holding one of them rewrote the
        // sentence around it -- and values are arbitrary data now that a
        // graph's parameters reach `git.gdiff.param`.
        text = text.replace(`{${k}}`, () => String(v));
      }
    }
    return text;
  },

  tn: (nodeName: string, field: string, fallback: string) => {
    const { locale } = get();
    if (locale === 'en') return fallback;

    const nodeT = nodeMessages[locale]?.[nodeName];
    if (!nodeT) return fallback;

    if (field === 'description') {
      return nodeT.description ?? fallback;
    }

    // field = "param.xxx"
    if (field.startsWith('param.')) {
      const paramName = field.slice(6);
      return nodeT.params?.[paramName] ?? fallback;
    }

    return fallback;
  },
}));

export type { TranslationKey };
