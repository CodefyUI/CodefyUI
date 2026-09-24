import { create } from 'zustand';
import en from './locales/en';
import zhTW from './locales/zh-TW';
import zhTWNodes from './nodeLocales/zh-TW';
import zhTWExamples from './exampleLocales/zh-TW';
import type { TranslationKey } from './locales/en';
import type { NodeTranslations } from './nodeLocales/types';
import type { ExampleTranslations } from './exampleLocales/types';

export type Locale = 'en' | 'zh-TW';

const messages: Record<Locale, Record<TranslationKey, string>> = {
  en,
  'zh-TW': zhTW,
};

const nodeMessages: Partial<Record<Locale, NodeTranslations>> = {
  'zh-TW': zhTWNodes,
};

const exampleMessages: Partial<Record<Locale, ExampleTranslations>> = {
  'zh-TW': zhTWExamples,
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
  tn: (
    nodeName: string,
    field: 'description' | 'details' | `param.${string}`,
    fallback: string,
  ) => string;
  /**
   * Translate an example's description, keyed by the `path` from
   * `/api/examples/list`. Falls back to `fallback` (the English text the
   * backend read out of `graph.json`), so an example added since this build
   * -- or one a third-party plugin ships -- still reads, in English.
   */
  te: (examplePath: string, fallback: string) => string;
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

    if (field === 'description' || field === 'details') {
      return nodeT[field] ?? fallback;
    }

    // field = "param.xxx"
    if (field.startsWith('param.')) {
      const paramName = field.slice(6);
      return nodeT.params?.[paramName] ?? fallback;
    }

    return fallback;
  },

  te: (examplePath: string, fallback: string) => {
    const { locale } = get();
    if (locale === 'en') return fallback;
    return exampleMessages[locale]?.[examplePath]?.description ?? fallback;
  },
}));

// The page's language is the UI's (#504). A screen reader picks its voice from
// `<html lang>`, so a page that always said "en" had the Chinese UI read out
// with English pronunciation (WCAG 3.1.1); Chrome also picks its CJK fallback
// font from it. Set here rather than where a locale is chosen, so the first
// load, the language menu, a workspace import and a direct `setState` are all
// covered. The locale code is used as is: `zh-TW` is valid BCP 47 and the
// language-plus-region form screen readers pick a voice by.
document.documentElement.lang = useI18n.getState().locale;
useI18n.subscribe((state, prev) => {
  if (state.locale !== prev.locale) document.documentElement.lang = state.locale;
});

export type { TranslationKey };
