import { describe, expect, it, beforeEach, afterEach, vi } from 'vitest';

// Note: getInitialLocale runs once at module load. To exercise its branches we
// re-import the module under different localStorage / navigator conditions using
// vi.resetModules() + dynamic import. The "static" describe blocks below import
// the module normally for the runtime store methods (t / tn / setLocale).

describe('useI18n store (runtime methods)', () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it('setLocale persists to localStorage and updates state', async () => {
    const { useI18n } = await import('./index');
    useI18n.getState().setLocale('zh-TW');
    expect(useI18n.getState().locale).toBe('zh-TW');
    expect(localStorage.getItem('codefyui-locale')).toBe('zh-TW');

    useI18n.getState().setLocale('en');
    expect(useI18n.getState().locale).toBe('en');
    expect(localStorage.getItem('codefyui-locale')).toBe('en');
  });

  describe('t()', () => {
    it('returns the locale-specific translation when present', async () => {
      const { useI18n } = await import('./index');
      useI18n.getState().setLocale('zh-TW');
      const en = (await import('./locales/en')).default;
      // A zh-TW value that differs from English for the same key.
      expect(useI18n.getState().t('toolbar.run')).not.toBe(en['toolbar.run']);
    });

    it('falls back to English when the active locale lacks the key', async () => {
      const { useI18n } = await import('./index');
      const en = (await import('./locales/en')).default;
      // Force a locale whose message table is missing (covers the `?? messages.en` arm).
      useI18n.setState({ locale: 'fr' as never });
      expect(useI18n.getState().t('toolbar.run')).toBe(en['toolbar.run']);
    });

    it('falls back to the raw key when neither locale nor English has it', async () => {
      const { useI18n } = await import('./index');
      useI18n.getState().setLocale('en');
      const missingKey = 'nonexistent.key' as never;
      expect(useI18n.getState().t(missingKey)).toBe('nonexistent.key');
    });

    it('interpolates {var} placeholders when vars are provided', async () => {
      const { useI18n } = await import('./index');
      useI18n.getState().setLocale('en');
      // 'toolbar.save.success' === 'Graph "{name}" saved successfully.'
      const out = useI18n.getState().t('toolbar.save.success', { name: 'MyGraph' });
      expect(out).toContain('MyGraph');
      expect(out).not.toContain('{name}');
    });

    it('coerces numeric vars to strings', async () => {
      const { useI18n } = await import('./index');
      useI18n.getState().setLocale('en');
      // Use a key with a placeholder; pass a number to hit String(v).
      const out = useI18n.getState().t('toolbar.save.success', { name: 42 });
      expect(out).toContain('42');
    });

    it('returns text unchanged when no vars are passed', async () => {
      const { useI18n } = await import('./index');
      useI18n.getState().setLocale('en');
      const out = useI18n.getState().t('toolbar.run');
      expect(typeof out).toBe('string');
      expect(out.length).toBeGreaterThan(0);
    });
  });

  describe('tn()', () => {
    it('returns the fallback immediately when locale is en', async () => {
      const { useI18n } = await import('./index');
      useI18n.getState().setLocale('en');
      expect(useI18n.getState().tn('Start', 'description', 'FALLBACK')).toBe('FALLBACK');
    });

    it('returns the fallback when no node translation table exists for the node', async () => {
      const { useI18n } = await import('./index');
      useI18n.getState().setLocale('zh-TW');
      expect(
        useI18n.getState().tn('NodeThatDoesNotExist', 'description', 'FB'),
      ).toBe('FB');
    });

    it('returns the translated description when present', async () => {
      const { useI18n } = await import('./index');
      useI18n.getState().setLocale('zh-TW');
      const out = useI18n.getState().tn('Start', 'description', 'FB');
      expect(out).not.toBe('FB');
      expect(out.length).toBeGreaterThan(0);
    });

    // Covered in the mocked-node-table block below (every real zh-TW node has a
    // description, so the `nodeT.description ?? fallback` fallback arm requires a
    // synthetic node table).

    it('returns the translated param when present', async () => {
      const { useI18n } = await import('./index');
      useI18n.getState().setLocale('zh-TW');
      // DecisionTreeClassifier has a translated `max_depth` param.
      const out = useI18n.getState().tn('DecisionTreeClassifier', 'param.max_depth', 'FB');
      expect(out).not.toBe('FB');
    });

    it('returns fallback for an unknown param name', async () => {
      const { useI18n } = await import('./index');
      useI18n.getState().setLocale('zh-TW');
      expect(
        useI18n.getState().tn('DecisionTreeClassifier', 'param.__nope__', 'FB'),
      ).toBe('FB');
    });

    it('returns fallback for a node that has no params table', async () => {
      const { useI18n } = await import('./index');
      useI18n.getState().setLocale('zh-TW');
      // Start has a description but no params object → optional-chain miss.
      expect(useI18n.getState().tn('Start', 'param.anything', 'FB')).toBe('FB');
    });

    it('returns fallback for a field that is neither description nor param.*', async () => {
      const { useI18n } = await import('./index');
      useI18n.getState().setLocale('zh-TW');
      expect(
        useI18n.getState().tn('Start', 'somethingElse' as never, 'FB'),
      ).toBe('FB');
    });
  });
});

describe('tn() with a synthetic node table (description fallback arm)', () => {
  beforeEach(() => {
    vi.resetModules();
    localStorage.clear();
  });

  afterEach(() => {
    vi.resetModules();
    vi.doUnmock('./nodeLocales/zh-TW');
  });

  it('falls back when a node entry has params but no description', async () => {
    // Inject a zh-TW node table whose entry deliberately omits `description`,
    // so `nodeT.description ?? fallback` exercises its fallback (right) arm.
    vi.doMock('./nodeLocales/zh-TW', () => ({
      default: {
        FakeNode: { params: { foo: '譯文' } },
      },
    }));
    const { useI18n } = await import('./index');
    useI18n.getState().setLocale('zh-TW');
    expect(useI18n.getState().tn('FakeNode', 'description', 'FB')).toBe('FB');
    // Sanity: the param lookup on the same synthetic node still resolves.
    expect(useI18n.getState().tn('FakeNode', 'param.foo', 'FB')).toBe('譯文');
  });
});

describe('getInitialLocale (module init branches)', () => {
  const originalLanguage = Object.getOwnPropertyDescriptor(navigator, 'language');

  beforeEach(() => {
    vi.resetModules();
    localStorage.clear();
  });

  afterEach(() => {
    vi.resetModules();
    localStorage.clear();
    if (originalLanguage) {
      Object.defineProperty(navigator, 'language', originalLanguage);
    }
  });

  it('uses a valid stored locale', async () => {
    localStorage.setItem('codefyui-locale', 'zh-TW');
    const { useI18n } = await import('./index');
    expect(useI18n.getState().locale).toBe('zh-TW');
  });

  it('ignores a stored value that is not a known locale and falls through to navigator', async () => {
    localStorage.setItem('codefyui-locale', 'klingon');
    Object.defineProperty(navigator, 'language', { value: 'en-US', configurable: true });
    const { useI18n } = await import('./index');
    expect(useI18n.getState().locale).toBe('en');
  });

  it('falls back to zh-TW when navigator.language starts with "zh"', async () => {
    Object.defineProperty(navigator, 'language', { value: 'zh-CN', configurable: true });
    const { useI18n } = await import('./index');
    expect(useI18n.getState().locale).toBe('zh-TW');
  });

  it('falls back to en for a non-zh navigator language', async () => {
    Object.defineProperty(navigator, 'language', { value: 'fr-FR', configurable: true });
    const { useI18n } = await import('./index');
    expect(useI18n.getState().locale).toBe('en');
  });
});
