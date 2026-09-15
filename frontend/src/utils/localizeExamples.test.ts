import { describe, it, expect, afterEach } from 'vitest';
import { renderHook } from '@testing-library/react';
import { useI18n } from '../i18n';
import zhTWExamples from '../i18n/exampleLocales/zh-TW';
import {
  useLocalizedExamples,
  exampleMatches,
  displayWidth,
  truncateToWidth,
  type LocalizedExample,
} from './localizeExamples';
import type { ExampleSummary } from '../api/rest';

function summary(partial: Partial<ExampleSummary> & { path: string }): ExampleSummary {
  return {
    name: 'Example',
    description: 'An English description.',
    category: 'LLM',
    node_count: 3,
    edge_count: 2,
    ...partial,
  };
}

function localized(partial: Partial<LocalizedExample> & { path: string }): LocalizedExample {
  const base = summary(partial);
  return { ...base, descriptionEn: base.description, ...partial };
}

afterEach(() => {
  useI18n.setState({ locale: 'en' });
});

describe('useLocalizedExamples', () => {
  // The path a translation is keyed by has to be a path that actually ships,
  // or the table silently translates nothing. One real key, read out of the
  // table itself, keeps this test honest as the table grows.
  const [translatedPath] = Object.keys(zhTWExamples);

  it('leaves the English text alone in the en locale', () => {
    const examples = [summary({ path: translatedPath, description: 'English text.' })];
    const { result } = renderHook(() => useLocalizedExamples(examples));
    expect(result.current[0].description).toBe('English text.');
    expect(result.current[0].descriptionEn).toBe('English text.');
  });

  it('swaps in the zh-TW description and keeps the English for search', () => {
    useI18n.setState({ locale: 'zh-TW' });
    const examples = [summary({ path: translatedPath, description: 'English text.' })];
    const { result } = renderHook(() => useLocalizedExamples(examples));
    expect(result.current[0].description).toBe(zhTWExamples[translatedPath].description);
    expect(result.current[0].descriptionEn).toBe('English text.');
  });

  it('falls back to English for an example the table does not know', () => {
    // A third-party plugin's example, or one added since this build: the
    // gallery still lists it, reading in English, rather than going blank.
    useI18n.setState({ locale: 'zh-TW' });
    const examples = [summary({ path: 'plugin:someone-elses/Cool/Demo', description: 'Theirs.' })];
    const { result } = renderHook(() => useLocalizedExamples(examples));
    expect(result.current[0].description).toBe('Theirs.');
  });

  it('never translates the name', () => {
    useI18n.setState({ locale: 'zh-TW' });
    const examples = [summary({ path: translatedPath, name: 'Train CNN on MNIST' })];
    const { result } = renderHook(() => useLocalizedExamples(examples));
    expect(result.current[0].name).toBe('Train CNN on MNIST');
  });
});

describe('exampleMatches', () => {
  const example = localized({
    path: 'LLM/Demo',
    name: 'Word Embedding Analogy',
    description: '用向量算類比：king - man + woman。',
    descriptionEn: 'Vector arithmetic over word embeddings.',
    category: 'LLM',
  });

  it('matches the displayed Chinese', () => {
    expect(exampleMatches(example, '類比')).toBe(true);
  });

  it('still matches the original English while the UI is in Chinese', () => {
    // An engineer reading the UI in Chinese types `embedding` as readily as
    // 「詞向量」, and the docs and node names are English either way.
    expect(exampleMatches(example, 'embeddings')).toBe(true);
  });

  it('matches the name and the category', () => {
    expect(exampleMatches(example, 'analogy')).toBe(true);
    expect(exampleMatches(example, 'llm')).toBe(true);
  });

  it('does not match an unrelated query', () => {
    expect(exampleMatches(example, 'diffusion')).toBe(false);
  });
});

describe('displayWidth', () => {
  it('counts Latin characters as one column', () => {
    expect(displayWidth('hello')).toBe(5);
  });

  it('counts CJK ideographs and full-width punctuation as two', () => {
    expect(displayWidth('向量')).toBe(4);
    expect(displayWidth('，')).toBe(2);
  });

  it('measures a mixed string by its parts', () => {
    expect(displayWidth('ResNet 殘差')).toBe('ResNet '.length + 4);
  });
});

describe('truncateToWidth', () => {
  it('returns a string that already fits, with no ellipsis', () => {
    expect(truncateToWidth('short', 80)).toBe('short');
  });

  it('cuts English at the column count, as the old code-point cut did', () => {
    const text = 'a'.repeat(100);
    expect(truncateToWidth(text, 80)).toBe('a'.repeat(80) + '...');
  });

  it('cuts Chinese at half the character count, so the card still fits', () => {
    // 80 ideographs are 160 columns and would run off a card sized for 80.
    const text = '字'.repeat(100);
    expect(truncateToWidth(text, 80)).toBe('字'.repeat(40) + '...');
  });

  it('never splits a surrogate pair', () => {
    // Iterating by code point rather than by UTF-16 unit; a half-emoji is a
    // replacement glyph on screen.
    const text = '𠜎'.repeat(50);
    const cut = truncateToWidth(text, 9);
    expect(cut).toBe('𠜎'.repeat(4) + '...');
  });

  it('drops the whitespace it would otherwise leave before the ellipsis', () => {
    expect(truncateToWidth('abc def ghijkl', 8)).toBe('abc def...');
  });
});
