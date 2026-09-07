import { it, expect, vi } from 'vitest';
import { render, waitFor } from '@testing-library/react';
import { MathText } from './MathText';

// Own file on purpose: the module-level promise cache in MathText starts
// empty here, so the first mount is the one whose chunk fails. A factory that
// throws makes the dynamic import() reject, and vitest calls it again on the
// next import, which is what lets one test cover the loader's catch (reset and
// rethrow), the effect's catch, and the retry on a later mount.
//
// Vitest wraps the factory's error ("[vitest] There was an error when mocking
// a module..."), so nothing below asserts on the message: the contract under
// test is the fallback and the retry, never the error text.
let attempts = 0;
vi.mock('./katexRenderer', () => {
  attempts += 1;
  throw new Error('chunk unavailable');
});

it('keeps the raw source when the chunk fails and lets a later mount retry', async () => {
  const first = render(<MathText text="value $x$ end" />);
  await waitFor(() => expect(attempts).toBe(1));
  await new Promise((resolve) => setTimeout(resolve, 0));
  expect(first.container.querySelector('.katex')).toBeNull();
  expect(first.container.textContent).toBe('value $x$ end');
  first.unmount();

  const second = render(<MathText text="$$y$$" />);
  await waitFor(() => expect(attempts).toBe(2));
  await new Promise((resolve) => setTimeout(resolve, 0));
  expect(second.container.querySelector('.katex')).toBeNull();
  expect(second.container.textContent).toBe('$$y$$');
});
