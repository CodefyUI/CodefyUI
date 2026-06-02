import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { TokenChipsView } from './TokenChipsView';
import { useI18n } from '../../i18n';
import type { OutputData, ListOutput } from '../../types';

function list(values: unknown[] | undefined): ListOutput {
  return {
    type: 'list',
    run_id: 'r',
    node_id: 'n',
    port: 'p',
    length: values?.length ?? 0,
    values,
  };
}

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('TokenChipsView', () => {
  it('returns null when tokens is not a list (renders nothing)', () => {
    const { container } = render(
      <TokenChipsView tokens={null} tokenIds={null} offsets={null} />,
    );
    expect(container.firstChild).toBeNull();
  });

  it('returns null when tokens is a non-list OutputData', () => {
    const scalar: OutputData = { type: 'scalar', run_id: 'r', node_id: 'n', port: 'p', value: 1 };
    const { container } = render(
      <TokenChipsView tokens={scalar} tokenIds={null} offsets={null} />,
    );
    expect(container.firstChild).toBeNull();
  });

  it('returns null when list has no values (values undefined → null)', () => {
    const { container } = render(
      <TokenChipsView tokens={list(undefined)} tokenIds={null} offsets={null} />,
    );
    // asListValues returns list.values ?? null → null → component returns null
    expect(container.firstChild).toBeNull();
  });

  it('renders the empty-output message when token list is empty', () => {
    render(<TokenChipsView tokens={list([])} tokenIds={null} offsets={null} />);
    expect(screen.getByText('No tokens — input text was empty.')).toBeInTheDocument();
  });

  it('renders chips with count header for non-empty tokens', () => {
    render(
      <TokenChipsView
        tokens={list(['Hello', 'World'])}
        tokenIds={list([1, 2])}
        offsets={list([[0, 5], [5, 10]])}
      />,
    );
    expect(screen.getByText('2 tokens')).toBeInTheDocument();
    expect(screen.getByText('Hello')).toBeInTheDocument();
    expect(screen.getByText('World')).toBeInTheDocument();
  });

  it('handles missing/invalid id and offset values per chip', () => {
    render(
      <TokenChipsView
        tokens={list(['a', 'b', 'c'])}
        // idValues: number ok for a, non-number for b, missing for c
        tokenIds={list([10, 'x'])}
        // offsets: valid 2-tuple for a, wrong-length for b, non-number for c, missing further
        offsets={list([[0, 1], [0], ['s', 2]])}
      />,
    );
    expect(screen.getByText('a')).toBeInTheDocument();
    expect(screen.getByText('b')).toBeInTheDocument();
    expect(screen.getByText('c')).toBeInTheDocument();
  });

  it('renders with null tokenIds and offsets (optional chaining null branches)', () => {
    render(
      <TokenChipsView tokens={list(['x'])} tokenIds={null} offsets={null} />,
    );
    expect(screen.getByText('x')).toBeInTheDocument();
  });

  it('coerces non-string token values via String()', () => {
    render(
      <TokenChipsView tokens={list([42, true])} tokenIds={null} offsets={null} />,
    );
    expect(screen.getByText('42')).toBeInTheDocument();
    expect(screen.getByText('true')).toBeInTheDocument();
  });
});
