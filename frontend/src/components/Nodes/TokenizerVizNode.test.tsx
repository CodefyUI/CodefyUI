import { describe, it, expect, beforeEach } from 'vitest';
import { screen } from '@testing-library/react';
import { renderWithFlow } from '../../test/utils';
import { useI18n } from '../../i18n';
import { useTabStore } from '../../store/tabStore';
import type { NodeData, NodeDefinition, OutputSummary } from '../../types';
import TokenizerVizNode from './TokenizerVizNode';

const flowProps = {
  zIndex: 0,
  isConnectable: true,
  positionAbsoluteX: 0,
  positionAbsoluteY: 0,
  dragging: false,
  draggable: false,
  selectable: true,
  deletable: true,
} as const;

function def(): NodeDefinition {
  return {
    node_name: 'Tokenizer',
    category: 'IO',
    description: '',
    inputs: [],
    outputs: [{ name: 'tokens', data_type: 'LIST', description: '', optional: false }],
    params: [],
  };
}

function data(): NodeData {
  return { label: 'Tokenizer', type: 'Tokenizer', params: {}, definition: def(), executionStatus: 'idle' };
}

const NODE_ID = 'tok1';

function seedSummary(summary: Record<string, OutputSummary> | undefined) {
  const id = 'tab-tok';
  useTabStore.setState((s) => ({
    activeTabId: id,
    tabs: [
      {
        ...s.tabs[0],
        id,
        name: 'Tab',
        nodes: [],
        edges: [],
        outputSummaries: summary ? { [NODE_ID]: summary } : ({} as Record<string, Record<string, OutputSummary>>),
      },
    ],
  }));
}

function renderNode() {
  return renderWithFlow(
    <TokenizerVizNode id={NODE_ID} type="tokenizerNode" data={data()} selected={false} {...flowProps} />,
  );
}

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  seedSummary(undefined);
});

describe('TokenizerVizNode', () => {
  it('shows the run hint when there are no token summaries', () => {
    seedSummary(undefined);
    renderNode();
    expect(screen.getByText(useI18n.getState().t('tokenizer.runHint'))).toBeTruthy();
  });

  it('shows the run hint when tokens is present but empty', () => {
    seedSummary({ tokens: { type: 'list', values: [] } });
    renderNode();
    expect(screen.getByText(useI18n.getState().t('tokenizer.runHint'))).toBeTruthy();
  });

  it('shows the run hint when tokens.values is not an array', () => {
    seedSummary({ tokens: { type: 'list' } });
    renderNode();
    expect(screen.getByText(useI18n.getState().t('tokenizer.runHint'))).toBeTruthy();
  });

  it('renders token chips with ids and valid offsets', () => {
    seedSummary({
      tokens: { type: 'list', values: ['Hello', 'World'], length: 2 },
      token_ids: { type: 'list', values: [10, 20] },
      offsets: { type: 'list', values: [[0, 5], [5, 10]] },
    });
    const { container } = renderNode();
    // displayText converts no whitespace here; both tokens visible
    expect(screen.getByText('Hello')).toBeTruthy();
    expect(screen.getByText('World')).toBeTruthy();
    expect(container.querySelectorAll('span[class*="chip"]').length).toBe(2);
  });

  it('handles non-array ids/offsets (idVal/off undefined branches)', () => {
    seedSummary({
      tokens: { type: 'list', values: ['a', 'b'] },
      token_ids: { type: 'list' }, // not an array
      offsets: { type: 'list' }, // not an array
    });
    const { container } = renderNode();
    expect(container.querySelectorAll('span[class*="chip"]').length).toBe(2);
  });

  it('treats a non-number id as undefined and a malformed offset as undefined', () => {
    seedSummary({
      tokens: { type: 'list', values: ['a'] },
      token_ids: { type: 'list', values: ['not-a-number'] },
      // offset present but wrong length / wrong element types
      offsets: { type: 'list', values: [[0]] },
    });
    const { container } = renderNode();
    expect(container.querySelectorAll('span[class*="chip"]').length).toBe(1);
  });

  it('treats an offset with non-number elements as undefined', () => {
    seedSummary({
      tokens: { type: 'list', values: ['a'] },
      offsets: { type: 'list', values: [['x', 'y']] },
    });
    const { container } = renderNode();
    expect(container.querySelectorAll('span[class*="chip"]').length).toBe(1);
  });

  it('renders the truncation hint when there are more than 64 tokens', () => {
    const many = Array.from({ length: 70 }, (_, i) => `t${i}`);
    seedSummary({
      tokens: { type: 'list', values: many, length: 70 },
    });
    const { container } = renderNode();
    // Only first 64 chips rendered inline
    expect(container.querySelectorAll('span[class*="chip"]').length).toBe(64);
    expect(
      screen.getByText(useI18n.getState().t('tokenizer.truncatedInline', { shown: 64, total: 70 })),
    ).toBeTruthy();
  });

  it('uses chips.length as totalLen when summary length is absent (no truncation)', () => {
    seedSummary({
      tokens: { type: 'list', values: ['a', 'b', 'c'] }, // no `length`
    });
    const { container } = renderNode();
    expect(container.querySelectorAll('span[class*="chip"]').length).toBe(3);
    // total === shown → not truncated
    expect(screen.queryByText(/showing first/)).toBeNull();
  });

  it('does not truncate when totalLen equals the number of chips', () => {
    seedSummary({
      tokens: { type: 'list', values: ['a', 'b'], length: 2 },
    });
    renderNode();
    expect(screen.queryByText(/showing first/)).toBeNull();
  });
});
