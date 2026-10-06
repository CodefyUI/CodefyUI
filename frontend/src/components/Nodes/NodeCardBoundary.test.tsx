import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import type { NodeProps, NodeTypes } from '@xyflow/react';
import { nodeProps } from '../../test/utils';
import { useI18n } from '../../i18n';
import { withNodeCardBoundary, withNodeCardBoundaries } from './NodeCardBoundary';

interface CardData extends Record<string, unknown> {
  label: string;
  type: string;
  /** Thrown by the card when present. */
  boom?: unknown;
}

let attempts = 0;

function Card({ data }: NodeProps) {
  attempts += 1;
  const d = data as CardData;
  if ('boom' in d) throw d.boom;
  return <div>card {d.label}</div>;
}

const Guarded = withNodeCardBoundary(Card);

function propsOf(data: CardData, id = 'n1') {
  return nodeProps({ id, type: 'baseNode', data });
}

beforeEach(() => {
  attempts = 0;
  useI18n.setState({ locale: 'en' });
  // React reports each error a boundary catches through console.error. Only
  // that report is dropped: anything else, an act() warning above all, still
  // reaches the console and the act-warnings gate.
  const consoleError = console.error;
  vi.spyOn(console, 'error').mockImplementation((...args: unknown[]) => {
    const caughtReport = args.some(
      (arg) => typeof arg === 'string' && arg.startsWith('React will try to recreate'),
    );
    if (!caughtReport) consoleError(...args);
  });
});

afterEach(() => {
  vi.restoreAllMocks();
});

const failed = () => useI18n.getState().t('node.cardFailed');

describe('withNodeCardBoundary', () => {
  it('draws a card that does not throw exactly as it is', () => {
    const { container } = render(<Guarded {...propsOf({ label: 'A', type: 'X' })} />);
    expect(container.innerHTML).toBe('<div>card A</div>');
  });

  it('draws a card that throws as a box naming the node, with the error in its tooltip', () => {
    render(
      <Guarded
        {...propsOf({ label: 'My heatmap', type: 'AttentionHeatmap', boom: new TypeError('row is not iterable') })}
      />,
    );
    expect(screen.getByText('My heatmap')).toBeTruthy();
    expect(screen.getByText('AttentionHeatmap')).toBeTruthy();
    const box = screen.getByText(failed()).parentElement as HTMLElement;
    expect(box.getAttribute('title')).toBe('TypeError: row is not iterable');
  });

  it('names the node once when its label is its type', () => {
    render(<Guarded {...propsOf({ label: 'Note', type: 'note', boom: new Error('x') })} />);
    expect(screen.getByText('Note')).toBeTruthy();
    expect(screen.queryByText('note')).toBeNull();
  });

  it('falls back to the type, then the node id, when there is no label', () => {
    const { unmount } = render(<Guarded {...propsOf({ label: '', type: 'Tokenizer', boom: 1 })} />);
    expect(screen.getAllByText('Tokenizer').length).toBe(1);
    unmount();
    render(<Guarded {...propsOf({ label: '', type: '', boom: 1 }, 'node-7')} />);
    expect(screen.getByText('node-7')).toBeTruthy();
  });

  it('names a layers-editor card by its layer type', () => {
    const LayerCard = withNodeCardBoundary(Card);
    render(
      <LayerCard
        {...nodeProps({ id: 'lin1', type: 'layerNode', data: { layerType: 'Conv2d', boom: new Error('x') } })}
      />,
    );
    expect(screen.getByText('Conv2d')).toBeTruthy();
    expect(screen.queryByText('lin1')).toBeNull();
  });

  it('says it in zh-TW', () => {
    useI18n.setState({ locale: 'zh-TW' });
    render(<Guarded {...propsOf({ label: 'A', type: 'X', boom: new Error('x') })} />);
    expect(screen.getByText('這個節點無法顯示')).toBeTruthy();
  });

  it('shows the box for a thrown value that is not an Error, null included', () => {
    for (const boom of ['plain text', null]) {
      const { unmount } = render(<Guarded {...propsOf({ label: 'A', type: 'X', boom })} />);
      const box = screen.getByText(failed()).parentElement as HTMLElement;
      expect(box.getAttribute('title')).toBe(String(boom));
      unmount();
    }
  });

  it('keeps the cards beside a broken one drawn', () => {
    render(
      <>
        <Guarded {...propsOf({ label: 'Broken', type: 'X', boom: new Error('x') }, 'a')} />
        <Guarded {...propsOf({ label: 'Fine', type: 'X' }, 'b')} />
      </>,
    );
    expect(screen.getByText('card Fine')).toBeTruthy();
    expect(screen.getAllByText(failed()).length).toBe(1);
  });

  it('draws the card again when the node data changes, and does not retry the same data', () => {
    const broken: CardData = { label: 'A', type: 'X', boom: new Error('x') };
    const { rerender } = render(<Guarded {...propsOf(broken)} />);
    expect(screen.getByText(failed())).toBeTruthy();

    // Same data, another prop: the box stays and the card is not drawn again.
    const before = attempts;
    rerender(<Guarded {...propsOf(broken)} selected />);
    expect(attempts).toBe(before);
    expect(screen.getByText(failed())).toBeTruthy();

    // New data that still throws: the box again, not a loop.
    rerender(<Guarded {...propsOf({ ...broken })} />);
    expect(screen.getByText(failed())).toBeTruthy();

    // New data the card can draw: the card is back.
    rerender(<Guarded {...propsOf({ label: 'A', type: 'X' })} />);
    expect(screen.getByText('card A')).toBeTruthy();
    expect(screen.queryByText(failed())).toBeNull();
  });
});

describe('withNodeCardBoundaries', () => {
  it('wraps every entry of a nodeTypes map and keeps its keys', () => {
    const types = withNodeCardBoundaries({ first: Card, second: Card } as NodeTypes);
    expect(Object.keys(types)).toEqual(['first', 'second']);
    expect(types.first.displayName).toBe('NodeCardBoundary(Card)');
    const First = types.first;
    render(<First {...propsOf({ label: 'A', type: 'X', boom: new Error('x') })} />);
    expect(screen.getByText(failed())).toBeTruthy();
  });
});
