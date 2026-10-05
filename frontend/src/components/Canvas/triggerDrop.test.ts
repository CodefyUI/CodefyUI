import { describe, it, expect, vi, afterEach } from 'vitest';
import { Position, type Edge, type FinalConnectionState, type InternalNode } from '@xyflow/react';
import { triggerDropConnection } from './triggerDrop';

// A released wire as React Flow hands it to onConnectEnd: by default the
// Start node's trigger, released where no handle was in reach.
function released(over: Partial<FinalConnectionState> = {}): FinalConnectionState {
  return {
    isValid: null,
    from: { x: 0, y: 0 },
    fromHandle: {
      id: 'trigger',
      nodeId: 'start',
      type: 'source',
      position: Position.Right,
      x: 0,
      y: 0,
      width: 13,
      height: 13,
    },
    fromPosition: Position.Right,
    fromNode: { id: 'start' } as InternalNode,
    to: { x: 0, y: 0 },
    toHandle: null,
    toPosition: null,
    toNode: null,
    pointer: { x: 0, y: 0 },
    ...over,
  };
}

/**
 * A card as React Flow renders it: the node wrapper carries the id, and a
 * card that can be triggered holds a `__trigger` target handle. Returns the
 * label inside it, which is what the pointer is over in the middle of a card.
 */
function cardLabel(id: string, { trigger = true } = {}): Element {
  const node = document.createElement('div');
  node.className = 'react-flow__node';
  node.setAttribute('data-id', id);
  if (trigger) {
    const handle = document.createElement('div');
    handle.className = 'react-flow__handle target';
    handle.setAttribute('data-handleid', '__trigger');
    node.append(handle);
  }
  const label = document.createElement('span');
  node.append(label);
  return label;
}

/** A document whose `elementFromPoint` answers `hit` wherever it is asked. */
function docHitting(hit: Element | null) {
  const elementFromPoint = vi.fn((_x: number, _y: number) => hit);
  return { doc: { elementFromPoint } as unknown as Document, elementFromPoint };
}

const mouseUp = (x: number, y: number) => new MouseEvent('mouseup', { clientX: x, clientY: y });

const TRIGGER_TO_N2 = { source: 'start', sourceHandle: 'trigger', target: 'n2', targetHandle: '__trigger' };

describe('triggerDropConnection', () => {
  it("connects a trigger released over a card to that card's __trigger", () => {
    const { doc, elementFromPoint } = docHitting(cardLabel('n2'));
    expect(triggerDropConnection(mouseUp(40, 50), released({ isValid: false }), [], doc)).toEqual(TRIGGER_TO_N2);
    expect(elementFromPoint).toHaveBeenCalledWith(40, 50);
  });

  it('connects it when no handle was in reach of the release either', () => {
    const { doc } = docHitting(cardLabel('n2'));
    expect(triggerDropConnection(mouseUp(40, 50), released({ isValid: null }), [], doc)).toEqual(TRIGGER_TO_N2);
  });

  it('leaves a wire React Flow has already connected alone', () => {
    const { doc, elementFromPoint } = docHitting(cardLabel('n2'));
    expect(triggerDropConnection(mouseUp(40, 50), released({ isValid: true }), [], doc)).toBeNull();
    expect(elementFromPoint).not.toHaveBeenCalled();
  });

  it.each<[string, FinalConnectionState['fromHandle']]>([
    ['a data output', { ...released().fromHandle!, id: 'out' }],
    // Grabbing a trigger wire by its Start end drags it from the `__trigger` end.
    ['the __trigger end of a trigger wire', { ...released().fromHandle!, id: '__trigger', type: 'target' }],
    ['no handle at all', null],
  ])('ignores a wire dragged from %s', (_, fromHandle) => {
    const { doc } = docHitting(cardLabel('n2'));
    expect(triggerDropConnection(mouseUp(40, 50), released({ fromHandle }), [], doc)).toBeNull();
  });

  it('connects nothing over the empty canvas', () => {
    const pane = document.createElement('div');
    pane.className = 'react-flow__pane';
    expect(triggerDropConnection(mouseUp(40, 50), released(), [], docHitting(pane).doc)).toBeNull();
    expect(triggerDropConnection(mouseUp(40, 50), released(), [], docHitting(null).doc)).toBeNull();
  });

  it('does not connect the Start node to itself', () => {
    // Given a `__trigger` handle here so that only the self rule refuses it.
    const { doc } = docHitting(cardLabel('start'));
    expect(triggerDropConnection(mouseUp(40, 50), released(), [], doc)).toBeNull();
  });

  it('connects nothing to a card that takes no trigger, such as a Start node or a note', () => {
    const { doc } = docHitting(cardLabel('start2', { trigger: false }));
    expect(triggerDropConnection(mouseUp(40, 50), released(), [], doc)).toBeNull();
  });

  it('connects nothing to a card this Start node already triggers', () => {
    const { doc } = docHitting(cardLabel('n2'));
    const existing: Edge = { id: 't', source: 'start', target: 'n2', targetHandle: '__trigger' };
    expect(triggerDropConnection(mouseUp(40, 50), released(), [existing], doc)).toBeNull();
  });

  it("is not stopped by another Start node's trigger or by a data wire into the card", () => {
    const { doc } = docHitting(cardLabel('n2'));
    const others: Edge[] = [
      { id: 't2', source: 'start2', target: 'n2', targetHandle: '__trigger' },
      { id: 'd', source: 'start', target: 'n2', sourceHandle: 'out', targetHandle: 'in' },
    ];
    expect(triggerDropConnection(mouseUp(40, 50), released(), others, doc)).toEqual(TRIGGER_TO_N2);
  });

  it('reads a touch release from the touch that ended', () => {
    const { doc, elementFromPoint } = docHitting(cardLabel('n2'));
    // A touchend lists no current touches; the lifted finger is in changedTouches.
    const touchEnd = { touches: [], changedTouches: [{ clientX: 7, clientY: 8 }] } as unknown as TouchEvent;
    expect(triggerDropConnection(touchEnd, released(), [], doc)).toEqual(TRIGGER_TO_N2);
    expect(elementFromPoint).toHaveBeenCalledWith(7, 8);
  });

  it('connects nothing for a touch release that names no touch', () => {
    const { doc } = docHitting(cardLabel('n2'));
    const touchEnd = { touches: [], changedTouches: [] } as unknown as TouchEvent;
    expect(triggerDropConnection(touchEnd, released(), [], doc)).toBeNull();
  });

  describe('without a document passed in', () => {
    afterEach(() => {
      delete (document as { elementFromPoint?: unknown }).elementFromPoint;
    });

    it("asks the page's own document", () => {
      // jsdom has no elementFromPoint of its own.
      const label = cardLabel('n2');
      Object.defineProperty(document, 'elementFromPoint', { configurable: true, value: () => label });
      expect(triggerDropConnection(mouseUp(40, 50), released(), [])).toEqual(TRIGGER_TO_N2);
    });
  });
});
