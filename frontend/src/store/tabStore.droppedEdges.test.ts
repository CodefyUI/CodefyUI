/**
 * Phase 4B review: edges a folder save dropped (to a missing node, or a
 * second one between the same ports) come off the canvas too, so the next
 * save does not report them again -- without an undo frame, and without
 * making a tab that matched its file look unsaved.
 */
import { describe, it, expect, beforeEach } from 'vitest';
import type { Edge } from '@xyflow/react';

import { useTabStore } from './tabStore';
import type { SubgraphDefinition } from '../types';

const store = () => useTabStore.getState();
const tab = () => useTabStore.getState().getActiveTab();

const edge = (id: string): Edge => ({ id, source: 'a', target: 'b' });

beforeEach(() => {
  useTabStore.setState({ tabs: [], activeTabId: null as unknown as string, clipboard: null });
  store().addTab('Tab 1');
});

describe('removeDroppedEdges', () => {
  it('takes the dropped edges off the canvas and out of a block', () => {
    store().setEdges([edge('keep'), edge('ghost'), edge('dup')]);
    store().setSubgraphs([{
      id: 'blk', name: 'b', description: '', nodes: [],
      edges: [{ id: 'in1', source: 'x', target: 'y' }, { id: 'in2', source: 'x', target: 'z' }],
      interface: { inputs: [], outputs: [], triggerTargets: [] },
    } as unknown as SubgraphDefinition]);
    const id = tab().id;
    store().markTabSaved(id, tab().revision);
    const undoDepth = tab().undoStack?.length ?? 0;

    store().removeDroppedEdges(id, [
      { block: null, id: 'ghost' }, { block: null, id: 'dup' }, { block: 'blk', id: 'in2' },
    ]);

    expect(tab().edges.map((e) => e.id)).toEqual(['keep']);
    expect(tab().subgraphs[0].edges.map((e: { id: string }) => e.id)).toEqual(['in1']);
    expect(tab().savedRevision).toBe(tab().revision);
    expect(tab().undoStack?.length ?? 0).toBe(undoDepth);
  });

  it('leaves a tab with unsaved edits unsaved', () => {
    store().setEdges([edge('keep'), edge('ghost')]);
    const id = tab().id;
    store().markTabSaved(id, tab().revision - 1);
    store().removeDroppedEdges(id, [{ block: null, id: 'ghost' }]);
    expect(tab().edges.map((e) => e.id)).toEqual(['keep']);
    expect(tab().savedRevision).not.toBe(tab().revision);
  });
});
