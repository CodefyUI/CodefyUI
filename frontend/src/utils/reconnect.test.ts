import { describe, it, expect } from 'vitest';
import { computeDetachedEndpoint } from './reconnect';

// `handleType` semantics (verified against @xyflow/react 12.10.1
// EdgeUpdateAnchors): it names the end that STAYS connected — the drag-origin
// handle — so the detached endpoint is always the OPPOSITE end.
describe('computeDetachedEndpoint', () => {
  const edge = { source: 'a', target: 'b', sourceHandle: 'out', targetHandle: 'in' };

  it("handleType 'source' (source stays) -> the grabbed/detached endpoint is the target end", () => {
    expect(computeDetachedEndpoint(edge, 'source')).toEqual({
      nodeId: 'b',
      handleId: 'in',
      type: 'target',
    });
  });

  it("handleType 'target' (target stays) -> the grabbed/detached endpoint is the source end", () => {
    expect(computeDetachedEndpoint(edge, 'target')).toEqual({
      nodeId: 'a',
      handleId: 'out',
      type: 'source',
    });
  });

  it('returns null when the detached end has no handle id', () => {
    expect(computeDetachedEndpoint({ source: 'a', target: 'b' }, 'source')).toBeNull();
    expect(computeDetachedEndpoint({ source: 'a', target: 'b' }, 'target')).toBeNull();
  });

  it('a missing handle id on the STAYING end does not matter', () => {
    // Only the detached end's handle id is required.
    expect(
      computeDetachedEndpoint({ source: 'a', target: 'b', targetHandle: 'in' }, 'source'),
    ).toEqual({ nodeId: 'b', handleId: 'in', type: 'target' });
    expect(
      computeDetachedEndpoint({ source: 'a', target: 'b', sourceHandle: 'out' }, 'target'),
    ).toEqual({ nodeId: 'a', handleId: 'out', type: 'source' });
  });
});
