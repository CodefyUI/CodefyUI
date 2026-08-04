import { describe, it, expect } from 'vitest';
import { render } from '@testing-library/react';
import { EdgeLaneProvider, useEdgeLane } from './EdgeLaneContext';
import { SOLO_LANE, type LaneEdgeInput } from '../../utils/edgeLanes';

function wire(id: string, source: string, target: string): LaneEdgeInput {
  return { id, source, target, sourceHandle: 'out', targetHandle: 'in' };
}

describe('useEdgeLane', () => {
  it('falls back to the solo lane with no provider above it', () => {
    let seen: unknown;
    function Probe() {
      seen = useEdgeLane('anything');
      return null;
    }
    render(<Probe />);
    expect(seen).toEqual(SOLO_LANE);
  });

  it('falls back to the solo lane for an id the map does not carry', () => {
    let seen: unknown;
    function Probe() {
      seen = useEdgeLane('missing');
      return null;
    }
    render(
      <EdgeLaneProvider edges={[wire('e1', 'A', 'B')]}>
        <Probe />
      </EdgeLaneProvider>,
    );
    expect(seen).toEqual(SOLO_LANE);
  });

  it('hands each edge its slot', () => {
    const lanes: Record<string, number> = {};
    function Probe({ id }: { id: string }) {
      lanes[id] = useEdgeLane(id).outSlot;
      return null;
    }
    const edges = [wire('e1', 'A', 'B'), wire('e2', 'A', 'C')];
    render(
      <EdgeLaneProvider edges={edges}>
        <Probe id="e1" />
        <Probe id="e2" />
      </EdgeLaneProvider>,
    );
    expect(lanes.e1).not.toBe(lanes.e2);
  });
});

describe('EdgeLaneProvider recomputation', () => {
  /**
   * The provider is an ancestor of every edge on the canvas, so a lane map that
   * changed identity on each render would re-render all of them. Lanes read
   * topology only, and the memo is keyed on a signature of exactly that, so the
   * updates a canvas actually spends its time on - selecting an edge, recolouring
   * one, `applyEdgeChanges` handing back a fresh array, a node drag - must leave
   * the map alone.
   *
   * `probe` is one element object reused across renders, so React bails out of
   * re-rendering it unless the context value itself changed. Counting its renders
   * therefore counts context changes, not parent renders.
   */
  function setup() {
    let renders = 0;
    function Probe() {
      renders += 1;
      useEdgeLane('e1');
      return null;
    }
    const probe = <Probe />;
    return { probe, renders: () => renders };
  }

  it('keeps the map identical when a fresh array carries the same wiring', () => {
    const { probe, renders } = setup();
    const first = [wire('e1', 'A', 'B'), wire('e2', 'A', 'C')];
    const { rerender } = render(<EdgeLaneProvider edges={first}>{probe}</EdgeLaneProvider>);
    expect(renders()).toBe(1);

    // What `applyEdgeChanges` produces when an edge is merely selected: new array,
    // new objects, same wiring.
    const reselected = first.map((e) => ({ ...e }));
    rerender(<EdgeLaneProvider edges={reselected}>{probe}</EdgeLaneProvider>);
    expect(renders()).toBe(1);

    // And again, to be sure the first pass was not a fluke of mount ordering.
    rerender(<EdgeLaneProvider edges={first.map((e) => ({ ...e }))}>{probe}</EdgeLaneProvider>);
    expect(renders()).toBe(1);
  });

  it('rebuilds the map when an edge is added', () => {
    const { probe, renders } = setup();
    const first = [wire('e1', 'A', 'B')];
    const { rerender } = render(<EdgeLaneProvider edges={first}>{probe}</EdgeLaneProvider>);
    expect(renders()).toBe(1);

    rerender(
      <EdgeLaneProvider edges={[...first, wire('e2', 'A', 'C')]}>{probe}</EdgeLaneProvider>,
    );
    expect(renders()).toBe(2);
  });

  it('rebuilds the map when an edge is rewired', () => {
    const { probe, renders } = setup();
    const { rerender } = render(
      <EdgeLaneProvider edges={[wire('e1', 'A', 'B')]}>{probe}</EdgeLaneProvider>,
    );
    rerender(<EdgeLaneProvider edges={[wire('e1', 'A', 'C')]}>{probe}</EdgeLaneProvider>);
    expect(renders()).toBe(2);
  });
});
