import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { act, renderHook } from '@testing-library/react';
import type { PackItem, PackItemStatus, PackSummary } from '../api/rest';
import type { NodeDefinition, ParamDefinition } from '../types';
import { _resetPackStoreForTesting, usePackStore } from '../store/packStore';
import {
  isRequirementAvailable,
  itemTitle,
  missingRequirementForOption,
  nodeMissingPack,
  packTitle,
  parseRequirement,
  usePackAvailability,
  type PackIndex,
} from './packAvailability';

function makeItem(id: string, status: PackItemStatus): PackItem {
  return {
    id,
    kind: 'hf',
    repo_id: `org/${id}`,
    url: null,
    size_bytes: 1024,
    license: 'apache-2.0',
    status,
  };
}

function makePack(over: Partial<PackSummary> & { id: string }): PackSummary {
  return {
    title: over.id,
    description: '',
    install_mode: 'live',
    status: 'not_installed',
    pip_ready: false,
    usable: false,
    depends_on: [],
    blocked_by: [],
    pip: [],
    items: [],
    size_bytes_total: 0,
    install_command: null,
    ...over,
  };
}

function index(...packs: PackSummary[]): PackIndex {
  return Object.fromEntries(packs.map((pack) => [pack.id, pack]));
}

function makeParam(over: Partial<ParamDefinition>): ParamDefinition {
  return {
    name: 'model',
    param_type: 'select',
    default: '',
    description: '',
    options: [],
    min_value: null,
    max_value: null,
    ...over,
  };
}

function makeNode(over: Partial<NodeDefinition>): NodeDefinition {
  return {
    node_name: 'TextEmbedding',
    category: 'LLM',
    description: '',
    inputs: [],
    outputs: [],
    params: [],
    ...over,
  };
}

describe('parseRequirement', () => {
  it('handles bare, pack:item and malformed values', () => {
    expect(parseRequirement('sentence-embeddings')).toEqual({
      packId: 'sentence-embeddings',
      itemId: null,
    });
    expect(parseRequirement('sentence-embeddings:all-MiniLM-L6-v2')).toEqual({
      packId: 'sentence-embeddings',
      itemId: 'all-MiniLM-L6-v2',
    });

    // Malformed values keep the WHOLE string as the pack id, which no catalog
    // knows, which is what makes them available rather than a dead option.
    expect(parseRequirement('rag:')).toEqual({ packId: 'rag:', itemId: null });
    expect(parseRequirement(':model')).toEqual({ packId: ':model', itemId: null });
    expect(parseRequirement('a:b:c')).toEqual({ packId: 'a:b:c', itemId: null });
    expect(parseRequirement('')).toEqual({ packId: '', itemId: null });

    // Surrounding whitespace is a typo in a node definition, not a pack id.
    expect(parseRequirement('  rag : qwen  ')).toEqual({ packId: 'rag', itemId: 'qwen' });
  });
});

describe('isRequirementAvailable', () => {
  const unusable = index(makePack({ id: 'rag', usable: false }));
  const req = { packId: 'rag', itemId: null };

  it('treats an unloaded catalog, an unsupported server and an unknown pack as available', () => {
    // No requirement at all.
    expect(isRequirementAvailable(unusable, true, false, null)).toBe(true);
    // The catalog has not answered yet -- never grey out on a guess.
    expect(isRequirementAvailable(unusable, false, false, req)).toBe(true);
    // A server that predates the Package Center.
    expect(isRequirementAvailable(unusable, true, true, req)).toBe(true);
    // A pack id this catalog has never heard of (older/newer server, plugin).
    expect(isRequirementAvailable({}, true, false, req)).toBe(true);
    // The control: everything known, and the pack really is not usable.
    expect(isRequirementAvailable(unusable, true, false, req)).toBe(false);
  });

  it('a pack-only requirement follows pack.usable', () => {
    const usable = index(makePack({ id: 'rag', usable: true, pip_ready: false }));
    expect(isRequirementAvailable(usable, true, false, req)).toBe(true);

    // `usable` is the whole answer for a bare pack: a pack whose pip half is
    // ready and whose files are all downloaded is still unusable if the
    // server says so (a dependency of its own is missing).
    const half = index(
      makePack({
        id: 'rag',
        usable: false,
        pip_ready: true,
        items: [makeItem('qwen', 'present')],
      }),
    );
    expect(isRequirementAvailable(half, true, false, req)).toBe(false);
  });

  it('a pack:item requirement needs pip_ready and that item present', () => {
    const itemReq = { packId: 'sentence-embeddings', itemId: 'minilm' };
    const build = (pipReady: boolean, status: PackItemStatus) =>
      index(
        makePack({
          id: 'sentence-embeddings',
          usable: false,
          pip_ready: pipReady,
          items: [makeItem('minilm', status), makeItem('bge', 'missing')],
        }),
      );

    // Both halves present -- and note `usable` is false throughout, because a
    // per-item requirement asks about ONE model, not the whole pack.
    expect(isRequirementAvailable(build(true, 'present'), true, false, itemReq)).toBe(true);
    // The library is missing, so the downloaded weights cannot be loaded.
    expect(isRequirementAvailable(build(false, 'present'), true, false, itemReq)).toBe(false);
    // The library is there but this model was never fetched.
    expect(isRequirementAvailable(build(true, 'missing'), true, false, itemReq)).toBe(false);
    // Mid-download is not yet usable.
    expect(isRequirementAvailable(build(true, 'downloading'), true, false, itemReq)).toBe(false);
  });

  it('unknown item id is available', () => {
    // A node naming a model the installed catalog does not list is a version
    // skew between node and server; greying the option out would be a guess.
    const byId = index(
      makePack({
        id: 'sentence-embeddings',
        pip_ready: true,
        items: [makeItem('minilm', 'present')],
      }),
    );
    expect(
      isRequirementAvailable(byId, true, false, {
        packId: 'sentence-embeddings',
        itemId: 'not-in-this-catalog',
      }),
    ).toBe(true);
  });
});

describe('missingRequirementForOption', () => {
  const byId = index(
    makePack({
      id: 'sentence-embeddings',
      pip_ready: true,
      usable: true,
      items: [makeItem('minilm', 'present'), makeItem('bge', 'missing')],
    }),
  );

  it('reads option_packs', () => {
    const param = makeParam({
      options: ['hash', 'minilm', 'bge'],
      option_packs: {
        minilm: 'sentence-embeddings:minilm',
        bge: 'sentence-embeddings:bge',
      },
    });

    // No entry for this option -- it works on a base install.
    expect(missingRequirementForOption(param, 'hash', byId, true, false)).toBeNull();
    // Entry present and satisfied.
    expect(missingRequirementForOption(param, 'minilm', byId, true, false)).toBeNull();
    // Entry present and NOT satisfied -- the requirement comes back so the
    // editor can name the pack and the model in the tooltip.
    expect(missingRequirementForOption(param, 'bge', byId, true, false)).toEqual({
      packId: 'sentence-embeddings',
      itemId: 'bge',
    });
    // The catalog has not loaded: nothing is greyed out.
    expect(missingRequirementForOption(param, 'bge', byId, false, false)).toBeNull();
  });

  it('is null for a param with no option_packs at all', () => {
    expect(missingRequirementForOption(makeParam({}), 'bge', byId, true, false)).toBeNull();
    expect(
      missingRequirementForOption(makeParam({ option_packs: null }), 'bge', byId, true, false),
    ).toBeNull();
    // An empty string is a param author's mistake, not a requirement.
    expect(
      missingRequirementForOption(
        makeParam({ option_packs: { bge: '   ' } }),
        'bge',
        byId,
        true,
        false,
      ),
    ).toBeNull();
    // `option_packs` arrives as parsed JSON, so an option named after an
    // Object.prototype member must not pick up a method.
    expect(
      missingRequirementForOption(makeParam({ option_packs: {} }), 'toString', byId, true, false),
    ).toBeNull();
  });
});

describe('nodeMissingPack', () => {
  const byId = index(makePack({ id: 'rag', usable: false }));

  it('reads requires_pack', () => {
    expect(nodeMissingPack(makeNode({ requires_pack: 'rag' }), byId, true, false)).toEqual({
      packId: 'rag',
      itemId: null,
    });

    const usable = index(makePack({ id: 'rag', usable: true }));
    expect(nodeMissingPack(makeNode({ requires_pack: 'rag' }), usable, true, false)).toBeNull();

    // A base-install node, a node from a host that predates the key, and an
    // unknown definition are all fine.
    expect(nodeMissingPack(makeNode({ requires_pack: null }), byId, true, false)).toBeNull();
    expect(nodeMissingPack(makeNode({}), byId, true, false)).toBeNull();
    expect(nodeMissingPack(undefined, byId, true, false)).toBeNull();
    // Catalog not loaded -- no PACK badge on a guess.
    expect(nodeMissingPack(makeNode({ requires_pack: 'rag' }), byId, false, false)).toBeNull();
    // `byId` is built from parsed JSON, so a pack id that names an
    // Object.prototype member must read as UNKNOWN (available) rather than
    // resolve to an inherited function and grey the node out for good.
    expect(nodeMissingPack(makeNode({ requires_pack: 'toString' }), byId, true, false)).toBeNull();
    expect(
      nodeMissingPack(makeNode({ requires_pack: 'constructor:hasOwnProperty' }), byId, true, false),
    ).toBeNull();
    expect(packTitle(byId, 'toString')).toBe('toString');
  });
});

describe('packTitle / itemTitle', () => {
  const byId = index(makePack({ id: 'rag', title: 'RAG stack' }));

  it('falls back to the id before the catalog is loaded', () => {
    expect(packTitle(byId, 'rag')).toBe('RAG stack');
    expect(packTitle(byId, 'word-vectors')).toBe('word-vectors');
    expect(packTitle({}, 'rag')).toBe('rag');
  });

  it('names the item when the requirement has one', () => {
    expect(itemTitle(byId, { packId: 'rag', itemId: 'qwen2.5-0.5b' })).toBe('qwen2.5-0.5b');
    expect(itemTitle(byId, { packId: 'rag', itemId: null })).toBe('RAG stack');
  });
});

describe('usePackAvailability', () => {
  beforeEach(() => {
    _resetPackStoreForTesting();
  });

  afterEach(() => {
    _resetPackStoreForTesting();
  });

  it('answers for a raw requirement string, including the empty ones', () => {
    usePackStore.setState({
      byId: index(makePack({ id: 'rag', usable: false })),
      loaded: true,
    });
    const { result } = renderHook(() => usePackAvailability());

    expect(result.current.loaded).toBe(true);
    expect(result.current.unsupported).toBe(false);
    expect(result.current.isAvailable('rag')).toBe(false);
    expect(result.current.isAvailable('word-vectors')).toBe(true);
    // Nothing to require is always available -- the commonest case by far.
    expect(result.current.isAvailable(null)).toBe(true);
    expect(result.current.isAvailable(undefined)).toBe(true);
    expect(result.current.isAvailable('')).toBe(true);
    expect(result.current.isAvailable('   ')).toBe(true);
  });

  it('re-renders only when the slices change', () => {
    usePackStore.setState({
      byId: index(makePack({ id: 'rag', usable: false })),
      loaded: true,
    });

    let renders = 0;
    const { result } = renderHook(() => {
      renders += 1;
      return usePackAvailability();
    });
    const firstIsAvailable = result.current.isAvailable;
    const before = renders;

    // An install starting somewhere else touches `loading`, `busy` and the
    // job -- none of which change what is greyed out. Every param field on
    // the canvas holds this hook, so an unrelated `set` must be free.
    act(() => {
      usePackStore.setState({ loading: true, busy: { rag: true }, error: 'nope' });
    });
    expect(renders).toBe(before);
    expect(result.current.isAvailable).toBe(firstIsAvailable);

    // A refresh rebuilds `byId`, which is exactly when the answer can change.
    act(() => {
      usePackStore.setState({ byId: index(makePack({ id: 'rag', usable: true })) });
    });
    expect(renders).toBe(before + 1);
    expect(result.current.isAvailable).not.toBe(firstIsAvailable);
    expect(result.current.isAvailable('rag')).toBe(true);

    // So is a server turning out not to have the Package Center.
    act(() => {
      usePackStore.setState({ unsupported: true });
    });
    expect(renders).toBe(before + 2);
    expect(result.current.unsupported).toBe(true);
  });
});
