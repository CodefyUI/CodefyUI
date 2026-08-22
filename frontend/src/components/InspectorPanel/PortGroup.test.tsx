import { describe, it, expect, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { NonTensorView, PortGroup, keyOf } from './PortGroup';
import type { PortMedia } from './portCaptures';
import { makeHighlight, shapesEqual } from './diff';
import { useI18n } from '../../i18n';
import type { OutputData, TensorOutput } from '../../types';

function tensor(
  partial: Partial<TensorOutput> & Pick<TensorOutput, 'full_shape' | 'values'>,
): TensorOutput {
  return {
    type: 'tensor',
    run_id: 'r',
    node_id: 'n',
    port: 'p',
    dtype: 'float32',
    slice: ':',
    sliced_shape: partial.full_shape,
    truncated: false,
    ...partial,
  };
}

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
});

describe('NonTensorView', () => {
  it('renders scalar values with an optional label', () => {
    const sc: OutputData = { type: 'scalar', run_id: 'r', node_id: 'n', port: 'p', value: 3.14 };
    render(<NonTensorView value={sc} label="In" />);
    expect(screen.getByText('In')).toBeInTheDocument();
    expect(screen.getByText('scalar')).toBeInTheDocument();
    expect(screen.getByText('3.14')).toBeInTheDocument();
  });

  it('renders string values', () => {
    const s: OutputData = { type: 'string', run_id: 'r', node_id: 'n', port: 'p', value: 'hello' };
    render(<NonTensorView value={s} />);
    expect(screen.getByText('hello')).toBeInTheDocument();
  });

  it('renders model values with params formatting', () => {
    const m: OutputData = {
      type: 'model',
      run_id: 'r',
      node_id: 'n',
      port: 'p',
      class: 'Linear',
      params: 12345,
      trainable: 12345,
      repr: 'Linear(...)',
    };
    render(<NonTensorView value={m} />);
    expect(screen.getByText(/Linear/)).toBeInTheDocument();
    expect(screen.getByText(/12,345/)).toBeInTheDocument();
  });

  it('falls back to Module and ? when model class/params are missing', () => {
    const m = {
      type: 'model',
      run_id: 'r',
      node_id: 'n',
      port: 'p',
      trainable: 0,
      repr: '',
    } as unknown as OutputData;
    render(<NonTensorView value={m} />);
    expect(screen.getByText(/Module · params \?/)).toBeInTheDocument();
  });

  it('renders repr for generic/list types', () => {
    const g: OutputData = {
      type: 'list',
      run_id: 'r',
      node_id: 'n',
      port: 'p',
      length: 0,
      repr: 'list(empty)',
    };
    render(<NonTensorView value={g} />);
    expect(screen.getByText('list(empty)')).toBeInTheDocument();
  });

  it('falls back to the type name when no repr exists', () => {
    const g = { type: 'weird', run_id: 'r', node_id: 'n', port: 'p' } as unknown as OutputData;
    render(<NonTensorView value={g} />);
    expect(screen.getAllByText('weird').length).toBeGreaterThanOrEqual(1);
  });
});

// ── A port that declared media renders what it produced, not its bytes ──────
// The value on a `media=MEDIA_IMAGE` port is a base64 PNG, and on a
// `media=MEDIA_VIDEO` port a reference dict. Neither survives the capture
// path: `/api/execution/outputs` truncates every string at 4000 chars (a plot
// is ~35 000), and a dict arrives as a `repr`. Both are drawn from the
// node_status stream instead.

describe('PortGroup — media ports', () => {
  const PNG =
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==';

  function imageMedia(data = PNG, format = 'png'): PortMedia {
    return { kind: 'image', image: { format, encoding: 'base64', data } };
  }

  function videoMedia(format = 'mp4', url = '/api/media/clip.mp4'): PortMedia {
    return { kind: 'video', video: { path: 'clip.mp4', url, format } };
  }

  function stringState(value: string) {
    return {
      loading: false,
      error: null,
      data: { type: 'string', run_id: 'r', node_id: 'n1', port: 'image', value } as OutputData,
    };
  }

  function renderPort(media: PortMedia, fetches = {}, port = 'image', displayName?: string) {
    return render(
      <PortGroup
        kind="output"
        title="Outputs (1)"
        ports={[{ nodeId: 'n1', port, ...(displayName ? { displayName } : {}) }]}
        fetches={fetches}
        media={{ [keyOf('n1', port)]: media }}
      />,
    );
  }

  it('renders the picture instead of the captured base64 string', () => {
    renderPort(imageMedia(), {
      [keyOf('n1', 'image')]: stringState('iVBORw0KGgoTRUNCATED'),
    });
    const img = screen.getByAltText('image') as HTMLImageElement;
    expect(img.src).toBe(`data:image/png;base64,${PNG}`);
    expect(screen.queryByText('iVBORw0KGgoTRUNCATED')).toBeNull();
  });

  it('honours the payload format rather than assuming png', () => {
    renderPort(imageMedia(PNG, 'svg+xml'));
    expect((screen.getByAltText('image') as HTMLImageElement).src).toContain(
      'data:image/svg+xml;base64,',
    );
  });

  it('shows the media with no capture at all (Record outputs off)', () => {
    renderPort(imageMedia());
    expect(screen.getByAltText('image')).toBeInTheDocument();
    // No pending-capture ellipsis alongside it — the picture IS the value.
    expect(screen.queryByText('…')).toBeNull();
  });

  it('replaces a capture error rather than stacking under it', () => {
    renderPort(imageMedia(), {
      [keyOf('n1', 'image')]: { loading: false, error: 'run data expired', data: null },
    });
    expect(screen.getByAltText('image')).toBeInTheDocument();
    expect(screen.queryByText('run data expired')).toBeNull();
  });

  it('leaves ports with no media on the ordinary value path', () => {
    render(
      <PortGroup
        kind="output"
        title="Outputs (2)"
        ports={[
          { nodeId: 'n1', port: 'image' },
          { nodeId: 'n1', port: 'caption' },
        ]}
        fetches={{ [keyOf('n1', 'caption')]: stringState('a loss curve') }}
        media={{ [keyOf('n1', 'image')]: imageMedia() }}
      />,
    );
    expect(screen.getByAltText('image')).toBeInTheDocument();
    expect(screen.getByText('a loss curve')).toBeInTheDocument();
  });

  it('uses the row label for alt text when the port has one', () => {
    renderPort(imageMedia(), {}, 'image', 'Plot.image');
    expect(screen.getByAltText('Plot.image')).toBeInTheDocument();
  });

  it('plays a video port in a <video> element instead of showing its repr', () => {
    renderPort(videoMedia(), {
      [keyOf('n1', 'video')]: {
        loading: false,
        error: null,
        data: {
          type: 'dict',
          run_id: 'r',
          node_id: 'n1',
          port: 'video',
          repr: "{'path': 'clip.mp4', 'url': '/api/media/clip.mp4'}",
        } as unknown as OutputData,
      },
    }, 'video');
    const video = screen.getByLabelText('video') as HTMLVideoElement;
    expect(video.tagName).toBe('VIDEO');
    expect(video.getAttribute('src')).toBe('/api/media/clip.mp4');
    expect(video.controls).toBe(true);
    expect(screen.queryByText(/'path': 'clip.mp4'/)).toBeNull();
  });

  it('puts a gif in an <img> — browsers refuse it as a video source', () => {
    renderPort(videoMedia('gif', '/api/media/clip.gif'), {}, 'video');
    const img = screen.getByAltText('video') as HTMLImageElement;
    expect(img.tagName).toBe('IMG');
    expect(img.getAttribute('src')).toBe('/api/media/clip.gif');
    expect(document.querySelector('video')).toBeNull();
  });
});

describe('diff helpers', () => {
  it('shapesEqual compares length and every dim', () => {
    expect(shapesEqual([2, 3], [2, 3])).toBe(true);
    expect(shapesEqual([2, 3], [2, 4])).toBe(false);
    expect(shapesEqual([2], [2, 1])).toBe(false);
  });

  it('makeHighlight scores differing cells and zeroes equal ones', () => {
    const inT = tensor({ full_shape: [2, 2], values: [[1, 2], [3, 4]] });
    const outT = tensor({ full_shape: [2, 2], values: [[1, 9], [3, 4]] });
    const fn = makeHighlight(inT, outT);
    expect(fn).toBeDefined();
    expect(fn!(0, 1)).toBeGreaterThan(0);
    expect(fn!(1, 0)).toBe(0);
  });

  it('makeHighlight returns undefined for non-array values', () => {
    const inT = tensor({ full_shape: [], values: 5 });
    const outT = tensor({ full_shape: [], values: 7 });
    expect(makeHighlight(inT, outT)).toBeUndefined();
  });

  it('makeHighlight handles 1D tensors and non-number cells', () => {
    const inT = tensor({ full_shape: [3], values: [1, 'x', 3] });
    const outT = tensor({ full_shape: [3], values: [2, 'y', 3] });
    const fn = makeHighlight(inT, outT);
    expect(fn).toBeDefined();
    expect(fn!(0, 0)).toBeGreaterThan(0);
    expect(fn!(0, 1)).toBe(0);
  });

  it('makeHighlight unwraps 3D+ values to the last two dims', () => {
    const inT = tensor({ full_shape: [1, 2, 2], values: [[[1, 2], [3, 4]]] });
    const outT = tensor({ full_shape: [1, 2, 2], values: [[[1, 8], [3, 4]]] });
    const fn = makeHighlight(inT, outT);
    expect(fn).toBeDefined();
    expect(fn!(0, 1)).toBeGreaterThan(0);
  });
});
