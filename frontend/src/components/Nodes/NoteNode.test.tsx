import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { screen, fireEvent, waitFor, act } from '@testing-library/react';
import { renderWithFlow } from '../../test/utils';
import { useI18n } from '../../i18n';
import { useTabStore } from '../../store/tabStore';
import type { NodeData } from '../../types';
import NoteNode from './NoteNode';

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

function noteData(overrides: Partial<NodeData> = {}): NodeData {
  return {
    label: 'Note',
    type: 'note',
    params: {},
    noteKind: 'text',
    noteContent: '',
    noteColor: '#3d3d1a',
    boundToNodeId: null,
    boundOffset: null,
    noteWidth: 200,
    ...overrides,
  };
}

function renderNote(data: NodeData, opts: { id?: string; selected?: boolean } = {}) {
  const { id = 'note1', selected = false } = opts;
  return renderWithFlow(
    <NoteNode id={id} type="noteNode" data={data} selected={selected} {...flowProps} />,
  );
}

let lastUpdate: { id: string; updates: Record<string, unknown> } | null = null;

function resetStores() {
  useI18n.setState({ locale: 'en' });
  lastUpdate = null;
  const id = 'tab-note';
  useTabStore.setState((s) => ({
    activeTabId: id,
    tabs: [{ ...s.tabs[0], id, name: 'Tab', nodes: [], edges: [] }],
    // Spy on updateNoteData so we can assert without diffing nodes.
    updateNoteData: (nodeId: string, updates: Record<string, unknown>) => {
      lastUpdate = { id: nodeId, updates };
    },
  }));
}

beforeEach(() => {
  resetStores();
  vi.useFakeTimers({ toFake: ['requestAnimationFrame', 'cancelAnimationFrame'] });
});

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
});

describe('NoteNode', () => {
  it('renders a text note with content and the accent bar color', () => {
    const { container } = renderNote(noteData({ noteContent: 'hello world' }));
    expect(screen.getByText('hello world')).toBeTruthy();
    const accent = container.querySelector('[class*="accentBar"]') as HTMLElement;
    // #3d3d1a → rgb(61, 61, 26)
    expect(accent.style.background).toBe('rgb(61, 61, 26)');
  });

  it('uses default color, kind and width when fields are absent', () => {
    const { container } = renderNote(
      noteData({ noteColor: undefined, noteKind: undefined, noteWidth: undefined }),
    );
    const note = container.querySelector('[class*="note"]') as HTMLElement;
    expect(note.style.width).toBe('200px');
    const accent = container.querySelector('[class*="accentBar"]') as HTMLElement;
    expect(accent.style.background).toBe('rgb(61, 61, 26)');
    // default kind is text → contentEditable div present
    expect(container.querySelector('[class*="textContent"]')).toBeTruthy();
  });

  it('renders the bound indicator only when boundToNodeId is set', () => {
    const { container, rerender } = renderNote(noteData());
    expect(container.querySelector('[class*="bindIcon"]')).toBeNull();
    rerender(
      <NoteNode id="note1" type="noteNode" data={noteData({ boundToNodeId: 'x' })} selected={false} {...flowProps} />,
    );
    expect(container.querySelector('[class*="bindIcon"]')).toBeTruthy();
  });

  it('applies selected styling (border + box-shadow)', () => {
    const { container } = renderNote(noteData(), { selected: true });
    const note = container.querySelector('[class*="note"]') as HTMLElement;
    expect(note.style.borderColor).toBe('rgb(136, 136, 136)'); // #888 normalized
    // box-shadow keeps the 3-digit hex literal in jsdom.
    expect(note.style.boxShadow).toBe('0 0 0 1px #888');
  });

  it('does not apply selected styling when not selected', () => {
    const { container } = renderNote(noteData());
    const note = container.querySelector('[class*="note"]') as HTMLElement;
    expect(note.style.borderColor).toBe('');
    expect(note.style.boxShadow).toBe('');
  });

  // ── Text editing ──
  it('double-click on a text note enters edit mode and focuses content', () => {
    const { container } = renderNote(noteData({ noteContent: 'abc' }));
    const note = container.querySelector('[class*="note"]') as HTMLElement;
    const content = container.querySelector('[class*="textContent"]') as HTMLElement;
    const focusSpy = vi.spyOn(content, 'focus');
    fireEvent.doubleClick(note);
    // editing class applied
    expect(note.className).toMatch(/editing/);
    expect(content.getAttribute('contenteditable')).toBe('true');
    // requestAnimationFrame callback focuses the content div + selects text.
    act(() => {
      vi.runAllTimers();
    });
    expect(focusSpy).toHaveBeenCalled();
  });

  it('rAF focus block tolerates a null selection', () => {
    const { container } = renderNote(noteData());
    const orig = window.getSelection;
    window.getSelection = () => null as unknown as Selection;
    const note = container.querySelector('[class*="note"]') as HTMLElement;
    fireEvent.doubleClick(note);
    expect(() => act(() => vi.runAllTimers())).not.toThrow();
    window.getSelection = orig;
  });

  it('blur exits edit mode and persists the note content', () => {
    const { container } = renderNote(noteData());
    const note = container.querySelector('[class*="note"]') as HTMLElement;
    const content = container.querySelector('[class*="textContent"]') as HTMLElement;
    fireEvent.doubleClick(note);
    // Simulate typed text then blur.
    content.innerText = 'edited text';
    fireEvent.blur(content);
    expect(note.className).not.toMatch(/editing/);
    expect(lastUpdate).toEqual({ id: 'note1', updates: { noteContent: 'edited text' } });
  });

  it('blur with no content element text falls back to empty string', () => {
    const { container } = renderNote(noteData());
    const content = container.querySelector('[class*="textContent"]') as HTMLElement;
    // innerText defaults to '' in jsdom; blur without editing still calls update.
    fireEvent.blur(content);
    expect(lastUpdate?.updates).toEqual({ noteContent: '' });
  });

  it('Escape while editing exits edit mode and blurs', () => {
    const { container } = renderNote(noteData());
    const note = container.querySelector('[class*="note"]') as HTMLElement;
    const content = container.querySelector('[class*="textContent"]') as HTMLElement;
    fireEvent.doubleClick(note);
    const blurSpy = vi.spyOn(content, 'blur');
    fireEvent.keyDown(content, { key: 'Escape' });
    expect(blurSpy).toHaveBeenCalled();
  });

  it('non-Escape keydown while editing stops propagation but stays in edit mode', () => {
    const { container } = renderNote(noteData());
    const note = container.querySelector('[class*="note"]') as HTMLElement;
    const content = container.querySelector('[class*="textContent"]') as HTMLElement;
    fireEvent.doubleClick(note);
    fireEvent.keyDown(content, { key: 'a' });
    expect(note.className).toMatch(/editing/);
  });

  it('keydown while NOT editing is a no-op', () => {
    const { container } = renderNote(noteData());
    const content = container.querySelector('[class*="textContent"]') as HTMLElement;
    // Not in edit mode → handler short-circuits (editing === false branch).
    fireEvent.keyDown(content, { key: 'Escape' });
    const note = container.querySelector('[class*="note"]') as HTMLElement;
    expect(note.className).not.toMatch(/editing/);
  });

  // ── Image note ──
  it('renders an image note with src and maxHeight', () => {
    const { container } = renderNote(
      noteData({ noteKind: 'image', noteContent: 'data:image/png;base64,xyz', noteHeight: 123 }),
    );
    const img = container.querySelector('img') as HTMLImageElement;
    expect(img.getAttribute('src')).toBe('data:image/png;base64,xyz');
    expect(img.style.maxHeight).toBe('123px');
  });

  it('image note without content shows the placeholder; clicking it opens the file picker', () => {
    const { container } = renderNote(noteData({ noteKind: 'image', noteContent: '' }));
    expect(screen.getByText(useI18n.getState().t('note.imagePlaceholder'))).toBeTruthy();
    const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement;
    const clickSpy = vi.spyOn(fileInput, 'click');
    const placeholder = container.querySelector('[class*="imagePlaceholder"]') as HTMLElement;
    fireEvent.click(placeholder);
    expect(clickSpy).toHaveBeenCalled();
  });

  it('image note defaults maxHeight to 200 when noteHeight is absent', () => {
    const { container } = renderNote(
      noteData({ noteKind: 'image', noteContent: 'data:image/png;base64,xyz' }),
    );
    const img = container.querySelector('img') as HTMLImageElement;
    expect(img.style.maxHeight).toBe('200px');
  });

  it('double-click on an image note opens the file picker instead of editing', () => {
    const { container } = renderNote(noteData({ noteKind: 'image', noteContent: 'data:image/png;base64,xyz' }));
    const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement;
    const clickSpy = vi.spyOn(fileInput, 'click');
    fireEvent.doubleClick(container.querySelector('[class*="note"]') as HTMLElement);
    expect(clickSpy).toHaveBeenCalled();
  });

  // ── File change → resizeImage ──
  function installImageMocks(opts: { width: number; height: number; fail?: boolean }) {
    // FileReader → immediately produce a data URL.
    class FRMock {
      result: string | null = null;
      onload: (() => void) | null = null;
      onerror: ((e?: unknown) => void) | null = null;
      readAsDataURL() {
        this.result = 'data:image/png;base64,SRC';
        this.onload?.();
      }
    }
    vi.stubGlobal('FileReader', FRMock as unknown as typeof FileReader);

    // Image → fire onload (or onerror) on src assignment.
    class ImageMock {
      width = opts.width;
      height = opts.height;
      onload: (() => void) | null = null;
      onerror: ((e?: unknown) => void) | null = null;
      private _src = '';
      set src(_v: string) {
        this._src = _v;
        if (opts.fail) this.onerror?.(new Error('bad image'));
        else this.onload?.();
      }
      get src() {
        return this._src;
      }
    }
    vi.stubGlobal('Image', ImageMock as unknown as typeof Image);

    // canvas.getContext('2d') returns a stub with drawImage; toDataURL returns a URL.
    const drawImage = vi.fn();
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({ drawImage } as unknown as CanvasRenderingContext2D);
    vi.spyOn(HTMLCanvasElement.prototype, 'toDataURL').mockReturnValue('data:image/png;base64,RESIZED');
    return { drawImage };
  }

  function fireFileChange(container: HTMLElement, file: File | null) {
    const input = container.querySelector('input[type="file"]') as HTMLInputElement;
    if (file) {
      Object.defineProperty(input, 'files', { value: [file], configurable: true });
    } else {
      Object.defineProperty(input, 'files', { value: [], configurable: true });
    }
    fireEvent.change(input);
    return input;
  }

  it('uploading a small image stores the resized data URL', async () => {
    installImageMocks({ width: 100, height: 80 });
    const { container } = renderNote(noteData({ noteKind: 'image', noteContent: '' }));
    const file = new File(['x'], 'pic.png', { type: 'image/png' });
    const input = fireFileChange(container, file);
    await waitFor(() => expect(lastUpdate).not.toBeNull());
    expect(lastUpdate).toEqual({ id: 'note1', updates: { noteContent: 'data:image/png;base64,RESIZED' } });
    // input value reset so the same file can be re-selected
    expect(input.value).toBe('');
  });

  it('uploading a large image scales it down within MAX_IMAGE_DIM', async () => {
    const { drawImage } = installImageMocks({ width: 1600, height: 800 });
    const { container } = renderNote(noteData({ noteKind: 'image', noteContent: '' }));
    const file = new File(['x'], 'big.png', { type: 'image/png' });
    fireFileChange(container, file);
    await waitFor(() => expect(lastUpdate).not.toBeNull());
    // ratio = min(800/1600, 800/800) = 0.5 → 800 x 400
    expect(drawImage).toHaveBeenCalledWith(expect.anything(), 0, 0, 800, 400);
  });

  it('file change with no file selected is a no-op', async () => {
    installImageMocks({ width: 10, height: 10 });
    const { container } = renderNote(noteData({ noteKind: 'image', noteContent: '' }));
    fireFileChange(container, null);
    // No file → early return, no update.
    expect(lastUpdate).toBeNull();
  });

  it('image decode failure is silently ignored (no update)', async () => {
    installImageMocks({ width: 10, height: 10, fail: true });
    const { container } = renderNote(noteData({ noteKind: 'image', noteContent: '' }));
    const input = fireFileChange(container, new File(['x'], 'bad.png', { type: 'image/png' }));
    // Give the rejected promise a tick to settle.
    await Promise.resolve();
    await Promise.resolve();
    expect(lastUpdate).toBeNull();
    // input still reset
    await waitFor(() => expect(input.value).toBe(''));
  });
});
