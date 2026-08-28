import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, fireEvent, waitFor, act, within } from '@testing-library/react';
import type { ParamDefinition } from '../../types';
import type { PackItem, PackItemStatus, PackSummary } from '../../api/rest';
import { useToastStore } from '../../store/toastStore';
import { useUIStore } from '../../store/uiStore';
import { _resetPackStoreForTesting, usePackStore } from '../../store/packStore';
import { useI18n } from '../../i18n';

// Mock the REST file backends used by the model_file / image_file / data_file
// variants so we can drive list/upload/download success + failure paths
// deterministically.
vi.mock('../../api/rest', async (importOriginal) => ({
  // The REAL error class, not a stub: `packStore.refresh()` narrows a failed
  // catalog read with `err instanceof PackApiError`, and an undefined export
  // makes that line throw a TypeError instead of reporting the 404 an older
  // server answers with. This file pulls `packStore` in through
  // `usePackAvailability`, so the class has to survive the mock.
  PackApiError: (await importOriginal<typeof import('../../api/rest')>()).PackApiError,
  validateScript: vi.fn(),
  listModelFiles: vi.fn(),
  uploadModelFile: vi.fn(),
  downloadModelFile: vi.fn(),
  listImageFiles: vi.fn(),
  uploadImageFile: vi.fn(),
  downloadImageFile: vi.fn(),
  listDataFiles: vi.fn(),
  uploadDataFile: vi.fn(),
  downloadDataFile: vi.fn(),
  // Nothing here opens the Package Center, but the pack catalog is one
  // `refresh()` away through the store these tests seed, and an unstubbed
  // export would reach the network rather than fail loudly.
  listPacks: vi.fn(),
}));

import {
  listModelFiles,
  uploadModelFile,
  downloadModelFile,
  listImageFiles,
  uploadImageFile,
  downloadImageFile,
  listDataFiles,
  uploadDataFile,
  downloadDataFile,
} from '../../api/rest';
vi.mock('./ScriptCodeField', () => ({
  ScriptCodeField: ({ param, displayLabel }: any) => (
    <div data-testid="script-code-field" data-param={param.name}>
      {displayLabel}
    </div>
  ),
}));

import { ParamField } from './ParamField';

const mkParam = (over: Partial<ParamDefinition>): ParamDefinition => ({
  name: 'p',
  param_type: 'string',
  default: '',
  description: '',
  options: [],
  min_value: null,
  max_value: null,
  ...over,
});

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  useToastStore.setState({ toasts: [] });
  // The select branch reads the pack catalog. Every case that does not seed
  // one runs against an EMPTY catalog, which is the base install: nothing is
  // greyed out and no hint appears.
  _resetPackStoreForTesting();
  useUIStore.setState({ packCenterOpen: false, packCenterFocusPackId: null });
  // Clear accumulated call history so per-test "not called" assertions don't
  // pick up calls from earlier tests (these are factory vi.fn()s, not spies).
  [
    listModelFiles,
    uploadModelFile,
    downloadModelFile,
    listImageFiles,
    uploadImageFile,
    downloadImageFile,
    listDataFiles,
    uploadDataFile,
    downloadDataFile,
  ].forEach((fn) => vi.mocked(fn).mockReset());
  vi.mocked(listModelFiles).mockResolvedValue([{ filename: 'm1.pt' } as any]);
  vi.mocked(listImageFiles).mockResolvedValue([{ filename: 'a.png' } as any]);
  vi.mocked(listDataFiles).mockResolvedValue([{ filename: 'grades.csv' } as any]);
  vi.mocked(uploadModelFile).mockResolvedValue({ filename: 'up.pt' } as any);
  vi.mocked(uploadImageFile).mockResolvedValue({ filename: 'up.png' } as any);
  vi.mocked(uploadDataFile).mockResolvedValue({ filename: 'up.csv' } as any);
  vi.mocked(downloadModelFile).mockResolvedValue(undefined as any);
  vi.mocked(downloadImageFile).mockResolvedValue(undefined as any);
  vi.mocked(downloadDataFile).mockResolvedValue(undefined as any);
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('ParamField — string (default) branch', () => {
  it('renders a text input and fires onChange', () => {
    const onChange = vi.fn();
    render(<ParamField param={mkParam({ name: 'title' })} value="hi" onChange={onChange} />);
    const input = screen.getByDisplayValue('hi') as HTMLInputElement;
    expect(input.type).toBe('text');
    fireEvent.change(input, { target: { value: 'bye' } });
    expect(onChange).toHaveBeenCalledWith('title', 'bye');
  });

  it('falls back to param.default then empty string when value is nullish', () => {
    const { rerender } = render(
      <ParamField param={mkParam({ default: 'dft' })} value={undefined} onChange={() => {}} />,
    );
    expect(screen.getByDisplayValue('dft')).toBeTruthy();
    rerender(
      <ParamField param={mkParam({ default: undefined })} value={undefined} onChange={() => {}} />,
    );
    // No default and no value → empty string controlled input.
    const inputs = screen.getAllByRole('textbox') as HTMLInputElement[];
    expect(inputs[0].value).toBe('');
  });

  it('uses the explicit label prop when provided, else the param name', () => {
    const { rerender } = render(
      <ParamField param={mkParam({ name: 'raw' })} value="" onChange={() => {}} label="Pretty" />,
    );
    expect(screen.getByText('Pretty')).toBeTruthy();
    rerender(<ParamField param={mkParam({ name: 'raw' })} value="" onChange={() => {}} />);
    expect(screen.getByText('raw')).toBeTruthy();
  });
});

describe('ParamField — secret branch', () => {
  it('renders a masked password input, shows the session-only hint, and fires onChange', () => {
    const onChange = vi.fn();
    const { container } = render(
      <ParamField
        param={mkParam({ name: 'openai_api_key', param_type: 'secret' })}
        value="sk-abc"
        onChange={onChange}
      />,
    );
    const input = container.querySelector('input[type="password"]') as HTMLInputElement;
    expect(input).toBeTruthy();
    expect(input.value).toBe('sk-abc');
    // The English hint steers users to the environment variable.
    expect(
      screen.getByText('Session only - cleared on save. Prefer the environment variable.'),
    ).toBeTruthy();
    fireEvent.change(input, { target: { value: 'sk-xyz' } });
    expect(onChange).toHaveBeenCalledWith('openai_api_key', 'sk-xyz');
  });

  it('falls back to param.default then empty string when value is nullish', () => {
    const { rerender, container } = render(
      <ParamField param={mkParam({ param_type: 'secret', default: 'dflt' })} value={undefined} onChange={() => {}} />,
    );
    expect((container.querySelector('input[type="password"]') as HTMLInputElement).value).toBe('dflt');
    rerender(
      <ParamField param={mkParam({ param_type: 'secret', default: undefined })} value={undefined} onChange={() => {}} />,
    );
    expect((container.querySelector('input[type="password"]') as HTMLInputElement).value).toBe('');
  });
});

describe('ParamField — bool branch', () => {
  it('renders a checkbox bound to Boolean(value) and emits checked', () => {
    const onChange = vi.fn();
    render(
      <ParamField param={mkParam({ name: 'flag', param_type: 'bool' })} value={true} onChange={onChange} />,
    );
    const cb = screen.getByRole('checkbox') as HTMLInputElement;
    expect(cb.checked).toBe(true);
    fireEvent.click(cb);
    expect(onChange).toHaveBeenCalledWith('flag', false);
  });

  it('coerces non-boolean value via Boolean()', () => {
    render(
      <ParamField param={mkParam({ name: 'flag', param_type: 'bool' })} value={0} onChange={() => {}} />,
    );
    expect((screen.getByRole('checkbox') as HTMLInputElement).checked).toBe(false);
  });
});

describe('ParamField — select branch', () => {
  it('renders options and emits the chosen value', () => {
    const onChange = vi.fn();
    render(
      <ParamField
        param={mkParam({ name: 'mode', param_type: 'select', options: ['a', 'b', 'c'], default: 'a' })}
        value="b"
        onChange={onChange}
      />,
    );
    const sel = screen.getByRole('combobox') as HTMLSelectElement;
    expect(sel.value).toBe('b');
    expect(screen.getAllByRole('option').map((o) => o.textContent)).toEqual(['a', 'b', 'c']);
    fireEvent.change(sel, { target: { value: 'c' } });
    expect(onChange).toHaveBeenCalledWith('mode', 'c');
  });

  it('falls back to param.default when value is nullish', () => {
    render(
      <ParamField
        param={mkParam({ name: 'mode', param_type: 'select', options: ['x', 'y'], default: 'y' })}
        value={undefined}
        onChange={() => {}}
      />,
    );
    expect((screen.getByRole('combobox') as HTMLSelectElement).value).toBe('y');
  });
});

describe('ParamField — int / float numeric branch', () => {
  it('int uses step=1 and parseInt onChange', () => {
    const onChange = vi.fn();
    render(
      <ParamField param={mkParam({ name: 'n', param_type: 'int' })} value={3} onChange={onChange} />,
    );
    const input = screen.getByRole('spinbutton') as HTMLInputElement;
    expect(input.step).toBe('1');
    fireEvent.change(input, { target: { value: '7' } });
    expect(onChange).toHaveBeenCalledWith('n', 7);
  });

  it('float uses step=any and parseFloat onChange', () => {
    const onChange = vi.fn();
    render(
      <ParamField param={mkParam({ name: 'r', param_type: 'float' })} value={1.5} onChange={onChange} />,
    );
    const input = screen.getByRole('spinbutton') as HTMLInputElement;
    expect(input.step).toBe('any');
    fireEvent.change(input, { target: { value: '2.25' } });
    expect(onChange).toHaveBeenCalledWith('r', 2.25);
  });

  it('falls back to default then 0 when value is nullish', () => {
    const { rerender } = render(
      <ParamField param={mkParam({ name: 'n', param_type: 'int', default: 5 })} value={undefined} onChange={() => {}} />,
    );
    expect((screen.getByRole('spinbutton') as HTMLInputElement).value).toBe('5');
    rerender(
      <ParamField param={mkParam({ name: 'n', param_type: 'int', default: undefined })} value={undefined} onChange={() => {}} />,
    );
    expect((screen.getByRole('spinbutton') as HTMLInputElement).value).toBe('0');
  });

  it('shows "Range" hint and error class when below min with both bounds', () => {
    const { container } = render(
      <ParamField
        param={mkParam({ name: 'n', param_type: 'int', min_value: 0, max_value: 10 })}
        value={-3}
        onChange={() => {}}
      />,
    );
    expect(screen.getByText('Range: 0 — 10')).toBeTruthy();
    expect(container.querySelector('input')?.className).toMatch(/inputError|Error/);
  });

  it('shows "Range" hint when above max with both bounds', () => {
    render(
      <ParamField
        param={mkParam({ name: 'n', param_type: 'int', min_value: 0, max_value: 10 })}
        value={99}
        onChange={() => {}}
      />,
    );
    expect(screen.getByText('Range: 0 — 10')).toBeTruthy();
  });

  it('shows "Min" hint when only a min bound is violated', () => {
    render(
      <ParamField
        param={mkParam({ name: 'n', param_type: 'float', min_value: 1, max_value: null })}
        value={0}
        onChange={() => {}}
      />,
    );
    expect(screen.getByText('Min: 1')).toBeTruthy();
  });

  it('shows "Max" hint when only a max bound is violated', () => {
    render(
      <ParamField
        param={mkParam({ name: 'n', param_type: 'float', min_value: null, max_value: 5 })}
        value={8}
        onChange={() => {}}
      />,
    );
    expect(screen.getByText('Max: 5')).toBeTruthy();
  });

  it('renders no hint and no error class when within range', () => {
    const { container } = render(
      <ParamField
        param={mkParam({ name: 'n', param_type: 'int', min_value: 0, max_value: 10 })}
        value={5}
        onChange={() => {}}
      />,
    );
    expect(screen.queryByText(/Range|Min|Max/)).toBeNull();
    expect(container.querySelector('input')?.className || '').not.toMatch(/inputError/);
  });

  it('treats NaN values as not-out-of-range (no hint)', () => {
    // Number('abc') → NaN; isNaN guard keeps outOfRange false even with bounds.
    render(
      <ParamField
        param={mkParam({ name: 'n', param_type: 'float', min_value: 0, max_value: 10 })}
        value={'abc'}
        onChange={() => {}}
      />,
    );
    expect(screen.queryByText(/Range|Min|Max/)).toBeNull();
  });
});

describe('ParamField — tensor_grid branch (delegates to TensorGridEditor)', () => {
  it('renders the TensorGridEditor with the display label', () => {
    render(
      <ParamField
        param={mkParam({ name: 'grid', param_type: 'tensor_grid' })}
        value={null}
        onChange={() => {}}
        label="My Tensor"
        siblingParams={{ shape: '2,2' }}
      />,
    );
    // TensorGridEditor renders the label text; assert it mounted via the label.
    expect(screen.getByText('My Tensor')).toBeTruthy();
  });
});

describe('ParamField — model_file FileField', () => {
  it('lists files on mount and lets the user select one', async () => {
    const onChange = vi.fn();
    render(
      <ParamField param={mkParam({ name: 'ckpt', param_type: 'model_file' })} value="" onChange={onChange} />,
    );
    await waitFor(() => expect(listModelFiles).toHaveBeenCalled());
    await screen.findByRole('option', { name: 'm1.pt' });
    fireEvent.change(screen.getByRole('combobox'), { target: { value: 'm1.pt' } });
    expect(onChange).toHaveBeenCalledWith('ckpt', 'm1.pt');
  });

  it('clicking the upload (↑) button opens the hidden file input', async () => {
    const { container } = render(
      <ParamField param={mkParam({ name: 'ckpt', param_type: 'model_file' })} value="" onChange={() => {}} />,
    );
    await waitFor(() => expect(listModelFiles).toHaveBeenCalled());
    const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement;
    const clickSpy = vi.spyOn(fileInput, 'click').mockImplementation(() => {});
    fireEvent.click(screen.getByTitle('Upload model file'));
    expect(clickSpy).toHaveBeenCalled();
  });

  it('uploads a file, refreshes, and emits the returned filename', async () => {
    const onChange = vi.fn();
    const { container } = render(
      <ParamField param={mkParam({ name: 'ckpt', param_type: 'model_file' })} value="" onChange={onChange} />,
    );
    await waitFor(() => expect(listModelFiles).toHaveBeenCalled());
    const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement;
    const file = new File(['x'], 'new.pt');
    fireEvent.change(fileInput, { target: { files: [file] } });
    await waitFor(() => expect(uploadModelFile).toHaveBeenCalledWith(file));
    await waitFor(() => expect(onChange).toHaveBeenCalledWith('ckpt', 'up.pt'));
    // List refreshed after upload (mount + post-upload).
    expect(vi.mocked(listModelFiles).mock.calls.length).toBeGreaterThanOrEqual(2);
    // Input value cleared in finally.
    expect(fileInput.value).toBe('');
  });

  it('clears the input value in finally only when the ref is still mounted', async () => {
    // Unmount the field while the upload is still in flight. When the promise
    // settles, the finally block runs with fileInputRef.current === null
    // (React detaches the ref on unmount), exercising the false branch of the
    // `if (fileInputRef.current)` guard without throwing.
    let resolveUpload: (v: { filename: string }) => void = () => {};
    vi.mocked(uploadModelFile).mockReturnValueOnce(
      new Promise<{ filename: string }>((res) => {
        resolveUpload = res;
      }) as any,
    );
    const { container, unmount } = render(
      <ParamField param={mkParam({ name: 'ckpt', param_type: 'model_file' })} value="" onChange={() => {}} />,
    );
    await waitFor(() => expect(listModelFiles).toHaveBeenCalled());
    const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement;
    fireEvent.change(fileInput, { target: { files: [new File(['x'], 'new.pt')] } });
    await waitFor(() => expect(uploadModelFile).toHaveBeenCalled());
    // Unmount before the upload settles → ref becomes null.
    unmount();
    await act(async () => {
      resolveUpload({ filename: 'up.pt' });
      await Promise.resolve();
    });
    // Reaching here without a throw means the null-ref branch was handled.
    expect(uploadModelFile).toHaveBeenCalledTimes(1);
  });

  it('no-ops upload when no file is selected', async () => {
    const { container } = render(
      <ParamField param={mkParam({ name: 'ckpt', param_type: 'model_file' })} value="" onChange={() => {}} />,
    );
    await waitFor(() => expect(listModelFiles).toHaveBeenCalled());
    const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement;
    fireEvent.change(fileInput, { target: { files: [] } });
    expect(uploadModelFile).not.toHaveBeenCalled();
  });

  it('shows an error toast (err.message) when upload rejects', async () => {
    vi.mocked(uploadModelFile).mockRejectedValueOnce(new Error('disk full'));
    const { container } = render(
      <ParamField param={mkParam({ name: 'ckpt', param_type: 'model_file' })} value="" onChange={() => {}} />,
    );
    await waitFor(() => expect(listModelFiles).toHaveBeenCalled());
    const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement;
    fireEvent.change(fileInput, { target: { files: [new File(['x'], 'bad.pt')] } });
    await waitFor(() => {
      expect(useToastStore.getState().toasts.some((t) => t.message === 'disk full' && t.type === 'error')).toBe(true);
    });
  });

  it('falls back to the i18n message when the upload error has no message', async () => {
    vi.mocked(uploadModelFile).mockRejectedValueOnce({});
    const { container } = render(
      <ParamField param={mkParam({ name: 'ckpt', param_type: 'model_file' })} value="" onChange={() => {}} />,
    );
    await waitFor(() => expect(listModelFiles).toHaveBeenCalled());
    const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement;
    fireEvent.change(fileInput, { target: { files: [new File(['x'], 'bad.pt')] } });
    await waitFor(() => {
      expect(useToastStore.getState().toasts.some((t) => t.message === 'Upload failed')).toBe(true);
    });
  });

  it('downloads the selected file', async () => {
    render(
      <ParamField param={mkParam({ name: 'ckpt', param_type: 'model_file' })} value="m1.pt" onChange={() => {}} />,
    );
    await waitFor(() => expect(listModelFiles).toHaveBeenCalled());
    const dl = screen.getByTitle('Download selected file');
    fireEvent.click(dl);
    await waitFor(() => expect(downloadModelFile).toHaveBeenCalledWith('m1.pt'));
  });

  it('download no-ops when no value is selected (button disabled)', async () => {
    render(
      <ParamField param={mkParam({ name: 'ckpt', param_type: 'model_file' })} value="" onChange={() => {}} />,
    );
    await waitFor(() => expect(listModelFiles).toHaveBeenCalled());
    const dl = screen.getByTitle('Download selected file') as HTMLButtonElement;
    expect(dl.disabled).toBe(true);
    // Force-fire the handler to exercise the early-return guard on empty value.
    fireEvent.click(dl);
    expect(downloadModelFile).not.toHaveBeenCalled();
  });

  it('shows an error toast when download rejects (err.message)', async () => {
    vi.mocked(downloadModelFile).mockRejectedValueOnce(new Error('404'));
    render(
      <ParamField param={mkParam({ name: 'ckpt', param_type: 'model_file' })} value="m1.pt" onChange={() => {}} />,
    );
    await waitFor(() => expect(listModelFiles).toHaveBeenCalled());
    fireEvent.click(screen.getByTitle('Download selected file'));
    await waitFor(() => {
      expect(useToastStore.getState().toasts.some((t) => t.message === '404')).toBe(true);
    });
  });

  it('falls back to i18n message when download error has no message', async () => {
    vi.mocked(downloadModelFile).mockRejectedValueOnce({});
    render(
      <ParamField param={mkParam({ name: 'ckpt', param_type: 'model_file' })} value="m1.pt" onChange={() => {}} />,
    );
    await waitFor(() => expect(listModelFiles).toHaveBeenCalled());
    fireEvent.click(screen.getByTitle('Download selected file'));
    await waitFor(() => {
      expect(useToastStore.getState().toasts.some((t) => t.message === 'Download failed')).toBe(true);
    });
  });

  it('refresh button re-lists files', async () => {
    render(
      <ParamField param={mkParam({ name: 'ckpt', param_type: 'model_file' })} value="" onChange={() => {}} />,
    );
    await waitFor(() => expect(listModelFiles).toHaveBeenCalled());
    const before = vi.mocked(listModelFiles).mock.calls.length;
    fireEvent.click(screen.getByTitle('Refresh file list'));
    await waitFor(() => expect(vi.mocked(listModelFiles).mock.calls.length).toBe(before + 1));
  });

  it('renders nullish value as the empty placeholder option', async () => {
    render(
      <ParamField param={mkParam({ name: 'ckpt', param_type: 'model_file' })} value={undefined} onChange={() => {}} />,
    );
    await waitFor(() => expect(listModelFiles).toHaveBeenCalled());
    expect((screen.getByRole('combobox') as HTMLSelectElement).value).toBe('');
  });
});

describe('ParamField — image_file FileField backend', () => {
  it('uses the image backend list/upload', async () => {
    const onChange = vi.fn();
    const { container } = render(
      <ParamField param={mkParam({ name: 'img', param_type: 'image_file' })} value="" onChange={onChange} />,
    );
    await waitFor(() => expect(listImageFiles).toHaveBeenCalled());
    await screen.findByRole('option', { name: 'a.png' });
    const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement;
    expect(fileInput.accept).toContain('.png');
    fireEvent.change(fileInput, { target: { files: [new File(['x'], 'n.png')] } });
    await waitFor(() => expect(uploadImageFile).toHaveBeenCalled());
    await waitFor(() => expect(onChange).toHaveBeenCalledWith('img', 'up.png'));
  });
});

describe('ParamField — data_file FileField backend', () => {
  it('uses the data backend list/upload', async () => {
    const onChange = vi.fn();
    const { container } = render(
      <ParamField param={mkParam({ name: 'path', param_type: 'data_file' })} value="" onChange={onChange} />,
    );
    await waitFor(() => expect(listDataFiles).toHaveBeenCalled());
    await screen.findByRole('option', { name: 'grades.csv' });
    const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement;
    expect(fileInput.accept).toContain('.csv');
    fireEvent.change(fileInput, { target: { files: [new File(['a,b\n1,2\n'], 'n.csv')] } });
    await waitFor(() => expect(uploadDataFile).toHaveBeenCalled());
    await waitFor(() => expect(onChange).toHaveBeenCalledWith('path', 'up.csv'));
  });

  it('starts empty on a fresh node rather than inheriting a previous pick', async () => {
    // Regression guard: the dropdown lists every uploaded file, but a newly
    // dragged node's value must stay '' until the learner picks one.
    render(
      <ParamField param={mkParam({ name: 'path', param_type: 'data_file' })} value="" onChange={() => {}} />,
    );
    await waitFor(() => expect(listDataFiles).toHaveBeenCalled());
    await screen.findByRole('option', { name: 'grades.csv' });
    expect((screen.getByRole('combobox') as HTMLSelectElement).value).toBe('');
  });
});

describe('ParamField — code params (core#131)', () => {
  const SCRIPT = ['def run(inputs, params):', '    return 1', ''].join('\n');

  it('renders the script editor rather than a one-line text input', () => {
    render(
      <ParamField
        param={mkParam({ name: 'code', param_type: 'code', default: SCRIPT })}
        value={SCRIPT}
        onChange={() => {}}
      />,
    );
    const field = screen.getByTestId('script-code-field');
    expect(field.getAttribute('data-param')).toBe('code');
    // The string default arm would have produced an <input type="text">,
    // which turns a multi-line script into an unreadable single line.
    expect(screen.queryByRole('textbox')).toBeNull();
  });
});

// ── select options gated on an optional pack (PR 2, F7) ───────────────────

function packItem(id: string, status: PackItemStatus): PackItem {
  return {
    id,
    kind: 'hf',
    repo_id: `org/${id}`,
    url: null,
    size_bytes: 1024,
    license: null,
    status,
  };
}

function packSummary(over: Partial<PackSummary> & { id: string }): PackSummary {
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

/** Put a catalog in the store, the way a finished `refresh()` would. */
function seedPacks(...packs: PackSummary[]) {
  usePackStore.setState({
    loaded: true,
    unsupported: false,
    packs,
    byId: Object.fromEntries(packs.map((pack) => [pack.id, pack])),
  });
}

const wordVectors = () =>
  packSummary({ id: 'word-vectors', title: 'Word vectors', pip_ready: true, usable: false });

const embeddingParam = (over: Partial<ParamDefinition> = {}) =>
  mkParam({
    name: 'model',
    param_type: 'select',
    default: 'demo-16d',
    options: ['demo-16d', 'glove-50d'],
    option_packs: { 'glove-50d': 'word-vectors' },
    ...over,
  });

const hintFor = (select: HTMLElement): HTMLElement => {
  const id = select.getAttribute('aria-describedby');
  expect(id).toBeTruthy();
  return document.getElementById(id as string) as HTMLElement;
};

describe('ParamField — select options gated on an optional pack', () => {
  it('disables options whose pack is missing and suffixes the pack name', () => {
    seedPacks(wordVectors());
    render(<ParamField param={embeddingParam()} value="demo-16d" onChange={() => {}} />);

    const options = screen.getAllByRole('option') as HTMLOptionElement[];
    expect(options.map((o) => o.value)).toEqual(['demo-16d', 'glove-50d']);
    expect(options[0].disabled).toBe(false);
    expect(options[0].textContent).toBe('demo-16d');
    // The reason travels with the option: a learner reading the list should
    // not have to open the Package Center to find out why one is greyed.
    expect(options[1].disabled).toBe(true);
    expect(options[1].textContent).toBe('glove-50d — needs pack: Word vectors');
  });

  it('suffixes the model id for a pack:item requirement', () => {
    seedPacks(
      packSummary({
        id: 'sentence-embeddings',
        title: 'Sentence embeddings',
        pip_ready: true,
        // "partial": one model is downloaded, the other never was.
        usable: false,
        items: [packItem('all-MiniLM-L6-v2', 'present'), packItem('all-mpnet-base-v2', 'missing')],
      }),
    );
    render(
      <ParamField
        param={mkParam({
          name: 'model',
          param_type: 'select',
          default: 'all-MiniLM-L6-v2',
          options: ['all-MiniLM-L6-v2', 'all-mpnet-base-v2'],
          option_packs: {
            'all-MiniLM-L6-v2': 'sentence-embeddings:all-MiniLM-L6-v2',
            'all-mpnet-base-v2': 'sentence-embeddings:all-mpnet-base-v2',
          },
        })}
        value="all-MiniLM-L6-v2"
        onChange={() => {}}
      />,
    );

    const options = screen.getAllByRole('option') as HTMLOptionElement[];
    // A downloaded model inside a pack the catalog still calls partial loads
    // fine, so only the model that is actually absent greys out.
    expect(options[0].disabled).toBe(false);
    expect(options[0].textContent).toBe('all-MiniLM-L6-v2');
    expect(options[1].disabled).toBe(true);
    expect(options[1].textContent).toBe('all-mpnet-base-v2 — needs model: all-mpnet-base-v2');
  });

  it('keeps the current value selectable even when its requirement is missing and shows the install link', () => {
    seedPacks(wordVectors());
    const onChange = vi.fn();
    render(<ParamField param={embeddingParam()} value="glove-50d" onChange={onChange} />);

    const select = screen.getByRole('combobox') as HTMLSelectElement;
    expect(select.value).toBe('glove-50d');
    // A saved graph may hold it. Disabling it would make the browser drop the
    // selection, i.e. silently rewrite the graph the moment the panel opened.
    const chosen = screen.getAllByRole('option')[1] as HTMLOptionElement;
    expect(chosen.disabled).toBe(false);
    expect(onChange).not.toHaveBeenCalled();

    const hint = hintFor(select);
    expect(hint.className).toContain('hintWarning');
    expect(hint).toHaveTextContent('"glove-50d" needs the Word vectors pack.');
    // The accessible name carries the pack, because a node with two gated
    // selects shows two of these and "Install pack, Install pack" is not a
    // choice anyone can make from a list of controls.
    const link = within(hint).getByRole('button', { name: 'Install pack: Word vectors' });
    // The visible label stays short — the pack is already named in the
    // sentence this button sits at the end of.
    expect(link).toHaveTextContent('Install pack');
  });

  it('names the model in the warning for a pack:item requirement on the current value', () => {
    seedPacks(
      packSummary({
        id: 'sentence-embeddings',
        title: 'Sentence embeddings',
        pip_ready: true,
        items: [packItem('all-mpnet-base-v2', 'missing')],
      }),
    );
    render(
      <ParamField
        param={mkParam({
          name: 'model',
          param_type: 'select',
          default: 'all-mpnet-base-v2',
          options: ['all-mpnet-base-v2'],
          option_packs: { 'all-mpnet-base-v2': 'sentence-embeddings:all-mpnet-base-v2' },
        })}
        value="all-mpnet-base-v2"
        onChange={() => {}}
      />,
    );

    expect(hintFor(screen.getByRole('combobox'))).toHaveTextContent(
      '"all-mpnet-base-v2" needs the model all-mpnet-base-v2 from the Sentence embeddings pack.',
    );
  });

  it('shows the muted hint when other options are unavailable', () => {
    seedPacks(wordVectors());
    render(<ParamField param={embeddingParam()} value="demo-16d" onChange={() => {}} />);

    const hint = hintFor(screen.getByRole('combobox'));
    expect(hint).toHaveTextContent('Greyed-out options need an optional pack.');
    // Nothing is wrong with what this node is set to, so it is not a warning.
    expect(hint.className).not.toContain('hintWarning');
  });

  it('opens the Package Center focused on the pack from the link', () => {
    seedPacks(wordVectors());
    render(<ParamField param={embeddingParam()} value="demo-16d" onChange={() => {}} />);

    // The value is fine here; it is a SIBLING option that needs the pack, and
    // the link still names the one it would install.
    fireEvent.click(screen.getByRole('button', { name: 'Install pack: Word vectors' }));

    expect(useUIStore.getState().packCenterOpen).toBe(true);
    expect(useUIStore.getState().packCenterFocusPackId).toBe('word-vectors');
  });

  it('renders plain options when the catalog is unsupported', () => {
    // An older server has no Package Center at all; greying an option out
    // there would hide a node that works, with no way to find out why.
    usePackStore.setState({
      loaded: true,
      unsupported: true,
      packs: [],
      byId: { 'word-vectors': wordVectors() },
    });
    render(<ParamField param={embeddingParam()} value="demo-16d" onChange={() => {}} />);

    const options = screen.getAllByRole('option') as HTMLOptionElement[];
    expect(options.map((o) => o.textContent)).toEqual(['demo-16d', 'glove-50d']);
    expect(options.some((o) => o.disabled)).toBe(false);
    expect(screen.getByRole('combobox')).not.toHaveAttribute('aria-describedby');
    expect(screen.queryByRole('button', { name: /Install pack/ })).toBeNull();
  });

  it('renders plain options while no catalog has arrived', () => {
    render(<ParamField param={embeddingParam()} value="demo-16d" onChange={() => {}} />);

    expect(screen.getAllByRole('option').some((o) => (o as HTMLOptionElement).disabled)).toBe(
      false,
    );
    expect(screen.getByRole('combobox')).not.toHaveAttribute('aria-describedby');
  });

  it('leaves a select with no option_packs untouched', () => {
    seedPacks(wordVectors());
    render(
      <ParamField
        param={mkParam({ name: 'mode', param_type: 'select', options: ['a', 'b'], default: 'a' })}
        value="a"
        onChange={() => {}}
      />,
    );

    expect(screen.getAllByRole('option').map((o) => o.textContent)).toEqual(['a', 'b']);
    expect(screen.getByRole('combobox')).not.toHaveAttribute('aria-describedby');
  });
});
