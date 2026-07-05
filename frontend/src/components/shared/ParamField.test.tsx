import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import type { ParamDefinition } from '../../types';
import { useToastStore } from '../../store/toastStore';
import { useI18n } from '../../i18n';

// Mock the REST file backends used by the model_file / image_file variants so
// we can drive list/upload/download success + failure paths deterministically.
vi.mock('../../api/rest', () => ({
  listModelFiles: vi.fn(),
  uploadModelFile: vi.fn(),
  downloadModelFile: vi.fn(),
  listImageFiles: vi.fn(),
  uploadImageFile: vi.fn(),
  downloadImageFile: vi.fn(),
}));

import {
  listModelFiles,
  uploadModelFile,
  downloadModelFile,
  listImageFiles,
  uploadImageFile,
  downloadImageFile,
} from '../../api/rest';
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
  // Clear accumulated call history so per-test "not called" assertions don't
  // pick up calls from earlier tests (these are factory vi.fn()s, not spies).
  [
    listModelFiles,
    uploadModelFile,
    downloadModelFile,
    listImageFiles,
    uploadImageFile,
    downloadImageFile,
  ].forEach((fn) => vi.mocked(fn).mockReset());
  vi.mocked(listModelFiles).mockResolvedValue([{ filename: 'm1.pt' } as any]);
  vi.mocked(listImageFiles).mockResolvedValue([{ filename: 'a.png' } as any]);
  vi.mocked(uploadModelFile).mockResolvedValue({ filename: 'up.pt' } as any);
  vi.mocked(uploadImageFile).mockResolvedValue({ filename: 'up.png' } as any);
  vi.mocked(downloadModelFile).mockResolvedValue(undefined as any);
  vi.mocked(downloadImageFile).mockResolvedValue(undefined as any);
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
