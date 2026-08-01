import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';

// CodeMirror is loaded through a dynamic import so it never reaches the entry
// bundle. Mocking that module is also what lets these tests drive both halves
// of the contract: the editor mounting, and the textarea fallback that has to
// keep working when the chunk never arrives.
const handle = {
  setValue: vi.fn(),
  setErrorLine: vi.fn(),
  destroy: vi.fn(),
};
let lastOptions: Record<string, any> | null = null;
let failNextMount = false;

vi.mock('./codemirrorSetup', () => ({
  createPythonEditor: (options: Record<string, any>) => {
    if (failNextMount) throw new Error('chunk unavailable');
    lastOptions = options;
    return handle;
  },
}));

import { CodeEditor } from './CodeEditor';

beforeEach(() => {
  handle.setValue.mockReset();
  handle.setErrorLine.mockReset();
  handle.destroy.mockReset();
  lastOptions = null;
  failNextMount = false;
});

describe('CodeEditor', () => {
  it('mounts CodeMirror with the initial document', async () => {
    render(<CodeEditor value="x = 1" onChange={() => {}} ariaLabel="code" />);
    await waitFor(() => expect(lastOptions).not.toBeNull());
    expect(lastOptions!.doc).toBe('x = 1');
    expect(lastOptions!.ariaLabel).toBe('code');
    expect(lastOptions!.readOnly).toBe(false);
  });

  it('renders a usable textarea until the editor is ready', async () => {
    const onChange = vi.fn();
    render(<CodeEditor value="x = 1" onChange={onChange} ariaLabel="code" />);
    // Synchronously after mount the dynamic import has not resolved yet.
    const area = screen.getByLabelText('code') as HTMLTextAreaElement;
    expect(area.tagName).toBe('TEXTAREA');
    fireEvent.change(area, { target: { value: 'x = 2' } });
    expect(onChange).toHaveBeenCalledWith('x = 2');
    await waitFor(() => expect(lastOptions).not.toBeNull());
  });

  it('drops the fallback once the editor has mounted', async () => {
    render(<CodeEditor value="x = 1" onChange={() => {}} ariaLabel="code" />);
    await waitFor(() => expect(screen.queryByLabelText('code')).toBeNull());
  });

  it('keeps the textarea when the chunk fails to load', async () => {
    failNextMount = true;
    const onChange = vi.fn();
    render(<CodeEditor value="x = 1" onChange={onChange} ariaLabel="code" />);
    // Give the rejected import a turn; the textarea must survive it.
    await new Promise((resolve) => setTimeout(resolve, 0));
    const area = screen.getByLabelText('code') as HTMLTextAreaElement;
    fireEvent.change(area, { target: { value: 'still editable' } });
    expect(onChange).toHaveBeenCalledWith('still editable');
  });

  it('forwards edits made inside the editor', async () => {
    const onChange = vi.fn();
    render(<CodeEditor value="x = 1" onChange={onChange} ariaLabel="code" />);
    await waitFor(() => expect(lastOptions).not.toBeNull());
    lastOptions!.onChange('x = 3');
    expect(onChange).toHaveBeenCalledWith('x = 3');
  });

  it('pushes an externally changed value into the editor', async () => {
    const { rerender } = render(
      <CodeEditor value="x = 1" onChange={() => {}} ariaLabel="code" />,
    );
    await waitFor(() => expect(lastOptions).not.toBeNull());
    rerender(<CodeEditor value="x = 9" onChange={() => {}} ariaLabel="code" />);
    expect(handle.setValue).toHaveBeenCalledWith('x = 9');
  });

  it('marks and clears the error line', async () => {
    const { rerender } = render(
      <CodeEditor value="x = 1" onChange={() => {}} ariaLabel="code" errorLine={3} />,
    );
    await waitFor(() => expect(handle.setErrorLine).toHaveBeenCalledWith(3));
    rerender(
      <CodeEditor value="x = 1" onChange={() => {}} ariaLabel="code" errorLine={null} />,
    );
    expect(handle.setErrorLine).toHaveBeenLastCalledWith(null);
  });

  it('destroys the editor on unmount', async () => {
    const { unmount } = render(
      <CodeEditor value="x = 1" onChange={() => {}} ariaLabel="code" />,
    );
    await waitFor(() => expect(lastOptions).not.toBeNull());
    unmount();
    expect(handle.destroy).toHaveBeenCalled();
  });

  it('passes read-only through and keeps the fallback read-only too', () => {
    render(<CodeEditor value="x = 1" onChange={() => {}} ariaLabel="code" readOnly />);
    expect((screen.getByLabelText('code') as HTMLTextAreaElement).readOnly).toBe(true);
  });
});
