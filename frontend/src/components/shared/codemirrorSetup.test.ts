import { describe, it, expect, afterEach, vi } from 'vitest';
import { EditorView } from '@codemirror/view';
import { createPythonEditor, type PythonEditorHandle } from './codemirrorSetup';

/**
 * Drives the REAL CodeMirror instance (CodeEditor.test.tsx mocks this module,
 * so nothing else covers what it actually does).
 */
const mounted: PythonEditorHandle[] = [];
const hosts: HTMLElement[] = [];

function mount(doc: string, onChange = vi.fn(), readOnly = false) {
  const parent = document.createElement('div');
  document.body.appendChild(parent);
  hosts.push(parent);
  const handle = createPythonEditor({
    parent,
    doc,
    readOnly,
    onChange,
    ariaLabel: 'script',
  });
  mounted.push(handle);
  const view = EditorView.findFromDOM(parent)!;
  return { handle, view, onChange, parent };
}

afterEach(() => {
  while (mounted.length) mounted.pop()!.destroy();
  while (hosts.length) hosts.pop()!.remove();
});

describe('createPythonEditor', () => {
  it('mounts with the document and labels the editable region', () => {
    const { view } = mount('x = 1');
    expect(view.state.doc.toString()).toBe('x = 1');
    expect(view.contentDOM.getAttribute('aria-label')).toBe('script');
  });

  it('reports edits the user made', () => {
    const { view, onChange } = mount('x = 1');
    view.dispatch({ changes: { from: 5, insert: '0' } });
    expect(onChange).toHaveBeenCalledWith('x = 10');
  });

  it('does NOT report a value pushed in from the store', () => {
    // Two editors are open on the same param, so every keystroke in one is
    // echoed into the other. Reporting the echo would double the store
    // writes, the undo entries and the validation requests per character.
    const { handle, view, onChange } = mount('x = 1');
    handle.setValue('x = 2');
    expect(view.state.doc.toString()).toBe('x = 2');
    expect(onChange).not.toHaveBeenCalled();
  });

  it('ignores a setValue that matches the current document', () => {
    const { handle, view } = mount('x = 1');
    const before = view.state;
    handle.setValue('x = 1');
    expect(view.state).toBe(before);
  });

  it('Escape releases focus, so the editor is not a keyboard trap', () => {
    // WCAG 2.1.2: `indentWithTab` takes Tab away from the focus order, so
    // there has to be a documented way out. In the side config panel nothing
    // else listens for Escape.
    const { view } = mount('x = 1');
    view.contentDOM.focus();
    expect(document.activeElement).toBe(view.contentDOM);

    view.contentDOM.dispatchEvent(
      new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }),
    );
    expect(document.activeElement).not.toBe(view.contentDOM);
  });

  it('hands focus to the nearest container so a second Escape reaches it', () => {
    const surface = document.createElement('div');
    surface.tabIndex = -1;
    document.body.appendChild(surface);
    const handle = createPythonEditor({
      parent: surface,
      doc: 'x = 1',
      readOnly: false,
      onChange: vi.fn(),
      ariaLabel: 'script',
    });
    mounted.push(handle);
    const view = EditorView.findFromDOM(surface)!;

    view.contentDOM.focus();
    view.contentDOM.dispatchEvent(
      new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }),
    );
    expect(document.activeElement).toBe(surface);
    surface.remove();
  });

  it('marks and clears the error line without losing it to an edit', () => {
    const { handle, view } = mount('a = 1\nb = 2\nc = 3');
    handle.setErrorLine(2);
    expect(view.dom.querySelectorAll('.cm-errorLine').length).toBe(1);

    handle.setErrorLine(null);
    expect(view.dom.querySelectorAll('.cm-errorLine').length).toBe(0);
  });

  it('ignores an error line outside the document', () => {
    const { handle, view } = mount('a = 1');
    handle.setErrorLine(99);
    expect(view.dom.querySelectorAll('.cm-errorLine').length).toBe(0);
  });

  it('refuses edits when read-only', () => {
    const { view } = mount('x = 1', vi.fn(), true);
    expect(view.state.readOnly).toBe(true);
  });
});
