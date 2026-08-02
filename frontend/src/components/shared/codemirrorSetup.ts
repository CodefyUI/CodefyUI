/**
 * The CodeMirror 6 half of {@link CodeEditor} (core#131).
 *
 * Deliberately a SEPARATE module: `CodeEditor` reaches it through a dynamic
 * `import()`, and that import is the chunk boundary. Nothing here may be
 * imported statically from anywhere else, or CodeMirror lands in the entry
 * bundle and the lazy loading is undone silently.
 *
 * Only the pieces a small embedded editor needs are imported, so tree
 * shaking can drop the rest of the CodeMirror distribution: no search
 * panel, no autocompletion, no linting, no folding.
 */
import {
  Annotation,
  EditorState,
  StateEffect,
  StateField,
  type Extension,
} from '@codemirror/state';
import {
  Decoration,
  EditorView,
  keymap,
  lineNumbers,
  highlightActiveLine,
  highlightActiveLineGutter,
  type DecorationSet,
} from '@codemirror/view';
import { defaultKeymap, history, historyKeymap, indentWithTab } from '@codemirror/commands';
import { HighlightStyle, indentUnit, syntaxHighlighting } from '@codemirror/language';
import { python } from '@codemirror/lang-python';
import { tags } from '@lezer/highlight';

/** Crafted-dark syntax colours, matching the app's editor chrome. */
const highlightStyle = HighlightStyle.define([
  { tag: tags.keyword, color: '#c792ea' },
  { tag: [tags.controlKeyword, tags.moduleKeyword], color: '#c792ea' },
  { tag: [tags.definitionKeyword, tags.operatorKeyword], color: '#06b6d4' },
  { tag: [tags.function(tags.variableName), tags.function(tags.propertyName)], color: '#82aaff' },
  { tag: [tags.string, tags.special(tags.string)], color: '#c3e88d' },
  { tag: [tags.number, tags.bool, tags.null], color: '#f78c6c' },
  { tag: tags.comment, color: '#6b7280', fontStyle: 'italic' },
  { tag: [tags.className, tags.typeName], color: '#ffcb6b' },
  { tag: tags.propertyName, color: '#e5e7eb' },
  { tag: tags.operator, color: '#89ddff' },
  { tag: tags.variableName, color: '#e5e7eb' },
]);

const theme = EditorView.theme(
  {
    '&': {
      color: '#e5e7eb',
      backgroundColor: 'rgba(0, 0, 0, 0.25)',
      fontSize: '12px',
      height: '100%',
    },
    '.cm-content': {
      fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace',
      caretColor: '#06b6d4',
      padding: '6px 0',
    },
    '.cm-scroller': { overflow: 'auto', lineHeight: '1.55' },
    '.cm-gutters': {
      backgroundColor: 'transparent',
      color: '#4b5563',
      border: 'none',
      borderRight: '1px solid #1f2937',
    },
    '.cm-activeLine': { backgroundColor: 'rgba(6, 182, 212, 0.06)' },
    '.cm-activeLineGutter': { backgroundColor: 'transparent', color: '#9ca3af' },
    '&.cm-focused': { outline: 'none' },
    '&.cm-focused .cm-cursor': { borderLeftColor: '#06b6d4' },
    '&.cm-focused .cm-selectionBackground, .cm-selectionBackground, ::selection': {
      backgroundColor: 'rgba(6, 182, 212, 0.22)',
    },
    // The line the server's policy check pointed at.
    '.cm-errorLine': {
      backgroundColor: 'rgba(239, 68, 68, 0.16)',
      boxShadow: 'inset 2px 0 0 #ef4444',
    },
  },
  { dark: true },
);

const setErrorLine = StateEffect.define<number | null>();

/**
 * Marks a change this module made itself, rather than one the user typed.
 *
 * Two editors are mounted at once (the param column and the Code tab) and
 * both are controlled by the same store value, so every keystroke in one is
 * echoed into the other through `setValue`. Without this annotation the echo
 * looks exactly like typing: the update listener fires, calls `onChange`,
 * and writes the same text back to the store a second time — doubling the
 * store writes, the undo entries and the validation round-trips for every
 * character.
 */
const programmatic = Annotation.define<boolean>();

/**
 * Escape moves focus OUT of the editor (WCAG 2.1.2, No Keyboard Trap).
 *
 * `indentWithTab` deliberately takes Tab away from the browser's focus
 * order — correct for a code editor, and a trap without a documented way
 * out. In the Node Detail Modal a second Escape then closes the modal; in
 * the side config panel, where nothing else listens for Escape, this is the
 * only way out for a keyboard user.
 */
const escapeBlur = {
  key: 'Escape',
  run: (view: EditorView) => {
    view.contentDOM.blur();
    // Hand focus to the nearest container that can hold it, so the next
    // Escape reaches the modal instead of the document body.
    view.dom.closest<HTMLElement>('[tabindex]')?.focus();
    return true;
  },
};

/**
 * Highlight one line, by 1-based number.
 *
 * Held in a StateField rather than re-created per render so that ordinary
 * typing (which shifts every position after the edit) moves the mark with
 * the document instead of leaving it stranded on a line number that no
 * longer means anything.
 */
const errorLineField = StateField.define<DecorationSet>({
  create: () => Decoration.none,
  update(value, transaction) {
    let next = value.map(transaction.changes);
    for (const effect of transaction.effects) {
      if (!effect.is(setErrorLine)) continue;
      const line = effect.value;
      if (line === null || line < 1 || line > transaction.state.doc.lines) {
        next = Decoration.none;
      } else {
        next = Decoration.set([
          Decoration.line({ class: 'cm-errorLine' }).range(
            transaction.state.doc.line(line).from,
          ),
        ]);
      }
    }
    return next;
  },
  provide: (field) => EditorView.decorations.from(field),
});

export interface PythonEditorOptions {
  parent: HTMLElement;
  doc: string;
  readOnly: boolean;
  onChange: (next: string) => void;
  ariaLabel: string;
}

export interface PythonEditorHandle {
  setValue: (next: string) => void;
  setErrorLine: (line: number | null) => void;
  destroy: () => void;
}

export function createPythonEditor(options: PythonEditorOptions): PythonEditorHandle {
  const extensions: Extension[] = [
    lineNumbers(),
    highlightActiveLine(),
    highlightActiveLineGutter(),
    history(),
    // `escapeBlur` first so it wins over the default Escape binding
    // (simplifySelection); `indentWithTab` last, after the default keymap,
    // which has no Tab binding of its own.
    keymap.of([escapeBlur, ...defaultKeymap, ...historyKeymap, indentWithTab]),
    python(),
    indentUnit.of('    '),
    syntaxHighlighting(highlightStyle),
    errorLineField,
    theme,
    EditorView.lineWrapping,
    EditorView.editable.of(!options.readOnly),
    EditorState.readOnly.of(options.readOnly),
    EditorView.contentAttributes.of({
      'aria-label': options.ariaLabel,
      // Python is whitespace-significant; a helpful autocorrect is not.
      autocorrect: 'off',
      autocapitalize: 'off',
      spellcheck: 'false',
    }),
    EditorView.updateListener.of((update) => {
      if (!update.docChanged) return;
      // An echo of the store's own value is not an edit; reporting it would
      // write the same text back and re-run validation for nothing.
      if (update.transactions.some((tr) => tr.annotation(programmatic))) return;
      options.onChange(update.state.doc.toString());
    }),
  ];

  const view = new EditorView({
    parent: options.parent,
    state: EditorState.create({ doc: options.doc, extensions }),
  });

  return {
    setValue(next: string) {
      // Only when it actually differs: the parent echoes every keystroke
      // back, and dispatching that would reset the selection on every
      // character typed.
      if (next === view.state.doc.toString()) return;
      view.dispatch({
        changes: { from: 0, to: view.state.doc.length, insert: next },
        annotations: programmatic.of(true),
      });
    },
    setErrorLine(line: number | null) {
      view.dispatch({ effects: setErrorLine.of(line) });
    },
    destroy() {
      view.destroy();
    },
  };
}
