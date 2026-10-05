import { useState, type CSSProperties } from 'react';

interface NumberInputProps {
  /** The stored value: a number, a numeric string, or nothing yet. */
  value: unknown;
  /** Called with a finite number only, and only with an integer when `integer`. */
  onCommit: (n: number) => void;
  integer?: boolean;
  min?: number;
  max?: number;
  step?: number | 'any';
  className?: string;
  style?: CSSProperties;
  'aria-label'?: string;
}

/** The text a stored value is shown as. */
function asText(value: unknown): string {
  return value == null ? '' : String(value);
}

/** The number a draft commits, or null while it is still being typed. */
function parseDraft(draft: string, integer: boolean): number | null {
  // Number, not parseInt: parseInt reads "1e3" as 1 and "2.5" as 2. And
  // Number("") is 0, so an empty draft is ruled out first.
  if (draft.trim() === '') return null;
  const n = Number(draft);
  if (!Number.isFinite(n) || (integer && !Number.isInteger(n))) return null;
  return n;
}

/** Whether the draft already shows `value`, however it is written (1.50, 1e-3). */
function represents(draft: string, value: unknown): boolean {
  if (value == null) return draft.trim() === '';
  return draft.trim() !== '' && Number(draft) === Number(value);
}

/**
 * A number field that takes a number typed one key at a time.
 *
 * While a field holds "-", "-0." or "1e-", the browser reports its value as
 * "" because what is typed is not a number yet. A number input bound straight
 * to the stored value broke on that twice over: the "" was parsed and
 * committed (NaN from parseInt / parseFloat, 0 from Number), and React then
 * wrote the stored value back into the field, wiping the "-". Typing -1
 * stored 1 (UAT of 2.8.8).
 *
 * So the input is bound to a local draft, which is exactly what the browser
 * reports and leaves React nothing to write back, and only a finished number
 * is committed. A half-typed or cleared field writes nothing to the graph.
 */
export function NumberInput({
  value,
  onCommit,
  integer = false,
  min,
  max,
  step,
  className,
  style,
  'aria-label': ariaLabel,
}: NumberInputProps) {
  const [draft, setDraft] = useState(() => asText(value));
  // The stored value the draft was last brought in line with. Boxed, so no
  // value is ever taken for a state updater.
  const [seen, setSeen] = useState(() => ({ value }));

  // A change from outside (undo, a load, the other panel editing the same
  // param, the tensor grid's reshape) replaces the draft. Keyed on the stored
  // value CHANGING, never on it differing from the draft: while "-" is on
  // screen the draft is "" and the stored value has not moved, and deriving
  // the draft from it again would wipe the "-". Object.is, so a stored NaN is
  // not a new value on every render.
  if (!Object.is(value, seen.value)) {
    setSeen({ value });
    if (!represents(draft, value)) setDraft(asText(value));
  }

  return (
    <input
      type="number"
      value={draft}
      min={min}
      max={max}
      step={step}
      className={className}
      style={style}
      aria-label={ariaLabel}
      onChange={(e) => {
        setDraft(e.target.value);
        const n = parseDraft(e.target.value, integer);
        if (n !== null) onCommit(n);
      }}
      // Leaving the field shows what is stored: a field cleared and left gets
      // its number back, and 1.50 settles to 1.5.
      onBlur={() => setDraft(asText(value))}
    />
  );
}
