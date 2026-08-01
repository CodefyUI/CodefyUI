import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, fireEvent, act } from '@testing-library/react';
import type { ParamDefinition } from '../../types';
import { useI18n } from '../../i18n';

vi.mock('../../api/rest', () => ({ validateScript: vi.fn() }));

// The editor itself is covered by CodeEditor.test.tsx; here it is a plain
// textarea so a test can type without driving CodeMirror.
vi.mock('./CodeEditor', () => ({
  CodeEditor: ({ value, onChange, ariaLabel, errorLine }: any) => (
    <textarea
      aria-label={ariaLabel}
      data-error-line={errorLine ?? ''}
      value={value}
      onChange={(e) => onChange(e.target.value)}
    />
  ),
}));

import { validateScript } from '../../api/rest';
import { resetScriptValidationCache } from '../../hooks/useScriptValidation';
import { ScriptCodeField } from './ScriptCodeField';

const param: ParamDefinition = {
  name: 'code',
  param_type: 'code',
  default: 'def run(inputs, params):\n    return 1\n',
  description: '',
  options: [],
  min_value: null,
  max_value: null,
};

const ALLOWED = ['collections', 'json', 'math', 'numpy', 'torch'];

function ok(overrides: Record<string, unknown> = {}) {
  return {
    ok: true,
    error: null,
    line: null,
    defines_run: true,
    allowed_modules: ALLOWED,
    ...overrides,
  };
}

/** Let the debounce fire and the verdict promise settle. */
async function settle() {
  await act(async () => {
    vi.advanceTimersByTime(500);
    await Promise.resolve();
    await Promise.resolve();
  });
}

beforeEach(() => {
  vi.useFakeTimers();
  useI18n.setState({ locale: 'en' });
  resetScriptValidationCache();
  vi.mocked(validateScript).mockReset();
  vi.mocked(validateScript).mockResolvedValue(ok() as any);
});

afterEach(() => {
  vi.useRealTimers();
});

describe('ScriptCodeField', () => {
  it('validates the script and reports that the policy accepted it', async () => {
    render(
      <ScriptCodeField
        param={param}
        value={param.default}
        onChange={() => {}}
        displayLabel="code"
      />,
    );
    await settle();
    expect(validateScript).toHaveBeenCalledWith(param.default);
    expect(screen.getByTestId('script-policy-badge').textContent).toBe('policy OK');
  });

  it('surfaces a policy rejection with its line number as soon as it is typed', async () => {
    vi.mocked(validateScript).mockResolvedValue(
      ok({
        ok: false,
        error: "Importing 'requests' is not allowed in <PythonScript>. Write a custom node instead.",
        line: 2,
        defines_run: true,
      }) as any,
    );
    render(
      <ScriptCodeField
        param={param}
        value={'x = 1\nimport requests\n'}
        onChange={() => {}}
        displayLabel="code"
      />,
    );
    await settle();

    const alert = screen.getByRole('alert');
    expect(alert.textContent).toContain('Line 2:');
    expect(alert.textContent).toContain("Importing 'requests' is not allowed");
    expect(screen.getByTestId('script-policy-badge').textContent).toBe('rejected');
    // The editor gets the line so it can mark it.
    expect(screen.getByLabelText('code').getAttribute('data-error-line')).toBe('2');
  });

  it('re-validates after an edit and clears the banner when it passes', async () => {
    vi.mocked(validateScript).mockResolvedValueOnce(
      ok({ ok: false, error: 'nope', line: 1 }) as any,
    );
    const onChange = vi.fn();
    const { rerender } = render(
      <ScriptCodeField
        param={param}
        value={'import requests\n'}
        onChange={onChange}
        displayLabel="code"
      />,
    );
    await settle();
    expect(screen.getByRole('alert')).toBeTruthy();

    fireEvent.change(screen.getByLabelText('code'), {
      target: { value: 'import numpy\n' },
    });
    expect(onChange).toHaveBeenCalledWith('code', 'import numpy\n');

    // The parent owns the value; echo the edit back as the real store does.
    rerender(
      <ScriptCodeField
        param={param}
        value={'import numpy\n'}
        onChange={onChange}
        displayLabel="code"
      />,
    );
    await settle();
    expect(screen.queryByRole('alert')).toBeNull();
    expect(validateScript).toHaveBeenLastCalledWith('import numpy\n');
  });

  it('debounces: typing three characters costs one request', async () => {
    const { rerender } = render(
      <ScriptCodeField param={param} value="a" onChange={() => {}} displayLabel="code" />,
    );
    for (const value of ['ab', 'abc']) {
      act(() => {
        vi.advanceTimersByTime(100);
      });
      rerender(
        <ScriptCodeField param={param} value={value} onChange={() => {}} displayLabel="code" />,
      );
    }
    await settle();
    expect(validateScript).toHaveBeenCalledTimes(1);
    expect(validateScript).toHaveBeenCalledWith('abc');
  });

  it('reuses a cached verdict instead of asking again', async () => {
    const { rerender } = render(
      <ScriptCodeField param={param} value="a" onChange={() => {}} displayLabel="code" />,
    );
    await settle();
    rerender(
      <ScriptCodeField param={param} value="b" onChange={() => {}} displayLabel="code" />,
    );
    await settle();
    rerender(
      <ScriptCodeField param={param} value="a" onChange={() => {}} displayLabel="code" />,
    );
    await settle();
    expect(validateScript).toHaveBeenCalledTimes(2);
  });

  it('warns when the script defines no run() yet', async () => {
    vi.mocked(validateScript).mockResolvedValue(ok({ defines_run: false }) as any);
    render(
      <ScriptCodeField param={param} value="x = 1" onChange={() => {}} displayLabel="code" />,
    );
    await settle();
    expect(screen.getByText(/No run\(inputs, params\) defined yet/)).toBeTruthy();
  });

  it('lists the allowlist the server reported', async () => {
    render(
      <ScriptCodeField param={param} value="x = 1" onChange={() => {}} displayLabel="code" />,
    );
    await settle();
    expect(screen.getByText(/Imports allowed:/).textContent).toContain('numpy');
  });

  it('stays quiet and editable when the policy check cannot be reached', async () => {
    vi.mocked(validateScript).mockRejectedValue(new Error('offline'));
    render(
      <ScriptCodeField param={param} value="x = 1" onChange={() => {}} displayLabel="code" />,
    );
    await settle();
    expect(screen.getByTestId('script-policy-badge').textContent).toBe('check unavailable');
    expect(screen.queryByRole('alert')).toBeNull();
    expect(screen.getByLabelText('code')).toBeTruthy();
  });
});
