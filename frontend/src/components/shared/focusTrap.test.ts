import { describe, it, expect, vi } from 'vitest';
import { trapTab } from './focusTrap';

describe('trapTab', () => {
  it('does nothing without a container or anything focusable in it', () => {
    const event = { key: 'Tab', shiftKey: false, preventDefault: vi.fn() };
    trapTab(event as never, null);
    trapTab(event as never, document.createElement('div'));
    expect(event.preventDefault).not.toHaveBeenCalled();
  });

  it('ignores keys other than Tab', () => {
    const container = document.createElement('div');
    container.innerHTML = '<button>a</button>';
    document.body.appendChild(container);
    (container.querySelector('button') as HTMLButtonElement).focus();
    const event = { key: 'Enter', shiftKey: false, preventDefault: vi.fn() };
    trapTab(event as never, container);
    expect(event.preventDefault).not.toHaveBeenCalled();
    container.remove();
  });

  it('skips controls folded away in a closed details, never its summary (#845)', () => {
    const container = document.createElement('div');
    container.innerHTML = `
      <button id="first">first</button>
      <details id="outer" open>
        <summary id="outer-summary">outer</summary>
        <details id="inner">
          <summary id="inner-summary">inner</summary>
          <button id="folded">folded</button>
        </details>
        <button id="shown">shown</button>
      </details>
      <details id="closed"><button id="no-summary">no summary</button></details>`;
    document.body.appendChild(container);
    const byId = (id: string) => container.querySelector(`#${id}`) as HTMLElement;
    const tab = (shiftKey = false) => {
      const event = { key: 'Tab', shiftKey, preventDefault: vi.fn() };
      trapTab(event as never, container);
      return event.preventDefault;
    };
    // The last control is the one shown last, not the folded ones after it.
    byId('shown').focus();
    expect(tab()).toHaveBeenCalled();
    expect(byId('first')).toHaveFocus();
    expect(tab(true)).toHaveBeenCalled();
    expect(byId('shown')).toHaveFocus();
    // Folding the outer one hides everything in it but its own summary.
    (byId('outer') as HTMLDetailsElement).open = false;
    byId('outer-summary').focus();
    expect(tab()).toHaveBeenCalled();
    expect(byId('first')).toHaveFocus();
    byId('inner-summary').focus();
    expect(tab()).not.toHaveBeenCalled();
    container.remove();
  });
});
