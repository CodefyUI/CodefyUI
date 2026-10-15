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
});
