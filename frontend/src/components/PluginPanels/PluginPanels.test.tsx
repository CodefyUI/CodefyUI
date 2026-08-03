/**
 * The panel-persistence contract, tested where it is actually at risk.
 *
 * The published promise is "the element persists across tab switches"; the
 * dock's own rule (and the Node Detail Modal's) is "an inactive tab body is
 * UNMOUNTED". These tests hold both at once: they mount and unmount the
 * adapter the way the dock does, and check that what comes back is the same
 * element with the same children, not a rebuilt one.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { StrictMode, useState } from 'react';
import { render, screen, fireEvent, act } from '@testing-library/react';
import { useI18n } from '../../i18n';
import {
  _clearPluginPanels, registerPluginPanel, removePluginPanel,
} from '../../plugins/panels';
import { PluginDockPanel, PluginRightPanels } from './PluginPanels';

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  _clearPluginPanels();
});

afterEach(() => {
  _clearPluginPanels();
  vi.restoreAllMocks();
});

/** A host that mounts the adapter only while its "tab" is active. */
function DockHarness({ panelKey }: { panelKey: string }) {
  const [active, setActive] = useState(true);
  return (
    <div>
      <button type="button" onClick={() => setActive((v) => !v)}>toggle</button>
      {active && <PluginDockPanel panelKey={panelKey} />}
    </div>
  );
}

describe('PluginDockPanel', () => {
  it('attaches the plugin element while mounted', () => {
    const el = registerPluginPanel('p', { id: 'x', title: 'X' });
    el.textContent = 'plugin body';
    render(<PluginDockPanel panelKey="p:x" />);
    expect(screen.getByTestId('plugin-dock-panel-p:x').contains(el)).toBe(true);
    expect(el.isConnected).toBe(true);
  });

  it('detaches on unmount but keeps the element and its children alive', () => {
    const el = registerPluginPanel('p', { id: 'x', title: 'X' });
    const child = document.createElement('canvas');
    el.appendChild(child);

    render(<DockHarness panelKey="p:x" />);
    expect(el.isConnected).toBe(true);

    fireEvent.click(screen.getByText('toggle'));
    expect(el.isConnected).toBe(false);
    expect(el.firstChild).toBe(child);
    expect(screen.queryByTestId('plugin-dock-panel-p:x')).not.toBeInTheDocument();
  });

  it('re-attaches the SAME element on the way back — the contract', () => {
    const el = registerPluginPanel('p', { id: 'x', title: 'X' });
    // Stand-in for a plugin's React root: mounted once, never re-created.
    const root = document.createElement('div');
    root.textContent = 'expensive';
    el.appendChild(root);

    render(<DockHarness panelKey="p:x" />);
    fireEvent.click(screen.getByText('toggle'));   // leave the tab
    fireEvent.click(screen.getByText('toggle'));   // come back

    const host = screen.getByTestId('plugin-dock-panel-p:x');
    expect(host.firstChild).toBe(el);
    expect(el.firstChild).toBe(root);
    expect(el.textContent).toBe('expensive');
  });

  it('survives many switches without duplicating the element', () => {
    const el = registerPluginPanel('p', { id: 'x', title: 'X' });
    render(<DockHarness panelKey="p:x" />);
    for (let i = 0; i < 10; i += 1) {
      fireEvent.click(screen.getByText('toggle'));
      fireEvent.click(screen.getByText('toggle'));
    }
    const host = screen.getByTestId('plugin-dock-panel-p:x');
    expect(host.childNodes).toHaveLength(1);
    expect(host.firstChild).toBe(el);
    expect(document.querySelectorAll('#plugin-panel-p\\:x')).toHaveLength(1);
  });

  it('fires onShow when attached and onHide when detached', () => {
    const calls: string[] = [];
    registerPluginPanel('p', {
      id: 'x', title: 'X',
      onShow: () => calls.push('show'),
      onHide: () => calls.push('hide'),
    });
    render(<DockHarness panelKey="p:x" />);
    expect(calls).toEqual(['show']);
    fireEvent.click(screen.getByText('toggle'));
    expect(calls).toEqual(['show', 'hide']);
    fireEvent.click(screen.getByText('toggle'));
    expect(calls).toEqual(['show', 'hide', 'show']);
  });

  it('attaches exactly once under StrictMode, which double-invokes effects', () => {
    const el = registerPluginPanel('p', { id: 'x', title: 'X' });
    const calls: string[] = [];
    registerPluginPanel('p', {
      id: 'x', title: 'X',
      onShow: () => calls.push('show'),
      onHide: () => calls.push('hide'),
    });
    render(
      <StrictMode>
        <PluginDockPanel panelKey="p:x" />
      </StrictMode>,
    );
    const host = screen.getByTestId('plugin-dock-panel-p:x');
    expect(host.childNodes).toHaveLength(1);
    expect(host.firstChild).toBe(el);
    // The dev-only replay is visible to the plugin, so the callbacks have to
    // be idempotent — which is exactly what the contract asks of them.
    expect(calls[calls.length - 1]).toBe('show');
  });

  it('renders an empty host for a panel that is not registered', () => {
    expect(() => render(<PluginDockPanel panelKey="ghost:x" />)).not.toThrow();
    expect(screen.getByTestId('plugin-dock-panel-ghost:x').childNodes).toHaveLength(0);
  });

  it('a panel unregistered while attached leaves no element behind', () => {
    const el = registerPluginPanel('p', { id: 'x', title: 'X' });
    render(<PluginDockPanel panelKey="p:x" />);
    expect(el.isConnected).toBe(true);
    act(() => removePluginPanel('p', 'x'));
    expect(el.isConnected).toBe(false);
  });
});

describe('PluginRightPanels', () => {
  it('renders nothing at all when no plugin asked for a right panel', () => {
    registerPluginPanel('p', { id: 'bottom', title: 'Bottom' });
    const { container } = render(<PluginRightPanels />);
    expect(container).toBeEmptyDOMElement();
  });

  it('stacks every right panel with its title, in registration order', () => {
    registerPluginPanel('a', { id: 'x', title: 'First', dock: 'right' });
    registerPluginPanel('b', { id: 'y', title: 'Second', dock: 'right', icon: '+' });
    registerPluginPanel('c', { id: 'z', title: 'Not here', dock: 'bottom' });
    render(<PluginRightPanels />);

    const headings = screen.getAllByRole('complementary')[0].querySelectorAll('header');
    expect([...headings].map((h) => h.textContent)).toEqual(['First', '+Second']);
    expect(screen.queryByTestId('plugin-right-panel-c:z')).not.toBeInTheDocument();
  });

  it('attaches each element into its own section', () => {
    const first = registerPluginPanel('a', { id: 'x', title: 'First', dock: 'right' });
    const second = registerPluginPanel('b', { id: 'y', title: 'Second', dock: 'right' });
    render(<PluginRightPanels />);
    expect(screen.getByTestId('plugin-right-panel-a:x').firstChild).toBe(first);
    expect(screen.getByTestId('plugin-right-panel-b:y').firstChild).toBe(second);
  });

  it('picks up a panel registered after the first render', () => {
    render(<PluginRightPanels />);
    act(() => { registerPluginPanel('a', { id: 'x', title: 'Late', dock: 'right' }); });
    expect(screen.getByTestId('plugin-right-panel-a:x')).toBeInTheDocument();
  });

  it('detaches on unmount', () => {
    const el = registerPluginPanel('a', { id: 'x', title: 'First', dock: 'right' });
    const { unmount } = render(<PluginRightPanels />);
    expect(el.isConnected).toBe(true);
    unmount();
    expect(el.isConnected).toBe(false);
  });
});
