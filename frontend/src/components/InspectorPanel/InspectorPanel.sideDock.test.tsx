/**
 * The Inspector gives way to an open side panel (apiVersion 6) on a window
 * narrower than 1680px: it appears as its 36px strip so the canvas keeps its
 * width, and the user can still open it.
 */
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react';
import { InspectorPanel } from './InspectorPanel';
import { useI18n } from '../../i18n';
import { useTabStore } from '../../store/tabStore';
import { _clearPluginPanels, registerPluginPanel } from '../../plugins/panels';
import { _resetSideDock, setSidePanelOpen } from '../../plugins/sideDock';

function setWidth(width: number) {
  Object.defineProperty(window, 'innerWidth', { value: width, configurable: true, writable: true });
  window.dispatchEvent(new Event('resize'));
}

const collapsed = () => screen.queryByLabelText('Expand inspector') !== null;

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  window.localStorage.clear();
  useTabStore.setState({ tabs: [], activeTabId: null as unknown as string, clipboard: null });
  useTabStore.getState().addTab('t');
  _clearPluginPanels();
  _resetSideDock();
  registerPluginPanel('agent', { id: 'chat', title: 'Agent', dock: 'side' });
  setWidth(1470);
});

afterEach(() => {
  cleanup();
  _clearPluginPanels();
  _resetSideDock();
  window.localStorage.clear();
});

describe('InspectorPanel beside a side panel', () => {
  it('starts collapsed when a side panel is open on a 1470px window', () => {
    setSidePanelOpen('agent:chat', true);
    render(<InspectorPanel />);
    expect(collapsed()).toBe(true);
    // Still the user's to open.
    fireEvent.click(screen.getByLabelText('Expand inspector'));
    expect(collapsed()).toBe(false);
  });

  it('stays open with no side panel, or on a wide window', () => {
    render(<InspectorPanel />);
    expect(collapsed()).toBe(false);
    cleanup();
    setWidth(1920);
    setSidePanelOpen('agent:chat', true);
    render(<InspectorPanel />);
    expect(collapsed()).toBe(false);
  });

  it('collapses when the side panel opens, and comes back when it closes', () => {
    render(<InspectorPanel />);
    act(() => { setSidePanelOpen('agent:chat', true); });
    expect(collapsed()).toBe(true);
    act(() => { setSidePanelOpen('agent:chat', false); });
    expect(collapsed()).toBe(false);
  });

  it('does not undo a collapse the user made', () => {
    render(<InspectorPanel />);
    fireEvent.click(screen.getByLabelText('Collapse inspector'));
    act(() => { setSidePanelOpen('agent:chat', true); });
    act(() => { setSidePanelOpen('agent:chat', false); });
    expect(collapsed()).toBe(true);
  });
});
