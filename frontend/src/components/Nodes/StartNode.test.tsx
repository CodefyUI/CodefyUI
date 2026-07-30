import { describe, it, expect, beforeEach } from 'vitest';
import { screen } from '@testing-library/react';
import { StartNode } from './StartNode';
import { nodeProps, renderWithFlow } from '../../test/utils';
import { useI18n } from '../../i18n';
import { useUIStore } from '../../store/uiStore';

describe('StartNode', () => {
  beforeEach(() => {
    useI18n.setState({ locale: 'en' });
    useUIStore.setState({ reconnectingHandle: null });
  });

  it('renders the localized start label and a source handle', () => {
    const { container } = renderWithFlow(
      <StartNode {...nodeProps({ id: 's', type: 'start', data: {} })} />,
    );
    expect(screen.getByText(useI18n.getState().t('node.start.label'))).toBeTruthy();
    // The source handle is rendered by React Flow.
    expect(container.querySelector('.react-flow__handle')).toBeTruthy();
  });

  it('adds handleDetaching while its trigger source end is being detached', () => {
    useUIStore.setState({
      reconnectingHandle: { nodeId: 's', handleId: 'trigger', type: 'source' },
    });
    const { container } = renderWithFlow(
      <StartNode {...nodeProps({ id: 's', type: 'start', data: {} })} />,
    );
    const handle = container.querySelector('.react-flow__handle') as HTMLElement;
    expect(handle.className).toMatch(/handleDetaching/);
  });

  it('does not add handleDetaching when the reconnect concerns another node', () => {
    useUIStore.setState({
      reconnectingHandle: { nodeId: 'other', handleId: 'trigger', type: 'source' },
    });
    const { container } = renderWithFlow(
      <StartNode {...nodeProps({ id: 's', type: 'start', data: {} })} />,
    );
    const handle = container.querySelector('.react-flow__handle') as HTMLElement;
    expect(handle.className).not.toMatch(/handleDetaching/);
  });
});
