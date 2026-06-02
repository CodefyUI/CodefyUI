import { describe, it, expect } from 'vitest';
import { screen } from '@testing-library/react';
import { StartNode } from './StartNode';
import { nodeProps, renderWithFlow } from '../../test/utils';
import { useI18n } from '../../i18n';

describe('StartNode', () => {
  it('renders the localized start label and a source handle', () => {
    useI18n.setState({ locale: 'en' });
    const { container } = renderWithFlow(
      <StartNode {...nodeProps({ id: 's', type: 'start', data: {} })} />,
    );
    expect(screen.getByText(useI18n.getState().t('node.start.label'))).toBeTruthy();
    // The source handle is rendered by React Flow.
    expect(container.querySelector('.react-flow__handle')).toBeTruthy();
  });
});
