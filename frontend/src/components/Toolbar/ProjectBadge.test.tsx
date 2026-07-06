import { describe, it, expect, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ProjectBadge } from './ProjectBadge';
import { useProjectStore } from '../../store/projectStore';

beforeEach(() => useProjectStore.setState({ projectDir: null, projectName: null, loaded: false }));

describe('ProjectBadge', () => {
  it('renders nothing in non-project mode', () => {
    const { container } = render(<ProjectBadge />);
    expect(container.firstChild).toBeNull();
  });
  it('shows the project name in project mode', () => {
    useProjectStore.getState().setProject('/home/me/my-service');
    render(<ProjectBadge />);
    expect(screen.queryByText('my-service')).not.toBeNull();
  });
});
