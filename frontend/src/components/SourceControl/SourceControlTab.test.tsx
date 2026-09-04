import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { SourceControlTab } from './SourceControlTab';
import { useI18n } from '../../i18n';

/*
 * The placeholder panel: it exists so the rail entry, the panel slot and the
 * title can be wired and tested before the tab itself is built. What is pinned
 * here is the two things the shell depends on — the heading it shows, and that
 * opening the tab costs nothing yet. Both move on when the real panel replaces
 * the body.
 */

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('SourceControlTab (placeholder)', () => {
  it('renders the tab title', () => {
    render(<SourceControlTab />);
    expect(screen.getByText('Source Control')).toBeTruthy();
  });

  it('translates the title', () => {
    useI18n.setState({ locale: 'zh-TW' });
    render(<SourceControlTab />);
    expect(screen.getByText('版本控制')).toBeTruthy();
  });

  it('attaches nothing yet: no controls, no request, no poll', () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response('{}'));
    const intervalSpy = vi.spyOn(globalThis, 'setInterval');
    const listenSpy = vi.spyOn(document, 'addEventListener');

    const { unmount } = render(<SourceControlTab />);

    expect(screen.queryAllByRole('button')).toHaveLength(0);
    expect(fetchSpy).not.toHaveBeenCalled();
    // The real tab starts a 15s poll and listens for focus / visibilitychange
    // from its own effect. Until it does, opening the tab must be free.
    expect(intervalSpy).not.toHaveBeenCalled();
    expect(listenSpy).not.toHaveBeenCalled();
    unmount();
  });
});
