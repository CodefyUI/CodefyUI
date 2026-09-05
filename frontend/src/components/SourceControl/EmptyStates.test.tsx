import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { EmptyStates } from './EmptyStates';
import { useI18n } from '../../i18n';

/*
 * The four screens the tab shows when there is no repository to show. Each is
 * a sentence and the single next step; what is pinned here is that the next
 * step is the RIGHT one -- an Initialize button only where init can work, and
 * two commands where nothing in the browser can help at all.
 */

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
});

describe('EmptyStates: no project directory', () => {
  it('shows the two commands and the guide, and no button that cannot work', () => {
    render(
      <EmptyStates
        state="no_project"
        nestedToplevel={null}
        gitVersion={null}
        onInit={vi.fn()}
      />,
    );
    expect(screen.getByText('Source control needs a project directory.')).toBeTruthy();
    expect(screen.getByText('Create one and start the server on it:')).toBeTruthy();
    expect(screen.getByText('cdui project init my-project')).toBeTruthy();
    expect(screen.getByText('cdui start --project my-project')).toBeTruthy();
    expect(screen.queryByRole('button', { name: 'Initialize Repository' })).toBeNull();

    // Through `SCM_DOCS_PATH`, so this screen and the More menu's row cannot
    // point at two different pages.
    const guide = screen.getByRole('link', { name: 'Setup guide' });
    expect(guide.getAttribute('href')).toBe(
      'https://docs.codefyui.com/usage/source-control',
    );
    expect(guide.getAttribute('rel')).toBe('noopener noreferrer');
  });
});

describe('EmptyStates: not a repository', () => {
  it('offers the one button that fixes it', () => {
    const onInit = vi.fn();
    render(
      <EmptyStates
        state="not_repo"
        nestedToplevel={null}
        gitVersion="2.45.0"
        onInit={onInit}
      />,
    );
    expect(screen.getByText('This project is not a git repository yet.')).toBeTruthy();
    screen.getByRole('button', { name: 'Initialize Repository' }).click();
    expect(onInit).toHaveBeenCalledTimes(1);
  });

  it('says a repository already surrounds this one before offering to add another', () => {
    render(
      <EmptyStates
        state="not_repo"
        nestedToplevel="D:/work/monorepo"
        gitVersion="2.45.0"
        onInit={vi.fn()}
      />,
    );
    expect(
      screen.getByText(
        'It sits inside another repository (D:/work/monorepo); initializing creates a separate one here.',
      ),
    ).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Initialize Repository' })).toBeTruthy();
  });
});

describe('EmptyStates: the server computer\'s git', () => {
  it('reports a missing git with the one thing to do about it', () => {
    render(
      <EmptyStates
        state="git_missing"
        nestedToplevel={null}
        gitVersion={null}
        onInit={vi.fn()}
      />,
    );
    expect(screen.getByText('git is not installed on the server computer.')).toBeTruthy();
    expect(screen.getByText('Install it, then restart the server.')).toBeTruthy();
    expect(screen.queryByRole('button')).toBeNull();
  });

  it('names the version that is too old', () => {
    render(
      <EmptyStates
        state="git_too_old"
        nestedToplevel={null}
        gitVersion="2.19.1"
        onInit={vi.fn()}
      />,
    );
    expect(
      screen.getByText('git 2.19.1 is too old; 2.23 or newer is required.'),
    ).toBeTruthy();
  });

  it('still says something when the version itself could not be read', () => {
    render(
      <EmptyStates
        state="git_too_old"
        nestedToplevel={null}
        gitVersion={null}
        onInit={vi.fn()}
      />,
    );
    expect(screen.getByText('git ? is too old; 2.23 or newer is required.')).toBeTruthy();
  });
});
