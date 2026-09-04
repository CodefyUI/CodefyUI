import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react';
import { RemotesSection } from './RemotesSection';
import { useI18n } from '../../i18n';
import { _resetGitStoreForTesting, useGitStore } from '../../store/gitStore';
import { confirm, prompt } from '../../utils/dialog';
import type { RemoteInfo } from '../../api/git';

/*
 * The remote list. The URLs on these rows are DISPLAY strings -- the server
 * masks the credential half of one before it ever leaves the machine -- so the
 * property that matters most here is that nothing ever offers a masked URL
 * back to the server as if it were the real one.
 */
vi.mock('../../utils/dialog', () => ({
  confirm: vi.fn(async () => true),
  prompt: vi.fn(async () => null),
}));

const askedConfirm = vi.mocked(confirm);
const askedPrompt = vi.mocked(prompt);

function remote(name: string, url = `https://example.invalid/${name}.git`): RemoteInfo {
  return { name, fetch_url: url, push_url: url };
}

/** The store's own action types, so a fake cannot drift from the real one. */
type GitActions = ReturnType<typeof useGitStore.getState>;

let addRemote: ReturnType<typeof vi.fn<GitActions['addRemote']>>;
let setRemoteUrl: ReturnType<typeof vi.fn<GitActions['setRemoteUrl']>>;
let removeRemote: ReturnType<typeof vi.fn<GitActions['removeRemote']>>;
let setSectionOpen: ReturnType<typeof vi.fn<GitActions['setSectionOpen']>>;

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  _resetGitStoreForTesting();
  askedConfirm.mockReset();
  askedConfirm.mockResolvedValue(true);
  askedPrompt.mockReset();
  askedPrompt.mockResolvedValue(null);
  addRemote = vi.fn(async () => true);
  setRemoteUrl = vi.fn(async () => true);
  removeRemote = vi.fn(async () => true);
  setSectionOpen = vi.fn();
  useGitStore.setState({
    repoState: 'ready',
    remotes: [remote('origin')],
    sections: { branches: false, remotes: true, stashes: false },
    addRemote,
    setRemoteUrl,
    removeRemote,
    setSectionOpen,
  });
});

afterEach(() => {
  _resetGitStoreForTesting();
  vi.restoreAllMocks();
});

const section = () => screen.getByRole('region', { name: 'Remotes' });

describe('RemotesSection: the rows', () => {
  it('is the Remotes section, one row per remote, with its fetch URL', () => {
    render(<RemotesSection />);
    expect(within(section()).getByText('1')).toBeTruthy();
    expect(within(section()).getByText('origin')).toBeTruthy();
    expect(
      within(section()).getByText('https://example.invalid/origin.git'),
    ).toBeTruthy();
  });

  it('keeps the whole row readable when its URL is too long to fit', () => {
    // A URL is one unbroken token in a 180px column and is the first thing on
    // the row to be ellipsised, so it carries its own `title` -- the row's is
    // the remote's name.
    render(<RemotesSection />);
    expect(screen.getByRole('listitem').getAttribute('title')).toBe('origin');
    expect(
      screen.getByText('https://example.invalid/origin.git').getAttribute('title'),
    ).toBe('https://example.invalid/origin.git');
  });

  it('says a repository has no remote, but only once it has been read', () => {
    useGitStore.setState({ remotes: [] });
    const view = render(<RemotesSection />);
    expect(within(section()).getByText('No remote yet.')).toBeTruthy();

    view.unmount();
    // Null is "not read yet", never "none": a sentence here would be a claim
    // nobody has checked.
    useGitStore.setState({ remotes: null });
    render(<RemotesSection />);
    expect(screen.queryByText('No remote yet.')).toBeNull();
  });

  it('reports a list that could not be read, inside the section', () => {
    useGitStore.setState({
      remotes: null,
      refsError: { branches: null, remotes: 'Failed to fetch', stashes: null },
    });
    render(<RemotesSection />);
    expect(
      within(section()).getByText('Could not read repository status: Failed to fetch'),
    ).toBeTruthy();
    // One answer, not two: the reason replaces the empty sentence.
    expect(screen.queryByText('No remote yet.')).toBeNull();
  });
});

describe('RemotesSection: adding one', () => {
  it('asks for a name and then a URL, and adds what came back', async () => {
    askedPrompt
      .mockResolvedValueOnce('  upstream  ')
      .mockResolvedValueOnce('  git@github.com:owner/repo.git  ');
    render(<RemotesSection />);
    fireEvent.click(screen.getByRole('button', { name: 'Add Remote...' }));
    await waitFor(() =>
      expect(addRemote).toHaveBeenCalledWith('upstream', 'git@github.com:owner/repo.git'),
    );
    expect(askedPrompt.mock.calls[0][0].title).toBe('Remote name');
    expect(askedPrompt.mock.calls[1][0].title).toBe('Remote URL (https:// or git@...)');
  });

  it('adds nothing when the second question is cancelled', async () => {
    askedPrompt.mockResolvedValueOnce('upstream').mockResolvedValueOnce(null);
    render(<RemotesSection />);
    fireEvent.click(screen.getByRole('button', { name: 'Add Remote...' }));
    await waitFor(() => expect(askedPrompt).toHaveBeenCalledTimes(2));
    expect(addRemote).not.toHaveBeenCalled();
  });

  it('never asks for a URL when the name was cancelled', async () => {
    askedPrompt.mockResolvedValue(null);
    render(<RemotesSection />);
    fireEvent.click(screen.getByRole('button', { name: 'Add Remote...' }));
    await waitFor(() => expect(askedPrompt).toHaveBeenCalledTimes(1));
    expect(addRemote).not.toHaveBeenCalled();
  });

  it('refuses a name and a URL the server would refuse', async () => {
    askedPrompt.mockResolvedValueOnce('origin').mockResolvedValueOnce(null);
    render(<RemotesSection />);
    fireEvent.click(screen.getByRole('button', { name: 'Add Remote...' }));
    await waitFor(() => expect(askedPrompt).toHaveBeenCalledTimes(2));

    const name = askedPrompt.mock.calls[0][0].validate;
    expect(name?.('upstream')).toBeNull();
    expect(name?.('has space')).toBe('Invalid value.');
    expect(name?.('')).toBe('Invalid value.');

    const url = askedPrompt.mock.calls[1][0].validate;
    expect(url?.('https://github.com/owner/repo.git')).toBeNull();
    expect(url?.('git@github.com:owner/repo.git')).toBeNull();
    expect(url?.('github.com/owner/repo')).toBe('Use an https:// or SSH URL');
  });
});

describe('RemotesSection: changing and removing one', () => {
  it('asks for a URL with an EMPTY box, never the one on screen', async () => {
    askedPrompt.mockResolvedValue('https://example.invalid/moved.git');
    render(<RemotesSection />);
    fireEvent.click(screen.getByRole('button', { name: 'Change URL origin' }));
    await waitFor(() =>
      expect(setRemoteUrl).toHaveBeenCalledWith(
        'origin',
        'https://example.invalid/moved.git',
      ),
    );
    // The URL on the row is a DISPLAY string: the server masks the credential
    // half of one before it is served. Prefilling the box would offer that
    // mask back as if it were the URL.
    expect(askedPrompt.mock.calls[0][0].defaultValue).toBeUndefined();
  });

  it('asks before removing a remote, and removes it', async () => {
    render(<RemotesSection />);
    fireEvent.click(screen.getByRole('button', { name: 'Remove origin' }));
    await waitFor(() => expect(removeRemote).toHaveBeenCalledWith('origin'));
    expect(askedConfirm).toHaveBeenCalledWith({
      title: 'Remove remote origin?',
      confirmText: 'Remove',
      variant: 'danger',
    });
  });

  it('removes nothing when the question is answered no', async () => {
    askedConfirm.mockResolvedValue(false);
    render(<RemotesSection />);
    fireEvent.click(screen.getByRole('button', { name: 'Remove origin' }));
    await waitFor(() => expect(askedConfirm).toHaveBeenCalled());
    expect(removeRemote).not.toHaveBeenCalled();
  });

  it('puts focus back on the section heading when the row is gone', async () => {
    render(<RemotesSection />);
    fireEvent.click(screen.getByRole('button', { name: 'Remove origin' }));
    await waitFor(() =>
      expect(document.activeElement).toBe(
        screen.getByRole('button', { name: 'Remotes' }),
      ),
    );
  });
});
