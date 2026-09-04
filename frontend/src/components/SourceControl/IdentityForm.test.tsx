import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { act, render, screen, fireEvent, waitFor } from '@testing-library/react';
import { IdentityForm } from './IdentityForm';
import { useI18n } from '../../i18n';
import { _resetGitStoreForTesting, useGitStore } from '../../store/gitStore';
import type { Identity } from '../../api/git';

/*
 * Name and email, with where each of them comes from. The scope line is the
 * point of the form as much as the fields are: "for this project" and "from
 * global git config" are the difference between a commit signed as you and one
 * signed as whoever last used this machine.
 */

function identity(over: Partial<Identity> = {}): Identity {
  return { name: null, email: null, name_scope: null, email_scope: null, ...over };
}

/** The store's own action types, so a fake cannot drift from the real one. */
type GitActions = ReturnType<typeof useGitStore.getState>;

let saveIdentity: ReturnType<typeof vi.fn<GitActions['saveIdentity']>>;
let closeIdentityForm: ReturnType<typeof vi.fn<GitActions['closeIdentityForm']>>;

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  _resetGitStoreForTesting();
  saveIdentity = vi.fn(async () => true);
  closeIdentityForm = vi.fn();
  useGitStore.setState({ identityFormOpen: true, saveIdentity, closeIdentityForm });
});

afterEach(() => {
  _resetGitStoreForTesting();
  vi.restoreAllMocks();
});

const nameBox = () => screen.getByLabelText('Name');
const emailBox = () => screen.getByLabelText('Email');

describe('IdentityForm', () => {
  it('opens empty while the config read is still in flight', () => {
    render(<IdentityForm />);
    expect(screen.getByText('Commit identity')).toBeTruthy();
    expect(nameBox()).toHaveValue('');
    expect(emailBox()).toHaveValue('');
  });

  it('fills in from the identity the store reads, and says where each half came from', async () => {
    render(<IdentityForm />);
    useGitStore.setState({
      identity: identity({
        name: 'Ada',
        email: 'ada@example.com',
        name_scope: 'global',
        email_scope: 'local',
      }),
    });
    await waitFor(() => expect(nameBox()).toHaveValue('Ada'));
    expect(emailBox()).toHaveValue('ada@example.com');
    expect(screen.getByText('from global git config')).toBeTruthy();
    expect(screen.getByText('for this project')).toBeTruthy();
  });

  it('reads a system-wide value as a global one', async () => {
    render(<IdentityForm />);
    useGitStore.setState({
      identity: identity({ name: 'Ada', name_scope: 'system' }),
    });
    await waitFor(() => expect(nameBox()).toHaveValue('Ada'));
    expect(screen.getAllByText('from global git config')).toHaveLength(1);
  });

  it('says which half is not set at all', () => {
    useGitStore.setState({
      identity: identity({ email: 'ada@example.com', email_scope: 'local' }),
    });
    render(<IdentityForm />);
    expect(screen.getByText('Not set')).toBeTruthy();
    expect(screen.getByText('for this project')).toBeTruthy();
  });

  it('saves what was typed', () => {
    render(<IdentityForm />);
    fireEvent.change(nameBox(), { target: { value: 'Ada Lovelace' } });
    fireEvent.change(emailBox(), { target: { value: 'ada@example.com' } });
    fireEvent.click(screen.getByRole('button', { name: 'Save' }));
    expect(saveIdentity).toHaveBeenCalledWith({
      name: 'Ada Lovelace',
      email: 'ada@example.com',
    });
  });

  it('saves one half alone, which the store reads as leave the other alone', () => {
    render(<IdentityForm />);
    fireEvent.change(emailBox(), { target: { value: 'ada@example.com' } });
    fireEvent.click(screen.getByRole('button', { name: 'Save' }));
    expect(saveIdentity).toHaveBeenCalledWith({ name: '', email: 'ada@example.com' });
  });

  it('does not put a second config read over what is being typed', async () => {
    render(<IdentityForm />);
    act(() => {
      useGitStore.setState({ identity: identity({ name: 'Ada', name_scope: 'global' }) });
    });
    await waitFor(() => expect(nameBox()).toHaveValue('Ada'));
    fireEvent.change(nameBox(), { target: { value: 'Ada Lovelace' } });

    // A commit refused a second time for a missing identity reopens the form
    // and reads the config again. The seed is once only, or that answer would
    // land in the middle of the sentence being typed.
    act(() => {
      useGitStore.setState({ identity: identity({ name: 'Ada', name_scope: 'global' }) });
    });
    expect(nameBox()).toHaveValue('Ada Lovelace');
  });

  it('has nothing to save while both halves are blank', () => {
    render(<IdentityForm />);
    const save = screen.getByRole('button', { name: 'Save' });
    expect(save).toHaveAttribute('aria-disabled', 'true');
    fireEvent.click(save);
    expect(saveIdentity).not.toHaveBeenCalled();
  });

  it('turns Save on as soon as either half is filled', () => {
    render(<IdentityForm />);
    fireEvent.change(emailBox(), { target: { value: 'ada@example.com' } });
    expect(screen.getByRole('button', { name: 'Save' })).toHaveAttribute(
      'aria-disabled',
      'false',
    );
  });

  it('puts the caret in the first field, because nobody may have opened it', () => {
    // The form appears BY ITSELF when a commit is refused for a missing
    // identity, above the button that was just pressed. Without this it is a
    // paragraph that turned up somewhere on the page.
    render(<IdentityForm />);
    expect(document.activeElement).toBe(nameBox());
  });

  it('describes each field with the scope it would be written at', async () => {
    render(<IdentityForm />);
    useGitStore.setState({
      identity: identity({
        name: 'Ada',
        email: 'ada@example.com',
        name_scope: 'global',
        email_scope: 'local',
      }),
    });
    await waitFor(() => expect(nameBox()).toHaveValue('Ada'));
    // A reader who hears only the label and the value never learns which
    // identity they are about to commit under.
    expect(nameBox()).toHaveAccessibleDescription('from global git config');
    expect(emailBox()).toHaveAccessibleDescription('for this project');
  });

  it('closes without writing anything', () => {
    render(<IdentityForm />);
    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }));
    expect(closeIdentityForm).toHaveBeenCalledTimes(1);
    expect(saveIdentity).not.toHaveBeenCalled();
  });
});

describe('IdentityForm: what a save is allowed to write', () => {
  /** The form with the identity the config read answered, already seeded. */
  async function opened() {
    render(<IdentityForm />);
    useGitStore.setState({
      identity: identity({
        name: 'Ada',
        email: 'ada@example.com',
        name_scope: 'global',
        email_scope: 'global',
      }),
    });
    await waitFor(() => expect(nameBox()).toHaveValue('Ada'));
  }

  it('sends the half that changed and nothing about the other one', async () => {
    // The fields are seeded with the GLOBAL identity, so sending the seeded
    // email back writes it into this project's `.git/config` and pins the
    // repository to an address nobody chose. `git config --local user.email`
    // was set after a save that only meant to give the project a name.
    await opened();
    fireEvent.change(nameBox(), { target: { value: 'Grace' } });
    fireEvent.click(screen.getByRole('button', { name: 'Save' }));
    expect(saveIdentity).toHaveBeenCalledWith({ name: 'Grace', email: '' });
  });

  it('sends the email alone when that is the half that changed', async () => {
    await opened();
    fireEvent.change(emailBox(), { target: { value: 'grace@example.com' } });
    fireEvent.click(screen.getByRole('button', { name: 'Save' }));
    expect(saveIdentity).toHaveBeenCalledWith({ name: '', email: 'grace@example.com' });
  });

  it('has nothing to save while both halves still hold what was loaded', async () => {
    await opened();
    const save = screen.getByRole('button', { name: 'Save' });
    expect(save).toHaveAttribute('aria-disabled', 'true');
    fireEvent.click(save);
    expect(saveIdentity).not.toHaveBeenCalled();
  });

  it('measures a change against the first read, not a later one', async () => {
    await opened();
    fireEvent.change(nameBox(), { target: { value: 'Grace' } });

    // A commit refused a second time reads the config again. The fields
    // already ignore that answer; so must the comparison behind Save, or a
    // typed value that happens to match the new read stops being sent -- the
    // half of "seed once" that is not the text in the boxes.
    act(() => {
      useGitStore.setState({
        identity: identity({
          name: 'Grace',
          email: 'ada@example.com',
          name_scope: 'global',
          email_scope: 'global',
        }),
      });
    });
    fireEvent.click(screen.getByRole('button', { name: 'Save' }));
    expect(saveIdentity).toHaveBeenCalledWith({ name: 'Grace', email: '' });
  });

  it('still reads a half the user cleared as leave that one alone', async () => {
    await opened();
    fireEvent.change(nameBox(), { target: { value: '' } });
    fireEvent.change(emailBox(), { target: { value: 'grace@example.com' } });
    fireEvent.click(screen.getByRole('button', { name: 'Save' }));
    expect(saveIdentity).toHaveBeenCalledWith({ name: '', email: 'grace@example.com' });
  });
});
