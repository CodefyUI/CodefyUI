import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
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

  it('closes without writing anything', () => {
    render(<IdentityForm />);
    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }));
    expect(closeIdentityForm).toHaveBeenCalledTimes(1);
    expect(saveIdentity).not.toHaveBeenCalled();
  });
});
