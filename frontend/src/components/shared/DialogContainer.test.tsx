import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { DialogContainer } from './DialogContainer';
import { useDialogStore } from '../../store/dialogStore';
import { confirm, prompt } from '../../utils/dialog';
import { useI18n } from '../../i18n';

describe('DialogContainer', () => {
  beforeEach(() => {
    useDialogStore.setState({ active: null, resolve: null });
    useI18n.setState({ locale: 'en' });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('renders nothing when no dialog is active', () => {
    const { container } = render(<DialogContainer />);
    expect(container.querySelector('[role="dialog"]')).toBeNull();
  });

  it('renders confirm dialog with title and message', async () => {
    render(<DialogContainer />);
    confirm({ title: 'Delete graph?', message: 'This cannot be undone.' });
    expect(await screen.findByText('Delete graph?')).toBeTruthy();
    expect(await screen.findByText('This cannot be undone.')).toBeTruthy();
  });

  it('clicking confirm button resolves true', async () => {
    render(<DialogContainer />);
    const p = confirm({ title: 'OK?', confirmText: 'Yes' });
    fireEvent.click(await screen.findByText('Yes'));
    await expect(p).resolves.toBe(true);
  });

  it('clicking cancel button resolves false', async () => {
    render(<DialogContainer />);
    const p = confirm({ title: 'OK?', cancelText: 'No' });
    fireEvent.click(await screen.findByText('No'));
    await expect(p).resolves.toBe(false);
  });

  it('clicking backdrop resolves cancel', async () => {
    render(<DialogContainer />);
    const p = confirm({ title: 'X' });
    // Wait for portal to render the backdrop.
    await screen.findByText('X');
    const backdrop = document.querySelector('[role="dialog"]') as HTMLElement;
    fireEvent.click(backdrop);
    await expect(p).resolves.toBe(false);
  });

  it('Escape resolves cancel', async () => {
    render(<DialogContainer />);
    const p = confirm({ title: 'X' });
    await screen.findByText('X');
    fireEvent.keyDown(window, { key: 'Escape' });
    await expect(p).resolves.toBe(false);
  });

  it('Escape on a prompt resolves null (prompt-cancel branch)', async () => {
    render(<DialogContainer />);
    const p = prompt({ title: 'Name?', defaultValue: 'x' });
    await screen.findByText('Name?');
    fireEvent.keyDown(window, { key: 'Escape' });
    await expect(p).resolves.toBeNull();
  });

  it('ignores non-Escape keydowns while a dialog is open', async () => {
    render(<DialogContainer />);
    const p = confirm({ title: 'KeepOpen' });
    await screen.findByText('KeepOpen');
    fireEvent.keyDown(window, { key: 'Enter' });
    fireEvent.keyDown(window, { key: 'a' });
    // Dialog stays open; promise unresolved. Resolve it to clean up.
    expect(screen.getByText('KeepOpen')).toBeTruthy();
    fireEvent.keyDown(window, { key: 'Escape' });
    await expect(p).resolves.toBe(false);
  });

  it('renders prompt with input pre-filled with defaultValue', async () => {
    render(<DialogContainer />);
    prompt({ title: 'Rename', defaultValue: 'untitled' });
    // The prompt's own question names the box: one dialog asks for a branch
    // name and the next for a remote URL, and "Dialog input" told a reader
    // which of those they had landed in exactly never.
    const input = (await screen.findByRole('textbox', { name: 'Rename' })) as HTMLInputElement;
    expect(input.value).toBe('untitled');
  });

  it('typing + clicking confirm resolves with input value', async () => {
    render(<DialogContainer />);
    const p = prompt({ title: 'Name?', confirmText: 'OK' });
    const input = (await screen.findByRole('textbox', { name: 'Name?' })) as HTMLInputElement;
    fireEvent.change(input, { target: { value: 'alice' } });
    fireEvent.click(screen.getByText('OK'));
    await expect(p).resolves.toBe('alice');
  });

  it('cancelling prompt resolves null', async () => {
    render(<DialogContainer />);
    const p = prompt({ title: 'Name?', cancelText: 'Cancel' });
    fireEvent.click(await screen.findByText('Cancel'));
    await expect(p).resolves.toBeNull();
  });

  it('danger variant adds danger class to the confirm button', async () => {
    render(<DialogContainer />);
    confirm({ title: 'Delete?', confirmText: 'Delete', variant: 'danger' });
    const btn = (await screen.findByText('Delete')) as HTMLButtonElement;
    expect(btn.className).toContain('danger');
  });

  it('validate hook blocks submit and shows the error', async () => {
    render(<DialogContainer />);
    const p = prompt({
      title: 'Name?',
      validate: (v) => (v.trim() ? null : 'Required'),
    });
    fireEvent.click(await screen.findByText('OK'));
    expect(await screen.findByText('Required')).toBeTruthy();
    // Promise is not yet resolved — fix the input and retry.
    const input = (await screen.findByRole('textbox', { name: 'Name?' })) as HTMLInputElement;
    fireEvent.change(input, { target: { value: 'fine' } });
    fireEvent.click(screen.getByText('OK'));
    await expect(p).resolves.toBe('fine');
  });

  it('reports the refusal ON the box that was refused', async () => {
    // A message sitting under the input with nothing tying it to the input is
    // a message a screen reader reads only if the user happens to walk into
    // it. It is the input's description, and the input says it is invalid.
    render(<DialogContainer />);
    prompt({ title: 'Branch name', validate: () => 'Not a valid branch name' });
    const input = (await screen.findByRole('textbox', {
      name: 'Branch name',
    })) as HTMLInputElement;
    expect(input.getAttribute('aria-invalid')).toBe('false');
    expect(input.getAttribute('aria-describedby')).toBeNull();

    fireEvent.click(screen.getByText('OK'));
    const said = await screen.findByText('Not a valid branch name');
    expect(said.id).not.toBe('');
    expect(input.getAttribute('aria-invalid')).toBe('true');
    expect(input.getAttribute('aria-describedby')).toBe(said.id);

    // Typing again clears both halves, so the description does not outlive
    // the value it was about.
    fireEvent.change(input, { target: { value: 'main' } });
    expect(input.getAttribute('aria-invalid')).toBe('false');
    expect(input.getAttribute('aria-describedby')).toBeNull();
  });

  // ── Locale-aware fallback labels (#160) ─────────────────────────────────
  // cancelText is passed by no production caller at all, and confirmText's
  // English fallback ('OK' / 'Confirm') was a raw literal -- both used to
  // reach a zh-TW user unchanged.

  it('shows the zh-TW cancel and confirm labels on a confirm dialog with no override (#160)', async () => {
    useI18n.setState({ locale: 'zh-TW' });
    render(<DialogContainer />);
    confirm({ title: 'X' });
    expect(await screen.findByText('取消')).toBeTruthy();
    expect(await screen.findByText('確認')).toBeTruthy();
    // Never the raw English fallback for a zh-TW user.
    expect(screen.queryByText('Cancel')).toBeNull();
    expect(screen.queryByText('Confirm')).toBeNull();
  });

  it('shows the zh-TW OK label on a prompt dialog with no override (#160)', async () => {
    useI18n.setState({ locale: 'zh-TW' });
    render(<DialogContainer />);
    prompt({ title: 'X' });
    expect(await screen.findByText('確定')).toBeTruthy();
    expect(screen.queryByText('OK')).toBeNull();
  });
});
