import { describe, it, expect, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { DialogContainer } from './DialogContainer';
import { useDialogStore } from '../../store/dialogStore';
import { confirm, prompt } from '../../utils/dialog';

describe('DialogContainer', () => {
  beforeEach(() => {
    useDialogStore.setState({ active: null, resolve: null });
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

  it('renders prompt with input pre-filled with defaultValue', async () => {
    render(<DialogContainer />);
    prompt({ title: 'Rename', defaultValue: 'untitled' });
    const input = (await screen.findByLabelText('Dialog input')) as HTMLInputElement;
    expect(input.value).toBe('untitled');
  });

  it('typing + clicking confirm resolves with input value', async () => {
    render(<DialogContainer />);
    const p = prompt({ title: 'Name?', confirmText: 'OK' });
    const input = (await screen.findByLabelText('Dialog input')) as HTMLInputElement;
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
    const input = (await screen.findByLabelText('Dialog input')) as HTMLInputElement;
    fireEvent.change(input, { target: { value: 'fine' } });
    fireEvent.click(screen.getByText('OK'));
    await expect(p).resolves.toBe('fine');
  });
});
