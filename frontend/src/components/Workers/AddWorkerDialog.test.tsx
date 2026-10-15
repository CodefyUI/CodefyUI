import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { act, fireEvent, render, screen, within } from '@testing-library/react';
import { useI18n } from '../../i18n';
import { useToastStore } from '../../store/toastStore';
import { AGENT_START_COMMAND, AGENT_STOP_COMMAND, AddWorkerDialog } from './AddWorkerDialog';

const writeText = vi.fn();

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  useToastStore.setState({ toasts: [] });
  writeText.mockReset().mockResolvedValue(undefined);
  Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true });
});

afterEach(() => {
  Reflect.deleteProperty(navigator, 'clipboard');
});

function renderDialog(onClose = vi.fn()) {
  render(<AddWorkerDialog mode="same-host" onClose={onClose} />);
  return { onClose, dialog: screen.getByRole('dialog', { name: 'Add a worker' }) };
}

describe('AddWorkerDialog (same-host)', () => {
  it('prints the agent start and stop commands and the options', () => {
    const { dialog } = renderDialog();
    expect(dialog).toHaveAttribute('data-mode', 'same-host');
    expect(within(dialog).getByText(AGENT_START_COMMAND)).toBeInTheDocument();
    expect(within(dialog).getByText(AGENT_STOP_COMMAND)).toBeInTheDocument();
    expect(within(dialog).getByText('--name <name>')).toBeInTheDocument();
    expect(within(dialog).getByText('--devices cuda:0,cuda:1')).toBeInTheDocument();
    expect(within(dialog).getByText('--foreground')).toBeInTheDocument();
  });

  it('copies a command and says so', async () => {
    const { dialog } = renderDialog();
    const [copyStart, copyStop] = within(dialog).getAllByRole('button', { name: 'Copy' });
    await act(async () => fireEvent.click(copyStart));
    expect(writeText).toHaveBeenCalledWith('cdui agent start');
    await act(async () => fireEvent.click(copyStop));
    expect(writeText).toHaveBeenLastCalledWith('cdui agent stop');
    expect(useToastStore.getState().toasts.map((t) => t.message)).toEqual([
      'Command copied', 'Command copied',
    ]);
  });

  it('asks for a manual copy when the clipboard refuses', async () => {
    writeText.mockRejectedValue(new Error('denied'));
    const { dialog } = renderDialog();
    await act(async () => fireEvent.click(within(dialog).getAllByRole('button', { name: 'Copy' })[0]));
    const [toast] = useToastStore.getState().toasts;
    expect(toast).toMatchObject({ type: 'error', message: expect.stringMatching(/copy it by hand/) });
  });

  it('closes on Escape, the close button and a press on its backdrop only', () => {
    const { dialog, onClose } = renderDialog();
    expect(within(dialog).getByRole('button', { name: 'Close' })).toHaveFocus();
    fireEvent.keyDown(dialog, { key: 'Escape' });
    expect(onClose).toHaveBeenCalledTimes(1);
    fireEvent.click(within(dialog).getByRole('button', { name: 'Close' }));
    expect(onClose).toHaveBeenCalledTimes(2);
    fireEvent.mouseDown(dialog);
    expect(onClose).toHaveBeenCalledTimes(2);
    fireEvent.mouseDown(dialog.parentElement as HTMLElement);
    expect(onClose).toHaveBeenCalledTimes(3);
  });

  it('keeps Tab inside and does not let its keys reach the panel', () => {
    const outer = vi.fn();
    render(
      <div onKeyDown={outer}>
        <AddWorkerDialog mode="same-host" onClose={vi.fn()} />
      </div>,
    );
    const dialog = screen.getByRole('dialog', { name: 'Add a worker' });
    const close = within(dialog).getByRole('button', { name: 'Close' });
    const copies = within(dialog).getAllByRole('button', { name: 'Copy' });
    copies[1].focus();
    fireEvent.keyDown(dialog, { key: 'Tab' });
    expect(close).toHaveFocus();
    expect(outer).not.toHaveBeenCalled();
  });

  it('gives the focus back on close, unless the opener left the page', () => {
    const opener = document.createElement('button');
    document.body.appendChild(opener);
    opener.focus();
    const first = render(<AddWorkerDialog mode="same-host" onClose={vi.fn()} />);
    first.unmount();
    expect(opener).toHaveFocus();

    const second = render(<AddWorkerDialog mode="same-host" onClose={vi.fn()} />);
    opener.remove();
    second.unmount();
    expect(opener).not.toHaveFocus();
  });

  it('renders in zh-TW', () => {
    useI18n.setState({ locale: 'zh-TW' });
    render(<AddWorkerDialog mode="same-host" onClose={vi.fn()} />);
    expect(screen.getByRole('dialog', { name: '新增 worker' })).toBeInTheDocument();
    expect(screen.getAllByRole('button', { name: '複製' })).toHaveLength(2);
  });
});
