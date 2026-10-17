import { StrictMode, Suspense, useState } from 'react';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { useI18n } from '../../i18n';
import { useToastStore } from '../../store/toastStore';
import { lazyRetry } from './lazyRetry';

interface DialogProps {
  label: string;
  onClose: () => void;
}

function Dialog({ label }: DialogProps) {
  return <div role="dialog">{label}</div>;
}

/** Opens the lazy dialog from a button, as the app's call sites do. */
function Host({ Lazy }: { Lazy: React.ComponentType<DialogProps> }) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <button type="button" onClick={() => setOpen(true)}>Open</button>
      {open && (
        <Suspense fallback={null}>
          <Lazy label="Loaded" onClose={() => setOpen(false)} />
        </Suspense>
      )}
    </>
  );
}

let warn: ReturnType<typeof vi.spyOn>;

beforeEach(() => {
  useI18n.setState({ locale: 'en' });
  useToastStore.setState({ toasts: [] });
  warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
});

afterEach(() => {
  warn.mockRestore();
});

async function clickOpen() {
  await act(async () => {
    fireEvent.click(screen.getByRole('button', { name: 'Open' }));
  });
}

describe('lazyRetry', () => {
  it('renders the component once its chunk loads', async () => {
    const Lazy = lazyRetry(() => Promise.resolve(Dialog), vi.fn());
    render(<Host Lazy={Lazy} />);
    await clickOpen();
    expect(await screen.findByRole('dialog')).toHaveTextContent('Loaded');
  });

  it('turns a failed chunk into a toast and a close, then imports again on the next open', async () => {
    const load = vi.fn<() => Promise<React.ComponentType<DialogProps>>>()
      .mockRejectedValueOnce(new Error('Failed to fetch dynamically imported module'))
      .mockResolvedValueOnce(Dialog);
    const onFail = vi.fn((props: DialogProps) => props.onClose());
    const Lazy = lazyRetry(load, onFail);
    render(<Host Lazy={Lazy} />);

    await clickOpen();
    await waitFor(() => expect(onFail).toHaveBeenCalledTimes(1));
    expect(screen.queryByRole('dialog')).toBeNull();
    const { toasts } = useToastStore.getState();
    expect(toasts.map((toast) => [toast.type, toast.message])).toEqual([
      ['error', 'This part of the app could not be loaded. Try again; if it keeps failing, reload the page.'],
    ]);
    expect(warn).toHaveBeenCalledTimes(1);

    await clickOpen();
    expect(await screen.findByRole('dialog')).toHaveTextContent('Loaded');
    expect(load).toHaveBeenCalledTimes(2);
  });

  it('reports a failure once under StrictMode', async () => {
    const onFail = vi.fn((props: DialogProps) => props.onClose());
    const Lazy = lazyRetry(() => Promise.reject(new Error('404')), onFail);
    render(<StrictMode><Host Lazy={Lazy} /></StrictMode>);
    await clickOpen();
    await waitFor(() => expect(onFail).toHaveBeenCalledTimes(1));
    expect(useToastStore.getState().toasts).toHaveLength(1);
  });
});
