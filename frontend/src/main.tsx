import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import 'katex/dist/katex.min.css';
import './App.css';
import App from './App';
import { getSessionToken } from './api/_auth';
import { fetchDevices } from './api/rest';
import { useUIStore } from './store/uiStore';

const rootElement = document.getElementById('root');
if (!rootElement) throw new Error('Root element not found');

// Bootstrap the session token before any mutating request can fire. The
// auth_guard middleware on the backend rejects mutations without the token,
// so doing this first means user actions never race the bootstrap.
//
// We don't fail the whole app if bootstrap fails — the user still sees the
// UI, and individual mutation errors will surface naturally. This keeps the
// dev-mode experience reasonable when the backend is being restarted.
getSessionToken().catch((err) => {
  console.error('[CodefyUI] Auth bootstrap failed:', err);
});

// The global device defaulted to 'cpu' for anyone who had never opened
// Settings, and the backend's own answer — `describe_accelerator().default`,
// the best device present — was fetched but never read. On a shared lab box
// that meant every student trained on the CPU while the GPU sat idle, with
// nothing on screen suggesting otherwise.
//
// Done here rather than in SettingsPopover because that component only mounts
// when the popover is opened, which is exactly the thing the student does not
// know to do. An explicit choice still wins; see `adoptDefaultDevice`.
fetchDevices()
  .then((r) => useUIStore.getState().adoptDefaultDevice(r.default))
  .catch(() => {
    /* No device list — keep the CPU default, which always works. */
  });

createRoot(rootElement).render(
  <StrictMode>
    <App />
  </StrictMode>
);
