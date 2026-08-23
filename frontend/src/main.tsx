import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import 'katex/dist/katex.min.css';
import './App.css';
import App from './App';
import { getSessionToken } from './api/_auth';

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

createRoot(rootElement).render(
  <StrictMode>
    <App />
  </StrictMode>
);
