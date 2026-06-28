import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import cssInjectedByJsPlugin from 'vite-plugin-css-injected-by-js';
import { resolve } from 'path';

// Library build: bundle src/index.tsx (your React app + the SDK + React itself)
// into a single ESM file at ../frontend/index.js — the bundle the editor imports
// and whose default export it calls as activate(api). CSS is inlined into the JS
// so there is exactly one shippable file.
export default defineConfig({
  plugins: [react(), cssInjectedByJsPlugin()],
  define: { 'process.env.NODE_ENV': JSON.stringify('production') },
  build: {
    lib: {
      entry: resolve(__dirname, 'src/index.tsx'),
      formats: ['es'],
      fileName: () => 'index.js',
    },
    outDir: resolve(__dirname, '../frontend'),
    emptyOutDir: true,
    rollupOptions: { output: { inlineDynamicImports: true } },
    cssCodeSplit: false,
  },
});
