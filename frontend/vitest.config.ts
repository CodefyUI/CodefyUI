import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import path from 'path';
import { actWarningsGate } from './scripts/act-warnings.mjs';

export default defineConfig({
  // actWarningsGate: `vitest run` fails when a test file prints more act()
  // warnings than scripts/act-warnings.baseline.json allows (#505).
  plugins: [react(), actWarningsGate()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: './src/test/setup.ts',
    coverage: {
      provider: 'v8',
      reporter: ['text-summary', 'json-summary', 'html'],
      include: ['src/**/*.{ts,tsx}'],
      exclude: [
        'src/**/*.test.{ts,tsx}',
        'src/**/*.d.ts',
        'src/main.tsx',
        'src/vite-env.d.ts',
        'src/test/**',
      ],
    },
  },
});
