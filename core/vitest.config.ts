import { defineConfig } from 'vitest/config';
import { fileURLToPath, URL } from 'node:url';

export default defineConfig({
  resolve: {
    alias: { '@fusion/shared': fileURLToPath(new URL('../shared/src', import.meta.url)) },
  },
  test: {
    include: ['src/**/*.test.ts'],
  },
});
