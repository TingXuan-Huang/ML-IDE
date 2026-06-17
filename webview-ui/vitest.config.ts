import { defineConfig } from 'vitest/config';
import { fileURLToPath, URL } from 'node:url';

// store.ts + its formatters are pure TS over svelte/store — no DOM needed, so run in node.
// Mirror the build's @fusion/shared -> ../shared/src alias so types resolve from source.
export default defineConfig({
  resolve: {
    alias: { '@fusion/shared': fileURLToPath(new URL('../shared/src', import.meta.url)) },
  },
  test: { environment: 'node', include: ['src/**/*.test.ts'] },
});
