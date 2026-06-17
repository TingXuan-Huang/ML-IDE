import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';
import { fileURLToPath, URL } from 'node:url';

// Builds the cockpit webview to dist/assets/main.{js,css} with FIXED names so the
// extension can reference them via asWebviewUri (no content hashes to chase).
// base:'./' makes dist/index.html reference assets RELATIVELY, so the Electron host
// can load it over file:// (the VS Code host builds its own HTML, so it's unaffected).
export default defineConfig({
  base: './',
  plugins: [svelte()],
  resolve: {
    alias: { '@fusion/shared': fileURLToPath(new URL('../shared/src', import.meta.url)) },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    target: 'es2020',
    // The Monaco editor-core chunk is ~2.2 MB (code-split + lazy on desktop). Quiet the warning
    // at that expected size while still flagging any future regression that re-bloats it.
    chunkSizeWarningLimit: 2400,
    rollupOptions: {
      output: {
        entryFileNames: 'assets/main.js',
        assetFileNames: 'assets/main.[ext]',
        chunkFileNames: 'assets/[name].js',
      },
    },
  },
});
