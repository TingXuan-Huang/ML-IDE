import './app.css';
import { get } from 'svelte/store';
import App from './App.svelte';
import { applyHostMessage } from './store';
import { applyTheme, theme } from './theme';
import { post } from './vscode';
import type { HostMessage } from '@fusion/shared';

// The desktop (Electron) host has no VS Code theme, so inject the --vscode-* design tokens
// ourselves (light or dark, from the saved preference) BEFORE the app mounts — no flash.
// In the VS Code webview, window.fusionHost is absent and VS Code supplies the real theme.
applyTheme(get(theme));

const app = new App({ target: document.getElementById('app')! });
window.addEventListener('message', (e: MessageEvent<HostMessage>) => applyHostMessage(e.data));
post({ type: 'ready' });
if ((window as unknown as { fusionHost?: unknown }).fusionHost) post({ type: 'getTracingConfig' });

export default app;
