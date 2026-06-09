import './app.css';
import App from './App.svelte';
import { applyHostMessage } from './store';
import { post } from './vscode';
import type { HostMessage } from '@fusion/shared';

const app = new App({ target: document.getElementById('app')! });

window.addEventListener('message', (e: MessageEvent<HostMessage>) => applyHostMessage(e.data));
post({ type: 'ready' });

export default app;
