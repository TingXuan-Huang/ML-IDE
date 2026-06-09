import './app.css';
import App from './App.svelte';
import { applyHostMessage } from './store';
import { post } from './vscode';
import type { HostMessage } from '@fusion/shared';

// The desktop (Electron) host has no VS Code theme, so set the --vscode-* variables in
// the renderer itself. (A host-side insertCSS on 'dom-ready' proved unreliable.) In the
// VS Code webview, window.fusionHost is absent and VS Code supplies the real theme.
if ((window as unknown as { fusionHost?: unknown }).fusionHost) {
  const st = document.createElement('style');
  st.textContent =
    ':root{--vscode-foreground:#cccccc;--vscode-editor-background:#1e1e1e;' +
    '--vscode-editor-font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,monospace;--vscode-editor-font-size:13px;' +
    '--vscode-tab-inactiveBackground:#2d2d2d;--vscode-tab-inactiveForeground:#969696;--vscode-panel-border:#3c3c3c;' +
    '--vscode-focusBorder:#007fd4;--vscode-textLink-foreground:#3794ff;--vscode-textLink-activeForeground:#4daafc;' +
    '--vscode-descriptionForeground:#9d9d9d;--vscode-editorWidget-background:#252526;' +
    '--vscode-symbolIcon-functionForeground:#dcdcaa;--vscode-inputValidation-errorBackground:#5a1d1d;' +
    '--vscode-errorForeground:#f48771;--vscode-charts-blue:#4daafc;--vscode-button-background:#0e639c;' +
    '--vscode-button-foreground:#ffffff;--vscode-input-background:#3c3c3c;--vscode-input-foreground:#cccccc;}';
  (document.head || document.documentElement).appendChild(st);
}

const app = new App({ target: document.getElementById('app')! });
window.addEventListener('message', (e: MessageEvent<HostMessage>) => applyHostMessage(e.data));
post({ type: 'ready' });

export default app;
