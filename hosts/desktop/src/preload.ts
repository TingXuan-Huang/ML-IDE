// Fusion desktop — preload bridge.
//
// Recreates the VS Code webview contract so the Svelte cockpit runs UNCHANGED:
//   • renderer → host :  window.fusionHost.send(msg)   (the UI's post() routes here)
//   • host → renderer :  arrives as a normal window 'message' event
// contextIsolation is on, so the renderer never touches Node/Electron directly.
import { contextBridge, ipcRenderer } from 'electron';

contextBridge.exposeInMainWorld('fusionHost', {
  send: (msg: unknown) => ipcRenderer.send('renderer:message', msg),
});

// Re-emit host messages as window messages -> webview-ui's existing
// window.addEventListener('message', ...) handles them with zero changes.
ipcRenderer.on('host:message', (_event, msg) => {
  window.postMessage(msg, '*');
});
