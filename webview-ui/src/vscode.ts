// Host bridge — abstracts the transport so the SAME cockpit UI runs in every host:
//   • Electron renderer : window.fusionHost (exposed by the desktop preload over IPC)
//   • VS Code webview    : acquireVsCodeApi() (injected by VS Code)
//   • bare browser       : no host -> no-op (Playwright / vite dev)
// Host -> renderer messages always arrive as a window 'message' event (the VS Code
// webview posts them; the Electron preload re-emits IPC as window messages), so the
// receive side (main.ts's addEventListener) is identical everywhere — only post() forks.
import type { WebviewMessage } from '@fusion/shared';

interface Bridge {
  postMessage(msg: unknown): void;
}
declare function acquireVsCodeApi(): Bridge;

let bridge: Bridge | undefined;

const electron = (globalThis as unknown as { fusionHost?: { send(m: unknown): void } }).fusionHost;
if (electron) {
  bridge = { postMessage: (m) => electron.send(m) };
} else {
  try {
    bridge = acquireVsCodeApi();
  } catch {
    bridge = undefined; // standalone browser / dev harness — no host
  }
}

export function post(msg: WebviewMessage): void {
  bridge?.postMessage(msg);
}
