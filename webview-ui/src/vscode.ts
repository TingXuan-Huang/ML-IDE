// Thin bridge to the VS Code webview host. `acquireVsCodeApi` is injected by VS Code.
import type { WebviewMessage } from '@fusion/shared';

interface VsCodeApi {
  postMessage(msg: unknown): void;
}
declare function acquireVsCodeApi(): VsCodeApi;

let api: VsCodeApi | undefined;
try {
  api = acquireVsCodeApi();
} catch {
  api = undefined; // standalone (Playwright harness) — no host
}

export function post(msg: WebviewMessage): void {
  api?.postMessage(msg);
}
