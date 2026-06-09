// Fusion desktop — Electron main process (P0/P1 skeleton).
//
// This is the standalone host: it owns the window, spawns the SAME Python helper the
// VS Code extension uses (@fusion/core HelperClient), and folds its JSON into the SAME
// typed FileStructure (@fusion/core toFileStructure) the SAME Svelte cockpit renders.
// The only thing that changed vs the extension is this shell — proof the core is portable.
//
// Host↔UI contract mirrors the VS Code webview: renderer→host over IPC ('renderer:message'),
// host→renderer via webContents.send('host:message') which the preload re-emits as a
// window 'message' event — so webview-ui runs byte-for-byte unchanged.
import { app, BrowserWindow, ipcMain, dialog, Menu } from 'electron';
import * as path from 'path';
import * as fs from 'fs';
import {
  HelperClient,
  toFileStructure,
  moduleTraceToRaw,
  type RawStructure,
  type RawTrace,
  type TraceModuleResult,
} from '@fusion/core';
import type { CallGraph, HostMessage, WebviewMessage } from '@fusion/shared';

// hosts/desktop/dist/main.js -> up three to the repo root.
const REPO = path.resolve(__dirname, '..', '..', '..');
const LENS_HELPER = path.join(REPO, 'lens-helper');
const UI_INDEX = path.join(REPO, 'webview-ui', 'dist', 'index.html');

// The cockpit UI is themed via VS Code's `--vscode-*` CSS variables. VS Code injects
// those; outside it they're undefined and the UI renders unstyled. Theming is a HOST
// job, so the desktop host injects a default dark theme (Dark+ values). Scoped to this
// window only via insertCSS — it can never leak into the VS Code host.
const THEME_CSS = `:root{
  --vscode-foreground:#cccccc;
  --vscode-editor-background:#1e1e1e;
  --vscode-editor-font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,"Cascadia Code",monospace;
  --vscode-editor-font-size:13px;
  --vscode-tab-inactiveBackground:#2d2d2d;
  --vscode-tab-inactiveForeground:#969696;
  --vscode-panel-border:#3c3c3c;
  --vscode-focusBorder:#007fd4;
  --vscode-textLink-foreground:#3794ff;
  --vscode-textLink-activeForeground:#4daafc;
  --vscode-descriptionForeground:#9d9d9d;
  --vscode-editorWidget-background:#252526;
  --vscode-symbolIcon-functionForeground:#dcdcaa;
  --vscode-inputValidation-errorBackground:#5a1d1d;
  --vscode-errorForeground:#f48771;
  --vscode-charts-blue:#4daafc;
}`;

let win: BrowserWindow | undefined;
let helper: HelperClient | undefined;
// The currently-open Python file: its path + source lines (for toFileStructure's getLine).
let openFile: { path: string; lines: string[] } | undefined;

// Resolve a Python that actually EXISTS (same stale-conda-env guard as the extension).
function resolvePython(): string {
  const cands = [process.env.FUSION_PYTHON, '/opt/anaconda3/bin/python3', '/usr/local/bin/python3', 'python3'];
  for (const c of cands) {
    if (!c) continue;
    if (!c.includes('/')) return c; // bare name -> trust PATH
    if (fs.existsSync(c)) return c;
  }
  return 'python3';
}

function getHelper(): HelperClient {
  if (!helper) {
    helper = new HelperClient({
      python: resolvePython(),
      cwd: LENS_HELPER,
      onStderr: (l) => console.error('[lens-helper]', l),
    });
  }
  return helper;
}

function send(msg: HostMessage): void {
  win?.webContents.send('host:message', msg);
}

const lineAt = (n: number): string => openFile?.lines[n - 1] ?? '';

async function openPython(p: string): Promise<void> {
  const text = fs.readFileSync(p, 'utf8');
  openFile = { path: p, lines: text.split('\n') };
  win?.setTitle(`Fusion — ${path.basename(p)}`);
  send({ type: 'openDocument', path: p, text, language: 'python' }); // load into the Monaco pane
  const raw = await getHelper().request<RawStructure>('structure_file', { path: p });
  send({ type: 'activeFile', structure: toFileStructure(raw, lineAt, undefined) });
  try {
    send({ type: 'callGraph', graph: await getHelper().request<CallGraph>('callgraph_file', { path: p }) });
  } catch (e) {
    console.error('callgraph failed', e);
  }
}

// Editor saved (Cmd+S): write to disk, refresh source lines, re-structure. Shapes go
// stale until the user re-traces (we don't auto-run code on every keystroke).
async function saveDocument(p: string, text: string): Promise<void> {
  try {
    fs.writeFileSync(p, text, 'utf8');
  } catch (e) {
    send({ type: 'traceState', state: { phase: 'error', message: `save failed: ${String(e)}` } });
    return;
  }
  if (openFile && openFile.path === p) openFile.lines = text.split('\n');
  const raw = await getHelper().request<RawStructure>('structure_file', { path: p });
  send({ type: 'activeFile', structure: toFileStructure(raw, lineAt, undefined) });
}

async function pickPython(): Promise<void> {
  const r = await dialog.showOpenDialog({ properties: ['openFile'], filters: [{ name: 'Python', extensions: ['py'] }] });
  if (!r.canceled && r.filePaths[0]) await openPython(r.filePaths[0]);
}

// "Trace this file" = trace_module: build-check + synth-call every model/function
// (no __main__ needed; batch 2 matches per-function trace). See extension.ts for rationale.
async function runTrace(): Promise<void> {
  if (!openFile) {
    await pickPython();
    if (!openFile) return;
  }
  send({ type: 'traceState', state: { phase: 'tracing' } });
  try {
    const [raw, mod] = await Promise.all([
      getHelper().request<RawStructure>('structure_file', { path: openFile.path }),
      getHelper().request<TraceModuleResult>('trace_module', { path: openFile.path }),
    ]);
    send({ type: 'activeFile', structure: toFileStructure(raw, lineAt, moduleTraceToRaw(mod)) });
    send({ type: 'traceState', state: { phase: 'done', runId: String(stamp()) } });
    console.log(`[trace_module] ${mod.notes.length} traced, ${mod.problems.length} problem(s)`);
    mod.notes.forEach((n) => console.log('  •', n.note));
  } catch (e) {
    send({ type: 'traceState', state: { phase: 'error', message: String(e) } });
  }
}

async function runTraceFunction(name: string, line: number): Promise<void> {
  if (!openFile) return;
  send({ type: 'traceState', state: { phase: 'tracing', progress: name } });
  try {
    const [raw, res] = await Promise.all([
      getHelper().request<RawStructure>('structure_file', { path: openFile.path }),
      getHelper().request<RawTrace & { note?: string }>('trace_function', { path: openFile.path, name, line }),
    ]);
    send({ type: 'activeFile', structure: toFileStructure(raw, lineAt, res) });
    send({ type: 'traceState', state: { phase: 'done', runId: String(stamp()) } });
  } catch (e) {
    send({ type: 'traceState', state: { phase: 'error', message: String(e) } });
  }
}

async function pickDataFile(): Promise<void> {
  const r = await dialog.showOpenDialog({
    properties: ['openFile'],
    filters: [{ name: 'Data', extensions: ['csv', 'tsv', 'npy', 'parquet', 'pq'] }],
  });
  if (r.canceled || !r.filePaths[0]) return;
  try {
    const meta = await getHelper().request('load_file', { path: r.filePaths[0] });
    send({ type: 'dataView', meta: meta as never });
  } catch (e) {
    send({ type: 'traceState', state: { phase: 'error', message: String(e) } });
  }
}

// Monotonic run id without Date.now() (kept deterministic-friendly).
let _seq = 0;
const stamp = (): number => ++_seq;

async function onMessage(m: WebviewMessage): Promise<void> {
  switch (m.type) {
    case 'ready':
      break; // P1: wait for File ▸ Open. (P3 will restore the last workspace.)
    case 'requestTrace':
      await runTrace();
      break;
    case 'traceFunction':
      await runTraceFunction(m.name, m.line);
      break;
    case 'pickDataFile':
      await pickDataFile();
      break;
    case 'saveDocument':
      await saveDocument(m.path, m.text);
      break;
    case 'revealSymbol':
      break; // desktop scrolls Monaco in-renderer (revealInEditor); nothing to do host-side
  }
}

function buildMenu(): void {
  Menu.setApplicationMenu(
    Menu.buildFromTemplate([
      { role: 'appMenu' },
      {
        label: 'File',
        submenu: [
          { label: 'Open Python File…', accelerator: 'CmdOrCtrl+O', click: () => void pickPython() },
          { label: 'Open Data File…', accelerator: 'CmdOrCtrl+Shift+O', click: () => void pickDataFile() },
          { label: 'Trace Current File', accelerator: 'CmdOrCtrl+R', click: () => void runTrace() },
          { type: 'separator' },
          { role: 'close' },
        ],
      },
      { role: 'editMenu' },
      { label: 'View', submenu: [{ role: 'reload' }, { role: 'toggleDevTools' }, { role: 'togglefullscreen' }] },
      { role: 'windowMenu' },
    ]),
  );
}

function createWindow(): void {
  win = new BrowserWindow({
    width: 1400,
    height: 900,
    title: 'Fusion',
    backgroundColor: '#1e1e1e',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });
  // Inject the host theme as early as the DOM exists (minimal flash; window bg is dark).
  win.webContents.on('dom-ready', () => void win?.webContents.insertCSS(THEME_CSS));
  void win.loadFile(UI_INDEX);
  win.on('closed', () => (win = undefined));
}

ipcMain.on('renderer:message', (_e, m: WebviewMessage) => void onMessage(m));

void app.whenReady().then(() => {
  buildMenu();
  createWindow();
});
app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
});
app.on('window-all-closed', () => {
  helper?.dispose();
  if (process.platform !== 'darwin') app.quit();
});
