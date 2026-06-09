// Fusion Cockpit — extension host (M1 shell).
//
// Owns CockpitState (active file -> structure, debounced+cached), the editor-area
// WebviewPanel, the typed message bus, and the HelperClient wiring. The webview is
// a Svelte app built to webview-ui/dist, loaded via asWebviewUri (see BUILD_STATUS.md).
import * as vscode from 'vscode';
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

let panel: vscode.WebviewPanel | undefined;
let helper: HelperClient | undefined;
let debounce: ReturnType<typeof setTimeout> | undefined;
// The last REAL text editor. activeTextEditor becomes undefined when the cockpit
// webview has focus, so we can't read it at send time — track it as it changes.
let lastEditor: vscode.TextEditor | undefined;
let extensionDir: string | undefined; // <project>/extension — used to locate lens-helper
let log: vscode.OutputChannel | undefined;
const L = (s: string): void => log?.appendLine(s);

export function activate(context: vscode.ExtensionContext): void {
  lastEditor = vscode.window.activeTextEditor;
  extensionDir = context.extensionPath;
  log = vscode.window.createOutputChannel('Fusion Cockpit');
  context.subscriptions.push(log);
  L(`activated. extensionDir=${extensionDir}  activeEditor=${lastEditor?.document.fileName ?? 'none'}`);
  context.subscriptions.push(
    vscode.commands.registerCommand('fusion.openCockpit', () => openCockpit(context)),
    vscode.commands.registerCommand('fusion.traceFile', () => runTrace()),
    vscode.window.onDidChangeActiveTextEditor((ed) => {
      if (ed) lastEditor = ed; // keep the last real editor; ignore webview-focus (undefined)
      scheduleStructure();
    }),
    vscode.workspace.onDidSaveTextDocument(() => scheduleStructure()),
  );
}

export function deactivate(): void {
  helper?.dispose();
}

// --- helper client -------------------------------------------------------------
function getHelper(): HelperClient {
  if (!helper) {
    const py = resolvePython();
    const cwd = helperCwd();
    L(`helper: spawning python='${py}' cwd='${cwd}'`);
    helper = new HelperClient({ python: py, cwd, onStderr: (l) => L('helper.stderr: ' + l) });
  }
  return helper;
}

function helperCwd(): string | undefined {
  // Prefer a path relative to the extension (works regardless of which folder the
  // dev-host opened): <project>/extension/../lens-helper. Fall back to the workspace.
  if (extensionDir) return path.join(extensionDir, '..', 'lens-helper');
  const ws = vscode.workspace.workspaceFolders?.[0];
  return ws ? path.join(ws.uri.fsPath, 'lens-helper') : undefined;
}

// Resolve a Python interpreter that actually EXISTS — the user's selected env can be
// stale (a deleted conda env). Tries: our setting, the Python ext's selection,
// python.defaultInterpreterPath, then known-good fallbacks. Absolute paths must exist.
function resolvePython(): string {
  const cands: string[] = [];
  const ours = vscode.workspace.getConfiguration('fusion').get<string>('pythonPath');
  if (ours) cands.push(ours);
  try {
    const api = vscode.extensions.getExtension('ms-python.python')?.exports as
      | { environments?: { getActiveEnvironmentPath?: () => { path?: string } } }
      | undefined;
    const p = api?.environments?.getActiveEnvironmentPath?.()?.path;
    if (typeof p === 'string') cands.push(p);
  } catch {
    /* python ext not present */
  }
  const cfg = vscode.workspace.getConfiguration('python').get<string>('defaultInterpreterPath');
  if (cfg) cands.push(cfg);
  cands.push('/opt/anaconda3/bin/python3', '/usr/local/bin/python3', 'python3');
  for (const c of cands) {
    if (!c.includes('/')) return c; // bare name -> trust PATH
    if (fs.existsSync(c)) return c; // absolute -> must exist
  }
  return 'python3';
}

// --- panel ---------------------------------------------------------------------
function openCockpit(context: vscode.ExtensionContext): void {
  if (panel) {
    panel.reveal(vscode.ViewColumn.Beside);
    return;
  }
  panel = vscode.window.createWebviewPanel('fusionCockpit', 'Fusion Cockpit', vscode.ViewColumn.Beside, {
    enableScripts: true,
    retainContextWhenHidden: true,
    localResourceRoots: [vscode.Uri.joinPath(vscode.Uri.file(extensionDir!), '..', 'webview-ui', 'dist')],
  });
  panel.webview.html = html(panel.webview);
  panel.onDidDispose(() => (panel = undefined));
  panel.webview.onDidReceiveMessage((m: WebviewMessage) => onMessage(m));
  scheduleStructure(); // populate for whatever is already active
}

function post(msg: HostMessage): void {
  panel?.webview.postMessage(msg);
}

function onMessage(m: WebviewMessage): void {
  switch (m.type) {
    case 'ready':
      scheduleStructure();
      break;
    case 'requestTrace':
      runTrace();
      break;
    case 'traceFunction':
      runTraceFunction(m.name, m.line);
      break;
    case 'pickDataFile':
      pickDataFile();
      break;
    case 'revealSymbol':
      revealSymbol(m.path, m.line);
      break;
    // setPrimaryZone / toggleHintDensity are handled inside the webview for M1
  }
}

// --- structure (Mode 1, static) ------------------------------------------------
function scheduleStructure(): void {
  if (debounce) clearTimeout(debounce);
  debounce = setTimeout(() => void sendStructure(), 300); // debounce + (cache is by path below)
}

async function sendStructure(): Promise<void> {
  const ed = lastEditor;
  if (!panel) {
    L('structure: skipped (cockpit panel not open)');
    return;
  }
  if (!ed || ed.document.languageId !== 'python') {
    L(`structure: skipped (no python editor; lastEditor=${ed?.document.fileName ?? 'undefined'})`);
    return;
  }
  const p = ed.document.uri.fsPath;
  L(`structure: requesting ${p}`);
  try {
    const raw = await getHelper().request<RawStructure>('structure_file', { path: p });
    L(`structure: got ${raw.functions?.length ?? 0} functions`);
    post({ type: 'activeFile', structure: toFileStructure(raw, (n) => ed.document.lineAt(n - 1).text, undefined) });
    void sendCallGraph(p);
  } catch (e) {
    L(`structure: ERROR ${errMsg(e)}`);
    post({ type: 'traceState', state: { phase: 'error', message: errMsg(e) } });
  }
}

async function sendCallGraph(p: string): Promise<void> {
  if (!panel) return;
  try {
    const g = await getHelper().request<CallGraph>('callgraph_file', { path: p });
    L(`callgraph: ${g.nodes.length} nodes, ${g.edges.length} edges`);
    post({ type: 'callGraph', graph: g });
  } catch (e) {
    L(`callgraph: ERROR ${errMsg(e)}`);
  }
}

// --- trace (Mode 1, runtime shapes) --------------------------------------------
// "Trace this file" = trace_module: for every model class + zero-arg function in the
// file, build-check the constructor and synth-call forward (batch 2). No __main__
// needed (works on libraries), and the batch matches the per-function "▶ trace" so the
// two never disagree. Each crashing function shows its own red line.
async function runTrace(): Promise<void> {
  const ed = lastEditor;
  if (!panel || !ed || ed.document.languageId !== 'python') {
    vscode.window.showWarningMessage('Open a Python file to trace.');
    return;
  }
  const p = ed.document.uri.fsPath;
  post({ type: 'traceState', state: { phase: 'tracing' } });
  try {
    const [raw, mod] = await Promise.all([
      getHelper().request<RawStructure>('structure_file', { path: p }),
      getHelper().request<TraceModuleResult>('trace_module', { path: p }),
    ]);
    post({
      type: 'activeFile',
      structure: toFileStructure(raw, (n) => ed.document.lineAt(n - 1).text, moduleTraceToRaw(mod)),
    });
    // Caught shape crashes are a SUCCESS of the tool (shown inline on the bad lines).
    post({ type: 'traceState', state: { phase: 'done', runId: String(Date.now()) } });
    L(`trace_module: ${mod.notes.length} traced, ${mod.problems.length} problem(s)`);
    mod.notes.forEach((n) => L(`  • ${n.note}`));
    const msg = mod.problems.length
      ? `Fusion ▶ ${mod.problems.length} shape problem(s) — see red lines`
      : `Fusion ▶ traced ${mod.notes.filter((n) => n.note.includes('randn')).length} function(s), synth batch 2`;
    vscode.window.setStatusBarMessage(msg, 6000);
  } catch (e) {
    post({ type: 'traceState', state: { phase: 'error', message: errMsg(e) } });
  }
}

// Call ONE function directly (no __main__, no debugger). The helper auto-synthesizes
// an input and returns `note` = the exact call it ran, so the result is reproducible
// and the input is never hidden. Empty records (e.g. "needs args") => show the reason.
async function runTraceFunction(name: string, line: number): Promise<void> {
  const ed = lastEditor;
  if (!panel || !ed || ed.document.languageId !== 'python') {
    vscode.window.showWarningMessage('Open a Python file to trace.');
    return;
  }
  const p = ed.document.uri.fsPath;
  post({ type: 'traceState', state: { phase: 'tracing', progress: name } });
  L(`traceFunction: ${name}@${line} in ${p}`);
  try {
    const [raw, res] = await Promise.all([
      getHelper().request<RawStructure>('structure_file', { path: p }),
      getHelper().request<RawTrace & { note?: string }>('trace_function', { path: p, name, line }),
    ]);
    post({ type: 'activeFile', structure: toFileStructure(raw, (n) => ed.document.lineAt(n - 1).text, res) });
    post({ type: 'traceState', state: { phase: 'done', runId: String(Date.now()) } });
    const captured = Object.keys(res.records ?? {}).length;
    L(`traceFunction: note='${res.note ?? ''}' error='${res.error ?? ''}' lines=${captured}`);
    if (res.note) {
      // Shapes filled in -> a quiet status line. Nothing captured (couldn't auto-call)
      // -> a visible info toast so the reason isn't silently swallowed.
      if (captured > 0) vscode.window.setStatusBarMessage(`Fusion ▶ ${res.note}`, 6000);
      else vscode.window.showInformationMessage(`Fusion: ${res.note}`);
    }
  } catch (e) {
    L(`traceFunction: ERROR ${errMsg(e)}`);
    post({ type: 'traceState', state: { phase: 'error', message: errMsg(e) } });
  }
}

// --- data viz (Mode 2) ---------------------------------------------------------
async function pickDataFile(): Promise<void> {
  const picked = await vscode.window.showOpenDialog({
    canSelectMany: false,
    filters: { Data: ['csv', 'tsv', 'npy', 'parquet', 'pq'] },
  });
  if (!picked?.[0] || !panel) return;
  try {
    const meta = await getHelper().request<Record<string, unknown>>('load_file', {
      path: picked[0].fsPath,
    });
    post({ type: 'dataView', meta: meta as never });
  } catch (e) {
    post({ type: 'traceState', state: { phase: 'error', message: errMsg(e) } });
  }
}

function revealSymbol(p: string, line: number): void {
  vscode.workspace.openTextDocument(p).then((doc) => {
    vscode.window.showTextDocument(doc, { viewColumn: vscode.ViewColumn.One }).then((ed) => {
      const pos = new vscode.Position(Math.max(0, line - 1), 0);
      ed.revealRange(new vscode.Range(pos, pos), vscode.TextEditorRevealType.InCenter);
      ed.selection = new vscode.Selection(pos, pos);
    });
  });
}


function errMsg(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

// --- webview: load the built Svelte bundle (webview-ui/dist) ------------------
function distAssets(): vscode.Uri {
  return vscode.Uri.joinPath(vscode.Uri.file(extensionDir!), '..', 'webview-ui', 'dist', 'assets');
}

function html(webview: vscode.Webview): string {
  const assets = distAssets();
  const js = webview.asWebviewUri(vscode.Uri.joinPath(assets, 'main.js'));
  const css = webview.asWebviewUri(vscode.Uri.joinPath(assets, 'main.css'));
  const nonce = Array.from({ length: 20 }, () => Math.floor(Math.random() * 36).toString(36)).join('');
  const csp = [
    `default-src 'none'`,
    `img-src ${webview.cspSource} data:`,
    `style-src ${webview.cspSource} 'unsafe-inline'`,
    `script-src 'nonce-${nonce}'`,
    `connect-src ${webview.cspSource}`,
  ].join('; ');
  return `<!DOCTYPE html><html><head><meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="${csp}">
<link rel="stylesheet" href="${css}">
</head><body><div id="app"></div>
<script type="module" nonce="${nonce}" src="${js}"></script>
</body></html>`;
}
