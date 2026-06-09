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
import { app, BrowserWindow, ipcMain, dialog, Menu, protocol } from 'electron';
import * as path from 'path';
import * as fs from 'fs';
import * as os from 'os';
import {
  HelperClient,
  AgentClient,
  loadAgentConfig,
  saveAgentConfig,
  toFileStructure,
  moduleTraceToRaw,
  type AgentConfig,
  type RawStructure,
  type RawTrace,
  type TraceModuleResult,
} from '@fusion/core';
import type { AgentConfigWire, CallGraph, HostMessage, WebviewMessage } from '@fusion/shared';

// hosts/desktop/dist/main.js -> up three to the repo root.
const REPO = path.resolve(__dirname, '..', '..', '..');
const LENS_HELPER = path.join(REPO, 'lens-helper');
const UI_DIST = path.join(REPO, 'webview-ui', 'dist');

// Serve the built UI over a PRIVILEGED custom scheme (fusion://) instead of file://.
// Chromium blocks ES-module <script type="module"> loads over file:// (CORS), which a
// Vite bundle is — so file:// gives a blank window. A standard+secure scheme behaves
// like https: modules, dynamic import() (Monaco code-split), and workers all load.
const UI_SCHEME = 'fusion';
const MIME: Record<string, string> = {
  '.html': 'text/html',
  '.js': 'text/javascript',
  '.mjs': 'text/javascript',
  '.css': 'text/css',
  '.json': 'application/json',
  '.map': 'application/json',
  '.svg': 'image/svg+xml',
  '.wasm': 'application/wasm',
  '.ttf': 'font/ttf',
  '.woff': 'font/woff',
  '.woff2': 'font/woff2',
};
protocol.registerSchemesAsPrivileged([
  // corsEnabled is required: Vite emits <script type="module" crossorigin>, and a
  // crossorigin module fetch over a custom scheme without CORS gets blocked -> blank page.
  { scheme: UI_SCHEME, privileges: { standard: true, secure: true, supportFetchAPI: true, stream: true, corsEnabled: true } },
]);

// Content-Security-Policy for the served document. Strict (no unsafe-eval): self scripts,
// inline styles (Monaco + our theme inject <style>), self/blob workers (Monaco's worker).
const CSP = [
  "default-src 'none'",
  "script-src 'self'",
  "style-src 'self' 'unsafe-inline'",
  "img-src 'self' data:",
  "font-src 'self' data:",
  "worker-src 'self' blob:",
  "child-src 'self' blob:",
  "connect-src 'self'",
].join('; ');

async function serveUi(request: Request): Promise<Response> {
  const { pathname } = new URL(request.url); // fusion://app/assets/main.js -> /assets/main.js
  const rel = pathname === '/' ? 'index.html' : decodeURIComponent(pathname).replace(/^\/+/, '');
  const file = path.join(UI_DIST, rel);
  if (!file.startsWith(UI_DIST)) return new Response('forbidden', { status: 403 }); // no path traversal
  try {
    const data = await fs.promises.readFile(file);
    const ext = path.extname(file).toLowerCase();
    const headers: Record<string, string> = {
      'content-type': MIME[ext] ?? 'application/octet-stream',
      'access-control-allow-origin': '*',
    };
    if (ext === '.html') headers['content-security-policy'] = CSP; // CSP applies to the document
    return new Response(data, { headers });
  } catch {
    return new Response('not found', { status: 404 });
  }
}

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
    const traced: RawTrace = { ...res, notes: res.note ? [{ line, note: res.note }] : [] };
    send({ type: 'activeFile', structure: toFileStructure(raw, lineAt, traced) });
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

// --- agent (CLI coding assistant) ----------------------------------------------
// Config lives at ~/.fusion/agent.json (written by the settings page in a later step);
// absent => the claude preset. Re-read per run so config edits take effect immediately.
const AGENT_CONFIG_FILE = path.join(os.homedir(), '.fusion', 'agent.json');
const agentRuns = new Map<string, AbortController>();

// Give the agent the open file as context so answers are about the user's code.
function agentPrompt(userText: string): string {
  if (!openFile) return userText;
  const code = openFile.lines.join('\n');
  return `You are an assistant inside an ML/DS IDE. The user is editing this Python file (${path.basename(
    openFile.path,
  )}):\n\n\`\`\`python\n${code}\n\`\`\`\n\nUser: ${userText}`;
}

async function runAgent(id: string, text: string): Promise<void> {
  const ctrl = new AbortController();
  agentRuns.set(id, ctrl);
  try {
    const client = new AgentClient(loadAgentConfig(AGENT_CONFIG_FILE));
    const full = await client.run(agentPrompt(text), {
      signal: ctrl.signal,
      onChunk: (delta) => send({ type: 'agentChunk', id, delta }),
    });
    send({ type: 'agentDone', id, text: full });
  } catch (e) {
    send({ type: 'agentError', id, message: e instanceof Error ? e.message : String(e) });
  } finally {
    agentRuns.delete(id);
  }
}

// --- agent config (settings page) ----------------------------------------------
function toWire(c: AgentConfig): AgentConfigWire {
  return {
    kind: c.kind,
    command: c.command,
    args: c.args,
    promptVia: c.promptVia,
    trust: c.trust ?? 'review',
    model: c.model,
  };
}

// Settings "Test": run the (unsaved) config with a tiny prompt to confirm it's reachable.
async function testAgentConfig(id: string, cfg: AgentConfig): Promise<void> {
  try {
    const out = await new AgentClient(cfg).run('Reply with exactly: ok', { timeoutMs: 25000 });
    send({ type: 'agentTestResult', id, ok: true, message: out.trim().slice(0, 200) || '(empty reply)' });
  } catch (e) {
    send({ type: 'agentTestResult', id, ok: false, message: e instanceof Error ? e.message : String(e) });
  }
}

// --- trace-assist: agent authors a `# fusion:` directive -----------------------
const extractDirectives = (text: string): string =>
  text
    .split('\n')
    .map((l) => l.trim())
    .filter((l) => /^#\s*fusion:\s*(input|model)\s*=/.test(l))
    .join('\n');

const traceAssistPrompt = (file: string, src: string, name: string, reason: string): string =>
  [
    `Our PyTorch shape tracer can't auto-call \`${name}\` in ${file} (reason: ${reason || 'needs an input'}).`,
    `Write a "# fusion:" directive that makes it traceable:`,
    `  # fusion: input = <expr>      (forward input; a (a, b) tuple supplies multiple positional args)`,
    `  # fusion: model = Class(...)  (only if the model needs constructor args)`,
    `Use torch.randn / torch.randint with realistic shapes and dtypes for THIS model.`,
    `Reply with ONLY the directive line(s) — no prose, no code fences.`,
    '',
    '```python',
    src,
    '```',
  ].join('\n');

const retryPrompt = (file: string, src: string, name: string, prev: string, error: string): string =>
  [
    `Your "# fusion:" directive for \`${name}\` in ${file}:`,
    prev,
    `FAILED when traced: ${error}`,
    `Fix the shapes / dtypes / number of args. Reply with ONLY the corrected "# fusion:" line(s).`,
    '',
    '```python',
    src,
    '```',
  ].join('\n');

// Trace a CANDIDATE directive against a temp copy (never touches the user's file) to verify
// it works. Returns an error string if it still fails, or null if it traced clean.
let _tmpN = 0;
async function tryDirective(srcLines: string[], line: number, directive: string, name: string): Promise<string | null> {
  const lines = [...srcLines];
  const idx = Math.max(0, line - 1);
  const indent = (lines[idx] || '').match(/^\s*/)?.[0] ?? '';
  const dir = directive
    .split('\n')
    .map((d) => d.trim())
    .filter(Boolean)
    .map((d) => indent + d);
  lines.splice(idx, 0, ...dir);
  const tmp = path.join(os.tmpdir(), `fusion-try-${process.pid}-${_tmpN++}.py`);
  fs.writeFileSync(tmp, lines.join('\n'), 'utf8');
  try {
    const res = await getHelper().request<{ error?: string | null; records?: Record<string, unknown> }>(
      'trace_function',
      { path: tmp, name, line: 0 }, // line 0 -> find by name (the directive shifted the line)
    );
    if (res.error) return String(res.error);
    if (!res.records || Object.keys(res.records).length === 0) return 'traced but captured no shapes';
    return null;
  } catch (e) {
    return e instanceof Error ? e.message : String(e);
  } finally {
    try {
      fs.unlinkSync(tmp);
    } catch {
      /* ignore */
    }
  }
}

// agent <-> tracer loop: propose a directive, TEST it, feed errors back, retry up to 3 rounds.
async function runTraceAssist(id: string, p: string, name: string, line: number): Promise<void> {
  let reason = '';
  try {
    const res = await getHelper().request<{ note?: string }>('trace_function', { path: p, name, line });
    reason = res.note ?? '';
  } catch {
    /* ignore */
  }
  const srcLines = openFile && openFile.path === p ? [...openFile.lines] : fs.readFileSync(p, 'utf8').split('\n');
  const src = srcLines.join('\n');
  const cfg = loadAgentConfig(AGENT_CONFIG_FILE);
  const client = new AgentClient(cfg);

  let directive = '';
  let err: string | null = reason || 'needs an input';
  const MAX = 3;
  for (let attempt = 0; attempt < MAX; attempt++) {
    const prompt =
      attempt === 0
        ? traceAssistPrompt(path.basename(p), src, name, reason)
        : retryPrompt(path.basename(p), src, name, directive, err ?? '');
    let full = '';
    try {
      full = await client.run(prompt, { onChunk: (delta) => send({ type: 'agentChunk', id, delta }) });
    } catch (e) {
      send({ type: 'agentError', id, message: e instanceof Error ? e.message : String(e) });
      return;
    }
    directive = extractDirectives(full);
    if (!directive) {
      send({ type: 'agentDone', id, text: full });
      return;
    }
    err = await tryDirective(srcLines, line, directive, name);
    if (!err) break; // verified — traces clean
    if (attempt < MAX - 1) send({ type: 'agentChunk', id, delta: `\n[attempt ${attempt + 1} failed: ${err} — retrying]\n` });
  }

  send({ type: 'agentDone', id, text: `Proposed for ${name}()${err ? ` (still failing: ${err})` : ' — verified ✓'}:\n${directive}` });
  if ((cfg.trust ?? 'review') === 'auto') {
    await insertDirective(p, line, directive);
  } else {
    send({
      type: 'directiveProposed',
      forFunction: name,
      path: p,
      line,
      directive,
      explanation: err ? `still failing: ${err}` : 'verified ✓ — traces clean',
    });
  }
}

// Write a directive just above `line` (matching indentation), refresh the editor, re-trace.
async function insertDirective(p: string, line: number, text: string): Promise<void> {
  const src = fs.readFileSync(p, 'utf8').split('\n');
  const idx = Math.max(0, line - 1);
  const indent = (src[idx] || '').match(/^\s*/)?.[0] ?? '';
  const dir = text
    .split('\n')
    .map((d) => d.trim())
    .filter(Boolean)
    .map((d) => indent + d);
  src.splice(idx, 0, ...dir);
  const newText = src.join('\n');
  fs.writeFileSync(p, newText, 'utf8');
  if (openFile && openFile.path === p) openFile.lines = newText.split('\n');
  send({ type: 'openDocument', path: p, text: newText, language: 'python' });
  await runTrace(); // trace_module now honors the new directive
}

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
    case 'agentPrompt':
      await runAgent(m.id, m.text);
      break;
    case 'agentCancel':
      agentRuns.get(m.id)?.abort();
      break;
    case 'getAgentConfig':
      send({ type: 'agentConfig', config: toWire(loadAgentConfig(AGENT_CONFIG_FILE)) });
      break;
    case 'saveAgentConfig':
      saveAgentConfig(AGENT_CONFIG_FILE, m.config as AgentConfig);
      send({ type: 'agentConfig', config: m.config });
      break;
    case 'testAgentConfig':
      await testAgentConfig(m.id, m.config as AgentConfig);
      break;
    case 'traceAssist':
      await runTraceAssist(m.id, m.path, m.name, m.line);
      break;
    case 'insertDirective':
      await insertDirective(m.path, m.line, m.text);
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
  // Surface load failures instead of a silent blank window.
  win.webContents.on('did-fail-load', (_e, code, desc, url) =>
    console.error(`[fusion] renderer failed to load: ${code} ${desc} ${url}`),
  );
  win.webContents.on('render-process-gone', (_e, d) => console.error('[fusion] render-process-gone', d));
  void win.loadURL(`${UI_SCHEME}://app/index.html`);
  win.on('closed', () => (win = undefined));
}

ipcMain.on('renderer:message', (_e, m: WebviewMessage) => void onMessage(m));

void app.whenReady().then(() => {
  protocol.handle(UI_SCHEME, serveUi);
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
