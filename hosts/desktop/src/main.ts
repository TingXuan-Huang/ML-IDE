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
  parseRemembers,
  stripRememberLines,
  appendMemoryFact,
  memoryPromptBlock,
  formatTranscript,
  type AgentConfig,
  type RawStructure,
  type RawTrace,
  type TraceModuleResult,
} from '@fusion/core';
import type { AgentConfigWire, CallGraph, CompareResult, Conversation, ConversationMeta, HostMessage, ModuleSummary, PaperView, ProjectGraphWire, TracingConfigWire, WebviewMessage } from '@fusion/shared';

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
// Last file-trace's dim-symbol map (value -> B/L/D…). Persisted so a per-function ▶ trace
// keeps the abstract view; cleared when a different file is opened.
let _lastDimNames: Record<string, string> = {};

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
  _lastDimNames = {}; // new file -> drop the previous model's dim symbols
  _lastPaper = undefined; // and the previous paper view
  win?.setTitle(`Fusion — ${path.basename(p)}`);
  send({ type: 'openDocument', path: p, text, language: 'python' }); // load into the Monaco pane
  const raw = await getHelper().request<RawStructure>('structure_file', { path: p });
  send({ type: 'activeFile', structure: toFileStructure(raw, lineAt, undefined) });
  try {
    send({ type: 'callGraph', graph: await getHelper().request<CallGraph>('callgraph_file', { path: p }) });
  } catch (e) {
    console.error('callgraph failed', e);
  }
  if (loadTracing().autoTrace) void runTrace(); // trace as soon as the file opens
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

// --- open-a-folder (project mode) ----------------------------------------------
let workspaceRoot: string | undefined;
async function openFolder(dir: string): Promise<void> {
  try {
    const { root, files } = await getHelper().request<{ root: string; files: string[] }>('list_folder', { path: dir });
    workspaceRoot = root;
    win?.setTitle(`Fusion — ${path.basename(root)}/`);
    send({ type: 'folderTree', root, files });
  } catch (e) {
    send({ type: 'traceState', state: { phase: 'error', message: `open folder failed: ${String(e)}` } });
  }
}
async function pickFolder(): Promise<void> {
  const r = await dialog.showOpenDialog({ properties: ['openDirectory'] });
  if (!r.canceled && r.filePaths[0]) await openFolder(r.filePaths[0]);
}
async function runProjectGraph(): Promise<void> {
  if (!workspaceRoot) return;
  try {
    send({ type: 'projectGraph', graph: await getHelper().request<ProjectGraphWire>('project_graph', { path: workspaceRoot }) });
  } catch (e) {
    console.error('[fusion] project_graph failed', e);
  }
}

// #2 — torchinfo-style model summary for the open file (lazy; Summary tab). Merely viewing
// the tab with no file open shows an empty state (no surprise file dialog); a transport
// failure surfaces as an error summary so the tab never sticks on the spinner.
async function runModelSummary(): Promise<void> {
  if (!openFile) {
    send({ type: 'modelSummary', summary: { path: '', target: '', rows: [], totalParams: 0, trainableParams: 0, paramBytes: 0, dims: {}, error: 'Open a Python file to see its model summary.' } });
    return;
  }
  try {
    const summary = await getHelper().request<ModuleSummary>('module_summary', { path: openFile.path, projectRoot: workspaceRoot ?? '' });
    send({ type: 'modelSummary', summary });
  } catch (e) {
    send({ type: 'modelSummary', summary: { path: openFile.path, target: '', rows: [], totalParams: 0, trainableParams: 0, paramBytes: 0, dims: {}, error: `summary failed: ${String(e)}` } });
  }
}

// #8 — paper-reading view for the open file (lazy; Paper tab). View failures go through the
// paper channel (problems) so the Paper tab shows them, not the Explain prose panel.
let _lastPaper: PaperView | undefined;
async function runPaper(): Promise<void> {
  if (!openFile) {
    send({ type: 'paperView', paper: { path: '', sections: [], dims: {}, problems: [{ line: 1, message: 'Open a Python file to read it as a paper.' }] } });
    return;
  }
  try {
    const paper = await getHelper().request<PaperView>('paper_module', { path: openFile.path, projectRoot: workspaceRoot ?? '' });
    _lastPaper = paper;
    send({ type: 'paperView', paper });
  } catch (e) {
    send({ type: 'paperView', paper: { path: openFile.path, sections: [], dims: {}, problems: [{ line: 1, message: `paper view failed: ${String(e)}` }] } });
  }
}

// #5 — pick a real data file and pin it as this function's input via a load() directive.
// The written path is portable: project-relative if the file is under the workspace, else
// relative to the open file's dir, else absolute (the escape hatch). realpath-resolved so a
// symlinked workspace still yields a clean relative path.
async function traceWithRealInput(p: string, name: string, line: number): Promise<void> {
  const r = await dialog.showOpenDialog({
    properties: ['openFile'],
    filters: [{ name: 'Data', extensions: ['npy', 'npz', 'pt', 'pth', 'csv', 'tsv'] }],
  });
  if (r.canceled || !r.filePaths[0]) return;
  const real = (q: string): string => {
    try {
      return fs.realpathSync(q);
    } catch {
      return q;
    }
  };
  const chosen = real(r.filePaths[0]);
  const portable = (base: string): string | null => {
    const rel = path.relative(real(base), chosen);
    return rel && !rel.startsWith('..') && !path.isAbsolute(rel) ? rel : null;
  };
  const rel = (workspaceRoot && portable(workspaceRoot)) || portable(path.dirname(p)) || chosen;
  await insertDirective(p, line, `# fusion: input = load("${rel.split(path.sep).join('/')}")`); // placed above the method; refreshes + re-traces
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
      getHelper().request<TraceModuleResult>('trace_module', { path: openFile.path, projectRoot: workspaceRoot ?? '' }),
    ]);
    _lastDimNames = mod.dims ?? {};
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
      getHelper().request<RawTrace & { note?: string }>('trace_function', { path: openFile.path, name, line, projectRoot: workspaceRoot ?? '' }),
    ]);
    // Merge this trace's dim symbols over the last file-trace's (fresher keys win), so the
    // abstract view survives — and improves with — a per-function trace.
    _lastDimNames = { ..._lastDimNames, ...(res.dims ?? {}) };
    const traced: RawTrace = { ...res, notes: res.note ? [{ line, note: res.note }] : [], dims: _lastDimNames };
    send({ type: 'activeFile', structure: toFileStructure(raw, lineAt, traced) });
    send({ type: 'traceState', state: { phase: 'done', runId: String(stamp()) } });
  } catch (e) {
    send({ type: 'traceState', state: { phase: 'error', message: String(e) } });
  }
}

// design B — faithful-port compare: trace both files as paper sections and diff matched forwards.
// A = this (open) file, B = the picked reference.
async function runCompare(pathA: string, pathB: string): Promise<void> {
  // Clear any stale diff + show "comparing…" while both files trace (can be multi-second).
  send({ type: 'compareResult', result: { pathA, pathB, matched: [], onlyA: [], onlyB: [], pending: true } });
  try {
    const result = await getHelper().request<CompareResult>('compare_traces', {
      pathA,
      pathB,
      projectRoot: workspaceRoot ?? '',
    });
    send({ type: 'compareResult', result });
  } catch (e) {
    // Route the failure back into the Compare zone (it shows result.error) rather than the
    // global trace spinner, so the zone never sticks loading.
    send({ type: 'compareResult', result: { pathA, pathB, matched: [], onlyA: [], onlyB: [], error: String(e) } });
  }
}

// Pick a reference .py and compare it against the currently-open file (A). Opens the file
// dialog (mirrors pickPython); A is always the open file, so the common case is one click.
async function pickReferenceAndCompare(): Promise<void> {
  if (!openFile) {
    await pickPython();
    if (!openFile) return;
  }
  const r = await dialog.showOpenDialog({
    title: 'Pick a reference implementation to compare against this file',
    properties: ['openFile'],
    filters: [{ name: 'Python', extensions: ['py'] }],
  });
  if (r.canceled || !r.filePaths[0]) return;
  send({ type: 'focus', zone: 'compare' });
  await runCompare(openFile.path, r.filePaths[0]);
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

// --- tracing config (Tracing settings tab) -------------------------------------
const TRACING_CONFIG_FILE = path.join(os.homedir(), '.fusion', 'tracing.json');
const DEFAULT_TRACING: TracingConfigWire = { density: 'changed', abstract: false, autoTrace: false, retries: 3, meta: false };
function loadTracing(): TracingConfigWire {
  try {
    // Drop legacy batch/seq keys from older config files (the synth now uses fixed dims).
    const { batch, seq, ...rest } = JSON.parse(fs.readFileSync(TRACING_CONFIG_FILE, 'utf8')) as Record<string, unknown>;
    void batch;
    void seq;
    return { ...DEFAULT_TRACING, ...rest };
  } catch {
    return { ...DEFAULT_TRACING };
  }
}
function saveTracing(c: TracingConfigWire): void {
  fs.mkdirSync(path.dirname(TRACING_CONFIG_FILE), { recursive: true });
  fs.writeFileSync(TRACING_CONFIG_FILE, JSON.stringify(c, null, 2));
}

// --- durable agent memory (Memory settings tab) --------------------------------
// A plain-markdown file injected into every agent prompt, so the CLI agent remembers
// project conventions across sessions. Global for now; per-project memory arrives with
// open-a-folder. Read per run so manual edits take effect immediately.
const MEMORY_FILE = path.join(os.homedir(), '.fusion', 'memory.md');
function loadMemory(): string {
  try {
    return fs.readFileSync(MEMORY_FILE, 'utf8');
  } catch {
    return '';
  }
}
function saveMemory(text: string): void {
  fs.mkdirSync(path.dirname(MEMORY_FILE), { recursive: true });
  fs.writeFileSync(MEMORY_FILE, text);
}
/** Append one fact (deduped); returns true if memory actually changed. */
function appendMemory(fact: string): boolean {
  const cur = loadMemory();
  const next = appendMemoryFact(cur, fact);
  if (next === cur) return false;
  saveMemory(next);
  return true;
}
/** Save facts the agent surfaced, then tell the webview (refresh the tab + a chat note). */
function rememberFacts(facts: string[]): void {
  const saved = facts.filter((f) => appendMemory(f));
  if (!saved.length) return;
  send({ type: 'agentMemory', text: loadMemory() });
  send({ type: 'memoryNote', text: `📌 Remembered: ${saved.join('; ')}` });
}

// --- saved conversations (chat history / resume) -------------------------------
// One JSON file per conversation under ~/.fusion/conversations/. The webview owns the
// live chat + autosaves after each turn; here we just persist, list, and serve them back.
const CONVO_DIR = path.join(os.homedir(), '.fusion', 'conversations');
const convoFile = (id: string): string => path.join(CONVO_DIR, `${id.replace(/[^a-zA-Z0-9_-]/g, '_')}.json`);
function saveConversation(c: Conversation): void {
  fs.mkdirSync(CONVO_DIR, { recursive: true });
  fs.writeFileSync(convoFile(c.id), JSON.stringify(c, null, 2));
}
function listConversations(): ConversationMeta[] {
  let files: string[] = [];
  try {
    files = fs.readdirSync(CONVO_DIR).filter((f) => f.endsWith('.json'));
  } catch {
    return [];
  }
  const out: ConversationMeta[] = [];
  for (const f of files) {
    try {
      const c = JSON.parse(fs.readFileSync(path.join(CONVO_DIR, f), 'utf8')) as Conversation;
      out.push({ id: c.id, title: c.title, updatedAt: c.updatedAt, count: c.messages?.length ?? 0 });
    } catch {
      /* skip unreadable */
    }
  }
  return out.sort((a, b) => b.updatedAt - a.updatedAt);
}
function loadConversation(id: string): Conversation | null {
  try {
    return JSON.parse(fs.readFileSync(convoFile(id), 'utf8')) as Conversation;
  } catch {
    return null;
  }
}
function deleteConversation(id: string): void {
  try {
    fs.unlinkSync(convoFile(id));
  } catch {
    /* already gone */
  }
}
// Serialize agent runs so file-level ✦ ask doesn't spawn N CLI processes at once.
let _agentChain: Promise<unknown> = Promise.resolve();
let _agentGen = 0; // bumped by Stop -> queued runs from an older gen are skipped
const queueAgent = (fn: () => Promise<void>): void => {
  const gen = _agentGen;
  _agentChain = _agentChain.then(() => (gen === _agentGen ? fn() : undefined)).catch((e) => console.error('[fusion] agent run failed', e));
};
// Active agent runs (chat + trace-assist), keyed by run id, so Stop can abort them.
const agentRuns = new Map<string, AbortController>();
// Stop: abort every in-flight agent + drop the queue.
function cancelAllAgents(): void {
  _agentGen += 1;
  for (const c of agentRuns.values()) c.abort();
  agentRuns.clear();
}

// Give the agent the open file as context so answers are about the user's code.
const REMEMBER_HINT =
  '\n\nIf you learned a durable fact about this project worth keeping (a model config, an ' +
  'input convention, a user preference), add it as a final line:  # fusion: remember <fact>';

function agentPrompt(userText: string, history?: Array<{ role: string; text: string }>): string {
  const mem = memoryPromptBlock(loadMemory());
  const transcript = formatTranscript(history ?? []);
  const convo = transcript ? `Conversation so far:\n${transcript}\n\n` : '';
  if (!openFile) return `${mem}${convo}User: ${userText}${REMEMBER_HINT}`;
  const code = openFile.lines.join('\n');
  return (
    `${mem}You are an assistant inside an ML/DS IDE. The user is editing this Python file (${path.basename(
      openFile.path,
    )}):\n\n\`\`\`python\n${code}\n\`\`\`\n\n${convo}User: ${userText}` + REMEMBER_HINT
  );
}

async function runAgent(id: string, text: string, history?: Array<{ role: string; text: string }>): Promise<void> {
  const ctrl = new AbortController();
  agentRuns.set(id, ctrl);
  try {
    const client = new AgentClient(loadAgentConfig(AGENT_CONFIG_FILE));
    const full = await client.run(agentPrompt(text, history), {
      signal: ctrl.signal,
      onChunk: (delta) => send({ type: 'agentChunk', id, delta }),
    });
    // Capture any `# fusion: remember <fact>` lines to durable memory, and show the reply
    // without them (they're memory, not prose).
    const facts = parseRemembers(full);
    send({ type: 'agentDone', id, text: facts.length ? stripRememberLines(full) : full });
    rememberFacts(facts);
  } catch (e) {
    send({ type: 'agentError', id, message: e instanceof Error ? e.message : String(e) });
  } finally {
    agentRuns.delete(id);
  }
}

// #8 — "Explain like a paper": serialize the traced forward pass + source, ask the agent
// for a Method-section description. Streams into the Paper panel (not the chat). Reuses
// the agent infra (memory injection, remember-loop, agentRuns Stop).
function paperExplainPrompt(paper: PaperView, src: string, mem: string): string {
  const sections = paper.sections
    .map((s) => `## ${s.module}  —  ${s.forwardNote}\n` + s.steps.map((st) => `  ${st.lhs} = ${st.op ?? st.shapes.map((sh) => `[${sh.shape.join(',')}]`).join(' ')}`).join('\n'))
    .join('\n\n');
  return (
    `${mem}You are writing the Method section of an ML paper. Below is the TRACED forward pass of a model — ordered tensor operations with shapes (B=batch, L=sequence, D=model dim, H=heads, dh=head dim where applicable). ` +
    `Describe the architecture in clear, concise paper prose, grouping by module and naming standard components (multi-head self-attention, residual connections, LayerNorm, MLP/feed-forward) where the ops imply them. Use inline math like softmax(QKᵀ/√dh)V where appropriate. Do NOT invent layers that aren't in the trace.\n\n` +
    `TRACED FORWARD PASS:\n${sections}\n\n\`\`\`python\n${src}\n\`\`\`` +
    REMEMBER_HINT
  );
}

// Tracked at ENQUEUE time so a cancel lands even while the run is still queued behind
// another agent run (cleared/superseded -> the dequeued run no-ops).
let _paperExplainId: string | undefined;
async function runPaperExplain(id: string): Promise<void> {
  if (id !== _paperExplainId) return; // cancelled or superseded while queued
  if (!_lastPaper || !_lastPaper.sections.length) {
    send({ type: 'paperExplainError', message: 'Trace the file first (open the Paper tab), then explain.' });
    return;
  }
  const ctrl = new AbortController();
  agentRuns.set(id, ctrl);
  try {
    const src = openFile ? openFile.lines.join('\n') : '';
    const client = new AgentClient(loadAgentConfig(AGENT_CONFIG_FILE));
    const full = await client.run(paperExplainPrompt(_lastPaper, src, memoryPromptBlock(loadMemory())), {
      signal: ctrl.signal,
      onChunk: (delta) => send({ type: 'paperExplainChunk', delta }),
    });
    const facts = parseRemembers(full);
    send({ type: 'paperExplainDone', text: facts.length ? stripRememberLines(full) : full });
    rememberFacts(facts);
  } catch (e) {
    send({ type: 'paperExplainError', message: e instanceof Error ? e.message : String(e) });
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
    timeoutMs: c.timeoutMs,
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
    `Our PyTorch shape tracer can't auto-call \`${name}\` in ${file}. Reason: ${reason || 'needs an input'}.`,
    `Write a "# fusion:" directive so it traces in ISOLATION:`,
    `  # fusion: model = <Class>(...)  — construct the class that DEFINES ${name} (NOT a different/outer model), with its real constructor args.`,
    `  # fusion: input = <expr>        — the argument(s) to ${name}; use a (a, b) tuple if it takes multiple args. Match its parameter list and dtypes exactly (torch.randn / torch.randint).`,
    `Reply with ONLY the "# fusion:" line(s) — no prose, no code fences.`,
    `If you find a reusable fact about this model (its constructor args or input shape/dtype), also add:  # fusion: remember <fact>`,
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

// Nearest top-level `class …:` above a method line (its def line), or null.
function classLineFor(lines: string[], methodLine: number): number | null {
  for (let i = methodLine - 1; i >= 1; i--) {
    if (/^class\s+\w/.test(lines[i - 1] ?? '')) return i;
  }
  return null;
}

// Place directives at the RIGHT spot so trace_module finds them: `# fusion: model =` goes
// above the enclosing CLASS (where _pick_directive looks), `# fusion: input =` above the
// METHOD. Returns a NEW lines array (bottom-up inserts so line numbers don't shift).
function placeDirectives(lines: string[], methodLine: number, text: string): string[] {
  const model: string[] = [];
  const input: string[] = [];
  for (const d of text.split('\n').map((s) => s.trim()).filter(Boolean)) {
    if (/^#\s*fusion:\s*model\s*=/.test(d)) model.push(d);
    else if (/^#\s*fusion:\s*input\s*=/.test(d)) input.push(d);
  }
  const out = [...lines];
  const edits: Array<{ at: number; dirs: string[] }> = [];
  if (input.length) edits.push({ at: methodLine, dirs: [...new Set(input)] });
  if (model.length) edits.push({ at: classLineFor(out, methodLine) ?? methodLine, dirs: [...new Set(model)] });
  edits.sort((a, b) => b.at - a.at); // bottom-up so the class edit's line number stays valid
  for (const e of edits) {
    // Strip any existing contiguous `# fusion:` lines directly above the target FIRST, so
    // re-inserting is idempotent — this is what stops directives piling up on every click.
    let lo = Math.max(0, e.at - 1); // 0-based index of the target line; we insert before it
    while (lo > 0 && /^\s*#\s*fusion:/.test(out[lo - 1] ?? '')) lo--;
    out.splice(lo, Math.max(0, e.at - 1) - lo); // remove the stale directive block
    const indent = (out[lo] ?? '').match(/^\s*/)?.[0] ?? '';
    out.splice(lo, 0, ...e.dirs.map((d) => indent + d));
  }
  return out;
}

// Trace a CANDIDATE directive against a temp copy (never touches the user's file) to verify
// it works. Returns an error string if it still fails, or null if it traced clean.
let _tmpN = 0;
async function tryDirective(srcLines: string[], line: number, directive: string, name: string): Promise<string | null> {
  const lines = placeDirectives(srcLines, line, directive);
  const tmp = path.join(os.tmpdir(), `fusion-try-${process.pid}-${_tmpN++}.py`);
  fs.writeFileSync(tmp, lines.join('\n'), 'utf8');
  try {
    const res = await getHelper().request<{ error?: string | null; records?: Record<string, unknown> }>(
      'trace_function',
      { path: tmp, name, line: 0, projectRoot: workspaceRoot ?? '' }, // line 0 -> find by name (the directive shifted the line)
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
  const ctrl = new AbortController();
  agentRuns.set(id, ctrl); // registered so Stop (agentCancelAll) can abort it
  try {
    let reason = '';
    try {
      const res = await getHelper().request<{ note?: string }>('trace_function', { path: p, name, line, projectRoot: workspaceRoot ?? '' });
      reason = res.note ?? '';
    } catch {
      /* ignore */
    }
    const srcLines = openFile && openFile.path === p ? [...openFile.lines] : fs.readFileSync(p, 'utf8').split('\n');
    const src = srcLines.join('\n');
    const cfg = loadAgentConfig(AGENT_CONFIG_FILE);
    const client = new AgentClient(cfg);

    const mem = memoryPromptBlock(loadMemory());
    const facts: string[] = [];
    let directive = '';
    let err: string | null = reason || 'needs an input';
    const MAX = Math.max(1, loadTracing().retries);
    for (let attempt = 0; attempt < MAX; attempt++) {
      const prompt =
        mem +
        (attempt === 0
          ? traceAssistPrompt(path.basename(p), src, name, reason)
          : retryPrompt(path.basename(p), src, name, directive, err ?? ''));
      let full = '';
      try {
        full = await client.run(prompt, { signal: ctrl.signal, onChunk: (delta) => send({ type: 'agentChunk', id, delta }) });
      } catch (e) {
        send({ type: 'agentError', id, message: e instanceof Error ? e.message : String(e) });
        return;
      }
      facts.push(...parseRemembers(full));
      directive = extractDirectives(full);
      if (!directive) {
        send({ type: 'agentDone', id, text: full });
        rememberFacts(facts);
        return;
      }
      err = await tryDirective(srcLines, line, directive, name);
      if (!err) break; // verified — traces clean
      if (attempt < MAX - 1) send({ type: 'agentChunk', id, delta: `\n[attempt ${attempt + 1} failed: ${err} — retrying]\n` });
    }

    send({ type: 'agentDone', id, text: `Proposed for ${name}()${err ? ` (still failing: ${err})` : ' — verified ✓'}:\n${directive}` });
    // A verified directive is a hard-won, reusable fact — remember it (with any the agent volunteered).
    if (!err && directive) facts.push(`To trace ${name}(): ${directive.split('\n').map((s) => s.trim()).filter(Boolean).join('  ')}`);
    rememberFacts(facts);
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
  } finally {
    agentRuns.delete(id);
  }
}

// Place directives correctly (model→class, input→method), refresh the editor, re-trace.
async function insertDirective(p: string, line: number, text: string): Promise<void> {
  const newText = placeDirectives(fs.readFileSync(p, 'utf8').split('\n'), line, text).join('\n');
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
      queueAgent(() => runAgent(m.id, m.text, m.history));
      break;
    case 'agentCancel':
      agentRuns.get(m.id)?.abort();
      break;
    case 'agentCancelAll':
      cancelAllAgents();
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
    case 'getTracingConfig':
      send({ type: 'tracingConfig', config: loadTracing() });
      break;
    case 'saveTracingConfig':
      saveTracing(m.config);
      send({ type: 'tracingConfig', config: m.config });
      // No re-trace: density/abstract are client-side display toggles and retries is agent-side;
      // none change trace OUTPUT, so a re-trace would be a pointless round-trip that also resets
      // the Summary/Paper caches. (autoTrace only matters at file-open.)
      break;
    case 'getMemory':
      send({ type: 'agentMemory', text: loadMemory() });
      break;
    case 'saveMemory':
      saveMemory(m.text);
      send({ type: 'agentMemory', text: m.text });
      break;
    case 'saveConversation':
      saveConversation(m.conversation);
      break;
    case 'listConversations':
      send({ type: 'conversationList', items: listConversations() });
      break;
    case 'loadConversation': {
      const c = loadConversation(m.id);
      if (c) send({ type: 'conversation', conversation: c });
      break;
    }
    case 'deleteConversation':
      deleteConversation(m.id);
      send({ type: 'conversationList', items: listConversations() });
      break;
    case 'pickFolder':
      await pickFolder();
      break;
    case 'openFileInFolder': {
      // Containment: only open files genuinely under the workspace (a `..`-laden message
      // can't read outside it). Mirrors serveUi's traversal guard.
      if (workspaceRoot) {
        const f = path.resolve(workspaceRoot, m.path);
        if (f === workspaceRoot || f.startsWith(workspaceRoot + path.sep)) await openPython(f);
      }
      break;
    }
    case 'requestProjectGraph':
      await runProjectGraph();
      break;
    case 'traceWithRealInput': // #5
      await traceWithRealInput(m.path, m.name, m.line);
      break;
    case 'requestModelSummary': // #2
      await runModelSummary();
      break;
    case 'requestPaper': // #8
      await runPaper();
      break;
    case 'explainPaper': // #8
      _paperExplainId = m.id; // mark before queueing so a cancel can target it
      queueAgent(() => runPaperExplain(m.id));
      break;
    case 'cancelPaperExplain': // #8
      if (_paperExplainId) agentRuns.get(_paperExplainId)?.abort();
      _paperExplainId = undefined; // also drops a still-queued run (its id no longer matches)
      break;
    case 'compareFiles': // design B — faithful-port compare (re-compare a known pair)
      await runCompare(m.pathA, m.pathB);
      break;
    case 'pickCompareFile': // design B — pick a reference, compare against the open file
      await pickReferenceAndCompare();
      break;
    case 'traceAssist':
      queueAgent(() => runTraceAssist(m.id, m.path, m.name, m.line)); // serialized (file-level ✦ ask)
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
          { label: 'Open Folder…', accelerator: 'CmdOrCtrl+Shift+F', click: () => void pickFolder() },
          { label: 'Open Data File…', accelerator: 'CmdOrCtrl+Shift+O', click: () => void pickDataFile() },
          { label: 'Trace Current File', accelerator: 'CmdOrCtrl+R', click: () => void runTrace() },
          { label: 'Compare With Reference…', accelerator: 'CmdOrCtrl+Shift+C', click: () => void pickReferenceAndCompare() },
          { type: 'separator' },
          { role: 'close' },
        ],
      },
      { role: 'editMenu' },
      // reload re-bound off CmdOrCtrl+R so it doesn't collide with 'Trace Current File'.
      { label: 'View', submenu: [{ role: 'reload', accelerator: 'CmdOrCtrl+Shift+R' }, { role: 'toggleDevTools' }, { role: 'togglefullscreen' }] },
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
