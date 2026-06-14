import { derived, get, writable } from 'svelte/store';
import type {
  AgentConfigWire,
  CallGraph,
  ChatRecordMessage,
  ConversationMeta,
  DataMeta,
  FileStructure,
  FunctionBlock,
  HostMessage,
  ModuleSummary,
  PaperView,
  ProjectGraphWire,
  TracingConfigWire,
  TraceState,
  Zone,
} from '@fusion/shared';
import { post } from './vscode';

export const structure = writable<FileStructure | null>(null);
export const graph = writable<CallGraph | null>(null);
export const data = writable<DataMeta | null>(null);
export const trace = writable<TraceState>({ phase: 'idle' });
export const zone = writable<Zone>('blocks');

// --- open-a-folder (project mode) ----------------------------------------------
export const folder = writable<{ root: string; files: string[] } | null>(null);
export const projectGraph = writable<ProjectGraphWire | null>(null);
export const pickFolder = (): void => post({ type: 'pickFolder' });
export const openTreeFile = (relPath: string): void => post({ type: 'openFileInFolder', path: relPath });
export const requestProjectGraph = (): void => post({ type: 'requestProjectGraph' });

// --- #2 model summary · #8 paper-reading · #5 real-input -----------------------
export const modelSummary = writable<ModuleSummary | null>(null);
export const paper = writable<PaperView | null>(null);
export const paperExplain = writable<{ text: string; streaming: boolean; error?: boolean } | null>(null);
export const requestModelSummary = (): void => post({ type: 'requestModelSummary' });
export const requestPaper = (): void => post({ type: 'requestPaper' });
export function explainPaper(): void {
  paperExplain.set({ text: '', streaming: true });
  post({ type: 'explainPaper', id: nextAgentId() });
}
export const cancelPaperExplain = (): void => {
  post({ type: 'cancelPaperExplain' });
  // Clean stop: keep the partial text, mark it [stopped] (a later abort-error is ignored).
  paperExplain.update((p) => (p?.streaming ? { ...p, streaming: false, text: p.text ? `${p.text}\n[stopped]` : '[stopped]' } : p));
};
/** #5 — pick a real data file as this function's input (writes `# fusion: input = load(...)`). */
export const traceWithRealInput = (path: string, name: string, line: number): void =>
  post({ type: 'traceWithRealInput', path, name, line });

// --- standalone editor (desktop host only) -------------------------------------
// True when running inside the Electron host (preload exposed window.fusionHost).
// In VS Code there's no embedded editor — the host owns it — so the cockpit renders alone.
export const isDesktop =
  typeof window !== 'undefined' && !!(window as unknown as { fusionHost?: unknown }).fusionHost;

export const doc = writable<{ path: string; text: string; language: string } | null>(null);
export const caretLine = writable<number>(0); // 1-based editor caret line -> cockpit highlights its function
// A bumpable reveal signal: the editor scrolls to `.line` whenever `.seq` changes
// (a plain line number wouldn't re-fire when you click the same symbol twice).
export const revealTarget = writable<{ line: number; seq: number }>({ line: 0, seq: 0 });
let _seq = 0;

/** Reveal a source line: scroll the embedded editor (desktop) or ask the host (VS Code). */
export function revealInEditor(path: string, line: number): void {
  if (isDesktop) revealTarget.set({ line, seq: ++_seq });
  else post({ type: 'revealSymbol', path, line });
}

// --- agent chat ----------------------------------------------------------------
export interface ChatMsg {
  id?: string; // matches the agent run id (agent messages only)
  role: 'user' | 'agent';
  text: string;
  streaming?: boolean;
  error?: boolean;
  directive?: { path: string; line: number; text: string }; // a proposed # fusion: directive (Insert)
}
export const chat = writable<ChatMsg[]>([]);
export const agentBusy = writable<boolean>(false);
export const agentConfig = writable<AgentConfigWire | null>(null);
export const settingsOpen = writable<boolean>(false);
let _agentId = 0;
export const nextAgentId = (): string => `a${++_agentId}`;

const updateMsg = (ms: ChatMsg[], id: string, fn: (m: ChatMsg) => ChatMsg): ChatMsg[] =>
  ms.map((m) => (m.id === id ? fn(m) : m));

/** Ask the agent to make a function traceable (writes a `# fusion:` directive).
 *  `name` is the bare method name (for the trace lookup); `label` is the display name
 *  (class-qualified, e.g. "MultiHeadSelfAttention.forward") so the chat is unambiguous. */
export function askTrace(path: string, name: string, line: number, label = name): void {
  const id = nextAgentId();
  chat.update((ms) => [
    ...ms,
    { role: 'user', text: `✦ make ${label}() traceable` },
    { id, role: 'agent', text: '', streaming: true },
  ]);
  agentBusy.set(true);
  post({ type: 'traceAssist', id, path, name, line });
}

/** Apply an agent-proposed directive (review mode -> user clicked Insert). */
export function applyDirective(d: { path: string; line: number; text: string }): void {
  post({ type: 'insertDirective', path: d.path, line: d.line, text: d.text });
}

/** Stop — cancel every running + queued agent run (chat + trace-assist) and close out the chat. */
export function cancelAllAgents(): void {
  post({ type: 'agentCancelAll' });
  chat.update((ms) => ms.map((m) => (m.streaming ? { ...m, streaming: false, text: m.text ? `${m.text}\n[stopped]` : '[stopped]' } : m)));
  agentBusy.set(false);
}

const qualName = (f: FunctionBlock): string => (f.className ? `${f.className}.${f.name}` : f.name);

/** Ask the agent to resolve EVERY function in the file that couldn't auto-trace.
 *  Auto-traces the file first if needed — a file where every model needs constructor args
 *  captures ZERO shapes, and that's exactly when ✦ ask is needed, so we must NOT gate on
 *  hasShapes (which would be a dead end). We gate on whether a trace has *completed*. */
let _askFilePending = false;
export function askTraceFile(): void {
  const s = get(structure);
  if (!s) return;
  const phase = get(trace).phase;
  if (phase === 'done') {
    resolveFileTargets(s);
    return;
  }
  // Not traced yet (idle/error) or mid-trace -> trace first, then resolve when it lands.
  _askFilePending = true;
  chat.update((ms) => [...ms, { role: 'agent', text: '✦ Tracing the file first, then resolving what couldn’t auto-trace…' }]);
  if (phase !== 'tracing') post({ type: 'requestTrace', path: s.path });
}

/** Fire ✦ ask for every non-dunder function that has no shapes (couldn't auto-trace). */
function resolveFileTargets(s: FileStructure): void {
  const dunder = (n: string): boolean => n.startsWith('__') && n.endsWith('__');
  const targets = s.functions.filter((f) => !dunder(f.name) && !f.lines.some((l) => l.shapes.length));
  if (!targets.length) {
    chat.update((ms) => [...ms, { role: 'agent', text: '✦ Every function already traced — nothing to resolve.' }]);
    return;
  }
  chat.update((ms) => [
    ...ms,
    { role: 'agent', text: `✦ Resolving ${targets.length} function${targets.length > 1 ? 's' : ''}: ${targets.map(qualName).join(', ')}` },
  ]);
  for (const f of targets) askTrace(s.path, f.name, f.startLine, qualName(f));
}

/** Persist agent settings (config page). */
export function saveAgentConfig(cfg: AgentConfigWire): void {
  post({ type: 'saveAgentConfig', config: cfg });
  settingsOpen.set(false);
}

// Settings "Test connection" — run the unsaved config once and report ✓/✗.
export const agentTest = writable<{ pending: boolean; ok?: boolean; message?: string } | null>(null);
export function testAgent(cfg: AgentConfigWire): void {
  agentTest.set({ pending: true });
  post({ type: 'testAgentConfig', id: nextAgentId(), config: cfg });
}

// --- tracing config (Tracing settings tab) -------------------------------------
export const tracingConfig = writable<TracingConfigWire | null>(null);
// Show every tensor vs only changed shapes — read by Blocks + the editor.
export const density = derived(tracingConfig, ($t) => $t?.density ?? 'changed');
// Abstract view: relabel concrete dims with symbols (B/L/D) — read by Blocks + the editor.
export const abstract = derived(tracingConfig, ($t) => $t?.abstract ?? false);

/** Relabel a concrete shape with symbols when abstract mode is on, e.g. [2,16,128] -> "B, L, D".
 *  Dims with no known symbol stay numeric. `on=false` just joins the numbers. */
export function fmtShape(shape: number[], dims: Record<string, string> | undefined, on: boolean): string {
  return shape.map((d) => (on && dims?.[String(d)]) || String(d)).join(', ');
}

/** Relabel the [2, 16, 128] shape groups inside an op note (matmul/reshape/… annotations)
 *  when abstract mode is on. Only bracketed groups are touched — parenthesized op args
 *  like permute(0, 2, 1, 3) are axis indices and must stay numeric. */
export function fmtOp(op: string, dims: Record<string, string> | undefined, on: boolean): string {
  if (!on || !dims) return op;
  return op.replace(/\[([0-9,\s]+)\]/g, (_, inner: string) => {
    const parts = inner.split(',').map((t) => t.trim()).filter(Boolean);
    return '[' + parts.map((d) => dims[d] ?? d).join(', ') + ']';
  });
}
export function saveTracingConfig(cfg: TracingConfigWire): void {
  post({ type: 'saveTracingConfig', config: cfg });
  settingsOpen.set(false);
}

// --- durable agent memory (Memory settings tab) --------------------------------
export const memory = writable<string>('');
export function saveMemory(text: string): void {
  post({ type: 'saveMemory', text });
  settingsOpen.set(false);
}

// --- saved conversations (chat history / resume) -------------------------------
let _convSeq = 0;
const newConvId = (): string => `c${Date.now()}-${++_convSeq}`;
export const conversationId = writable<string>(newConvId());
export const conversations = writable<ConversationMeta[]>([]); // the history list

const convTitle = (ms: ChatMsg[]): string => {
  const u = ms.find((m) => m.role === 'user' && m.text.trim());
  const t = (u?.text ?? '').trim().replace(/\s+/g, ' ');
  return t ? (t.length > 60 ? `${t.slice(0, 57)}…` : t) : 'New chat';
};
// Drop transient fields (streaming placeholders, run ids) for persistence.
const toRecords = (ms: ChatMsg[]): ChatRecordMessage[] =>
  ms
    .filter((m) => !m.streaming && m.text.trim())
    .map((m) => ({ role: m.role, text: m.text, error: m.error, directive: m.directive }));

/** Autosave the current chat as a conversation (called after each turn settles). */
function persistConversation(): void {
  const ms = get(chat);
  if (!ms.some((m) => m.role === 'user')) return; // nothing worth saving yet
  post({
    type: 'saveConversation',
    conversation: { id: get(conversationId), title: convTitle(ms), updatedAt: Date.now(), count: ms.length, messages: toRecords(ms) },
  });
}

/** Bounded prior turns of THIS conversation, for the agent's context on the next prompt. */
export function chatHistory(): Array<{ role: string; text: string }> {
  return get(chat)
    .filter((m) => !m.streaming && m.text.trim() && !m.directive)
    .map((m) => ({ role: m.role, text: m.text }));
}

/** Archive the current chat and start a fresh conversation. */
export function newConversation(): void {
  persistConversation();
  conversationId.set(newConvId());
  chat.set([]);
}
export const loadConversations = (): void => post({ type: 'listConversations' });
export const resumeConversation = (id: string): void => post({ type: 'loadConversation', id });
export function deleteConversation(id: string): void {
  post({ type: 'deleteConversation', id });
  conversations.update((cs) => cs.filter((c) => c.id !== id));
}

/** Apply a message from the host to the stores. */
export function applyHostMessage(m: HostMessage): void {
  switch (m.type) {
    case 'activeFile':
      structure.set(m.structure);
      // A re-trace can change shapes/params -> invalidate the lazy Summary + Paper caches so
      // an open tab re-fetches (its reactive guard re-requests when the store goes null).
      modelSummary.set(null);
      paper.set(null);
      paperExplain.set(null);
      break;
    case 'openDocument':
      doc.set({ path: m.path, text: m.text, language: m.language });
      paper.set(null); // new file -> drop the previous paper view + its explanation
      paperExplain.set(null);
      break;
    case 'agentMemory':
      memory.set(m.text);
      break;
    case 'memoryNote':
      chat.update((ms) => [...ms, { role: 'agent', text: m.text }]);
      break;
    case 'traceState':
      trace.set(m.state);
      // A file-level ✦ ask that had to trace first: resolve now that the trace landed.
      // (activeFile arrives just before this 'done', so `structure` is already the traced one.)
      if (m.state.phase === 'done' && _askFilePending) {
        _askFilePending = false;
        const s = get(structure);
        if (s) resolveFileTargets(s);
      } else if (m.state.phase === 'error') {
        _askFilePending = false;
      }
      break;
    case 'callGraph':
      graph.set(m.graph);
      break;
    case 'dataView':
      data.set(m.meta);
      zone.set('data');
      break;
    case 'focus':
      zone.set(m.zone);
      break;
    case 'agentConfig':
      agentConfig.set(m.config);
      break;
    case 'agentTestResult':
      agentTest.set({ pending: false, ok: m.ok, message: m.message });
      break;
    case 'tracingConfig':
      tracingConfig.set(m.config);
      break;
    case 'directiveProposed':
      chat.update((ms) => [
        ...ms,
        {
          role: 'agent',
          text: `Insert into ${m.path.split('/').pop()} to trace ${m.forFunction}()?`,
          directive: { path: m.path, line: m.line, text: m.directive },
        },
      ]);
      break;
    // Late messages from an already-stopped run (streaming:false) are ignored, so a
    // trailing chunk or abort-error never clobbers the "[stopped]" text the user saw.
    case 'agentChunk':
      chat.update((ms) => updateMsg(ms, m.id, (x) => (x.streaming ? { ...x, text: x.text + m.delta } : x)));
      break;
    case 'agentDone':
      chat.update((ms) => updateMsg(ms, m.id, (x) => (x.streaming ? { ...x, text: m.text || x.text, streaming: false } : x)));
      agentBusy.set(false);
      persistConversation();
      break;
    case 'agentError':
      chat.update((ms) => updateMsg(ms, m.id, (x) => (x.streaming ? { ...x, text: m.message, streaming: false, error: true } : x)));
      agentBusy.set(false);
      persistConversation();
      break;
    case 'conversationList':
      conversations.set(m.items);
      break;
    case 'folderTree':
      folder.set({ root: m.root, files: m.files });
      projectGraph.set(null); // rebuilt lazily when the Project tab is next viewed
      break;
    case 'projectGraph':
      projectGraph.set(m.graph);
      break;
    case 'modelSummary':
      modelSummary.set(m.summary);
      break;
    case 'paperView':
      paper.set(m.paper);
      break;
    // Only apply while streaming, so a late chunk/done/abort-error can't clobber a [stopped] panel.
    case 'paperExplainChunk':
      paperExplain.update((p) => (p?.streaming ? { ...p, text: p.text + m.delta } : p));
      break;
    case 'paperExplainDone':
      paperExplain.update((p) => (p?.streaming ? { text: m.text || p.text, streaming: false } : p));
      break;
    case 'paperExplainError':
      paperExplain.update((p) => (p?.streaming ? { text: m.message, streaming: false, error: true } : p));
      break;
    case 'conversation':
      conversationId.set(m.conversation.id);
      chat.set(m.conversation.messages.map((r) => ({ role: r.role, text: r.text, error: r.error, directive: r.directive })));
      break;
  }
}
