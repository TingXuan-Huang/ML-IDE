import { derived, get, writable } from 'svelte/store';
import type {
  AgentConfigWire,
  CallGraph,
  DataMeta,
  FileStructure,
  HostMessage,
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

/** Ask the agent to make a function traceable (writes a `# fusion:` directive). */
export function askTrace(path: string, name: string, line: number): void {
  const id = nextAgentId();
  chat.update((ms) => [
    ...ms,
    { role: 'user', text: `✦ make ${name}() traceable` },
    { id, role: 'agent', text: '', streaming: true },
  ]);
  agentBusy.set(true);
  post({ type: 'traceAssist', id, path, name, line });
}

/** Apply an agent-proposed directive (review mode -> user clicked Insert). */
export function applyDirective(d: { path: string; line: number; text: string }): void {
  post({ type: 'insertDirective', path: d.path, line: d.line, text: d.text });
}

/** Ask the agent to resolve EVERY function in the file that couldn't auto-trace. */
export function askTraceFile(): void {
  const s = get(structure);
  if (!s) return;
  if (!s.hasShapes) {
    chat.update((ms) => [
      ...ms,
      { role: 'agent', text: '✦ Trace the file first (▶ Trace this file). Then ✦ ask resolves the functions that couldn’t auto-trace.' },
    ]);
    return;
  }
  const dunder = (n: string): boolean => n.startsWith('__') && n.endsWith('__');
  const targets = s.functions.filter((f) => !dunder(f.name) && !f.lines.some((l) => l.shapes.length));
  if (!targets.length) {
    chat.update((ms) => [...ms, { role: 'agent', text: '✦ Every function already traced — nothing to resolve.' }]);
    return;
  }
  for (const f of targets) askTrace(s.path, f.name, f.startLine);
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
export function saveTracingConfig(cfg: TracingConfigWire): void {
  post({ type: 'saveTracingConfig', config: cfg });
  settingsOpen.set(false);
}

/** Apply a message from the host to the stores. */
export function applyHostMessage(m: HostMessage): void {
  switch (m.type) {
    case 'activeFile':
      structure.set(m.structure);
      break;
    case 'openDocument':
      doc.set({ path: m.path, text: m.text, language: m.language });
      break;
    case 'traceState':
      trace.set(m.state);
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
    case 'agentChunk':
      chat.update((ms) => updateMsg(ms, m.id, (x) => ({ ...x, text: x.text + m.delta })));
      break;
    case 'agentDone':
      chat.update((ms) => updateMsg(ms, m.id, (x) => ({ ...x, text: m.text || x.text, streaming: false })));
      agentBusy.set(false);
      break;
    case 'agentError':
      chat.update((ms) => updateMsg(ms, m.id, (x) => ({ ...x, text: m.message, streaming: false, error: true })));
      agentBusy.set(false);
      break;
  }
}
