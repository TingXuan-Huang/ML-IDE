import { writable } from 'svelte/store';
import type { CallGraph, DataMeta, FileStructure, HostMessage, TraceState, Zone } from '@fusion/shared';
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
}
export const chat = writable<ChatMsg[]>([]);
export const agentBusy = writable<boolean>(false);
export const agentInfo = writable<{ kind: string; command: string } | null>(null);
let _agentId = 0;
export const nextAgentId = (): string => `a${++_agentId}`;

const updateMsg = (ms: ChatMsg[], id: string, fn: (m: ChatMsg) => ChatMsg): ChatMsg[] =>
  ms.map((m) => (m.id === id ? fn(m) : m));

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
      agentInfo.set({ kind: m.kind, command: m.command });
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
