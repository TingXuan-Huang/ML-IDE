import { writable } from 'svelte/store';
import type { CallGraph, DataMeta, FileStructure, HostMessage, TraceState, Zone } from '@fusion/shared';

export const structure = writable<FileStructure | null>(null);
export const graph = writable<CallGraph | null>(null);
export const data = writable<DataMeta | null>(null);
export const trace = writable<TraceState>({ phase: 'idle' });
export const zone = writable<Zone>('blocks');

/** Apply a message from the extension host to the stores. */
export function applyHostMessage(m: HostMessage): void {
  switch (m.type) {
    case 'activeFile':
      structure.set(m.structure);
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
  }
}
