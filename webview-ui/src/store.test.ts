import { describe, expect, it, beforeEach } from 'vitest';
import { get } from 'svelte/store';
import {
  applyHostMessage,
  chat,
  compareResult,
  fmtMeta,
  fmtOp,
  fmtShape,
  modelSummary,
  paper,
  structure,
} from './store';

const DIMS = { '2': 'B', '16': 'L', '128': 'D', '4': 'H' };

describe('formatters', () => {
  it('fmtShape relabels concrete dims when abstract is on, numeric when off', () => {
    expect(fmtShape([2, 16, 128], DIMS, true)).toBe('B, L, D');
    expect(fmtShape([2, 16, 128], DIMS, false)).toBe('2, 16, 128');
    expect(fmtShape([2, 7, 128], DIMS, true)).toBe('B, 7, D'); // unknown dim stays numeric
  });

  it('fmtOp relabels [..] shape groups but leaves permute(..) axis args numeric', () => {
    const out = fmtOp('permute(0,2,1,3) [2,4]→[4,2]', DIMS, true);
    expect(out).toContain('permute(0,2,1,3)'); // parenthesized axes untouched
    expect(out).toContain('[B, H]');
    expect(out).toContain('[H, B]');
  });

  it('fmtMeta drops the cpu default and humanizes bytes; shows accelerators', () => {
    expect(fmtMeta({ dtype: 'float32', device: 'cpu', bytes: 256 })).toBe('float32 · 256B');
    expect(fmtMeta({ dtype: 'float32', device: 'cuda:0', bytes: 48 * 1024 * 1024 })).toBe('float32 · cuda:0 · 48MB');
    expect(fmtMeta({ dtype: 'float16', bytes: 1024 })).toBe('float16 · 1KB');
  });
});

describe('applyHostMessage reducer', () => {
  beforeEach(() => {
    structure.set(null);
    modelSummary.set({} as never);
    paper.set({} as never);
    compareResult.set({} as never);
    chat.set([]);
  });

  it('activeFile sets the structure and invalidates the lazy caches', () => {
    applyHostMessage({
      type: 'activeFile',
      structure: { path: '/m.py', language: 'python', functions: [], hasShapes: false },
    });
    expect(get(structure)?.path).toBe('/m.py');
    expect(get(modelSummary)).toBeNull();
    expect(get(paper)).toBeNull();
    expect(get(compareResult)).toBeNull(); // stale diff cleared on a re-trace
  });

  it('drops a late agentChunk for an already-stopped (non-streaming) run', () => {
    chat.set([{ id: 'a1', role: 'agent', text: 'partial', streaming: false }]);
    applyHostMessage({ type: 'agentChunk', id: 'a1', delta: ' MORE' });
    expect(get(chat)[0].text).toBe('partial'); // streaming:false -> the chunk is ignored
  });

  it('appends a streamed agentChunk while the run is still streaming', () => {
    chat.set([{ id: 'a2', role: 'agent', text: 'a', streaming: true }]);
    applyHostMessage({ type: 'agentChunk', id: 'a2', delta: 'b' });
    expect(get(chat)[0].text).toBe('ab');
  });
});
