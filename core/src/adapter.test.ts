import { describe, expect, it } from 'vitest';
import { moduleTraceToRaw, toFileStructure, type RawStructure, type RawTrace, type TraceModuleResult } from './adapter';

const raw: RawStructure = {
  path: '/x/model.py',
  functions: [
    {
      name: 'forward',
      startLine: 1,
      endLine: 3,
      params: [{ name: 'self' }, { name: 'x', type: 'Tensor' }],
      returns: 'Tensor',
      intermediates: [{ line: 2, name: 'h' }],
    },
  ],
};
const lines: Record<number, string> = {
  1: 'def forward(self, x):',
  2: '    h = self.fc(x)',
  3: '    return h',
};
const getLine = (n: number) => lines[n] ?? '';

describe('toFileStructure', () => {
  it('maps structure with no trace (no shapes, hasShapes false)', () => {
    const s = toFileStructure(raw, getLine);
    expect(s.language).toBe('python');
    expect(s.hasShapes).toBe(false);
    expect(s.functions).toHaveLength(1);
    const fn = s.functions[0];
    expect(fn.params.map((p) => p.name)).toEqual(['self', 'x']);
    expect(fn.returns).toBe('Tensor');
    expect(fn.lines.map((l) => l.text)).toEqual([lines[1], lines[2], lines[3]]);
    expect(fn.lines.every((l) => l.shapes.length === 0)).toBe(true);
  });

  it('attaches traced shapes per line with the changed flag', () => {
    const trace: RawTrace = {
      records: { '2': { h: { shape: [8, 32], dtype: 'float32', changed: true } } },
      error: null,
      crashLine: null,
    };
    const s = toFileStructure(raw, getLine, trace);
    expect(s.hasShapes).toBe(true);
    const l2 = s.functions[0].lines.find((l) => l.line === 2)!;
    expect(l2.shapes).toEqual([{ varName: 'h', shape: [8, 32], dtype: 'float32', changed: true }]);
  });

  it('marks the crash line with a shape-mismatch problem and leaves others clean', () => {
    const trace: RawTrace = {
      records: { '2': { h: { shape: [8, 32], dtype: 'float32', changed: true } } },
      error: 'RuntimeError: mat1 and mat2 shapes cannot be multiplied (8x32 and 64x4)',
      crashLine: 3,
    };
    const fn = toFileStructure(raw, getLine, trace).functions[0];
    const crashed = fn.lines.find((l) => l.line === 3)!;
    expect(crashed.problem?.kind).toBe('mismatch');
    expect(crashed.problem?.message).toContain('cannot be multiplied');
    expect(fn.lines.find((l) => l.line === 2)!.problem).toBeUndefined();
  });

  it('pins a provenance note to its function by def line', () => {
    const trace: RawTrace = { records: {}, notes: [{ line: 1, note: 'M().forward(randn(2, 8))' }] };
    const fn = toFileStructure(raw, getLine, trace).functions[0];
    expect(fn.traceInput).toBe('M().forward(randn(2, 8))');
  });

  it('marks multiple crash lines from trace_module crashes[]', () => {
    const trace: RawTrace = {
      records: {},
      crashes: [
        { line: 1, message: 'build Bad(): RuntimeError mismatch' },
        { line: 3, message: 'Bad.forward: RuntimeError cannot be multiplied' },
      ],
    };
    const fn = toFileStructure(raw, getLine, trace).functions[0];
    expect(fn.lines.find((l) => l.line === 1)!.problem?.message).toContain('build Bad()');
    expect(fn.lines.find((l) => l.line === 3)!.problem?.message).toContain('cannot be multiplied');
    expect(fn.lines.find((l) => l.line === 2)!.problem).toBeUndefined();
  });

  it('flows dims, op notes, device/bytes, and the nonFinite flag into the structure', () => {
    const trace: RawTrace = {
      records: { '2': { h: { shape: [2, 32], dtype: 'float32', changed: true, device: 'cuda:0', bytes: 256 } } },
      ops: { '2': 'matmul [2,4] · [4,32] → [2,32]' },
      dims: { '2': 'B', '32': 'D' },
      nonFinite: { line: 2, var: 'h', kind: 'nan' },
    };
    const s = toFileStructure(raw, getLine, trace);
    expect(s.dimNames).toEqual({ '2': 'B', '32': 'D' });
    expect(s.nonFinite).toEqual({ line: 2, var: 'h', kind: 'nan' });
    const l2 = s.functions[0].lines.find((l) => l.line === 2)!;
    expect(l2.op).toContain('matmul');
    expect(l2.shapes[0]).toMatchObject({ varName: 'h', device: 'cuda:0', bytes: 256 });
  });
});

describe('moduleTraceToRaw', () => {
  it('maps problems->crashes and passes notes/ops/dims/nonFinite end-to-end', () => {
    const mod: TraceModuleResult = {
      records: { '2': { h: { shape: [2, 32], dtype: 'float32', changed: true } } },
      problems: [{ line: 3, message: 'Bad.forward: RuntimeError boom' }],
      notes: [{ label: 'M', line: 1, note: 'M().forward(randn(2, 4))' }],
      ops: { '2': 'matmul x' },
      dims: { '2': 'B' },
      nonFinite: { line: 2, var: 'h', kind: 'nan' },
    };
    const rawTrace = moduleTraceToRaw(mod);
    expect(rawTrace.crashes).toEqual([{ line: 3, message: 'Bad.forward: RuntimeError boom' }]);
    expect(rawTrace.notes).toEqual([{ line: 1, note: 'M().forward(randn(2, 4))' }]);
    expect(rawTrace.nonFinite).toEqual({ line: 2, var: 'h', kind: 'nan' });
    // end-to-end: a trace_module result renders crashes + provenance through toFileStructure
    const fn = toFileStructure(raw, getLine, rawTrace).functions[0];
    expect(fn.lines.find((l) => l.line === 3)!.problem?.message).toContain('boom');
    expect(fn.traceInput).toBe('M().forward(randn(2, 4))'); // provenance pins to the function, by def line
  });
});
