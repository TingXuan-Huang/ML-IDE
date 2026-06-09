// Pure helper-JSON -> @fusion/shared adapter. No vscode dependency, so it's
// unit-testable with Vitest (see adapter.test.ts).
import type { BlockLine, FileStructure, FunctionBlock } from '@fusion/shared';

export interface RawStructure {
  path: string;
  functions: Array<{
    name: string;
    startLine: number;
    endLine: number;
    params: Array<{ name: string; type?: string | null }>;
    returns?: string | null;
    intermediates: Array<{ line: number; name: string }>;
  }>;
}

export interface RawTrace {
  records: Record<string, Record<string, { shape: number[]; dtype: string; changed: boolean }>>;
  // Single crash (trace_file / trace_function): one crashing line + its message.
  error?: string | null;
  crashLine?: number | null;
  // Multiple crashes (trace_module traces many functions): one entry per crashing line.
  crashes?: Array<{ line: number; message: string }>;
  // Provenance per traced function (the exact call used, or a "needs # fusion:" hint),
  // keyed by the function's def line so it pins to the right block.
  notes?: Array<{ line: number; note: string }>;
  // matmul/broadcast op notes keyed by source line.
  ops?: Record<string, string>;
}

/**
 * Fold the helper's structure JSON (+ optional trace records) into a typed
 * FileStructure. `getLine(n)` returns the 1-based source line text — passed in
 * so this module never imports vscode. Crash lines (single OR multiple) become
 * per-line `problem`s the cockpit renders inline.
 */
export function toFileStructure(
  raw: RawStructure,
  getLine: (line1Based: number) => string,
  trace?: RawTrace,
): FileStructure {
  const hasShapes = !!trace && Object.keys(trace.records).length > 0;
  // Collapse single + multiple crashes into one line -> message map.
  const problemAt = new Map<number, string>();
  if (trace?.crashLine != null && trace.error) problemAt.set(trace.crashLine, trace.error);
  for (const c of trace?.crashes ?? []) problemAt.set(c.line, c.message);
  // Provenance notes keyed by function def line (pins to fn.startLine below).
  const noteAt = new Map<number, string>();
  for (const n of trace?.notes ?? []) noteAt.set(n.line, n.note);

  const functions: FunctionBlock[] = raw.functions.map((fn) => {
    const lines: BlockLine[] = [];
    for (let ln = fn.startLine; ln <= fn.endLine; ln++) {
      const rec = trace?.records[String(ln)];
      const shapes = rec
        ? Object.entries(rec).map(([varName, v]) => ({
            varName,
            shape: v.shape,
            dtype: v.dtype,
            changed: v.changed,
          }))
        : [];
      const problem = problemAt.has(ln)
        ? { kind: 'mismatch' as const, message: problemAt.get(ln)! }
        : undefined;
      lines.push({ line: ln, text: getLine(ln), shapes, problem, op: trace?.ops?.[String(ln)] });
    }
    return {
      name: fn.name,
      startLine: fn.startLine,
      endLine: fn.endLine,
      params: fn.params.map((p) => ({ name: p.name, type: p.type ?? undefined })),
      returns: fn.returns ?? undefined,
      lines,
      traceInput: noteAt.get(fn.startLine),
    };
  });
  return { path: raw.path, language: 'python', functions, hasShapes };
}

/** Shape of the helper's `trace_module` result (whole-file: build-check + synth forward). */
export interface TraceModuleResult {
  records: RawTrace['records'];
  problems: Array<{ line: number; message: string }>; // one per crashing function
  notes: Array<{ label: string; line: number; note: string }>; // call used per target (provenance)
  ops?: Record<string, string>; // matmul/broadcast notes by line
}

/** Fold a trace_module result into a RawTrace so toFileStructure can render it. */
export function moduleTraceToRaw(m: TraceModuleResult): RawTrace {
  return {
    records: m.records,
    crashes: m.problems,
    notes: m.notes.map((n) => ({ line: n.line, note: n.note })),
    ops: m.ops,
  };
}
