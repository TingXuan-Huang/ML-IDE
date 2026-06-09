// @fusion/shared — the ONE message protocol between the extension host and the
// cockpit webview. Both sides import these types, so the messages they exchange
// cannot drift (compiler-enforced). Bump PROTOCOL_VERSION on any breaking change.
export const PROTOCOL_VERSION = 1;

export type Zone = 'blocks' | 'graph' | 'data';
export type HintDensity = 'smart' | 'all';

// ---- data models --------------------------------------------------------------
export interface ShapeRecord {
  varName: string;
  shape: number[]; // e.g. [32, 768]
  dtype: string; // e.g. "float32"
  changed: boolean; // shape differs from this var's value on the previous line (smart density)
}

export interface BlockLine {
  line: number; // 1-based source line
  text: string; // source text, for the blocks-zone render
  shapes: ShapeRecord[]; // tensors live on/after this line
  problem?: { kind: 'mismatch' | 'broadcast'; message: string };
}

export interface FunctionBlock {
  name: string;
  startLine: number;
  endLine: number;
  params: Array<{ name: string; type?: string }>;
  returns?: string;
  lines: BlockLine[];
}

export interface FileStructure {
  path: string;
  language: string;
  functions: FunctionBlock[];
  hasShapes: boolean; // false = static only (pre-trace); true = trace populated
}

export interface CallGraphNode {
  id: string;
  label: string;
  line: number;
}
export interface CallGraphEdge {
  from: string;
  to: string;
}
export interface CallGraph {
  focus: string;
  nodes: CallGraphNode[];
  edges: CallGraphEdge[];
  sparse: boolean; // true => label "static calls only; dynamic edges may be missing"
}

export interface ColumnMeta {
  name: string;
  dtype: string;
  nulls: number;
}
export interface DataMeta {
  path: string;
  kind: 'table' | 'ndarray' | 'image' | 'unknown';
  shape?: number[];
  dtype?: string;
  columns?: ColumnMeta[];
  header?: string[];
  rowSample: number; // rows actually transported
  sample?: unknown[]; // flattened values (ndarray) or rows (table) for display
  stats?: Record<string, number>;
  arrowUri?: string; // asWebviewUri to a temp .arrow file (large payloads)
  bytes?: Uint8Array; // inline (small payloads)
  note?: string; // degraded / "couldn't fully parse" note
}

export type TraceState =
  | { phase: 'idle' }
  | { phase: 'tracing'; progress?: string }
  | { phase: 'done'; runId: string }
  | { phase: 'error'; message: string }
  | { phase: 'canceled' };

// ---- host -> webview -----------------------------------------------------------
export type HostMessage =
  | { type: 'init'; version: number }
  | { type: 'activeFile'; structure: FileStructure }
  | { type: 'traceState'; state: TraceState }
  | { type: 'callGraph'; graph: CallGraph }
  | { type: 'dataView'; meta: DataMeta }
  | { type: 'focus'; zone: Zone }
  | { type: 'hintDensity'; mode: HintDensity };

// ---- webview -> host -----------------------------------------------------------
export type WebviewMessage =
  | { type: 'ready' }
  | { type: 'requestTrace'; path: string }
  | { type: 'cancelTrace' }
  | { type: 'revealSymbol'; path: string; line: number }
  | { type: 'pickDataFile' }
  | { type: 'setPrimaryZone'; zone: Zone }
  | { type: 'toggleHintDensity' };
