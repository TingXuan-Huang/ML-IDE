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
  op?: string; // matmul/broadcast note, e.g. "matmul [2,8] · [8,4] → [2,4]"
}

export interface FunctionBlock {
  name: string;
  startLine: number;
  endLine: number;
  params: Array<{ name: string; type?: string }>;
  returns?: string;
  lines: BlockLine[];
  // Provenance pinned by the last trace: the exact call used to produce these shapes
  // (e.g. "Net().forward(randn(2, 16))"), or a hint when it couldn't auto-trace
  // (e.g. "forward() needs 2 args — add  # fusion: input = (...)").
  traceInput?: string;
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

// ---- agent config (wire copy of @fusion/core AgentConfig; keeps the protocol core-free) ----
export interface AgentConfigWire {
  kind: string;
  command: string;
  args: string[];
  promptVia: 'stdin' | 'arg';
  trust: 'review' | 'auto';
  model?: string;
}

// ---- host -> webview -----------------------------------------------------------
export type HostMessage =
  | { type: 'init'; version: number }
  | { type: 'activeFile'; structure: FileStructure }
  // openDocument carries the RAW source for the standalone editor (Monaco). Sent only
  // when a new file is opened — NOT on every structure/trace refresh — so editor text +
  // cursor survive re-traces. (VS Code host never sends this; it has its own editor.)
  | { type: 'openDocument'; path: string; text: string; language: string }
  | { type: 'traceState'; state: TraceState }
  | { type: 'callGraph'; graph: CallGraph }
  | { type: 'dataView'; meta: DataMeta }
  | { type: 'focus'; zone: Zone }
  | { type: 'hintDensity'; mode: HintDensity }
  // ---- agent (CLI coding assistant) ----
  | { type: 'agentChunk'; id: string; delta: string } // streamed stdout delta
  | { type: 'agentDone'; id: string; text: string } // run finished OK (full text)
  | { type: 'agentError'; id: string; message: string }
  | { type: 'agentConfig'; config: AgentConfigWire } // the full configured agent (for the settings page)
  | { type: 'agentTestResult'; id: string; ok: boolean; message: string } // settings "Test" result
  // trace-assist: the agent proposed a `# fusion:` directive for a function (review mode)
  | { type: 'directiveProposed'; forFunction: string; path: string; line: number; directive: string; explanation: string };

// ---- webview -> host -----------------------------------------------------------
export type WebviewMessage =
  | { type: 'ready' }
  | { type: 'requestTrace'; path: string }
  | { type: 'traceFunction'; path: string; name: string; line: number }
  | { type: 'cancelTrace' }
  | { type: 'revealSymbol'; path: string; line: number }
  // Standalone editor saved (Cmd+S) -> host writes to disk + re-structures.
  | { type: 'saveDocument'; path: string; text: string }
  | { type: 'pickDataFile' }
  | { type: 'setPrimaryZone'; zone: Zone }
  | { type: 'toggleHintDensity' }
  // ---- agent ----
  | { type: 'agentPrompt'; id: string; text: string } // user asked the agent something
  | { type: 'agentCancel'; id: string }
  | { type: 'getAgentConfig' }
  | { type: 'saveAgentConfig'; config: AgentConfigWire } // settings page saved
  | { type: 'testAgentConfig'; id: string; config: AgentConfigWire } // settings "Test" — run the unsaved config
  // trace-assist: ask the agent to author a `# fusion:` directive for a function
  | { type: 'traceAssist'; id: string; path: string; name: string; line: number }
  // apply an agent-proposed directive (review mode -> user clicked Insert)
  | { type: 'insertDirective'; path: string; line: number; text: string };
