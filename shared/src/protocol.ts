// @fusion/shared — the ONE message protocol between the extension host and the
// cockpit webview. Both sides import these types, so the messages they exchange
// cannot drift (compiler-enforced). Bump PROTOCOL_VERSION on any breaking change.
export const PROTOCOL_VERSION = 1;

export type Zone = 'blocks' | 'graph' | 'summary' | 'data' | 'project' | 'paper' | 'compare';
export type HintDensity = 'smart' | 'all';

// ---- data models --------------------------------------------------------------
export interface ShapeRecord {
  varName: string;
  shape: number[]; // e.g. [32, 768]
  dtype: string; // e.g. "float32"
  changed: boolean; // shape differs from this var's value on the previous line (smart density)
  // The rest of what `print(x.shape, x.dtype, x.device)` carries — optional for
  // backward-compat with traces produced before the Tensor Inspector add-on.
  // (requires_grad is intentionally omitted: the trace runs under torch.no_grad(), so every
  // captured activation reads False — the Summary tab's trainable-param count covers grad.)
  device?: string; // e.g. "cpu", "cuda:0" (numpy arrays have no device -> "")
  bytes?: number; // memory footprint = numel × element_size
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
  className?: string; // enclosing class for methods, so N identically-named `forward`s are distinguishable
  startLine: number;
  endLine: number;
  params: Array<{ name: string; type?: string }>;
  returns?: string;
  // Static per-param input-rank constraints: 'exact' (B,L,D = x.shape pins rank 3),
  // 'min' (transpose(1,2) needs rank ≥ 3), or 'free' (rank-polymorphic in THIS function).
  shapeReqs?: Array<{ name: string; kind: 'exact' | 'min' | 'free'; rank?: number | null; via?: string | null }>;
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
  // Abstract view: concrete dim VALUE (as string) -> symbol (B/L/D/C/H/W), from the
  // last trace's input dims. The cockpit relabels shapes with these when abstract mode is on.
  dimNames?: Record<string, string>;
  // The first tensor this trace saw contain NaN — the print(x.isnan()) killer. Surfaced as an
  // inline banner + editor marker on that line. NaN only (Inf is often an intentional mask);
  // absent = clean. The `kind` union leaves room for a future opt-in Inf mode.
  nonFinite?: { line: number; var: string; kind: 'nan' | 'inf' };
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

// Cross-file import graph with a precomputed layered (Sugiyama) layout. Nodes carry x/y;
// the cockpit draws the node-link diagram on a width×height canvas (pan + zoom).
export interface ProjectNode {
  id: string; // posix relpath under the workspace root
  label: string; // basename
  x: number;
  y: number;
}
export interface ProjectGraphWire {
  root: string;
  files: string[];
  focus: string;
  nodes: ProjectNode[];
  edges: CallGraphEdge[];
  sparse: boolean;
  width: number;
  height: number;
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
  timeoutMs?: number; // inactivity timeout (ms); 0 = none. Raise for slow local models.
}

// ---- saved agent conversations (chat history / resume) ----
export interface ChatRecordMessage {
  role: 'user' | 'agent';
  text: string;
  error?: boolean;
  directive?: { path: string; line: number; text: string };
}
export interface ConversationMeta {
  id: string;
  title: string;
  updatedAt: number; // epoch ms
  count: number; // message count
}
export interface Conversation extends ConversationMeta {
  messages: ChatRecordMessage[];
}

// ---- #2 model summary (torchinfo-style Summary tab) ----
export interface ModuleSummaryRow {
  name: string; // qualified module path, e.g. "encoder.blocks.0.attn.qkv"
  cls: string; // class name, e.g. "Linear"
  depth: number; // nesting depth (drives indentation)
  outShape?: number[] | null; // forward output shape (first tensor), null if unavailable
  params: number;
  trainable: number;
  pctParams: number; // params / total * 100, precomputed for the share bar
}
export interface ModuleSummary {
  path: string;
  target: string; // the exact call used, e.g. 'Net().forward(randn(2, 16))'
  rows: ModuleSummaryRow[];
  totalParams: number;
  trainableParams: number;
  paramBytes: number;
  dims?: Record<string, string>; // abstract-view dim-symbol map (reused for relabeling)
  error?: string | null; // set when the model couldn't build / forward crashed (params still shown)
}

// ---- #8 paper-reading mode (Paper tab) ----
export interface PaperStep {
  line: number;
  lhs: string; // assigned variable name(s), e.g. "q, k, v"
  op?: string | null; // the matmul/reshape/softmax note from the tracer ([2,16,128]-group form)
  shapes: ShapeRecord[]; // the lhs vars' shapes on this line
  changed: boolean;
}
export interface PaperSection {
  module: string; // "ClassName.forward"
  forwardNote: string; // provenance, e.g. "Attn().forward(randn(2,16,128))"
  params: string[]; // forward parameter names
  startLine: number;
  steps: PaperStep[];
}
export interface PaperView {
  path: string;
  sections: PaperSection[];
  dims?: Record<string, string>; // dim-symbol map (key matches the Python paper_module() result) — drives the abstract toggle
  problems: Array<{ line: number; module?: string; message: string }>; // module: structural name (compare_traces folds crashes by it)
  skipped?: Array<{ line: number; module: string; message: string }>; // forwards that couldn't be auto-invoked (compare_traces folds these)
}

// ---- tracing config (Tracing settings tab) ----
export interface TracingConfigWire {
  density: 'changed' | 'all'; // show only changed shapes, or every tensor
  abstract: boolean; // relabel concrete dims with symbols (B/L/D) — the "abstract" view
  autoTrace: boolean; // trace a file as soon as it's opened
  retries: number; // ✦ ask agent<->tracer rounds
  meta?: boolean; // show per-tensor dtype · device · grad · memory inline (Tensor Inspector)
}

// ---- design B: faithful-port compare ----
// Two implementations of the same architecture, matched by `ClassName.forward` (then a
// conservative structural rename-pair) and aligned step-by-step. A CompareStep is a PaperStep
// plus a crashed/message pair (so the Python side reuses _paper_step verbatim and the webview
// reuses PaperView's renderer).
export interface CompareStep {
  line: number;
  lhs: string;
  op?: string | null;
  shapes: ShapeRecord[];
  text?: string; // the stripped source line (so arg/activation differences are visible)
  changed?: boolean; // present on real (paper) steps
  crashed?: boolean; // a synthetic step standing in for a forward that couldn't be traced
  message?: string; // crash/skip message when crashed
}
// A pre-aligned diff row: the A step beside the B step (one side absent = an inserted/removed op).
export interface CompareRow {
  a?: CompareStep | null;
  b?: CompareStep | null;
  diverge: boolean; // a one-sided row, or a side that couldn't be traced
}
export interface CompareModule {
  module: string; // matched name
  diverges: boolean; // any diverging row (incl. a side that couldn't be traced)
  divergeStep: number | null; // index of the first diverging row in `rows`; null = matches
  rows: CompareRow[]; // the aligned A-vs-B step diff (LCS by op-kind + batch-normalized shape)
  matchNote?: string | null; // "not traceable in B (reference)", etc.
}
export interface CompareResult {
  pathA: string; // A = this file (the open editor file)
  pathB: string; // B = the reference you picked
  matched: CompareModule[]; // forwards present in BOTH files (incl. crash/skip folds)
  onlyA: string[]; // module names genuinely only in this file
  onlyB: string[]; // module names genuinely only in the reference
  dimsA?: Record<string, string>; // A's dim-symbol map (relabel A's column when abstract is on)
  dimsB?: Record<string, string>; // B's dim-symbol map (relabel B's column)
  problems?: Array<{ line: number; module?: string; message: string }>;
  note?: string; // diff-method disclaimer (rendered)
  error?: string; // a transport/trace failure — renders in-zone instead of hanging the spinner
  pending?: boolean; // set while a compare is in flight, so the zone shows "comparing…"
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
  | { type: 'tracingConfig'; config: TracingConfigWire } // current tracing settings
  | { type: 'agentMemory'; text: string } // current durable agent memory (for the Memory tab)
  | { type: 'memoryNote'; text: string } // a chat note that facts were remembered
  | { type: 'conversationList'; items: ConversationMeta[] } // saved conversations (history picker)
  | { type: 'conversation'; conversation: Conversation } // a resumed conversation's full transcript
  | { type: 'folderTree'; root: string; files: string[] } // opened-folder file list (explorer)
  | { type: 'projectGraph'; graph: ProjectGraphWire } // cross-file import graph (Project tab)
  | { type: 'modelSummary'; summary: ModuleSummary } // #2 torchinfo-style summary (Summary tab)
  | { type: 'paperView'; paper: PaperView } // #8 paper-reading view (Paper tab)
  | { type: 'paperExplainChunk'; delta: string } // #8 streamed "explain like a paper" prose
  | { type: 'paperExplainDone'; text: string }
  | { type: 'paperExplainError'; message: string }
  | { type: 'compareResult'; result: CompareResult } // design B scaffold — faithful-port compare
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
  // user asked the agent something; `history` is the prior turns of THIS conversation
  // (bounded) so the agent has context when resuming/continuing.
  | { type: 'agentPrompt'; id: string; text: string; history?: Array<{ role: string; text: string }> }
  | { type: 'agentCancel'; id: string }
  | { type: 'agentCancelAll' } // Stop — abort every running + queued agent run
  | { type: 'getAgentConfig' }
  | { type: 'saveAgentConfig'; config: AgentConfigWire } // settings page saved
  | { type: 'testAgentConfig'; id: string; config: AgentConfigWire } // settings "Test" — run the unsaved config
  | { type: 'getTracingConfig' }
  | { type: 'saveTracingConfig'; config: TracingConfigWire }
  | { type: 'getMemory' } // request the durable agent memory (Memory settings tab)
  | { type: 'saveMemory'; text: string } // overwrite the durable agent memory
  // ---- saved conversations ----
  | { type: 'saveConversation'; conversation: Conversation } // persist (autosave after each turn)
  | { type: 'listConversations' } // request the history list
  | { type: 'loadConversation'; id: string } // resume a specific conversation
  | { type: 'deleteConversation'; id: string }
  // ---- open-a-folder (project mode) ----
  | { type: 'pickFolder' } // open the folder dialog (host-side)
  | { type: 'openFileInFolder'; path: string } // open a file from the explorer (relative to root)
  | { type: 'requestProjectGraph' }
  // ---- #5 real-input · #2 summary · #8 paper ----
  | { type: 'traceWithRealInput'; path: string; name: string; line: number } // pick a data file -> # fusion: input = load(...)
  | { type: 'requestModelSummary' } // lazy build of the Summary tab
  | { type: 'requestPaper' } // lazy build of the Paper tab
  | { type: 'explainPaper'; id: string } // run the agent "explain like a paper"
  | { type: 'cancelPaperExplain' }
  | { type: 'compareFiles'; pathA: string; pathB: string } // design B — request a faithful-port compare
  | { type: 'pickCompareFile' } // design B — open a file dialog for the reference, compare against the open file
  // trace-assist: ask the agent to author a `# fusion:` directive for a function
  | { type: 'traceAssist'; id: string; path: string; name: string; line: number }
  // apply an agent-proposed directive (review mode -> user clicked Insert)
  | { type: 'insertDirective'; path: string; line: number; text: string };
