// @fusion/core — host-agnostic glue shared by every Fusion host (VS Code extension,
// Electron desktop app). NO vscode and NO electron imports live here: just the warm
// JSON-RPC client to the Python sidecar and the pure structure/trace adapter. Hosts
// import these and supply their own UI shell + file/editor access.
export { HelperClient } from './helperClient';
export type { HelperOptions } from './helperClient';
export { toFileStructure, moduleTraceToRaw } from './adapter';
export type { RawStructure, RawTrace, TraceModuleResult } from './adapter';
export { AgentClient, AGENT_PRESETS, defaultAgentConfig, loadAgentConfig, saveAgentConfig } from './agentClient';
export type { AgentKind, AgentConfig, AgentRunOptions } from './agentClient';
export { parseRemembers, stripRememberLines, appendMemoryFact, memoryPromptBlock } from './memory';
export { formatTranscript } from './conversation';
export type { Turn } from './conversation';
