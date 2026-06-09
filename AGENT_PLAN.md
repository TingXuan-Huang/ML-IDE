# Fusion — Agent integration plan

## Decision
Add a pluggable **CLI-agent** capability with two faces:
1. **Chat sidebar** — a side panel to converse with an agent about the open code (with the
   cockpit's structure/trace as context).
2. **Trace-assist** — when auto-synth can't figure out how to call a function, the agent
   **authors a `# fusion:` directive**, verified by an **agent↔tracer loop**.

The agent backend is the **user's choice** (Claude Code / Codex / custom command) set on a
**config page**. Applying agent-suggested inputs is a **trust preference**: *review-then-insert*
or *auto-apply-and-trace*.

## Why (the reframe)
Statically parsing arbitrary model architectures to synthesize inputs is brittle — heuristics +
magic comments still can't cover truly arbitrary models. An LLM that *reads the code* is the robust
escape hatch. Crucially, the agent does **not** become the per-trace runtime: it **writes a
`# fusion:` directive** (deterministic, reuses the directive + provenance system already built).
The tracer is the agent's **verification tool**: propose input → run `trace_function` → feed any
shape error back → revise → repeat until green. (Static call-graph building stays trace-driven —
the trace is ground truth there; the agent is for input-synthesis + explanation, not the graph.)

## Architecture (fits the existing host/core split)
```
AgentClient  (@fusion/core, pure Node — like HelperClient)   spawn the user's CLI, stream stdout
  ├─ adapter: claude    claude -p "<prompt>" --output-format stream-json
  ├─ adapter: codex     codex exec "<prompt>"
  ├─ adapter: custom    <user command template containing {prompt}>
  └─ adapter: mock      echoes a canned directive (for tests / no-CLI dev)
Trace-assist orchestrator (host)   build prompt -> AgentClient -> trace_function retry loop -> directive
Chat sidebar (webview)   transcript + streaming + input + agent picker + auto-attached file context
Config page  (webview)   pick agent kind + command + trust pref -> writes ~/.fusion/agent.json
```
- **Auth:** piggybacks on the user's CLI login — **no API keys in the app**.
- **Determinism:** the agent's output is a pinned `# fusion:` line, not a fresh runtime guess.

## Protocol additions (@fusion/shared)
- Webview→host: `agentPrompt {id,text}` · `agentCancel {id}` · `traceAssist {path,name,line}` ·
  `insertDirective {path,line,text}` · `getAgentConfig` · `saveAgentConfig {config}`
- Host→webview: `agentChunk {id,delta}` · `agentDone {id,text}` · `agentError {id,message}` ·
  `agentConfig {config}` · `directiveProposed {path,line,directive,forFunction}`

## Build order (v1 = chat sidebar + trace-assist)
1. **AgentClient + adapters + config** (core + host). Ship the **mock** adapter first so it's
   testable in CI; real claude/codex/custom adapters for the user's machine.
2. **Chat sidebar** (webview): collapsible right column `[editor | cockpit | chat]`; streaming
   transcript, input, agent picker, current-file context chip.
3. **Trace-assist**: an "✦ Ask agent" affordance on un-traceable functions (and a chat intent) →
   prompt with the source + failure reason → agent↔tracer retry loop → `# fusion:` directive;
   the **trust setting** gates insert (review | auto).
4. **Config/settings page**: pick agent + command + trust pref + "test connection".

## Risks / notes
- CLI agents can't be fully exercised in the dev sandbox (no auth/CLI) → build against the **mock
  adapter**; the user verifies real agents on their Mac.
- Streaming differs per agent (claude `stream-json` JSONL vs codex text) → the adapter normalizes;
  fall back to on-complete when streaming isn't available.
- We run LLM-generated code (the directive) → the **trust setting** gates it; the agent CLI itself
  runs with the user's own permissions.
- Large files → send the target function + imports (not the whole file) when it's big.
- **Desktop-first** (we own the window). The VS Code host can reuse AgentClient + trace-assist later.

## Out of scope (v1)
Autonomous multi-file edits/refactors · agent-built call graph (trace is ground truth) · MCP servers ·
conversation persistence across restarts (kept in-memory for v1).
