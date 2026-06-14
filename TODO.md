# Fusion — backlog / TODO

Roadmap context: `STANDALONE_PLAN.md` (P0–P8), `AGENT_PLAN.md` (agent v1), `BUILD_STATUS.md` (shipped).

## Requested features (2026-06-09)
1. **Agent memory / self-improving tracer.** Give the agent persistent memory so that when it
   recognizes a *common* code pattern (e.g. a recurring model/input shape it had to hand-write a
   `# fusion:` directive for), it proposes an edit to `lens-helper/lens_helper/tracer.py`
   (e.g. a new entry in `_synth_input`) — turning one-off fixes into permanent heuristics.
   Needs: a memory store (per-project `.fusion/memory.json`), a "propose tracer patch" agent
   intent, and a review-before-apply gate (reuse the trust setting).
2. **More visualization methods.** Expand the Data zone beyond histogram/schema — e.g. line/scatter,
   correlation heatmap, image-grid for tensors, embedding scatter (PCA/UMAP), per-layer activation
   stats. (Pluggable viz registry in the webview + helper-side summarizers.)
3. **Open a folder (workspace), not just one file.** File tree + multi-file. The call graph then
   offers a scope toggle: **(a)** whole folder (cross-file edges), **(b)** single file (current),
   **(c)** a few selected files. Needs: cross-file call graph in the helper (resolve imports), a
   workspace/file-tree in the desktop host (P3), and a scope control in the Graph zone.
4. **Light mode for the editor + UI.** A theme toggle (dark/light) — set the `--vscode-*` variables
   for a light palette and switch Monaco's theme (`vs` vs `vs-dark`). Persist the choice in config.

## Requested features (2026-06-09, batch 2)
5. **Trace with REAL input** (not synthesized). Capture the actual tensors a model is called with in
   the user's own code / training script and trace those, instead of `# fusion: input` guesses. Best
   once we can open a whole folder (find the entrypoint / call site). The `# fusion: input` directive
   is the manual stand-in until then. *(Depends on folder support — feature 3.)*
6. **Abstract / symbolic trace** (paper-reading add-on). Trace with NAMED dims — `(B, H, W, D)`,
   `(B, L, D)` — instead of concrete numbers, so shapes read like a paper. Pairs with a future
   paper-reading feature (map equations ↔ code). Likely a symbolic shape engine that tracks dim
   *names* through ops (or annotates concrete dims with inferred names).
7. **Annotate broadcast / matmul ops.** When a line does a matmul (`@`, `matmul`, `bmm`, `einsum`) or
   a broadcasting elementwise op (`*`/`+` between mismatched shapes), the trace should NOTE it, e.g.
   `matmul [2,8]·[8,4]→[2,4]`, `broadcast [2,1,8]*[2,8,8]→[2,8,8]`. Impl: AST-detect the op per line +
   compare runtime operand shapes → attach an `op` note to that BlockLine. Near-term, standalone.
8. **Paper-reading mode** (the abstract-trace consumer). Read an ML paper beside the code, map
   equations / architecture ↔ functions, and show the symbolic `(B,H,W,D)` trace next to the paper's
   notation. Big, later — depends on #6 (symbolic trace). Likely uses `reading-academic-paper-ml`.

## In-flight
- **Agent v1**: ① AgentClient ✅ · ② chat sidebar ✅ · ③ trace-assist ✅ · ④ config page ✅.
  - ③ "✦ ask" on a function → agent writes a `# fusion:` directive → review (Insert button) or
    auto (insert + trace) per the trust setting. *Next:* the agent↔tracer retry loop (feed shape
    errors back for a revised directive).
  - ④ ⚙ in the chat header → settings modal (agent kind/command/args/promptVia/trust) →
    `~/.fusion/agent.json`.
- **Desktop blank-screen**: FIXED (fusion:// protocol + renderer theme + Chat afterUpdate). ✅

## Shipped this session
- Inline shapes in the Monaco editor (ghost-text + red markers) · cockpit horizontal scroll ·
  draggable pane splitters · file-level **✦ ask** (resolve every un-traceable function).
- **Agent↔tracer retry loop** ✅ — trace-assist verifies each proposed directive on a temp copy
  and feeds shape errors back to the agent (≤3 rounds) before proposing the verified one.
- **#7 matmul/broadcast op notes** ✅ — annotated in Blocks + editor ghost-text.

## Remaining (bigger, queued)
- **#1 agent memory / self-improving tracer** · **#2 more viz** · **#3 open a folder + cross-file
  graph** · **#4 light mode** · **#5 real input** (needs folder) · **#6 symbolic trace** · **#8 paper-reading**.

## Known smaller items
- editor gutter decorations for shape problems · slim Monaco to python-only (3.3 MB → ~1 MB) ·
  remove redundant `extension/package-lock.json` · tree-sitter no-env structure · CI ·
  per-function ▶ trace could also surface op notes (only trace_module does today).

## Possible future TODOs from review (2026-06-11)
- **P1: Review gate for durable agent memory.** `# fusion: remember ...` facts from agent
  replies are currently stored and later injected as high-priority prompt context. Before
  shipping memory broadly, make remembers pending suggestions that the user approves, edits,
  or rejects before they become durable facts.
- **P1: Reset trace state on file changes.** Opening a new file should clear stale `done`
  trace state, pending file-level `✦ ask`, and any file-specific trace assumptions so the UI
  cannot treat a newly opened static structure as already traced.
- **P2: Validate desktop IPC file paths.** `openFileInFolder` and `insertDirective` should
  resolve paths against the current workspace root and reject absolute or `..` escapes before
  reading or writing files.
- **P2: Reload project graph when the opened folder changes.** If the Project tab is already
  mounted, clearing `projectGraph` on a new folder is not enough; request a fresh graph when
  the folder root changes, with an in-flight guard.
- **P2: Track agent busy state by active/queued run count.** File-level `✦ ask` can queue many
  runs, but the UI uses one boolean. Derive busy/Stop visibility from streaming messages or
  tracked run ids so the first completion does not hide later active work.
- **P3: Hydrate Settings from the latest agent config.** The settings modal should request
  `getAgentConfig` on mount and update its local form once, unless the user has started
  editing, to avoid saving defaults over an existing config.
- **P3: Decide nested-function structure policy.** `structure_file` now reports top-level
  functions and one level of class methods. If dropping nested functions is intentional,
  document it; if not, switch to an AST visitor that preserves enclosing context.
