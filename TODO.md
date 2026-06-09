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

## In-flight
- **Agent v1**: ① AgentClient ✅ · ② chat sidebar ✅ · ③ trace-assist (next) · ④ config page.
- **Desktop blank-screen**: theme now injected in the renderer (was gated on a host event that
  never fired). Remove debug instrumentation once confirmed working.

## Known smaller items
- Draggable splitter between panes · editor gutter decorations for shape problems ·
  slim Monaco to python-only (3.3 MB → ~1 MB) · remove redundant `extension/package-lock.json` ·
  tree-sitter no-env structure · CI.
