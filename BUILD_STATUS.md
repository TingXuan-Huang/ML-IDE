# Fusion IDE — build status

A VS Code "comprehension cockpit" for ML/DS. Full plan + all four review reports live in
`~/.gstack/projects/IDE/tingxuanhuang-unknown-design-20260608-221024.md` (20 tasks T1–T20).
This file tracks what's built; the design doc is the source of truth.

## Done & verified ✅ (backend spine — proven across both languages)

| Task | What | Verify in 30s |
|------|------|----------------|
| T1 | M0 spike: trace captures per-line tensor shapes + catches shape-mismatch crash | `python3 spike/trace_probe.py` |
| T1 | API probe (VS Code extension) — measures static fallback | open `spike/api-probe` in VS Code → Run ▸ Start Debugging → run "Fusion Spike: Probe APIs" |
| T2 | `@fusion/shared` — typed host↔webview message protocol | `cd shared && npm run build` |
| T3 | `lens-helper` warm Python sidecar: loaders (npy/csv/parquet) + trace core + JSON-RPC stdio | `cd lens-helper && printf '{"id":1,"method":"ping"}\n' \| python3 -m lens_helper` |
| T3 | `HelperClient` (TS): spawn helper + JSON-RPC + auto-restart on crash | `cd extension && npm run smoke` |
| T3 | helper `structure_file` (ast) + `trace_file` (exec under tracer, stdout redirected) | tested on `spike/sampledata/demo_model.py` |
| M1 | extension host + inline cockpit (variant A): active file → structure → trace → **inline shapes** | **VERIFIED LIVE in Windsurf F5** — `demo_model.py` traced, shows `h[8,32]`, `y[8,4]` inline (smart density) |
| T7 | Graph zone: `callgraph_file` (ast) → SVG node-link, click-to-reveal | **live in Graph tab**; `pipeline.py` → 6 nodes/3 edges |
| T14 | Shape-problem highlight: crash line (via `exception` event) → red line + mismatch msg + count; a caught crash = **success** (not a trace error) | tested `buggy.py` → crashLine 14, `mat1/mat2 … (8x32 and 64x4)` |
| — | Trace also captures **return-value shapes** (`return self.fc2(h)` → `return[8,4]`) via the `return` event | tested `pipeline.py` |
| NEW | **Trace this function** — call ONE function directly, **no `__main__`, no debugger**: per-function `▶ trace` → helper loads the module (main block won't run), instantiates the class, **auto-synthesizes** an input from the first `Linear`/`Conv` layer, traces just that call, and **always shows the input used** (`note`, e.g. `Encoder().forward(randn(2, 16))`). Empty capture (needs ctor/args) → visible info toast, not silent. | `cd lens-helper && python3 -c "from lens_helper import tracer;print(tracer.trace_function('../spike/sampledata/encoder.py','forward',0)[3])"` → `Encoder().forward(randn(2, 16))` |
| NEW | **Trace this file** = `trace_module` (replaces `trace_file` in the UI). For EVERY model class + zero-arg function in the file: **build-check** the constructor (catches `__init__` shape errors) + synth-call `forward` with **batch 2** (so file-trace and per-function trace AGREE — no more 8-vs-2). Works on **pure libraries** (no `__main__`). Each crashing function gets its own red line (`problems[]`). `trace_file` (run `__main__`) kept in the helper for a future "Run" action. | `cd lens-helper && python3 -c "from lens_helper import tracer;r=tracer.trace_module('../spike/sampledata/buggy.py');print(r['problems'])"` → `L14 … (2x32 and 64x4)` |

## How to run the cockpit (F5)
1. Open the **project ROOT** (`Desktop/IDE`) as the VS Code workspace (so `lens-helper` resolves), OR open `extension/`.
2. Run ▸ Start Debugging → "Run Fusion Cockpit". In the `[Extension Development Host]`:
3. `Cmd+Shift+P` → **"Fusion: Open Cockpit"** (opens beside the editor).
4. Open a Python model (`spike/sampledata/demo_model.py`) → Blocks zone shows structure → click **▶ Trace this file** → real shapes appear (smart density: only where shapes change). Data tab → browse a csv/npy.
5. **Trace a single function (no `__main__` needed):** open `spike/sampledata/encoder.py` (a *pure library* model — running the file does nothing). Each function header has a **`▶ trace`** on the right → click it on `forward` → the cockpit calls `Encoder().forward(randn(2, 16))` directly and fills in shapes. The synthesized call shows in the status bar so the input is never hidden.

## Repo layout (monorepo — npm workspaces)
```
shared/         @fusion/shared    — protocol.ts (types: zones, trace states, shape records, data meta)
core/           @fusion/core      — HelperClient (JSON-RPC) + toFileStructure adapter (+ vitest). No vscode/electron.
webview-ui/     fusion-webview-ui — Svelte cockpit (Blocks/Graph/Data). Host-agnostic bridge (src/vscode.ts).
extension/      fusion-cockpit    — VS Code host (thin: panel + msg bus + python resolve)
hosts/desktop/  @fusion/desktop   — Electron host (main + preload). Standalone app shell. → STANDALONE_PLAN.md
lens-helper/    Python core: lens_helper/{loaders,tracer,structure,callgraph,rpc,__main__}.py + tests/ (pytest)
spike/          throwaway: trace_probe.py (PROVEN), api-probe/ (F5), sampledata/ (demo_model, encoder, buggy, pipeline)
```
**Going standalone:** the full Electron-app → full-IDE roadmap (P0–P8) lives in **`STANDALONE_PLAN.md`**.
P0 (monorepo + `@fusion/core` extraction + Electron skeleton) is **done**; only the host shell differs between
VS Code and desktop — the Python tracer + Svelte cockpit are shared verbatim.

### Run the standalone desktop app
```
npm install            # once, at repo root (creates @fusion/* workspace links + fetches Electron)
npm run start:desktop  # builds core+host+UI, launches the Electron window
```
Then **File ▸ Open Python File…** (`spike/sampledata/encoder.py`) → cockpit shows structure → **▶ trace** a function.

## Next, in order
- **F5-verify M1** (above) and fix what the live run surfaces. Likely tweaks: helper `cwd` (assumes
  `lens-helper/` is in the workspace root), the interpreter (`python.defaultInterpreterPath`, else `python3`).
- ~~T7 Graph zone~~ **DONE (M1, ast-based):** `callgraph_file` (helper, intra-file calls) → SVG node-link in
  the Graph tab, click-to-reveal. Tested: `pipeline.py` → 6 nodes, 3 edges. CallHierarchy/Cytoscape = later upgrade.
- ~~Data zone histogram~~ **DONE** — `.npy` → value histogram; `.csv` → schema + rows + first-numeric-column histogram; "pick another" link.
- ~~Svelte migration~~ **DONE** — `webview-ui/` is a Vite+Svelte app (App + Blocks/Graph/Data components + a
  store + vscode bridge), built to `dist/assets/main.{js,css}` (fixed names) and loaded via `asWebviewUri` +
  nonce'd CSP (`cspSource` in style/connect/img, nonce on script) + `localResourceRoots`. The inline HTML is
  gone. `npm run compile` (in `extension/`) now builds tsc + the webview bundle, so F5 never serves a stale UI.
  NOTE: Svelte templates can't contain TS `as` casts — keep casts in `<script>`.
- **tree-sitter structure** — replace the ast-via-helper path so blocks render with NO Python env (locked decision).
- **T9 tests** — ✅ **unit layer DONE.** Python: `cd lens-helper && python3 -m pytest tests/ -q` (13 — tracer
  shapes/return/crash/stdout-isolation, **trace_function** no-`__main__`/zero-arg/needs-args, loaders, callgraph,
  structure). TS: `cd extension && npm test` (3 — the `toFileStructure` adapter, vscode-free/testable). `out/`
  excludes tests.
  Remaining test layers: @testing-library/svelte component tests (webview zones), @vscode/test-electron
  integration (helper-client RPC + structure/trace round-trip), Playwright on the standalone webview app.
- **T8 debounce tuning, T10 degraded paths, T11 CI, tree-sitter no-env structure.**

## Caveats found while building (M1)
- `trace_file` can't trace a file that itself calls `sys.settrace` (nested-tracer conflict) — rare for ML
  code; use a normal model. Real product: detect + warn, or a non-settrace mechanism later.
- Structure uses Python `ast` in the helper (M1) — needs the env; tree-sitter (no-env) is the locked upgrade.
- The helper's stdout IS the JSON-RPC channel: `trace_file` redirects the traced code's stdout/stderr so user
  `print()`s can't corrupt it. Keep that invariant for any new exec path.

## Fixed live during the first F5 (extension.ts)
- **Stale interpreter:** the user's selected Python env can be a *deleted* conda env (`…/envs/plot/bin/python`
  → ENOENT, silent hang). `resolvePython()` skips non-existent absolute paths and falls back to a real one;
  override with the `fusion.pythonPath` setting. Spawn errors now settle pending requests (no hang).
- **Webview focus:** `vscode.window.activeTextEditor` is `undefined` while the cockpit panel has focus, so
  read a tracked `lastEditor` (updated on `onDidChangeActiveTextEditor`) instead of reading it at send time.
- **Debug aid:** the extension logs to the **"Fusion Cockpit"** output channel (spawn cmd, requests, errors) —
  the fastest way to diagnose a blank cockpit.

## Resolved
- F5 API probe ran: Pylance ACTIVE but `DocumentSymbols: 0` (the probed file sits OUTSIDE the dev-host
  workspace root, and no interpreter was selected). Read: the static LSP path is fragile here — which
  **confirms** the locked decisions: use bundled **tree-sitter** (zero interpreter/LSP dependency) for the
  structure skeleton, and let the **trace carry the shapes** (proven). No plan change. For a clean LSP
  reading later: open the project ROOT as the dev-host workspace + select the Python interpreter
  (bottom-left), then re-run the probe.

Resume the gstack thread anytime with `/context-restore`.
