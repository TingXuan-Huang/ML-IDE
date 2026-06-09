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

## How to run the cockpit (F5)
1. Open the **project ROOT** (`Desktop/IDE`) as the VS Code workspace (so `lens-helper` resolves), OR open `extension/`.
2. Run ▸ Start Debugging → "Run Fusion Cockpit". In the `[Extension Development Host]`:
3. `Cmd+Shift+P` → **"Fusion: Open Cockpit"** (opens beside the editor).
4. Open a Python model (`spike/sampledata/demo_model.py`) → Blocks zone shows structure → click **▶ Trace this file** → real shapes appear (smart density: only where shapes change). Data tab → browse a csv/npy.

## Repo layout
```
shared/        @fusion/shared — protocol.ts (zones, hint density, trace states, shape records, data meta)
lens-helper/   lens_helper/{loaders,tracer,rpc,__main__}.py  + pyproject.toml
extension/     src/helperClient.ts (+ smoke)  — VS Code host; entry/UI wiring still TODO
spike/         throwaway: trace_probe.py (PROVEN), api-probe/ (F5), sampledata/
```

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
- **T8 debounce tuning, T9 tests (incl. standalone Playwright), T10 degraded paths, T11 CI.**

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
