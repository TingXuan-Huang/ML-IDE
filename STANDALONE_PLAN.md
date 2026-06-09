# Fusion — standalone app plan (Electron → full IDE)

**Decision (2026-06-09):** build a **standalone desktop app** (Electron + Monaco), grow it into a
**full IDE shell**, and **keep the VS Code extension alive** — both hosts driven by one shared core.
Tauri stays on the table as a later speed/size swap (smaller binary on modest machines).

## Why standalone (turning "VS Code is inefficient for ML/DS" into specs)
VS Code makes you bounce between editor → terminal → matplotlib popups → debugger; env/kernel switching
is painful; data viz is external; large tensor pipelines are opaque. Fusion owns the **whole loop**:
edit → trace shapes → see data, in one window. The shape-flow cockpit already exists; the app gives it a home.

## The architecture that makes this cheap
Only the **host shell** was ever VS Code-specific. Everything below is host-agnostic and already built:

```
HOST SHELL        VS Code ext  |  Electron app  (← swap/add hosts here)
─────────────────────────────────────────────────────────────────────
@fusion/core      HelperClient (spawn lens-helper, JSON-RPC) + toFileStructure adapter   [no vscode/electron]
@fusion/shared    typed host↔UI message protocol                                          [pure types]
webview-ui        Svelte cockpit (Blocks / Graph / Data)                                  [plain Vite web app]
lens-helper       Python: tensor-shape tracer, loaders, ast structure/callgraph, JSON-RPC [plain Python]
```

The host↔UI contract is `postMessage`: renderer→host `post(msg)`, host→renderer arrives as a window
`message` event. The Electron **preload** re-emits IPC as window messages, so **the UI runs unchanged**.

## Repo layout (monorepo, npm workspaces)
```
shared/         @fusion/shared   — protocol (types only)
core/           @fusion/core     — HelperClient + adapter (+ vitest)   ← NEW, extracted this phase
webview-ui/     fusion-webview-ui — Svelte cockpit (shared renderer)
extension/      fusion-cockpit   — VS Code host (thin)
hosts/desktop/  @fusion/desktop  — Electron host (thin)                ← NEW
lens-helper/    Python core (pytest)
spike/          throwaway demos + sampledata
```

## Roadmap (each phase ships something runnable)

- **P0 — Monorepo + shared core** ✅ *this phase.* Workspaces; extract `@fusion/core`; host-agnostic UI
  bridge; Electron skeleton that loads the cockpit + spawns the helper. Extension keeps working.
- **P1 — Standalone MVP.** Native window runs the cockpit: `File ▸ Open .py` → structure + call graph →
  `▶ trace` → real shapes. Data file → viz. *(= VS Code parity, no VS Code.)*
- **P2 — Editor (Monaco)** ✅ *core done.* Monaco embedded in the desktop host (dynamically imported →
  code-split, so the VS Code webview stays Monaco-free at 26 kB). Two-pane shell `[editor | cockpit]` (desktop
  only; VS Code renders the cockpit alone). Wired: cockpit symbol-click → editor scrolls (`revealInEditor`);
  editor caret → cockpit highlights the enclosing function (`caretLine`); **Cmd+S** → host writes to disk +
  re-structures. *Next P2.x:* draggable splitter, editor decorations for shape problems, theme-match Monaco to
  the cockpit, slim Monaco to python-only.
- **P3 — File workspace.** Open folder, file tree, tabs, new/save, fs watcher → re-structure on save.
- **P4 — Command palette + settings + keybindings.** Quick-open (Cmd+P), commands (Cmd+Shift+P),
  settings UI — incl. a **first-class Python/conda env picker** (the thing VS Code does badly for ML).
- **P5 — Integrated terminal + run.** `node-pty` terminal; run-file / run-cell; output panel.
- **P6 — Git + diff.** status / stage / commit; Monaco diff view.
- **P7 — ML/DS superpowers (the moat).** notebook/cell execution, run/experiment tracker, tensor
  inspector, dataset browser, attach to a **remote kernel** (cluster GPU). This is why it's not just VS Code.
- **P8 — Packaging.** electron-builder, signing, auto-update. Then evaluate **Tauri** swap for size/speed.

**Cross-cutting:** the VS Code extension ships from the same `@fusion/core` + `webview-ui` — features land once.

## Honest scope note
A *full* IDE is P2–P6 of real work (file tree, tabs, palette, terminal, git) on top of P1. The plan front-loads
a usable app at **P1** so it's dogfoodable immediately, then layers the shell. We are **not** rebuilding an LSP,
a debugger, or an extension marketplace — Monaco + the trace engine cover the ML/DS workflow without them.

## Run (after `npm install` at the repo root once)
- **Desktop app:** `cd hosts/desktop && npm install && npm start`  (first `npm install` fetches the Electron binary)
- **VS Code extension:** unchanged — open the repo, Run ▸ "Run Fusion Cockpit" (F5).
- **Tests:** `npm run test:ui` (vitest, core adapter) · `npm run test:py` (pytest, lens-helper).
