# Contributing to Fusion

Thanks for looking. Fusion is an early, experimental ML-code comprehension IDE; issues and PRs are
welcome.

## Setup

Prerequisites: **Node 18+** and **Python 3.10+**.

```bash
npm install                          # installs all workspaces (npm-workspaces monorepo)
pip install -e 'lens-helper[test]'   # the Python helper + its test deps (pytest, torch, numpy, ...)
```

## Running it

- **Desktop app**: `npm run start:desktop` (builds core + webview, launches Electron).
- **VS Code extension**: `npm run build:vscode`, then press F5 in `extension/` and run
  "Fusion: Open Cockpit".

## Tests (CI runs all of these)

```bash
npm run test:ui        # vitest — @fusion/core (adapter, client, validators, directives, ...)
npm run test:webview   # vitest — the webview store + formatters
npm run test:py        # pytest — the lens-helper tracer / compare / project / rpc
```

Plus the type/build checks CI runs:
`npm run build:core && npm run build:host -w @fusion/desktop && npm run compile:ext -w fusion-cockpit && npm run build:ui`.

## Layout

See the README's "Repo layout" and "How it works" sections. In short: one host-agnostic core
(`@fusion/shared` protocol + `@fusion/core` client/adapter) drives two thin hosts (the VS Code
extension and the Electron desktop app) plus a Python sidecar (`lens-helper`). Most features land as
a helper method + a protocol message + a Svelte zone, with no host-specific code.

## Conventions

- Commits follow [Conventional Commits](https://www.conventionalcommits.org/)
  (`feat:`, `fix:`, `perf:`, `refactor:`, `test:`, `chore:`, `docs:`).
- The webview imports only `@fusion/shared` (types). `@fusion/core` pulls in `child_process`, so it
  is host-only — never import it from `webview-ui`.
- The host ↔ webview message switches use a `never`-exhaustiveness `default`, so a new message
  variant is a compile error until it's handled explicitly. Keep it that way.
- The tracer runs user code under `sys.settrace` + a watchdog. Keep it duck-typed (no hard `torch`
  import) and cheap on the per-line path.
