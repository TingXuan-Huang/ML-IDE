# Fusion

**A comprehension cockpit for machine-learning code.** Fusion traces the tensor shapes flowing
through a model and shows them *inline, next to the code*, so you can read and trust a forward pass
without littering it with `print(x.shape)`.

> Status: early and experimental (n=1, built as a personal tool). The backend tracer and the
> shared core are solid and well-tested; the desktop shell and the newest add-ons are evolving.

---

## The idea

Reading ML code you didn't write is hard in a way that ordinary code isn't. The logic is short, but
the *shapes* are everything, and they're invisible. To understand `q = self.proj(x).view(B, L, H,
D).transpose(1, 2)` you have to simulate it in your head or instrument it by hand. So everyone does
the same thing: paste `print(x.shape)` (and `print(x.dtype, x.device)`, and `print(x.isnan().any())`)
through the forward pass, run it, read the numbers, delete the prints, and lose the map the moment
they clean up.

Fusion's wedge is to **kill that loop**: trust unfamiliar model code without hand-instrumenting it.
It runs your model under a tracer, captures the shape (and dtype, device, memory) of every tensor on
every line, and renders the result as annotations on the source. The mental model you'd build with a
dozen throwaway prints is just *there*, and it survives edits.

Conventional tools each cover a slice of this and stop: `torchinfo` prints a static summary, Netron
renders a static graph, TensorBoard shows the graph plus scalars, `jaxtyping` annotates shapes you
write by hand. None of them trace *live* shapes *inline in your source* with named dimensions and a
mapping back to the paper. That gap is what Fusion is built around.

Four kinds of reader share that one job, and Fusion is aimed at all of them:

- **Re-implementers** porting a model from a paper or another repo (does my version match the
  reference?).
- **New hires / students** onboarding onto a large unfamiliar model.
- **Engineers** localizing a shape mismatch or a NaN in a training pipeline.
- **Self-learners** reading reference implementations to actually understand them.

The design principle throughout: own the whole loop in one window (edit → trace shapes → see data),
and make the trace carry everything a `print()` would.

---

## What it does

**Shape tracing**
- Runs a model (or a single `forward`) under `sys.settrace` and captures every tensor local's shape
  per source line. No code changes required; the tracer duck-types tensors, so it never imports
  torch unless your own code does.
- Renders shapes **inline in the Monaco editor** (gray ghost-text) and in a **Blocks** view, with a
  "smart density" mode that only shows a shape when it *changes*.
- **Auto-synthesizes inputs** (dtype/layer-aware: `Embedding` → token ids, `Conv` → an image,
  `Linear` → a vector), or you pin the exact input with a `# fusion:` magic comment.
- Catches a shape-mismatch crash and marks the offending line (a caught crash is a *success*, not a
  tool error).

**Reading shapes like a paper**
- **Abstract / symbolic dims**: relabel concrete sizes as `B, L, D, H, dh, C, H, W` inferred from the
  input and the model's attributes, so `qkv[2, 16, 3, 4, 32]` reads as `qkv[B, L, 3, H, dh]`.
- **Op notes**: annotate `matmul` / `@` / `bmm` / `einsum` / `softmax` / reshape / permute /
  broadcast lines with what they did (`matmul [B,L,D]·[D,D] → [B,L,D]`).
- **Paper mode**: the forward pass rendered as an ordered, per-module sequence of named tensor ops,
  plus an optional agent "explain like a paper" prose pass.

**Comprehension at the model + project level**
- **Model summary** (torchinfo-style): per-module output shapes, parameter counts, parameter memory.
- **Project graph**: open a folder and see the cross-file import graph with a laid-out node-link
  diagram.
- **Data viz**: browse a `.csv` / `.npy` / `.parquet` and see schema + a value histogram.

**The newest add-ons (the "Tensor Inspector" + "Compare" suite)**
- **Tensor Inspector**: a toggle that appends the rest of what `print(x.shape, x.dtype, x.device)`
  carries — `h[B, L, D] float32 · cuda:0 · 48MB` — plus a **NaN sentinel** that flags the first
  tensor to go non-finite (no manual `isnan` print). NaN only, since `-inf` attention masks are
  intentional.
- **Faithful-port Compare**: pick a reference implementation and diff it, module by module, against
  the open file. Each `forward` is aligned by its op-kind + (batch-normalized) shape sequence, so a
  single inserted line is one gap rather than a cascade, and a wrong shape or op is flagged at the
  first diverging step. Every step shows its source line, so activation/arg differences are visible
  too.

**Agent assist** (optional, pluggable CLI agent: Claude Code, Codex, or any command)
- "✦ ask" a function and the agent writes a `# fusion:` directive to make it traceable, verified
  against the tracer before it touches your file.
- Durable per-project memory and resumable conversations.

---

## How it works

Only the **host shell** was ever editor-specific. Everything below it is host-agnostic, so the same
core drives both a VS Code extension and a standalone Electron app:

```
HOST SHELL        VS Code extension   |   Electron desktop app      (← add/swap hosts here)
──────────────────────────────────────────────────────────────────────────────────────────
@fusion/shared    typed host ↔ webview message protocol                      [pure types]
@fusion/core      HelperClient (spawns lens-helper, JSON-RPC) + the
                  raw-JSON → typed-structure adapter                          [no vscode/electron]
webview-ui        Svelte "cockpit": Blocks / Graph / Summary / Paper /
                  Data / Project / Compare zones                              [plain Vite web app]
lens-helper       Python: tensor-shape tracer, ast structure/callgraph,
                  loaders, project graph, JSON-RPC over stdio                 [plain Python]
```

- **The contract is `postMessage`.** The renderer posts a message to the host; the host replies as a
  window `message` event. In Electron, a preload script re-emits IPC as window messages, so the
  Svelte UI runs *unchanged* across both hosts.
- **The Python sidecar is a plain process.** `@fusion/core` spawns `lens-helper` and talks JSON-RPC
  over stdio, and respawns it if it dies. The helper never imports torch itself; it duck-types
  tensors from the user's own runtime.
- **The protocol is the single source of truth.** Both sides import the same `@fusion/shared` types,
  so the messages they exchange cannot drift (the compiler enforces it).

This separation is what makes a new feature cheap: most land as a helper method + a protocol message
+ a Svelte zone, with no host-specific code.

---

## Repo layout (monorepo, npm workspaces)

```
shared/         @fusion/shared     — the host ↔ webview protocol (types only)
core/           @fusion/core       — HelperClient (JSON-RPC) + structure/trace adapter (+ vitest)
webview-ui/     fusion-webview-ui  — Svelte cockpit (the shared renderer)
extension/      fusion-cockpit     — VS Code host (thin: panel + message bus + python resolve)
hosts/desktop/  @fusion/desktop    — Electron host (thin: window + preload + dialogs + menu)
lens-helper/    Python core        — tracer, loaders, ast structure/callgraph, project, JSON-RPC (pytest)
spike/          throwaway demos + sample data (demo_model.py, encoder.py, embeddings.npy, ...)
```

Design docs and history: `STANDALONE_PLAN.md` (the desktop plan), `BUILD_STATUS.md` (what's built),
`AGENT_PLAN.md` (agent integration), `TODO.md` (backlog).

---

## Build & run

Prerequisites: **Node 18+** and **Python 3.10+**. At runtime the helper duck-types tensors and
imports `torch` lazily (your model code brings its own torch); the test suite, however, imports
torch directly, so install the `test` extra before running the Python tests.

```bash
npm install

# Standalone desktop app (builds the core + webview, then launches Electron)
npm run start:desktop

# VS Code extension: build, then press F5 in extension/ and run "Fusion: Open Cockpit"
npm run build:vscode

# Tests
npm run test:ui                       # vitest over @fusion/core (adapter, client, memory, ...)
pip install -e 'lens-helper[test]'    # one-time: pytest + torch + numpy/pandas/pyarrow
npm run test:py                       # pytest over lens-helper (tracer, compare, project, ...)
```

Then open a Python model (e.g. `spike/sampledata/demo_model.py`), click **▶ Trace this file**, and
the shapes fill in. Open `spike/sampledata/encoder.py` (a pure-library model with no `__main__`) and
use the per-function **▶ trace** to trace just one `forward`.

---

## The `# fusion:` directive

When the tracer can't auto-synthesize an input (multi-arg forwards, models with constructor args),
pin it with a magic comment on the line(s) directly above the target:

```python
# fusion: model = Seq2Seq(vocab=500)                      # construct with specific ctor args
class Seq2Seq(nn.Module):
    # fusion: input = torch.randint(0, 500, (4, 10))      # exact input; a (a, b) tuple -> *args
    def forward(self, tokens): ...
```

`load("relpath")` inside a directive loads a real tensor from disk (`.npy` / `.npz` / `.pt` / `.csv`),
so you can trace with real data instead of synthesized input. Directives are eval'd in the module
namespace behind a small AST safety check (it rejects imports, dunder access, and `os`/`subprocess`;
it is a speed-bump, not a sandbox — only trace code you trust).

---

## Roadmap

- **Compare**: op-aware LCS alignment and the source-line diff are in; agent-driven matching of
  *renamed* classes is the next step.
- **Compute-aware + shareable**: a FLOPs / parameter / activation-memory budget in the summary, and
  exporting a traced model's shape-flow as an image or markdown to paste into a PR or paper.
- **Tauri** as a later size/speed swap for the desktop shell.

---

*Built with the Claude Code agent in the loop. Generated artifacts and design reviews informed the
architecture; the code and tests are the source of truth.*
