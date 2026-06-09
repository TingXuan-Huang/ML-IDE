# M0 Spike — feasibility probe (THROWAWAY)

Retires the two riskiest assumptions in the Fusion IDE plan before building T1–T20.
This whole folder is disposable; the knowledge is the deliverable.

## 1. Trace probe (Python) — PROVEN ✅

```bash
python3 spike/trace_probe.py
```

Captures every intermediate tensor's shape **per source line from ONE run** via `sys.settrace`,
and catches a shape-mismatch crash with the shapes leading into it.

Verified output (torch 2.3.0, numpy 1.26.4):
- **Working model:** `x [4,16] → h [4,32] → a [4,1] → y [4,8]` — the full shape flow, inline on the code.
- **Buggy model:** caught `RuntimeError: mat1 and mat2 shapes cannot be multiplied (4x32 and 64x8)`
  at the crash line, with `h [4,32]` captured right before it.
- **Bonus:** the silent-broadcast case (`a [4,1]` broadcasting against `h [4,32]`) is captured too —
  the data for the shape-problem-detection feature (CEO cherry-pick 3 / T14) is already there.

Key technique (durable): `sys.settrace` fires the `line` event *before* the line runs, so attribute
each frame's tensor locals to the **previous** line number. Scope the tracer to the target file
(`co_filename == THIS_FILE`) to skip torch internals.

## 2. VS Code API probe (TypeScript) — BUILT, run with F5

Measures how much the **static fallback** (Pyright hovers + CallHierarchy) actually gives, which
decides how hard the trace has to carry Mode 1 (the outside voice predicted: not much).

```bash
cd spike/api-probe && npm install && npm run compile   # already done — out/extension.js exists
```
Then: open `spike/api-probe/` in VS Code → press **F5** (launches the Extension Development Host) →
in the new window open a real torch model `.py` → focus it → `Cmd+Shift+P` →
**"Fusion Spike: Probe APIs on active file"** → read the **Fusion Spike** output channel.

What to look for:
- Function hovers like `(function) forward(x: Tensor) -> Tensor` → a type you can regex.
- Interior-var hovers mostly `(no type)` → static fallback is thin → **the trace carries Mode 1** (expected).
- CallHierarchy `out=[...]` listing callees → the scoped function graph (M4) is viable.

## M0 verdict

| Risk | Status |
|------|--------|
| Trace delivers real intermediate shapes from one run | **PROVEN** (real output above) |
| Trace catches shape-mismatch crashes with context | **PROVEN** |
| Static fallback (Pyright hovers) richness | Run the F5 probe to measure; plan already assumes thin |
| CallHierarchy gives usable graph edges | Run the F5 probe to confirm |

**The single biggest risk — does the centerpiece work — is retired.** Next: run the F5 probe to
size the static fallback, then proceed to **T2** (freeze the `shared/` message protocol) and scaffold
the real workspace (`extension/ webview-ui/ shared/ lens-helper/`).
