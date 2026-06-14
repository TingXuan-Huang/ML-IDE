"""Trace core for the cockpit's Mode 1 (shape-annotated blocks).

Runs a callable under sys.settrace and captures every tensor local's shape per source
line in the target file. The `line` event fires BEFORE the line runs, so locals are
attributed to the PREVIOUS line (the one that just produced them). Duck-types tensors
so the helper need not import torch unless the user's code does.

Inputs are auto-synthesized (dtype/layer-aware), or pinned by the user with a magic
comment directly above the def/class:
    # fusion: input = torch.randint(0, 1000, (4, 32))   # tuple -> *args for multi-input
    # fusion: model = MyModel(dim=8)                     # construct with specific args
"""
import re
import sys
from typing import Any, Callable, Dict, Optional, Tuple


def _is_tensor(v: Any) -> bool:
    root = type(v).__module__.split(".")[0]
    return root in ("torch", "numpy") and hasattr(v, "shape") and hasattr(v, "dtype")


def trace_callable(fn: Callable[[], Any], target_file: str):
    """Return (records, error).
    records: { lineno: { varname: {"shape":[...], "dtype":str, "changed":bool} } }
    error:   the Exception if fn() raised (e.g. a shape-mismatch RuntimeError), else None.
    """
    records: Dict[int, Dict[str, Any]] = {}
    prev_line: Dict[int, Optional[int]] = {}
    last_shape: Dict[str, Tuple[int, ...]] = {}
    crash_line: list = [None]  # first (deepest) exception line == the real crash site

    def snapshot(frame, lineno: int) -> None:
        slot = records.setdefault(lineno, {})
        for name, val in list(frame.f_locals.items()):
            if not _is_tensor(val):
                continue
            try:
                shape = tuple(int(d) for d in val.shape)
            except Exception:
                continue
            changed = last_shape.get(name) != shape
            last_shape[name] = shape
            slot[name] = {
                "shape": list(shape),
                "dtype": str(getattr(val, "dtype", "")).replace("torch.", ""),
                "changed": changed,
            }

    def tracer(frame, event, arg):
        if frame.f_code.co_filename != target_file:
            return None  # skip torch internals / other files
        fid = id(frame)
        if event == "call":
            prev_line[fid] = None
            return tracer
        if event in ("line", "return", "exception"):
            pl = prev_line.get(fid)
            if pl is not None:
                snapshot(frame, pl)
            if event == "exception" and crash_line[0] is None:
                crash_line[0] = frame.f_lineno  # raising line in the deepest traced frame
            if event == "line":
                prev_line[fid] = frame.f_lineno
            if event == "return" and _is_tensor(arg):
                # the returned value has no local name (e.g. `return self.fc2(h)`),
                # so attach its shape to the return line under a synthetic "return".
                try:
                    shape = tuple(int(d) for d in arg.shape)
                    changed = last_shape.get("return") != shape
                    last_shape["return"] = shape
                    records.setdefault(frame.f_lineno, {})["return"] = {
                        "shape": list(shape),
                        "dtype": str(getattr(arg, "dtype", "")).replace("torch.", ""),
                        "changed": changed,
                    }
                except Exception:
                    pass
        return tracer

    sys.settrace(tracer)
    try:
        fn()
        err: Optional[BaseException] = None
    except Exception as e:  # shape mismatch etc.
        err = e
    finally:
        sys.settrace(None)
    return records, err, crash_line[0]


def trace_file(path: str):
    """Exec a .py file under the tracer (opt-in "Trace this file").
    Returns (records, error_str_or_None, crash_line_or_None). Runs the module as
    __main__, so a file that runs a forward pass on execution gets real shapes."""
    import os
    import io
    import contextlib

    path = os.path.abspath(path)
    with open(path) as f:
        code = compile(f.read(), path, "exec")

    # The traced file's own print()/stderr must NOT pollute the JSON-RPC stdout
    # channel — capture and discard it while the module runs.
    sink = io.StringIO()

    def run():
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            exec(code, {"__name__": "__main__", "__file__": path})

    records, err, crash_line = trace_callable(run, path)
    return records, (f"{type(err).__name__}: {err}" if err else None), crash_line


# --- input synthesis & invocation ------------------------------------------------

_DIRECTIVE_RE = re.compile(r"^\s*#\s*fusion:\s*(input|model)\s*=\s*(.+?)\s*$")


class _CannotInvoke(Exception):
    """We can't auto-call a target (needs args / directive failed). The message explains
    why; it is NOT a runtime/shape error (those propagate from the traced call instead)."""


def _anchor_below(lines, i: int):
    """1-based line of the first real code line below 1-based line `i`, skipping blank
    lines, comments and decorators — i.e. the `def`/`class` header a directive attaches
    to (or None if there's no code below)."""
    for j in range(i, len(lines)):  # lines[j] is 1-based line j+1
        s = lines[j].strip()
        if not s or s.startswith("#") or s.startswith("@"):
            continue
        return j + 1
    return None


def _parse_directives(src: str):
    """Find `# fusion: input/model = <expr>` lines, each tagged with the line of the
    def/class it sits directly above (its 'anchor'). Anchoring binds a directive to ONE
    scope, so a stray/duplicate directive can't bleed into a neighbouring class/method."""
    lines = src.splitlines()
    out = []
    for i, line in enumerate(lines, start=1):
        m = _DIRECTIVE_RE.match(line)
        if m:
            out.append((i, m.group(1), m.group(2), _anchor_below(lines, i)))
    return out


def _pick_directive(directives, kind: str, def_line: int):
    """The `kind` directive ATTACHED to def_line (anchored directly above it; last one
    wins). Anchor-based, so a directive above a *different* scope is never mis-picked."""
    if not def_line:
        return None
    best = None
    for (ln, k, expr, anchor) in directives:
        if k == kind and anchor == def_line and (best is None or ln > best[0]):
            best = (ln, expr)
    return best[1] if best else None


def _looks_like_input_guess(err) -> bool:
    """True when a forward failure is most likely us guessing the input's DIMENSIONALITY
    wrong (not a real model bug) — e.g. unpacking `B, L, D = x.shape` from a 2-D synth
    tensor raises `ValueError: not enough values to unpack`."""
    msg = str(err).lower()
    return isinstance(err, ValueError) and ("unpack" in msg or "values to unpack" in msg)


# Conventional axis names by input rank (for the "abstract" shape view: relabel concrete
# dim VALUES with symbols). 2-D=(batch, features), 3-D=(batch, seq, model), 4-D=image.
_AXIS_NAMES = {2: ["B", "D"], 3: ["B", "L", "D"], 4: ["B", "C", "H", "W"]}


def _merge_dim_symbols(out: dict, args) -> None:
    """Map each forward-INPUT tensor's dim values -> a conventional symbol (B/L/D/C/H/W),
    first assignment wins. Only ranks 2-4 (typical inputs) anchor symbols; the webview then
    relabels every traced shape by value. A value that's genuinely two axes (square H=W)
    keeps its first symbol — a known, acceptable limit for a reading aid."""
    try:
        import torch
    except Exception:
        return
    tensors = [a for a in args if isinstance(a, torch.Tensor)]
    # Higher-rank inputs first: x(B,L,D) should claim L before a mask(B,L) is considered.
    for a in sorted(tensors, key=lambda t: -t.dim()):
        shape = list(a.shape)
        names = _AXIS_NAMES.get(len(shape))
        # A 2-D integer/bool tensor is token ids or a mask -> (B, L), not (B, features).
        if len(shape) == 2 and not a.dtype.is_floating_point:
            names = ["B", "L"]
        if not names:
            continue
        for i, v in enumerate(shape):
            out.setdefault(str(int(v)), names[i])


# Model-attribute names -> dim symbols, for dims that come from the MODEL rather than the
# input (the input anchors B/L/D; these add H, dh, V, … so qkv[2,16,3,4,32] can read as
# qkv[B, L, 3, H, dh]). Checked on every submodule plus one level of config objects.
_ATTR_SYMBOLS = [
    ({"n_heads", "num_heads", "n_head", "nhead", "heads"}, "H"),
    ({"head_dim", "d_head", "dim_head"}, "dh"),
    ({"d_model", "embed_dim", "embedding_dim", "hidden_size", "hidden_dim", "model_dim", "n_embd", "d_embedding"}, "D"),
    ({"vocab_size", "num_embeddings", "n_vocab"}, "V"),
    ({"num_classes", "n_classes", "out_classes"}, "C"),
]


def _merge_attr_symbols(out: dict, model) -> None:
    """Harvest dim symbols from int attributes with conventional names (n_heads -> H,
    head_dim -> dh, …) on the model, its submodules, and one level of config objects
    (self.cfg.d_model etc). setdefault — never overrides an input-derived symbol."""
    def scan(obj) -> None:
        for k, v in list(getattr(obj, "__dict__", {}).items()):
            if type(v) is int and v > 1:  # `type is int` also rejects bools
                for names, sym_ in _ATTR_SYMBOLS:
                    if k in names:
                        out.setdefault(str(v), sym_)

    try:
        mods = list(model.modules())
    except Exception:
        mods = [model]
    for m in mods:
        scan(m)
        for v in list(getattr(m, "__dict__", {}).values()):  # config objects (self.cfg.…)
            if hasattr(v, "__dict__") and not callable(v):
                try:
                    scan(v)
                except Exception:
                    pass


def _make_loader(root: str):
    """A `load("relpath")` callable for directives — loads a real data file into a tensor,
    resolving relative paths against the project root. Lazy import keeps loaders cheap."""
    def load(rel):
        from . import loaders

        return loaders.load_tensor(rel, root)

    return load


# Builtins/modules that only appear in a directive if it's trying to escape — never in a
# legitimate `randn(...)` / `Model(cfg)` / `load("x.npy")` expression.
_DIRECTIVE_DENY = {
    "__import__", "eval", "exec", "compile", "getattr", "setattr", "delattr",
    "globals", "locals", "vars", "os", "sys", "subprocess", "socket", "shutil",
    "importlib", "builtins", "breakpoint",
}


def _directive_is_safe(expr: str) -> bool:
    """A SPEED-BUMP (not a real sandbox — Python can't be fully sandboxed) against obviously
    dangerous directive expressions: import statements, dunder-attribute access (the classic
    `().__class__.__subclasses__()` escape), and calls into os/subprocess/eval/etc. This
    matters most in auto-trust mode, where an agent-proposed directive is eval'd without a
    human review. Real directives never trip it."""
    import ast

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return True  # let eval surface the real syntax error
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return False
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return False
        if isinstance(node, ast.Name) and node.id in _DIRECTIVE_DENY:
            return False
    return True


def _eval_expr(expr: str, g: dict):
    """Eval a directive expression in the module's namespace + torch conveniences + a
    `load()` helper for real-input tracing. Added to a COPY of g, never g itself, so the
    user's module namespace is never polluted. Rejects obviously-dangerous expressions
    (defense-in-depth for auto-trust mode). (Otherwise no more dangerous than exec'ing the
    user's own module, which we already do.)"""
    import torch

    if not _directive_is_safe(expr):
        raise ValueError(f"directive blocked for safety (imports / dunder access / os|subprocess): {expr!r}")
    ns = dict(g)
    ns.setdefault("torch", torch)
    for nm in ("randn", "randint", "zeros", "ones", "arange", "tensor", "full", "rand", "randn_like"):
        ns.setdefault(nm, getattr(torch, nm))
    ns.setdefault("load", _make_loader(_LOAD["root"]))
    return eval(expr, ns)


def _time_limit(seconds: float):
    """Watchdog so a runaway trace (infinite loop in a model's __init__/forward) can't hang
    the warm helper forever. In-process interruption is UNRELIABLE under sys.settrace (a
    signal- or trace-raised exception bypasses the traced frame's handlers), so we instead
    HARD-EXIT the helper from a daemon timer thread — which still runs while the main thread
    spins (the GIL time-slices). The host's HelperClient sees the dead process, fails the
    pending request, and respawns on the next call (auto-recovery). If the trace finishes in
    time the timer is cancelled and nothing happens."""
    import contextlib
    import os
    import sys
    import threading

    @contextlib.contextmanager
    def _cm():
        def _kill():
            sys.stderr.write(f"lens-helper: trace exceeded {seconds:g}s — killing the helper (host will respawn)\n")
            sys.stderr.flush()
            os._exit(2)

        timer = threading.Timer(seconds, _kill)
        timer.daemon = True
        timer.start()
        try:
            yield
        finally:
            timer.cancel()

    return _cm()


def _as_args(value):
    """A tuple directive means *multiple* positional args; anything else is one arg."""
    return tuple(value) if isinstance(value, tuple) else (value,)


def _required_pos(sig):
    return [
        p.name
        for p in sig.parameters.values()
        if p.default is p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]


# Synthesis dims, settable per trace via the Tracing settings page (batch B, sequence S).
_SYNTH = {"batch": 2, "seq": 16}
# Project root for load("relpath") directives. Module-level (traces are sequential, single
# RPC at a time) instead of stashed in the user module's globals — keeps that namespace clean.
_LOAD = {"root": ""}


def _synth_input(model, rank_hint=None):
    """Guess an input tensor from the model's first consuming layer — DTYPE-AWARE:
    Embedding -> long indices, Conv -> image, RNN/LSTM/GRU -> sequence, Linear -> vector.
    `rank_hint=3` (from static `B, L, D = x.shape`-style analysis) makes the Linear case
    synthesize (B, S, in_features) instead of (B, in_features) — transformer forwards
    need rank 3, and Linear accepts any leading dims."""
    import torch
    import torch.nn as nn

    b, s = _SYNTH["batch"], _SYNTH["seq"]
    try:
        mods = list(model.modules())
    except Exception:
        mods = []
    for m in mods:
        if isinstance(m, nn.Embedding):
            return torch.randint(0, m.num_embeddings, (b, s)), f"randint(0, {m.num_embeddings}, ({b}, {s}))"
        if isinstance(m, nn.Linear):
            if rank_hint == 3:
                return torch.randn(b, s, m.in_features), f"randn({b}, {s}, {m.in_features})"
            return torch.randn(b, m.in_features), f"randn({b}, {m.in_features})"
        if isinstance(m, nn.Conv2d):
            return torch.randn(b, m.in_channels, 32, 32), f"randn({b}, {m.in_channels}, 32, 32)"
        if isinstance(m, nn.Conv1d):
            return torch.randn(b, m.in_channels, 32), f"randn({b}, {m.in_channels}, 32)"
        if isinstance(m, nn.Conv3d):
            return torch.randn(b, m.in_channels, 16, 16, 16), f"randn({b}, {m.in_channels}, 16, 16, 16)"
        if isinstance(m, (nn.RNN, nn.LSTM, nn.GRU)):
            if getattr(m, "batch_first", False):
                return torch.randn(b, s, m.input_size), f"randn({b}, {s}, {m.input_size})"
            return torch.randn(s, b, m.input_size), f"randn({s}, {b}, {m.input_size})"
    if rank_hint == 3:
        return torch.randn(b, s, 8), f"randn({b}, {s}, 8) [guess]"
    return torch.randn(b, 8), f"randn({b}, 8) [guess]"


def _set_synth(batch, seq):
    try:
        _SYNTH["batch"] = max(1, int(batch))
        _SYNTH["seq"] = max(1, int(seq))
    except (TypeError, ValueError):
        pass


def _resolve_args(inst, method, method_line, directives, g, rank_hint=None):
    """(args_tuple, inside_parens_str). Honors `# fusion: input =`, else auto-synthesizes
    one tensor (rank_hint steers the synth's dimensionality — see _synth_input).
    Raises _CannotInvoke for multi-arg with no directive, or a bad directive."""
    import inspect

    expr = _pick_directive(directives, "input", method_line)
    if expr is not None:
        try:
            val = _eval_expr(expr, g)
        except Exception as e:
            raise _CannotInvoke(f"input directive failed: {expr!r}: {type(e).__name__}: {e}")
        return _as_args(val), expr
    name = getattr(method, "__name__", "fn")
    required = _required_pos(inspect.signature(method))
    if len(required) == 0:
        return (), ""
    if len(required) > 1:
        raise _CannotInvoke(f"{name}() needs {len(required)} args {required} — add  # fusion: input = (...)")
    if inst is None:
        raise _CannotInvoke(f"{name}() needs an arg — add  # fusion: input = ...")
    x, ishape = _synth_input(inst, rank_hint)
    return (x,), ishape


def _fn_node_at(tree, line):
    """The FunctionDef at this 1-based def line, or None."""
    import ast

    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.lineno == line:
            return n
    return None


def _rank_hints(tree):
    """{class_name: exact rank of forward's first tensor param} for every class in the
    file whose forward statically pins its input rank (B, L, D = x.shape / permute / …)."""
    import ast

    from .structure import shape_reqs

    hints = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and sub.name == "forward":
                    for r in shape_reqs(sub, skip_first=True):
                        if r["kind"] == "exact":
                            hints[node.name] = r["rank"]
                        break  # only the first param anchors the input
    return hints


def _rank_hint_for(inst, own_node, hints):
    """Input-rank hint for synthesizing this method's input: the method's OWN static pin
    first; else the first SUBMODULE whose class (defined in this file) pins its forward —
    e.g. TransformerEncoder.forward(x) just loops blocks, but its MultiHeadSelfAttention
    does `B, L, D = x.shape`, so the encoder's input must be rank 3 too."""
    if own_node is not None:
        from .structure import shape_reqs

        for r in shape_reqs(own_node, skip_first=True):
            if r["kind"] == "exact":
                return r["rank"]
            break
    if inst is not None:
        try:
            mods = list(inst.modules())[1:]
        except Exception:
            mods = []
        for m in mods:
            h = (hints or {}).get(type(m).__name__)
            if h:
                return h
    return None


def _instantiate(g, cls, cls_name, class_line, directives):
    """(inst, construct_note). Honors `# fusion: model =`, else zero-arg cls().
    Raises _CannotInvoke if it needs ctor args (and no directive) or a directive fails."""
    import inspect

    expr = _pick_directive(directives, "model", class_line)
    if expr is not None:
        try:
            return _eval_expr(expr, g), expr
        except Exception as e:
            raise _CannotInvoke(f"model directive failed: {expr!r}: {type(e).__name__}: {e}")
    try:
        req = _required_pos(inspect.signature(cls))
    except (ValueError, TypeError):
        req = []
    if req:
        raise _CannotInvoke(f"{cls_name}() needs constructor args {req} — add  # fusion: model = {cls_name}(...)")
    return cls(), f"{cls_name}()"


def _seeded_call(callable_, model=None):
    """Wrap a call: deterministic seed, eval() the model, run under no_grad. Returns a
    thunk for trace_callable; seed/eval run just before the (traced) call. If torch
    isn't importable (a non-torch file), run the callable as-is."""
    try:
        import torch
    except Exception:
        return callable_

    def go():
        torch.manual_seed(0)
        if model is not None:
            try:
                model.eval()
            except Exception:
                pass
        with torch.no_grad():
            return callable_()

    return go


def _make_invocation(g, name, cls_name, method_line, class_line, directives, own_node=None, hints=None):
    """Return (thunk, note, inst, args) ready for trace_callable, or (None, reason, None, ())
    if we can't auto-call it. `note` is the exact call used (provenance), shown in the
    cockpit; `inst`/`args` feed the abstract view's dim-symbol harvest; own_node/hints
    feed the rank-aware input synth."""
    import inspect

    if cls_name is None:  # top-level function
        fn = g.get(name)
        if not callable(fn):
            return None, f"{name}: not found or not callable", None, ()
        try:
            args, ishape = _resolve_args(None, fn, method_line, directives, g)
        except _CannotInvoke as e:
            return None, str(e), None, ()
        return _seeded_call(lambda f=fn, a=args: f(*a)), f"{name}({ishape})", None, args

    cls = g.get(cls_name)
    if not inspect.isclass(cls):
        return None, f"class {cls_name} not found", None, ()

    if name == "__init__":  # trace construction itself (build check) — never a synth tensor
        expr = _pick_directive(directives, "model", class_line)
        cnote = expr if expr else f"{cls_name}()"
        build = (lambda: _eval_expr(expr, g)) if expr else cls
        return _seeded_call(build), cnote, None, ()

    try:
        inst, cnote = _instantiate(g, cls, cls_name, class_line, directives)
    except _CannotInvoke as e:
        return None, str(e), None, ()
    except Exception as e:
        return None, f"can't construct {cls_name}(): {type(e).__name__}: {e}", None, ()

    method = getattr(inst, name, None)
    if not callable(method):
        return None, f"{cls_name} has no method {name}", None, ()
    try:
        rank_hint = _rank_hint_for(inst, own_node, hints)
        args, ishape = _resolve_args(inst, method, method_line, directives, g, rank_hint)
    except _CannotInvoke as e:
        return None, str(e), None, ()
    return _seeded_call(lambda m=method, a=args: m(*a), model=inst), f"{cnote}.{name}({ishape})", inst, args


def trace_function(path: str, name: str, line: int, batch: int = 2, seq: int = 16, project_root: str = ""):
    """Call ONE function/method directly (no __main__, no debugger) and trace it.
    Auto-synthesizes inputs (or honors a `# fusion:` directive; `load("relpath")` in a
    directive loads real data relative to project_root).
    Returns {"records", "error", "crashLine", "note", "ops", "dims"} — same op
    annotations and dim-symbol map as trace_module, so a per-function ▶ trace gets
    matmul/reshape notes and the abstract view too."""
    import ast
    import contextlib
    import io
    import os

    _set_synth(batch, seq)

    path = os.path.abspath(path)
    src = open(path).read()
    import sys
    import types
    mod = types.ModuleType("fusion_traced")
    mod.__file__ = path
    mod.__dict__.update({"__name__": "fusion_traced", "__file__": path})
    sys.modules["fusion_traced"] = mod
    g = mod.__dict__
    _LOAD["root"] = project_root or os.path.dirname(path)  # where load("rel") directives resolve from
    sink = io.StringIO()
    try:
        try:
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                exec(compile(src, path, "exec"), g)  # load defs (and any top-level code)
        except Exception as e:
            return {"records": {}, "error": f"module load failed: {type(e).__name__}: {e}",
                    "crashLine": None, "note": "", "ops": {}, "dims": {}}

        directives = _parse_directives(src)
        tree = ast.parse(src, path)
        cls_name, class_line, method_line = _locate(tree, name, line)
        invoke, note, inst, args = _make_invocation(
            g, name, cls_name, method_line, class_line, directives,
            own_node=_fn_node_at(tree, method_line), hints=_rank_hints(tree))
        if invoke is None:
            # not a crash — we just couldn't auto-call it; the note explains why
            return {"records": {}, "error": None, "crashLine": None, "note": note, "ops": {}, "dims": {}}

        records, err, crash = trace_callable(invoke, path)
        dim_syms: Dict[str, str] = {}
        attr_syms: Dict[str, str] = {}
        _merge_dim_symbols(dim_syms, args or ())
        if inst is not None:
            _merge_attr_symbols(attr_syms, inst)
        return {"records": records, "error": (f"{type(err).__name__}: {err}" if err else None),
                "crashLine": crash, "note": note, "ops": _op_notes(tree, records),
                "dims": {**attr_syms, **dim_syms}}
    finally:
        # keep the module registered through construction + the traced call (registry
        # decorators / get_type_hints look up sys.modules[__module__]), then clean up.
        sys.modules.pop("fusion_traced", None)


def trace_module(path: str, batch: int = 2, seq: int = 16, project_root: str = ""):
    """Comprehensively trace a file WITHOUT needing a __main__.

    For every model class and zero-arg function DEFINED in the file:
      • construct the model under the tracer  -> a BUILD-time shape check (__init__),
      • call its forward() with an auto-synthesized input (consistent batch=2, so the
        shapes agree with the per-function "trace this function"), honoring `# fusion:`.
    The file's own __main__ is deliberately NOT run. Returns a dict:
      {"records": {line: {var: {...}}},
       "problems": [{"line","message"}],
       "notes":    [{"label","line","note"}]}   # note = exact call used (provenance)
    """
    import ast
    import contextlib
    import inspect
    import io
    import os

    _set_synth(batch, seq)
    path = os.path.abspath(path)
    src = open(path).read()
    import sys
    import types
    mod = types.ModuleType("fusion_traced")
    mod.__file__ = path
    mod.__dict__.update({"__name__": "fusion_traced", "__file__": path})
    sys.modules["fusion_traced"] = mod
    g = mod.__dict__
    _LOAD["root"] = project_root or os.path.dirname(path)  # where load("rel") directives resolve from
    sink = io.StringIO()
    try:
        try:
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                exec(compile(src, path, "exec"), g)
        except Exception as e:
            return {"records": {}, "problems": [{"line": 1, "message": f"module load failed: {type(e).__name__}: {e}"}], "notes": [], "ops": {}, "dims": {}}

        directives = _parse_directives(src)
        tree = ast.parse(src, path)
        hints = _rank_hints(tree)  # class -> statically pinned forward input rank
        records: Dict[int, Dict[str, Any]] = {}
        problems: list = []
        notes: list = []
        dim_syms: Dict[str, str] = {}  # input-derived: concrete dim value -> B/L/D… (wins)
        attr_syms: Dict[str, str] = {}  # model-derived: n_heads -> H, head_dim -> dh, …

        def merge(srcrec) -> None:
            for ln, vars_ in srcrec.items():
                records.setdefault(ln, {}).update(vars_)

        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                cls = g.get(node.name)
                if not inspect.isclass(cls) or not _is_model(cls):
                    continue
                expr = _pick_directive(directives, "model", node.lineno)
                if expr is None:
                    try:
                        if _required_pos(inspect.signature(cls)):
                            notes.append({"label": node.name, "line": node.lineno,
                                          "note": f"{node.name}() needs constructor args — add  # fusion: model = {node.name}(...)"})
                            continue
                    except (ValueError, TypeError):
                        pass
                # BUILD CHECK: construct under the tracer (catches shape errors in __init__).
                holder: list = []
                build = (lambda e=expr, h=holder: h.append(_eval_expr(e, g))) if expr else (lambda c=cls, h=holder: h.append(c()))
                recs, err, crash = trace_callable(_seeded_call(build), path)
                merge(recs)
                cnote = expr if expr else f"{node.name}()"
                if err is not None:
                    problems.append({"line": crash or node.lineno, "message": f"build {node.name}(): {type(err).__name__}: {err}"})
                    continue
                notes.append({"label": node.name, "line": node.lineno, "note": f"{cnote}  [build ok]"})
                inst = holder[0] if holder else None
                if inst is not None:
                    _merge_attr_symbols(attr_syms, inst)
                method = getattr(inst, "forward", None) if inst is not None else None
                if not callable(method):
                    continue
                fwd_line = _method_line(node, "forward")
                try:
                    args, ishape, _fwd_node = _resolve_forward(inst, method, node, fwd_line, directives, hints, g)
                except _CannotInvoke as e:
                    notes.append({"label": f"{node.name}.forward", "line": fwd_line, "note": str(e)})
                    continue
                recs2, err2, crash2 = trace_callable(_seeded_call(lambda m=method, a=args: m(*a), model=inst), path)
                merge(recs2)
                if err2 is not None:
                    # A failure that's just us guessing the input's DIMENSIONALITY wrong (e.g.
                    # `B, L, D = x.shape` on a 2-D synth tensor) isn't a model bug — make it a
                    # soft note. A genuine internal mismatch (matmul, etc.) stays a problem.
                    had_input = _pick_directive(directives, "input", fwd_line) is not None
                    if not had_input and _looks_like_input_guess(err2):
                        notes.append({"label": f"{node.name}.forward", "line": fwd_line,
                                      "note": f"couldn't auto-trace forward ({type(err2).__name__}: {err2}) — use ✦ ask to add an input"})
                    else:
                        problems.append({"line": crash2 or fwd_line, "message": f"{node.name}.forward: {type(err2).__name__}: {err2}"})
                else:
                    _merge_dim_symbols(dim_syms, args)
                    notes.append({"label": f"{node.name}.forward", "line": fwd_line, "note": f"{cnote}.forward({ishape})"})

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fn = g.get(node.name)
                if not callable(fn):
                    continue
                try:
                    args, ishape = _resolve_args(None, fn, node.lineno, directives, g)
                except _CannotInvoke:
                    continue  # needs args & no directive -> skip silently at file level
                recs, err, crash = trace_callable(_seeded_call(lambda f=fn, a=args: f(*a)), path)
                merge(recs)
                if err is not None:
                    problems.append({"line": crash or node.lineno, "message": f"{node.name}(): {type(err).__name__}: {err}"})
                else:
                    _merge_dim_symbols(dim_syms, args)
                    notes.append({"label": node.name, "line": node.lineno, "note": f"{node.name}({ishape})"})

        # Input-derived symbols (B/L/D anchors) win over model-attribute ones on collision.
        return {"records": records, "problems": problems, "notes": notes,
                "ops": _op_notes(tree, records), "dims": {**attr_syms, **dim_syms}}
    finally:
        # keep the module registered through construction + every traced forward
        # (registry decorators look up sys.modules[__module__]), then clean up.
        sys.modules.pop("fusion_traced", None)


def _exec_module(path: str, project_root: str):
    """Exec a file into a fresh `fusion_traced` module (registered in sys.modules so registry
    decorators resolve). Returns (g, src, load_error). CALLER must pop sys.modules in a
    finally. Sets _LOAD['root'] for load()-based real-input directives."""
    import contextlib
    import io
    import os
    import sys
    import types

    path = os.path.abspath(path)
    src = open(path).read()
    mod = types.ModuleType("fusion_traced")
    mod.__file__ = path
    mod.__dict__.update({"__name__": "fusion_traced", "__file__": path})
    sys.modules["fusion_traced"] = mod
    g = mod.__dict__
    _LOAD["root"] = project_root or os.path.dirname(path)
    sink = io.StringIO()
    try:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            exec(compile(src, path, "exec"), g)
        return g, src, None
    except Exception as e:
        return g, src, e


def _resolve_forward(inst, method, node, fwd_line, directives, hints, g):
    """(args, ishape, fwd_node) for a model's forward, honoring `# fusion: input =` and the
    rank hint. Raises _CannotInvoke if the input can't be resolved (multi-arg + no directive,
    bad directive, …). Shared by trace_module and _build_class — one place owns the
    forward-input resolution rule. The caller supplies fwd_line so it can still attribute a
    note on failure without recomputing it."""
    import ast

    fwd_node = next((s for s in node.body if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)) and s.name == "forward"), None)
    rank_hint = _rank_hint_for(inst, fwd_node, hints)
    args, ishape = _resolve_args(inst, method, fwd_line, directives, g, rank_hint)
    return args, ishape, fwd_node


def _build_class(node, g, directives, hints):
    """Build a model ClassDef instance + resolve its forward input (NO tracing). Returns
    (inst, method, args, ishape, cnote, fwd_node, fwd_line) or raises _CannotInvoke when the
    class can't be auto-built (ctor args + no `# fusion: model =`, no forward, multi-arg
    forward + no `# fusion: input =`, …). Reused by module_summary / paper_module so
    trace_module's own build path stays untouched."""
    import inspect

    cls = g.get(node.name)
    if not inspect.isclass(cls) or not _is_model(cls):
        raise _CannotInvoke(f"{node.name} is not a traceable model")
    expr = _pick_directive(directives, "model", node.lineno)
    if expr is None:
        try:
            if _required_pos(inspect.signature(cls)):
                raise _CannotInvoke(f"{node.name}() needs constructor args — add  # fusion: model = {node.name}(...)")
        except (ValueError, TypeError):
            pass
    inst = _seeded_call((lambda e=expr: _eval_expr(e, g)) if expr else cls)()
    cnote = expr if expr else f"{node.name}()"
    method = getattr(inst, "forward", None)
    if not callable(method):
        raise _CannotInvoke(f"{node.name} has no forward()")
    fwd_line = _method_line(node, "forward")
    args, ishape, fwd_node = _resolve_forward(inst, method, node, fwd_line, directives, hints, g)
    return inst, method, args, ishape, cnote, fwd_node, fwd_line


def _first_tensor_shape(out):
    """Shape of the first tensor found in a (possibly nested tuple/list/dict) output, else None."""
    import torch

    if isinstance(out, torch.Tensor):
        return [int(d) for d in out.shape]
    if isinstance(out, (tuple, list)):
        for o in out:
            s = _first_tensor_shape(o)
            if s is not None:
                return s
    if isinstance(out, dict):
        for o in out.values():
            s = _first_tensor_shape(o)
            if s is not None:
                return s
    return None


def _collect_summary(inst, args):
    """Run inst(*args) with a forward hook on every submodule -> executed-order rows
    (name, class, depth, output shape) + per-module param counts (recurse=False, so child
    params aren't double-counted). Returns (rows, total, trainable, bytes, error)."""
    import torch

    rows = []
    seen = set()
    handles = []

    def mk(name, m):
        def hook(_m, _i, out):
            if name in seen:
                return
            seen.add(name)
            rows.append({"name": name, "cls": type(_m).__name__, "depth": name.count(".") + 1,
                         "outShape": _first_tensor_shape(out)})
        return hook

    for name, m in inst.named_modules():
        if name:
            handles.append(m.register_forward_hook(mk(name, m)))
    err = None
    try:
        with torch.no_grad():
            try:
                inst.eval()
            except Exception:
                pass
            torch.manual_seed(0)
            inst(*args)
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    finally:
        for h in handles:
            h.remove()

    if not rows:  # pure-functional forward or crash-before-any-hook -> definition order, no shapes
        for name, m in inst.named_modules():
            if name:
                rows.append({"name": name, "cls": type(m).__name__, "depth": name.count(".") + 1, "outShape": None})

    by_name = dict(inst.named_modules())
    total = sum(p.numel() for p in inst.parameters())  # parameters() already dedups tied weights
    seen_ids: set = set()
    for r in rows:
        m = by_name.get(r["name"])
        # Dedup tied/shared params across modules so per-row sums reconcile with the total
        # (a weight-tied head + embedding must not each claim the same numel).
        prm = [p for p in (m.parameters(recurse=False) if m is not None else []) if id(p) not in seen_ids]
        seen_ids.update(id(p) for p in prm)
        r["params"] = int(sum(p.numel() for p in prm))
        r["trainable"] = int(sum(p.numel() for p in prm if p.requires_grad))
        r["pctParams"] = round(100.0 * r["params"] / total, 2) if total else 0.0
    trainable = sum(p.numel() for p in inst.parameters() if p.requires_grad)
    pbytes = sum(p.numel() * p.element_size() for p in inst.parameters())
    return rows, int(total), int(trainable), int(pbytes), err


def module_summary(path: str, batch: int = 2, seq: int = 16, project_root: str = ""):
    """torchinfo-style summary of the first auto-buildable model in the file: per-module
    output shape + param counts, totals, and the abstract dim-symbol map. Lazy (only the
    Summary tab requests it). Never raises — unbuildable / crashed forward returns an error
    string with whatever it could compute."""
    import ast
    import os
    import sys

    _set_synth(batch, seq)
    empty = {"path": os.path.abspath(path), "target": "", "rows": [], "totalParams": 0,
             "trainableParams": 0, "paramBytes": 0, "dims": {}}
    g, src, load_err = _exec_module(path, project_root)
    try:
        if load_err is not None:
            return {**empty, "error": f"module load failed: {type(load_err).__name__}: {load_err}"}
        directives = _parse_directives(src)
        tree = ast.parse(src, path)
        hints = _rank_hints(tree)
        classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
        # Summarize the COMPOSED top-level model, not a sub-block. Statically find the
        # "root" classes — those NOT instantiated by another class in the file — and build
        # only those (cheap: avoids constructing every sub-block). Fall back to all classes
        # if no root is buildable. Among the built, pick the one with the most parameters.
        names = {n.name for n in classes}
        instantiated = {
            sub.func.id
            for n in classes
            for sub in ast.walk(n)
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) and sub.func.id in names and sub.func.id != n.name
        }
        roots = [n for n in classes if n.name not in instantiated]

        def build_all(nodes):
            out, err = [], None
            for node in nodes:
                try:
                    built = _build_class(node, g, directives, hints)
                    out.append((sum(p.numel() for p in built[0].parameters()), built))
                except _CannotInvoke as e:
                    err = str(e)
                except Exception as e:
                    err = f"{type(e).__name__}: {e}"
            return out, err

        candidates, build_err = build_all(roots)
        if not candidates:  # roots needed ctor args / failed -> try the rest
            candidates, err2 = build_all([n for n in classes if n not in roots])
            build_err = err2 or build_err  # keep the roots' reason if the fallback had none
        if not candidates:
            return {**empty, "error": build_err or "no auto-buildable model class found"}
        candidates.sort(key=lambda c: c[0])
        inst, _method, args, ishape, cnote, _fwd_node, _fwd_line = candidates[-1][1]
        attr_syms, dim_syms = {}, {}
        _merge_attr_symbols(attr_syms, inst)
        _merge_dim_symbols(dim_syms, args)
        rows, total, trainable, pbytes, fwd_err = _collect_summary(inst, args)
        return {"path": os.path.abspath(path), "target": f"{cnote}.forward({ishape})", "rows": rows,
                "totalParams": total, "trainableParams": trainable, "paramBytes": pbytes,
                "dims": {**attr_syms, **dim_syms}, "error": fwd_err}
    finally:
        sys.modules.pop("fusion_traced", None)


def _iter_forward_stmts(node):
    """Assign / AugAssign / AnnAssign / Return statements of a forward, descending into
    control flow (if/for/while/with/try) but NOT into nested defs/lambdas/comprehensions —
    so a residual `x += attn(x)` is included and an inner helper's statements are not."""
    import ast

    STOP = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
    STMT = (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Return)
    for child in ast.iter_child_nodes(node):
        if isinstance(child, STOP):
            continue
        if isinstance(child, STMT):
            yield child
        else:
            yield from _iter_forward_stmts(child)


def _paper_step(stmt, records, ops):
    """One forward statement -> {line, lhs, op?, shapes, changed}, or None when the line
    has neither a recorded shape nor an op note."""
    import ast

    ln = stmt.lineno
    if isinstance(stmt, ast.Return):
        names, lhs = ["return"], "return"
    elif isinstance(stmt, (ast.AugAssign, ast.AnnAssign)):  # x += attn(x) / x: T = v
        names = [stmt.target.id] if isinstance(stmt.target, ast.Name) else []
        if not names:
            return None
        lhs = names[0]
    else:  # ast.Assign
        tgt = stmt.targets[0] if stmt.targets else None
        names = ([tgt.id] if isinstance(tgt, ast.Name)
                 else [e.id for e in tgt.elts if isinstance(e, ast.Name)] if isinstance(tgt, ast.Tuple) else [])
        if not names:
            return None
        lhs = ", ".join(names)
    rec = records.get(ln, {})
    shapes = [{"varName": n, "shape": rec[n]["shape"], "dtype": rec[n]["dtype"], "changed": rec[n].get("changed", False)}
              for n in names if n in rec]
    op = ops.get(ln)
    if not shapes and not op:
        return None
    return {"line": ln, "lhs": lhs, "op": op, "shapes": shapes, "changed": any(s["changed"] for s in shapes)}


def paper_module(path: str, batch: int = 2, seq: int = 16, project_root: str = ""):
    """Paper-reading view: the traced forward pass as an ordered, per-module sequence of
    named tensor ops with concrete shapes + the B/L/D dim-symbol map (the webview relabels
    via the shared abstract toggle, reusing fmtShape/fmtOp). Reuses _op_notes + the
    build/trace path; leaves trace_module untouched. Lazy (only the Paper tab requests it)."""
    import ast
    import os
    import sys

    _set_synth(batch, seq)
    g, src, load_err = _exec_module(path, project_root)
    try:
        if load_err is not None:
            return {"path": os.path.abspath(path), "sections": [], "dims": {},
                    "problems": [{"line": 1, "message": f"module load failed: {type(load_err).__name__}: {load_err}"}]}
        directives = _parse_directives(src)
        tree = ast.parse(src, path)
        hints = _rank_hints(tree)
        sections, problems, attr_syms, dim_syms = [], [], {}, {}
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            try:
                inst, method, args, ishape, cnote, fwd_node, fwd_line = _build_class(node, g, directives, hints)
            except _CannotInvoke:
                continue
            except Exception as e:
                problems.append({"line": node.lineno, "message": f"build {node.name}(): {type(e).__name__}: {e}"})
                continue
            if fwd_node is None:
                continue
            recs, err, crash = trace_callable(_seeded_call(lambda m=method, a=args: m(*a), model=inst), path)
            if err is not None:
                problems.append({"line": crash or fwd_line, "message": f"{node.name}.forward: {type(err).__name__}: {err}"})
                continue
            _merge_attr_symbols(attr_syms, inst)
            _merge_dim_symbols(dim_syms, args)
            ops = _op_notes(tree, recs)
            stmts = sorted(_iter_forward_stmts(fwd_node), key=lambda s: (s.lineno, getattr(s, "col_offset", 0)))
            steps = [st for st in (_paper_step(s, recs, ops) for s in stmts) if st]
            sections.append({"module": f"{node.name}.forward", "forwardNote": f"{cnote}.forward({ishape})",
                             "params": [a.arg for a in fwd_node.args.args if a.arg not in ("self", "cls")],
                             "startLine": fwd_line, "steps": steps})
        return {"path": os.path.abspath(path), "sections": sections, "dims": {**attr_syms, **dim_syms}, "problems": problems}
    finally:
        sys.modules.pop("fusion_traced", None)


def _is_model(cls) -> bool:
    """True if cls subclasses torch.nn.Module — checked via MRO names so we never import torch."""
    try:
        for base in cls.__mro__:
            if base.__name__ == "Module" and base.__module__.split(".")[0] == "torch":
                return True
    except Exception:
        pass
    return False


def _method_line(class_node, method_name: str) -> int:
    import ast

    for sub in class_node.body:
        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and sub.name == method_name:
            return sub.lineno
    return class_node.lineno


# --- per-line op annotation: parse the COMMON PyTorch ops from the AST and pair them
# --- with the shapes captured at trace time. One parser family per op category.
_RESHAPE_METHODS = {"reshape", "view", "view_as", "reshape_as", "flatten", "ravel", "squeeze",
                    "unsqueeze", "permute", "transpose", "t", "expand", "expand_as", "repeat",
                    "tile", "movedim", "moveaxis", "swapaxes", "swapdims", "contiguous", "roll"}
_REDUCE_METHODS = {"sum", "mean", "prod", "amax", "amin", "max", "min", "argmax", "argmin",
                   "norm", "std", "var", "logsumexp", "softmax", "log_softmax", "cumsum",
                   "cumprod", "any", "all"}
_SPLIT_METHODS = {"unbind", "chunk", "split", "tensor_split", "hsplit", "vsplit"}
_COMBINE_FUNCS = {"cat", "concat", "concatenate", "stack", "hstack", "vstack", "dstack"}
_MATMUL_FUNCS = {"matmul", "mm", "bmm", "mv"}
_NO_ARG_RENDER = _RESHAPE_METHODS - {"permute", "transpose", "squeeze", "unsqueeze", "movedim",
                                     "moveaxis", "swapaxes", "swapdims", "roll"}  # reshape/view args are often symbolic — the output shape says it


def _op_notes(tree, records):
    """Annotate common tensor ops per source line, pairing the AST op with the operand &
    result shapes captured at trace time. Covers: @ /matmul/mm/bmm/einsum/sdpa, broadcasting
    arithmetic, reshape/view/permute/transpose…, unbind/chunk/split, cat/stack, and dim
    reductions (softmax/sum/mean/…). Returns {lineno: note} with shapes as [2, 16, 128]
    groups (the webview relabels those in abstract mode). Unknown shapes render as ?."""
    import ast

    def shape_at(nm, ln, before=False):
        """Shape of local `nm` as recorded ON line ln (locals live after it ran), or the
        nearest earlier line. before=True skips ln itself — for `x = x.op()` the receiver's
        pre-assign shape only exists on an earlier line."""
        if nm is None:
            return None
        if not before:
            v = (records.get(ln) or {}).get(nm)
            if v:
                return v["shape"]
        best = None
        for l2, vars_ in records.items():
            if l2 < ln and nm in vars_ and (best is None or l2 > best):
                best = l2
        return records[best][nm]["shape"] if best is not None else None

    fmt = lambda s: "[" + ", ".join(str(d) for d in s) + "]" if s else "?"

    def int_arg(a):
        """Render an int-literal argument (incl. negatives); None if it isn't one."""
        if isinstance(a, ast.Constant) and type(a.value) is int:
            return str(a.value)
        if isinstance(a, ast.UnaryOp) and isinstance(a.op, ast.USub) and isinstance(a.operand, ast.Constant):
            return f"-{a.operand.value}"
        return None

    def render_call(name, call):
        """`permute(0, 2, 1, 3)` / `sum(dim=-1)` — int args + dim kwarg only; reshape-like
        ops render bare (their args are usually symbolic; the → shape carries the info)."""
        if name in _NO_ARG_RENDER:
            return name
        parts = [s for s in (int_arg(a) for a in call.args) if s is not None]
        for kw in call.keywords:
            s = int_arg(kw.value)
            if kw.arg == "dim" and s is not None:
                parts.append(f"dim={s}")
        return f"{name}({', '.join(parts)})" if parts else name

    def name_of(node):
        return node.id if isinstance(node, ast.Name) else None

    def method_chain(call):
        """Peel `recv.op1(...).op2(...)` into (receiver_node, [(name, call), …] source-order),
        following only RECOGNIZED tensor methods. Returns (None, []) if the outer call isn't one."""
        known = _RESHAPE_METHODS | _REDUCE_METHODS | _SPLIT_METHODS
        ops, cur = [], call
        while (isinstance(cur, ast.Call) and isinstance(cur.func, ast.Attribute)
               and cur.func.attr in known):
            ops.append((cur.func.attr, cur))
            cur = cur.func.value
        ops.reverse()
        return cur, ops

    sym = {ast.Mult: "*", ast.Add: "+", ast.Sub: "-", ast.Div: "/"}
    out = {}

    def emit(ln, note):
        out[ln] = f"{out[ln]}; {note}" if ln in out else note

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        ln = node.lineno
        tgt = node.targets[0]
        tgt_names = ([tgt.id] if isinstance(tgt, ast.Name)
                     else [e.id for e in tgt.elts if isinstance(e, ast.Name)] if isinstance(tgt, ast.Tuple)
                     else [])
        tshapes = [shape_at(n, ln) for n in tgt_names]
        tshape = fmt(tshapes[0]) if len(tshapes) == 1 and tshapes[0] else None
        val = node.value

        # ---- a @ b ------------------------------------------------------------------
        if isinstance(val, ast.BinOp) and isinstance(val.op, ast.MatMult):
            sa = fmt(shape_at(name_of(val.left), ln, before=name_of(val.left) in tgt_names))
            sb = fmt(shape_at(name_of(val.right), ln, before=name_of(val.right) in tgt_names))
            if tshape or (sa != "?" and sb != "?"):
                emit(ln, f"matmul {sa} · {sb} → {tshape or '?'}")

        # ---- broadcasting arithmetic: a + b, a * b, … with DIFFERENT shapes ------------
        elif isinstance(val, ast.BinOp) and type(val.op) in sym:
            la, lb = shape_at(name_of(val.left), ln, before=name_of(val.left) in tgt_names), \
                     shape_at(name_of(val.right), ln, before=name_of(val.right) in tgt_names)
            if la and lb and la != lb:
                emit(ln, f"broadcast {fmt(la)} {sym[type(val.op)]} {fmt(lb)} → {tshape or '?'}")

        elif isinstance(val, ast.Call):
            attr = _call_attr(val.func)

            # ---- torch.matmul / mm / bmm / mv ----------------------------------------
            if attr in _MATMUL_FUNCS and len(val.args) >= 2:
                ss = [fmt(shape_at(name_of(a), ln)) for a in val.args[:2]]
                if tshape or "?" not in ss:
                    emit(ln, f"{attr} {ss[0]} · {ss[1]} → {tshape or '?'}")

            # ---- torch.einsum("eq", a, b, …) -----------------------------------------
            elif attr == "einsum" and val.args and isinstance(val.args[0], ast.Constant):
                eq = val.args[0].value
                ss = [fmt(shape_at(name_of(a), ln)) for a in val.args[1:]]
                emit(ln, f"einsum('{eq}') {' · '.join(ss)} → {tshape or '?'}")

            # ---- F.scaled_dot_product_attention(q, k, v, …) --------------------------
            elif attr == "scaled_dot_product_attention" and len(val.args) >= 3:
                ss = [fmt(shape_at(name_of(a), ln)) for a in val.args[:3]]
                emit(ln, f"sdpa {ss[0]} · {ss[1]} · {ss[2]} → {tshape or '?'}")

            # ---- torch.cat([a, b], dim=…) / torch.stack -------------------------------
            elif attr in _COMBINE_FUNCS and val.args:
                first = val.args[0]
                elts = first.elts if isinstance(first, (ast.List, ast.Tuple)) else []
                ss = [fmt(shape_at(name_of(e), ln)) for e in elts]
                if ss and (tshape or "?" not in ss):
                    emit(ln, f"{render_call(attr, val)} {' + '.join(ss)} → {tshape or '?'}")

            # ---- F.softmax(x, dim=…) and friends (torch-level reduce) -----------------
            elif attr in _REDUCE_METHODS and val.args and name_of(val.args[0]):
                sin = fmt(shape_at(name_of(val.args[0]), ln, before=name_of(val.args[0]) in tgt_names))
                if sin != "?" or tshape:
                    emit(ln, f"{render_call(attr, val)} {sin} → {tshape or '?'}")

            # ---- tensor-method chains: x.reshape(…), qkv.unbind(2), y.permute(…).reshape(…)
            else:
                recv, ops = method_chain(val)
                if ops:
                    rname = name_of(recv)
                    sin = fmt(shape_at(rname, ln, before=rname in tgt_names))
                    rendered = " ∘ ".join(render_call(nm, c) for nm, c in ops)
                    left = f"{rendered} {sin}" if sin != "?" else rendered  # no dangling "?" input
                    if ops[-1][0] in _SPLIT_METHODS and len(tgt_names) > 1:  # q, k, v = ….unbind(2)
                        shapes = [fmt(s) for s in tshapes]
                        same = len(set(shapes)) == 1 and shapes[0] != "?"
                        tout = f"{len(shapes)} × {shapes[0]}" if same else " | ".join(shapes)
                        emit(ln, f"{left} → {tout}")
                    elif sin != "?" or tshape:
                        emit(ln, f"{left} → {tshape or '?'}")
    return out


def _call_attr(func):
    import ast

    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _locate(tree, name, line):
    """(cls_name, class_line, method_line) for function `name` (optionally matching a
    specific def line). cls_name/class_line are None for a top-level function."""
    import ast

    for node in tree.body:  # top-level function?
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name and line in (0, node.lineno):
            return None, None, node.lineno
    for node in ast.walk(tree):  # method in a class
        if isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and sub.name == name and line in (0, sub.lineno):
                    return node.name, node.lineno, sub.lineno
    for node in ast.walk(tree):  # fallback: any function with that name
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return None, None, node.lineno
    return None, None, line or 0
