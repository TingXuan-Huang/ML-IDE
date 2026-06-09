"""Trace core for the cockpit's Mode 1 (shape-annotated blocks).

Runs a callable under sys.settrace and captures every tensor local's shape per source
line in the target file. The `line` event fires BEFORE the line runs, so locals are
attributed to the PREVIOUS line (the one that just produced them). Duck-types tensors
so the helper need not import torch unless the user's code does.
"""
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


def trace_function(path: str, name: str, line: int):
    """Call ONE function/method directly (no __main__, no debugger) and trace it.
    Auto-synthesizes inputs. Returns (records, error_or_None, crash_line, note)."""
    import ast
    import contextlib
    import io
    import os

    path = os.path.abspath(path)
    src = open(path).read()
    g = {"__name__": "fusion_traced", "__file__": path}  # NOT __main__ -> the file's main block won't run
    sink = io.StringIO()
    try:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            exec(compile(src, path, "exec"), g)  # load defs (and any top-level code)
    except Exception as e:
        return {}, f"module load failed: {type(e).__name__}: {e}", None, ""

    cls_name = _enclosing_class(ast.parse(src, path), name, line)
    invoke, note = _make_invocation(g, name, cls_name)
    if invoke is None:
        # not a crash — we just couldn't auto-call it; the note explains why
        return {}, None, None, note

    records, err, crash = trace_callable(invoke, path)
    return records, (f"{type(err).__name__}: {err}" if err else None), crash, note


def trace_module(path: str):
    """Comprehensively trace a file WITHOUT needing a __main__.

    For every model class and zero-arg function DEFINED in the file:
      • construct the model under the tracer  -> a BUILD-time shape check (__init__),
      • call its forward() with an auto-synthesized input (consistent batch=2, so the
        shapes agree with the per-function "trace this function").
    The file's own __main__ is deliberately NOT run (no surprise training loops / side
    effects, and the batch stays consistent). Returns a dict:
      {"records": {line: {var: {...}}}, "problems": [{"line","message"}], "notes": [{"label","note"}]}
    """
    import ast
    import contextlib
    import inspect
    import io
    import os

    path = os.path.abspath(path)
    src = open(path).read()
    g = {"__name__": "fusion_traced", "__file__": path}  # __main__ block won't run
    sink = io.StringIO()
    try:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            exec(compile(src, path, "exec"), g)
    except Exception as e:
        return {"records": {}, "problems": [{"line": 1, "message": f"module load failed: {type(e).__name__}: {e}"}], "notes": []}

    tree = ast.parse(src, path)
    records: Dict[int, Dict[str, Any]] = {}
    problems: list = []
    notes: list = []

    def merge(srcrec) -> None:
        for ln, vars_ in srcrec.items():
            records.setdefault(ln, {}).update(vars_)

    def required_pos(sig) -> list:
        return [
            p.name
            for p in sig.parameters.values()
            if p.default is p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            cls = g.get(node.name)
            if not inspect.isclass(cls) or not _is_model(cls):
                continue
            # Skip models that need constructor args — we can't guess those (yet).
            try:
                if required_pos(inspect.signature(cls)):
                    notes.append({"label": node.name, "note": f"{node.name}(…) needs constructor args — skipped"})
                    continue
            except (ValueError, TypeError):
                pass
            # BUILD CHECK: construct under the tracer (catches shape errors in __init__).
            holder: list = []
            recs, err, crash = trace_callable(lambda c=cls, h=holder: h.append(c()), path)
            merge(recs)
            if err is not None:
                problems.append({"line": crash or node.lineno, "message": f"build {node.name}(): {type(err).__name__}: {err}"})
                continue
            notes.append({"label": f"{node.name}()", "note": f"{node.name}()  [build ok]"})
            inst = holder[0] if holder else None
            method = getattr(inst, "forward", None) if inst is not None else None
            if not callable(method):
                continue
            if len(required_pos(inspect.signature(method))) != 1:
                continue  # forward needs 0 or >1 args -> can't auto-synthesize a single tensor
            x, ishape = _synth_input(inst)
            recs2, err2, crash2 = trace_callable(lambda m=method, xx=x: m(xx), path)
            merge(recs2)
            if err2 is not None:
                problems.append({"line": crash2 or node.lineno, "message": f"{node.name}.forward: {type(err2).__name__}: {err2}"})
            else:
                notes.append({"label": f"{node.name}.forward", "note": f"{node.name}().forward({ishape})"})

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            fn = g.get(node.name)
            if not callable(fn) or required_pos(inspect.signature(fn)):
                continue  # needs args -> skip at file level
            recs, err, crash = trace_callable(lambda f=fn: f(), path)
            merge(recs)
            if err is not None:
                problems.append({"line": crash or node.lineno, "message": f"{node.name}(): {type(err).__name__}: {err}"})
            else:
                notes.append({"label": node.name, "note": f"{node.name}()"})

    return {"records": records, "problems": problems, "notes": notes}


def _is_model(cls) -> bool:
    """True if cls subclasses torch.nn.Module — checked via MRO names so we never import torch."""
    try:
        for base in cls.__mro__:
            if base.__name__ == "Module" and base.__module__.split(".")[0] == "torch":
                return True
    except Exception:
        pass
    return False


def _enclosing_class(tree, name, line):
    import ast

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and sub.name == name:
                    if line == 0 or sub.lineno == line:
                        return node.name
    return None


def _make_invocation(g, name, cls_name):
    import inspect

    if cls_name is None:  # top-level function
        fn = g.get(name)
        if not callable(fn):
            return None, f"{name}: not found or not callable"
        required = [
            p.name
            for p in inspect.signature(fn).parameters.values()
            if p.default is p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        if required:
            return None, f"{name}() needs args {required} — example-input support coming soon"
        return (lambda: fn()), f"called {name}()"

    cls = g.get(cls_name)
    if not inspect.isclass(cls):
        return None, f"class {cls_name} not found"
    try:
        inst = cls()
    except Exception as e:
        return None, f"can't construct {cls_name}() (needs constructor args): {e}"
    # __init__ is NOT a data path. Tracing it means tracing construction itself, so
    # re-run cls() under the tracer (this captures any tensors/buffers built in the
    # ctor) and never pass a synth tensor — it would land in a non-tensor ctor arg
    # (e.g. nn.Linear(d_in=<Tensor>, ...) -> TypeError).
    if name == "__init__":
        return (lambda: cls()), f"{cls_name}()"
    method = getattr(inst, name, None)
    if not callable(method):
        return None, f"{cls_name} has no method {name}"
    # Required positional params (self is already bound out of a bound method's sig).
    required = [
        p.name
        for p in inspect.signature(method).parameters.values()
        if p.default is p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    if not required:
        return (lambda: method()), f"{cls_name}().{name}()"
    if len(required) > 1:
        # We only auto-synthesize ONE tensor; >1 required arg can't be guessed safely.
        return None, f"{name}() needs {len(required)} args {required} — only single-tensor auto-input is supported"
    x, ishape = _synth_input(inst)
    return (lambda: method(x)), f"{cls_name}().{name}({ishape})"


def _synth_input(model):
    """Guess a tensor input from the model's first parametrized layer."""
    import torch
    import torch.nn as nn

    try:
        mods = list(model.modules())
    except Exception:
        mods = []
    for m in mods:
        if isinstance(m, nn.Linear):
            return torch.randn(2, m.in_features), f"randn(2, {m.in_features})"
        if isinstance(m, nn.Conv2d):
            return torch.randn(2, m.in_channels, 32, 32), f"randn(2, {m.in_channels}, 32, 32)"
        if isinstance(m, nn.Conv1d):
            return torch.randn(2, m.in_channels, 32), f"randn(2, {m.in_channels}, 32)"
    return torch.randn(2, 8), "randn(2, 8) [guess]"
