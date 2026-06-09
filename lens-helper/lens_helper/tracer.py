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
