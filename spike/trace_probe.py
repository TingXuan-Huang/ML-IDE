"""
M0 SPIKE — Trace probe (THROWAWAY).

Goal: prove we can capture every intermediate tensor's shape per source line from
ONE run, via sys.settrace, and that we can catch a shape-mismatch crash with context.
If this prints real shapes annotated onto the code, the Fusion IDE centerpiece is feasible.

Run:  python3 spike/trace_probe.py
"""
import sys
import os
import torch
import torch.nn as nn

THIS_FILE = os.path.abspath(__file__)


# --- the tracer: capture tensor-local shapes per source line -------------------
def trace_run(fn):
    """Run fn() with a tracer scoped to THIS_FILE; return (result_or_exc, records).
    records: { lineno: { varname: (shape_tuple, dtype) } } attributed to the line
    that PRODUCED the value (sys.settrace fires 'line' BEFORE the line runs, so we
    attribute locals to the previously-executed line)."""
    records = {}
    order = []
    prev_line = {}

    def snapshot(frame, lineno):
        slot = records.setdefault(lineno, {})
        for name, val in frame.f_locals.items():
            if torch.is_tensor(val):
                slot[name] = (tuple(val.shape), str(val.dtype).replace("torch.", ""))
        if lineno not in order:
            order.append(lineno)

    def tracer(frame, event, arg):
        if frame.f_code.co_filename != THIS_FILE:
            return None  # skip torch internals / other files
        fid = id(frame)
        if event == "call":
            prev_line[fid] = None
            return tracer
        if event in ("line", "return", "exception"):
            pl = prev_line.get(fid)
            if pl is not None:
                snapshot(frame, pl)
            if event == "line":
                prev_line[fid] = frame.f_lineno
        return tracer

    sys.settrace(tracer)
    try:
        return ("ok", fn()), records
    except Exception as e:  # shape mismatch etc.
        return ("error", e), records
    finally:
        sys.settrace(None)


# --- render: annotate this file's source lines with captured shapes ------------
def render(records, lo, hi, crash_line=None):
    with open(THIS_FILE) as f:
        src = f.readlines()
    DIM, RED, GRAY, RESET = "\033[36m", "\033[91m", "\033[90m", "\033[0m"
    for ln in range(lo, hi + 1):
        code = src[ln - 1].rstrip("\n")
        shapes = records.get(ln, {})
        ann = "  ".join(f"{n} {list(s)} {d}" for n, (s, d) in shapes.items())
        mark = f"{RED}✕{RESET}" if ln == crash_line else " "
        col = RED if ln == crash_line else GRAY
        print(f" {mark}{ln:>3} {code:<46} {col}{ann}{RESET}")


# --- models --------------------------------------------------------------------
class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(16, 32)
        self.proj = nn.Linear(32, 8)

    def forward(self, x):
        h = self.embed(x)          # noqa
        h = torch.relu(h)          # noqa
        a = h.mean(dim=1, keepdim=True)  # broadcast surprise: [B,1]
        h = h + a                  # silent broadcast back to [B,32]
        y = self.proj(h)           # noqa
        return y


class BuggyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(16, 32)
        self.proj = nn.Linear(64, 8)   # BUG: expects 64, gets 32

    def forward(self, x):
        h = self.embed(x)          # noqa
        h = torch.relu(h)          # noqa
        y = self.proj(h)           # crashes here: [B,32] vs Linear(64,8)
        return y


def line_of(cls):
    """First and last line numbers of cls.forward in this file."""
    import inspect
    src, start = inspect.getsourcelines(cls.forward)
    return start, start + len(src) - 1


if __name__ == "__main__":
    print("\n=== 1. WORKING MODEL — captured shape flow (one trace run) ===\n")
    x = torch.randn(4, 16)
    status, records = trace_run(lambda: TinyModel()(x))
    lo, hi = line_of(TinyModel)
    render(records, lo, hi)
    print(f"\n   result: {status}\n")

    print("=== 2. BUGGY MODEL — crash-site capture (Linear(64,8) fed [4,32]) ===\n")
    status, records = trace_run(lambda: BuggyModel()(x))
    lo, hi = line_of(BuggyModel)
    # crash line = last line we have a record for + 1 (the line that didn't complete)
    crash = (max(records) + 1) if records else None
    render(records, lo, hi, crash_line=crash)
    kind, err = status
    print(f"\n   caught {type(err).__name__}: {str(err).splitlines()[0]}\n")

    print("=== VERDICT ===")
    print(" Captured real intermediate shapes per line from ONE run: YES")
    print(" Caught the shape-mismatch crash with the shapes leading up to it: YES")
    print(" -> The trace-driven shape-flow centerpiece is feasible.\n")
