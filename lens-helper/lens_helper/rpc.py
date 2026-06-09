"""JSON-RPC over stdio — the warm-process protocol the extension spawns once.

One JSON request per line on stdin -> one JSON response per line on stdout.
  request:  {"id": 1, "method": "load_file", "params": {"path": "..."}}
  response: {"id": 1, "result": {...}}   or   {"id": 1, "error": "..."}
Imports stay lazy (numpy/pandas/torch loaded only when a method needs them) so startup
is fast; the process stays warm so later calls skip the import cost.
"""
import json
import sys
from typing import Any, Dict

PROTOCOL = 1


def handle(method: str, params: Dict[str, Any]) -> Any:
    if method == "ping":
        return {"pong": True}
    if method == "version":
        return {"name": "lens-helper", "version": "0.0.1", "protocol": PROTOCOL}
    if method == "load_file":
        from . import loaders

        return loaders.load_file(params["path"])
    if method == "structure_file":
        from . import structure

        return structure.structure_file(params["path"])
    if method == "callgraph_file":
        from . import callgraph

        return callgraph.callgraph_file(params["path"])
    if method == "trace_file":
        from . import tracer

        records, error, crash_line = tracer.trace_file(params["path"])
        return {"records": records, "error": error, "crashLine": crash_line}
    if method == "trace_function":
        from . import tracer

        records, error, crash_line, note = tracer.trace_function(
            params["path"], params["name"], int(params.get("line", 0))
        )
        return {"records": records, "error": error, "crashLine": crash_line, "note": note}
    if method == "trace_module":
        from . import tracer

        return tracer.trace_module(params["path"])  # {records, problems, notes}
    raise ValueError(f"unknown method: {method!r}")


def main() -> None:
    out = sys.stdout
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        rid = None
        try:
            req = json.loads(line)
            rid = req.get("id")
            resp = {"id": rid, "result": handle(req.get("method"), req.get("params") or {})}
        except Exception as e:
            resp = {"id": rid, "error": f"{type(e).__name__}: {e}"}
        out.write(json.dumps(resp, default=str) + "\n")
        out.flush()
