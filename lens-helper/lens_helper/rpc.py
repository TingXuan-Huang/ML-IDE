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
    # Tracing exec's the user's model + runs its forward — a watchdog hard-kills the helper
    # if that runs away (infinite loop); the host transparently respawns. See tracer._time_limit.
    TRACE_TIMEOUT = 60
    if method == "trace_file":
        from . import tracer

        with tracer._time_limit(TRACE_TIMEOUT):
            records, error, crash_line = tracer.trace_file(params["path"])
        return {"records": records, "error": error, "crashLine": crash_line}
    if method == "trace_function":
        from . import tracer

        with tracer._time_limit(TRACE_TIMEOUT):
            return tracer.trace_function(
                params["path"],
                params["name"],
                int(params.get("line", 0)),
                int(params.get("batch", 2)),
                int(params.get("seq", 16)),
                params.get("projectRoot", ""),
            )  # {records, error, crashLine, note, ops, dims}
    if method == "trace_module":
        from . import tracer

        with tracer._time_limit(TRACE_TIMEOUT):
            return tracer.trace_module(
                params["path"], int(params.get("batch", 2)), int(params.get("seq", 16)), params.get("projectRoot", "")
            )  # {records, problems, notes, ops, dims}
    if method == "module_summary":
        from . import tracer

        with tracer._time_limit(TRACE_TIMEOUT):
            return tracer.module_summary(
                params["path"], int(params.get("batch", 2)), int(params.get("seq", 16)), params.get("projectRoot", "")
            )  # {target, rows, totalParams, trainableParams, paramBytes, dims, error}
    if method == "paper_module":
        from . import tracer

        with tracer._time_limit(TRACE_TIMEOUT):
            return tracer.paper_module(
                params["path"], int(params.get("batch", 2)), int(params.get("seq", 16)), params.get("projectRoot", "")
            )  # {path, sections, dims, problems}
    if method == "compare_traces":  # design B scaffold — faithful-port compare
        from . import tracer

        with tracer._time_limit(TRACE_TIMEOUT):
            return tracer.compare_traces(
                params["pathA"], params["pathB"], params.get("projectRoot", "")
            )  # {pathA, pathB, matched, onlyA, onlyB, dimsA, dimsB, problems, note}
    if method == "list_folder":
        from . import project

        return project.list_python_files(params["path"])  # {root, files}
    if method == "project_graph":
        from . import project

        return project.project_graph(params["path"])  # {root, files, focus, nodes, edges, sparse}
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
