"""Static structure extraction via Python `ast` (M1 path for Mode 1 blocks).

NOTE: the eng-review-locked decision is to use bundled tree-sitter for the no-env
structure path (so blocks render without a Python interpreter). This ast-based version
is the faster M1 stand-in; swap to tree-sitter later. Tracked in BUILD_STATUS.md.
"""
import ast
from typing import Any, Dict, List


def structure_file(path: str) -> Dict[str, Any]:
    try:
        with open(path) as f:
            src = f.read()
        tree = ast.parse(src, path)
    except Exception as e:
        return {"path": path, "language": "python", "functions": [], "note": f"parse error: {e}"}

    functions: List[Dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        params = [
            {"name": a.arg, "type": _unparse(a.annotation)}
            for a in node.args.args
        ]
        intermediates = []
        for stmt in ast.walk(node):
            if isinstance(stmt, ast.Assign):
                for tgt in stmt.targets:
                    if isinstance(tgt, ast.Name):
                        intermediates.append({"line": stmt.lineno, "name": tgt.id})
        functions.append(
            {
                "name": node.name,
                "startLine": node.lineno,
                "endLine": getattr(node, "end_lineno", node.lineno),
                "params": params,
                "returns": _unparse(node.returns),
                "intermediates": intermediates,
            }
        )
    functions.sort(key=lambda f: f["startLine"])
    return {"path": path, "language": "python", "functions": functions}


def _unparse(node: Any):
    if node is None:
        return None
    try:
        return ast.unparse(node)
    except Exception:
        return None
