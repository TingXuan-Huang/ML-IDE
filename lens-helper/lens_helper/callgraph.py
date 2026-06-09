"""Intra-file call graph via Python `ast` (Mode = Graph zone).

Nodes = functions/methods defined in the file; edges = calls between them (by name,
including `self.method()`). Intra-file only for M1 — cross-module/CallHierarchy later.
"""
import ast
from typing import Any, Dict, Optional


def callgraph_file(path: str, focus: Optional[str] = None) -> Dict[str, Any]:
    try:
        with open(path) as f:
            tree = ast.parse(f.read(), path)
    except Exception as e:
        return {"focus": focus or "", "nodes": [], "edges": [], "sparse": True, "note": str(e)}

    funcs: Dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs.setdefault(node.name, node.lineno)

    nodes = [{"id": name, "label": name, "line": line} for name, line in funcs.items()]
    edges = []
    seen = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        caller = node.name
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call):
                callee = _call_name(sub.func)
                if callee and callee in funcs and callee != caller:
                    key = (caller, callee)
                    if key not in seen:
                        seen.add(key)
                        edges.append({"from": caller, "to": callee})

    return {
        "focus": focus or (nodes[0]["id"] if nodes else ""),
        "nodes": nodes,
        "edges": edges,
        "sparse": len(edges) == 0,
    }


def _call_name(func: ast.AST) -> Optional[str]:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr  # self.foo() / obj.foo() -> "foo"
    return None
