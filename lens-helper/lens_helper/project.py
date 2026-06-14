"""Project-level analysis (open-a-folder): list a folder's Python files and build a
CROSS-FILE import graph. Nodes = files; edges = an import that resolves to another file
IN the project (external deps like torch are ignored). Static `ast` only — same M1 path
as structure/callgraph.
"""
import ast
import os
from typing import Any, Dict, List, Optional, Set, Tuple

_IGNORE_DIRS = {
    ".git", "__pycache__", ".venv", "venv", "env", "node_modules", ".mypy_cache",
    ".pytest_cache", ".ipynb_checkpoints", "build", "dist", ".tox", ".eggs",
    "site-packages", ".idea", ".vscode", ".ruff_cache", "wandb", ".cache", ".hg",
}
_MAX_FILES = 4000


def _iter_py(folder: str):
    for root, dirs, files in os.walk(folder):
        dirs[:] = sorted(d for d in dirs if d not in _IGNORE_DIRS and not d.startswith("."))
        for f in sorted(files):
            if f.endswith(".py"):
                yield os.path.join(root, f)


def list_python_files(folder: str) -> Dict[str, Any]:
    """{root, files} — files are posix relpaths under root, sorted, junk dirs skipped."""
    folder = os.path.abspath(folder)
    rels: List[str] = []
    for p in _iter_py(folder):
        rels.append(os.path.relpath(p, folder).replace(os.sep, "/"))
        if len(rels) >= _MAX_FILES:
            break
    rels.sort()
    return {"root": folder, "files": rels}


def _module_name(rel: str, pkg_dirs: Set[str]) -> Tuple[str, str]:
    """(module_dotted, package_dotted) using Python's __init__-based package rules. E.g.
    src/cellfm/models/hvg_mlp.py (src not a package) -> ('cellfm.models.hvg_mlp',
    'cellfm.models'); a loose top-level foo.py -> ('foo', '')."""
    parts = rel[:-3].split("/")  # drop .py
    dirs, stem = parts[:-1], parts[-1]
    i = len(dirs)  # contiguous package dirs from the file's own dir upward = the module root
    while i > 0 and "/".join(dirs[:i]) in pkg_dirs:
        i -= 1
    pkg_parts = dirs[i:]
    if stem == "__init__":
        return ".".join(pkg_parts), ".".join(pkg_parts[:-1])
    return ".".join(pkg_parts + [stem]), ".".join(pkg_parts)


def _identities(rel: str, module: str) -> Set[str]:
    """Dotted names by which OTHER files might import this one: the __init__-aware module
    name, the raw folder-relative name, and (for a src/ layout) the src-stripped name — so
    both `from cellfm.x import y` and `from src.cellfm.x import y` resolve."""
    raw = rel[:-3].replace("/", ".")
    cands = {module, raw}
    if rel.startswith("src/"):
        cands.add(rel[4:-3].replace("/", "."))
    out: Set[str] = set()
    for c in cands:
        if c.endswith(".__init__"):
            c = c[:-9]
        if c and c != "__init__":
            out.add(c)
    return out


def _import_targets(tree: ast.AST, package: str) -> Set[str]:
    """Every dotted module name an import in `tree` could refer to (absolute + relative)."""
    targets: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                targets.add(a.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative: from .mod import x / from .. import y
                base = package.split(".") if package else []
                if node.level > 1:
                    base = base[: -(node.level - 1)] if node.level - 1 <= len(base) else []
                mod = node.module.split(".") if node.module else []
                if base or mod:
                    targets.add(".".join(base + mod))
                for n in node.names:
                    targets.add(".".join(base + mod + [n.name]))
            elif node.module:  # absolute: from a.b import c  (c may be a submodule)
                targets.add(node.module)
                for n in node.names:
                    targets.add(f"{node.module}.{n.name}")
    return targets


_NODE_W, _NODE_H, _GAP_X, _GAP_Y = 168, 28, 26, 64


def _layered_layout(node_ids: List[str], edges: List[Dict[str, str]]):
    """Sugiyama-style layered layout for a (mostly) DAG: break cycles, assign layers by
    longest path, reduce crossings with a barycenter sweep, center each layer. Top→bottom:
    importers on top, their dependencies below. Returns (pos{id:(x,y)}, width, height)."""
    if not node_ids:
        return {}, 0, 0
    idx = {nid: i for i, nid in enumerate(node_ids)}
    n = len(node_ids)
    succ: List[List[int]] = [[] for _ in range(n)]
    for e in edges:
        a, b = idx.get(e["from"]), idx.get(e["to"])
        if a is not None and b is not None and a != b and b not in succ[a]:
            succ[a].append(b)

    # iterative DFS -> mark back-edges (the ones that close a cycle); ignore them for layering
    WHITE, GRAY, BLACK = 0, 1, 2
    color = [WHITE] * n
    back = set()
    for s in range(n):
        if color[s] != WHITE:
            continue
        stack = [(s, iter(succ[s]))]
        color[s] = GRAY
        while stack:
            u, it = stack[-1]
            for v in it:
                if color[v] == WHITE:
                    color[v] = GRAY
                    stack.append((v, iter(succ[v])))
                    break
                if color[v] == GRAY:
                    back.add((u, v))
            else:
                color[u] = BLACK
                stack.pop()

    asucc = [[v for v in succ[u] if (u, v) not in back] for u in range(n)]
    apred: List[List[int]] = [[] for _ in range(n)]
    indeg = [0] * n
    for u in range(n):
        for v in asucc[u]:
            apred[v].append(u)
            indeg[v] += 1

    # longest-path layering via Kahn
    layer = [0] * n
    queue = [i for i in range(n) if indeg[i] == 0]
    head = 0
    rem = indeg[:]
    while head < len(queue):
        u = queue[head]
        head += 1
        for v in asucc[u]:
            if layer[u] + 1 > layer[v]:
                layer[v] = layer[u] + 1
            rem[v] -= 1
            if rem[v] == 0:
                queue.append(v)

    rows: Dict[int, List[int]] = {}
    for i in range(n):
        rows.setdefault(layer[i], []).append(i)
    max_layer = max(rows)
    for L in rows:
        rows[L].sort(key=lambda i: node_ids[i])  # stable, deterministic start

    # barycenter sweeps to reduce edge crossings (down then up)
    order_in = {i: j for L in rows for j, i in enumerate(rows[L])}
    for _ in range(4):
        for L in range(1, max_layer + 1):
            rows[L].sort(key=lambda i: (sum(order_in[p] for p in apred[i]) / len(apred[i])) if apred[i] else order_in[i])
            for j, i in enumerate(rows[L]):
                order_in[i] = j
        for L in range(max_layer - 1, -1, -1):
            rows[L].sort(key=lambda i: (sum(order_in[s] for s in asucc[i]) / len(asucc[i])) if asucc[i] else order_in[i])
            for j, i in enumerate(rows[L]):
                order_in[i] = j

    step_x = _NODE_W + _GAP_X
    max_row = max(len(r) for r in rows.values())
    width = max_row * step_x - _GAP_X
    pos: Dict[str, Any] = {}
    for L in range(max_layer + 1):
        row = rows[L]
        offset = (width - (len(row) * step_x - _GAP_X)) / 2
        for j, i in enumerate(row):
            pos[node_ids[i]] = (round(offset + j * step_x), L * (_NODE_H + _GAP_Y))
    height = (max_layer + 1) * (_NODE_H + _GAP_Y) - _GAP_Y
    return pos, width, height


def project_graph(folder: str) -> Dict[str, Any]:
    """{root, files, focus, nodes, edges, sparse} — CallGraph-shaped so the cockpit's graph
    renderer can draw it. Nodes are files (id = relpath); edges are in-project imports."""
    info = list_python_files(folder)
    folder, rels = info["root"], info["files"]
    pkg_dirs = {r.rsplit("/", 1)[0] if "/" in r else "" for r in rels if r.rsplit("/", 1)[-1] == "__init__.py"}

    ident: Dict[str, str] = {}  # dotted name -> relpath
    package_of: Dict[str, str] = {}
    for rel in rels:
        module, package = _module_name(rel, pkg_dirs)
        package_of[rel] = package
        for name in _identities(rel, module):
            ident.setdefault(name, rel)

    def resolve(name: str) -> Optional[str]:
        if name in ident:
            return ident[name]
        parts = name.split(".")  # `import a.b.c` where only a.b is a project module
        for k in range(len(parts) - 1, 0, -1):
            pre = ".".join(parts[:k])
            if pre in ident:
                return ident[pre]
        return None

    edges: List[Dict[str, str]] = []
    seen: Set[Tuple[str, str]] = set()
    for rel in rels:
        try:
            with open(os.path.join(folder, rel)) as f:
                tree = ast.parse(f.read(), rel)
        except Exception:
            continue
        for t in _import_targets(tree, package_of[rel]):
            dest = resolve(t)
            if dest and dest != rel and (rel, dest) not in seen:
                seen.add((rel, dest))
                edges.append({"from": rel, "to": dest})

    pos, width, height = _layered_layout(rels, edges)
    nodes = [
        {"id": rel, "label": rel.rsplit("/", 1)[-1], "line": 0,
         "x": pos.get(rel, (0, 0))[0], "y": pos.get(rel, (0, 0))[1]}
        for rel in rels
    ]
    return {
        "root": folder,
        "files": rels,
        "focus": rels[0] if rels else "",
        "nodes": nodes,
        "edges": edges,
        "sparse": len(edges) == 0,
        "width": width,
        "height": height,
    }
