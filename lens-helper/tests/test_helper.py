"""Unit tests for lens-helper. Run: cd lens-helper && python -m pytest tests/ -q"""
import os

import torch
import torch.nn as nn

from lens_helper import callgraph, loaders, project, structure, tracer

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "spike", "sampledata")


def _write(tmp_path, name, src):
    p = tmp_path / name
    p.write_text(src)
    return str(p)


def _tf(*a):
    """trace_function returns a dict now; legacy tests unpack the original 4 fields."""
    r = tracer.trace_function(*a)
    return r["records"], r["error"], r["crashLine"], r["note"]


# --- tracer ---------------------------------------------------------------------
def test_trace_callable_captures_named_shapes():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 8)

        def forward(self, x):
            h = self.fc(x)
            return h

    records, err, crash = tracer.trace_callable(lambda: Net()(torch.randn(2, 4)), __file__)
    assert err is None and crash is None
    assert any(any(v["shape"] == [2, 8] for v in rec.values()) for rec in records.values())


def test_trace_file_captures_return_value_shape(tmp_path):
    f = _write(
        tmp_path,
        "m.py",
        "import torch, torch.nn as nn\n"
        "class N(nn.Module):\n"
        "    def __init__(s):\n"
        "        super().__init__(); s.fc = nn.Linear(4, 8)\n"
        "    def forward(s, x):\n"
        "        return s.fc(x)\n"  # bare return, no local name
        "N()(torch.randn(3, 4))\n",
    )
    records, err, crash = tracer.trace_file(f)
    assert err is None
    # the return value [3,8] is captured under a synthetic 'return' name
    assert any("return" in rec and rec["return"]["shape"] == [3, 8] for rec in records.values())


def test_trace_file_catches_shape_mismatch_at_right_line(tmp_path):
    f = _write(
        tmp_path,
        "bug.py",
        "import torch, torch.nn as nn\n"            # 1
        "class B(nn.Module):\n"                      # 2
        "    def __init__(s):\n"                      # 3
        "        super().__init__()\n"                # 4
        "        s.a = nn.Linear(4, 8)\n"             # 5
        "        s.b = nn.Linear(16, 2)\n"            # 6
        "    def forward(s, x):\n"                    # 7
        "        h = s.a(x)\n"                         # 8
        "        y = s.b(h)\n"                         # 9  <- crash (8 vs 16)
        "        return y\n"                           # 10
        "B()(torch.randn(2, 4))\n",                   # 11
    )
    records, err, crash = tracer.trace_file(f)
    assert err is not None and "shapes cannot be multiplied" in err
    assert crash == 9


def test_trace_file_isolates_user_stdout(tmp_path, capsys):
    f = _write(tmp_path, "p.py", "import torch\nprint('SHOULD_NOT_LEAK')\ntorch.randn(2, 2)\n")
    tracer.trace_file(f)
    assert "SHOULD_NOT_LEAK" not in capsys.readouterr().out


# --- loaders --------------------------------------------------------------------
def test_load_npy():
    m = loaders.load_file(os.path.join(DATA, "embeddings.npy"))
    assert m["kind"] == "ndarray" and m["shape"] == [128, 16]
    assert "mean" in m["stats"]


def test_load_csv():
    m = loaders.load_file(os.path.join(DATA, "metrics.csv"))
    assert m["kind"] == "table"
    assert [c["name"] for c in m["columns"]] == ["epoch", "loss", "acc"]
    assert len(m["sample"]) > 0


def test_load_missing_degrades():
    m = loaders.load_file("/definitely/not/here.csv")
    assert m["kind"] == "unknown" and "not found" in m["note"]


# --- callgraph ------------------------------------------------------------------
def test_callgraph_edges():
    g = callgraph.callgraph_file(os.path.join(DATA, "pipeline.py"))
    names = {n["id"] for n in g["nodes"]}
    assert {"main", "load_batch", "build_model", "train_step"} <= names
    edges = {(e["from"], e["to"]) for e in g["edges"]}
    assert ("main", "load_batch") in edges and ("main", "train_step") in edges
    assert g["sparse"] is False


# --- structure ------------------------------------------------------------------
def test_structure_functions_and_params():
    s = structure.structure_file(os.path.join(DATA, "demo_model.py"))
    names = [f["name"] for f in s["functions"]]
    assert "forward" in names and "__init__" in names
    fwd = next(f for f in s["functions"] if f["name"] == "forward")
    assert any(p["name"] == "x" for p in fwd["params"])


def test_structure_syntax_error_degrades(tmp_path):
    f = _write(tmp_path, "broken.py", "def f(:\n")
    s = structure.structure_file(f)
    assert s["functions"] == [] and "note" in s


# --- project (open-a-folder: file list + cross-file import graph) ----------------
def _mk(tmp_path, rel, src=""):
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(src)
    return p


def test_list_python_files_skips_junk(tmp_path):
    _mk(tmp_path, "a.py")
    _mk(tmp_path, "pkg/b.py")
    _mk(tmp_path, ".venv/lib/junk.py")        # hidden dir -> skipped
    _mk(tmp_path, "__pycache__/c.py")          # ignore dir -> skipped
    _mk(tmp_path, "notes.txt", "hi")           # non-py -> skipped
    files = project.list_python_files(str(tmp_path))["files"]
    assert files == ["a.py", "pkg/b.py"], files


def test_project_graph_src_layout_absolute_and_relative(tmp_path):
    # A real src/ package layout: b imports a two ways (absolute + relative) -> one edge b->a.
    _mk(tmp_path, "src/cellfm/__init__.py")
    _mk(tmp_path, "src/cellfm/models/__init__.py")
    _mk(tmp_path, "src/cellfm/models/a.py", "import torch\nX = 1\n")
    _mk(
        tmp_path,
        "src/cellfm/models/b.py",
        "from cellfm.models.a import X\nfrom .a import X as Y\nimport torch.nn as nn\n",
    )
    g = project.project_graph(str(tmp_path))
    ids = {n["id"] for n in g["nodes"]}
    assert "src/cellfm/models/a.py" in ids and "src/cellfm/models/b.py" in ids
    edges = {(e["from"], e["to"]) for e in g["edges"]}
    assert ("src/cellfm/models/b.py", "src/cellfm/models/a.py") in edges, edges
    # external (torch) and self never produce edges
    assert all("torch" not in e["to"] for e in g["edges"])
    assert all(e["from"] != e["to"] for e in g["edges"])
    assert g["sparse"] is False


def test_project_graph_flat_layout(tmp_path):
    _mk(tmp_path, "a.py", "VAL = 1\n")
    _mk(tmp_path, "b.py", "import a\nprint(a.VAL)\n")
    _mk(tmp_path, "c.py", "from b import VAL\n")
    edges = {(e["from"], e["to"]) for e in project.project_graph(str(tmp_path))["edges"]}
    assert ("b.py", "a.py") in edges and ("c.py", "b.py") in edges, edges


def test_project_graph_no_in_project_edges_is_sparse(tmp_path):
    _mk(tmp_path, "solo.py", "import os\nimport numpy as np\n")
    g = project.project_graph(str(tmp_path))
    assert g["edges"] == [] and g["sparse"] is True


def test_layered_layout_orders_by_dependency_depth():
    # a imports b imports c -> importer a on top, dependency c at the bottom (higher y).
    ids = ["a.py", "b.py", "c.py"]
    edges = [{"from": "a.py", "to": "b.py"}, {"from": "b.py", "to": "c.py"}]
    pos, w, h = project._layered_layout(ids, edges)
    assert pos["a.py"][1] < pos["b.py"][1] < pos["c.py"][1], pos
    assert w > 0 and h > 0
    # deterministic
    assert project._layered_layout(ids, edges)[0] == pos


def test_layered_layout_survives_cycles():
    # a <-> b cycle must not hang or crash; both get coordinates.
    ids = ["a.py", "b.py"]
    edges = [{"from": "a.py", "to": "b.py"}, {"from": "b.py", "to": "a.py"}]
    pos, w, h = project._layered_layout(ids, edges)
    assert set(pos) == {"a.py", "b.py"} and h > 0


def test_project_graph_nodes_carry_positions(tmp_path):
    _mk(tmp_path, "a.py", "X=1")
    _mk(tmp_path, "b.py", "import a")
    g = project.project_graph(str(tmp_path))
    assert all("x" in n and "y" in n for n in g["nodes"])
    assert g["width"] > 0 and g["height"] > 0


# --- #5 real-input tracing: load() in directives -------------------------------
import numpy as _np  # noqa: E402


def test_load_tensor_npy(tmp_path):
    p = tmp_path / "x.npy"
    _np.save(p, _np.zeros((3, 16), dtype="float64"))
    t = loaders.load_tensor("x.npy", str(tmp_path))
    assert list(t.shape) == [3, 16] and str(t.dtype) == "torch.float32"  # float64 -> float32


def test_load_directive_input_traces(tmp_path):
    _np.save(tmp_path / "x.npy", _np.zeros((3, 8), dtype="float32"))
    f = _write(
        tmp_path,
        "m.py",
        "import torch, torch.nn as nn\n"
        "class N(nn.Module):\n"
        "    def __init__(s):\n"
        "        super().__init__(); s.fc = nn.Linear(8, 4)\n"
        "    # fusion: input = load(\"x.npy\")\n"
        "    def forward(s, x):\n"
        "        return s.fc(x)\n",
    )
    r = tracer.trace_function(f, "forward", 0, project_root=str(tmp_path))
    assert r["error"] is None, r["error"]
    assert 'load("x.npy")' in r["note"]
    assert any(any(v["shape"] == [3, 4] for v in rec.values()) for rec in r["records"].values())


def test_load_missing_file_is_soft_note(tmp_path):
    f = _write(
        tmp_path,
        "m.py",
        "import torch, torch.nn as nn\n"
        "class N(nn.Module):\n"
        "    def __init__(s):\n"
        "        super().__init__(); s.fc = nn.Linear(8, 4)\n"
        "    # fusion: input = load(\"nope.npy\")\n"
        "    def forward(s, x):\n"
        "        return s.fc(x)\n",
    )
    r = tracer.trace_function(f, "forward", 0, project_root=str(tmp_path))
    assert r["records"] == {} and "nope.npy" in r["note"]  # FileNotFoundError -> _CannotInvoke note, no crash


def test_time_limit_cancels_when_fast():
    # Completing within the limit must NOT fire the watchdog (no os._exit).
    import os

    fired = []
    real_exit, os._exit = os._exit, lambda c: fired.append(c)
    try:
        with tracer._time_limit(5):
            x = 1 + 1
    finally:
        os._exit = real_exit
    assert x == 2 and fired == []


def test_time_limit_kills_on_overrun(monkeypatch):
    # An overrun fires the daemon timer -> os._exit (stubbed so it doesn't kill pytest).
    import os
    import time

    fired = []
    monkeypatch.setattr(os, "_exit", lambda c: fired.append(c))
    with tracer._time_limit(0.2):
        time.sleep(0.6)  # exceed the limit -> the timer thread calls our stubbed _exit
    assert fired == [2]


def test_directive_safety_blocks_dangerous(tmp_path):
    f = _write(
        tmp_path,
        "m.py",
        "import torch, torch.nn as nn\n"
        "class N(nn.Module):\n"
        "    def __init__(s):\n"
        "        super().__init__(); s.fc = nn.Linear(8, 4)\n"
        "    # fusion: input = __import__('os').getcwd() or torch.randn(2, 8)\n"
        "    def forward(s, x):\n"
        "        return s.fc(x)\n",
    )
    r = tracer.trace_function(f, "forward", 0)
    assert r["records"] == {} and "blocked" in r["note"].lower()  # _CannotInvoke note, no exec
    assert tracer._directive_is_safe("torch.randn(2, 8)") and tracer._directive_is_safe('load("x.npy")')
    assert not tracer._directive_is_safe("().__class__.__bases__")  # dunder-attr escape blocked


def test_load_csv_returns_float_tensor(tmp_path):
    (tmp_path / "d.csv").write_text("a,b\n1,2\n3,4\n5,6\n")
    t = loaders.load_tensor("d.csv", str(tmp_path))
    assert list(t.shape) == [3, 2] and str(t.dtype) == "torch.float32"


def test_load_size_cap_rejects(tmp_path, monkeypatch):
    _np.save(tmp_path / "big.npy", _np.zeros((100,), dtype="float32"))
    monkeypatch.setattr(loaders, "MAX_TENSOR_BYTES", 10)  # tiny cap
    try:
        loaders.load_tensor("big.npy", str(tmp_path))
        assert False, "should have raised"
    except ValueError as e:
        assert "too big" in str(e)


# --- #2 model summary viz ------------------------------------------------------
def test_module_summary_param_counts(tmp_path):
    f = _write(
        tmp_path,
        "m.py",
        "import torch, torch.nn as nn\n"
        "class Net(nn.Module):\n"
        "    def __init__(s):\n"
        "        super().__init__(); s.a = nn.Linear(16, 64); s.b = nn.Linear(64, 8)\n"
        "    def forward(s, x):\n"
        "        return s.b(s.a(x))\n",
    )
    res = tracer.module_summary(f)
    assert res["error"] is None, res["error"]
    by = {r["name"]: r for r in res["rows"]}
    assert by["a"]["params"] == 16 * 64 + 64 and by["b"]["params"] == 64 * 8 + 8
    assert res["totalParams"] == sum(r["params"] for r in res["rows"])  # no double-counting (leaf modules)
    assert by["a"]["outShape"] == [2, 64] and by["b"]["outShape"] == [2, 8]


def test_module_summary_needs_ctor_args_is_soft(tmp_path):
    f = _write(
        tmp_path,
        "m.py",
        "import torch, torch.nn as nn\n"
        "class Need(nn.Module):\n"
        "    def __init__(s, dim):\n"
        "        super().__init__(); s.fc = nn.Linear(dim, 4)\n"
        "    def forward(s, x):\n"
        "        return s.fc(x)\n",
    )
    res = tracer.module_summary(f)
    assert res["rows"] == [] and res["error"] and "constructor args" in res["error"]


def test_module_summary_forward_crash_still_reports_params(tmp_path):
    f = _write(
        tmp_path,
        "m.py",
        "import torch, torch.nn as nn\n"
        "class Bad(nn.Module):\n"
        "    def __init__(s):\n"
        "        super().__init__(); s.a = nn.Linear(16, 8); s.b = nn.Linear(99, 2)\n"
        "    def forward(s, x):\n"
        "        return s.b(s.a(x))\n",  # 8 vs 99 mismatch
    )
    res = tracer.module_summary(f)
    assert res["error"] and "multiplied" in res["error"]
    assert res["totalParams"] > 0 and any(r["params"] > 0 for r in res["rows"])


# --- #8 paper-reading mode -----------------------------------------------------
def test_paper_module_groups_ops_per_module(tmp_path):
    f = _write(
        tmp_path,
        "m.py",
        "import torch, torch.nn as nn\n"
        "import torch.nn.functional as F\n"
        "class Blk(nn.Module):\n"
        "    def __init__(s):\n"
        "        super().__init__(); s.qkv = nn.Linear(8, 24)\n"
        "    # fusion: input = torch.randn(2, 6, 8)\n"
        "    def forward(s, x):\n"
        "        B, L, D = x.shape\n"
        "        qkv = s.qkv(x).reshape(B, L, 3, 8)\n"
        "        a = F.softmax(qkv, dim=-1)\n"
        "        return a\n",
    )
    res = tracer.paper_module(f)
    assert res["problems"] == [], res["problems"]
    sec = next(s for s in res["sections"] if s["module"] == "Blk.forward")
    ops = " | ".join(st["op"] for st in sec["steps"] if st["op"])
    assert "reshape" in ops and "softmax(dim=-1)" in ops
    assert res["dims"].get("2") == "B" and res["dims"].get("6") == "L" and res["dims"].get("8") == "D"
    # steps preserve source order
    assert [st["line"] for st in sec["steps"]] == sorted(st["line"] for st in sec["steps"])


def test_module_summary_picks_largest_model_not_first(tmp_path):
    # Submodule defined BEFORE the composed model -> summary must describe the composed one.
    f = _write(
        tmp_path,
        "m.py",
        "import torch, torch.nn as nn\n"
        "class Block(nn.Module):\n"
        "    def __init__(s):\n"
        "        super().__init__(); s.fc = nn.Linear(8, 8)\n"
        "    def forward(s, x):\n"
        "        return s.fc(x)\n"
        "class Model(nn.Module):\n"
        "    def __init__(s):\n"
        "        super().__init__(); s.a = Block(); s.b = Block(); s.head = nn.Linear(8, 4)\n"
        "    def forward(s, x):\n"
        "        return s.head(s.b(s.a(x)))\n",
    )
    res = tracer.module_summary(f)
    assert res["target"].startswith("Model("), res["target"]  # not Block()
    assert any(r["name"] == "head" for r in res["rows"])


def test_module_summary_tied_params_no_double_count(tmp_path):
    f = _write(
        tmp_path,
        "m.py",
        "import torch, torch.nn as nn\n"
        "class Tied(nn.Module):\n"
        "    def __init__(s):\n"
        "        super().__init__(); s.emb = nn.Embedding(10, 4); s.head = nn.Linear(4, 10, bias=False)\n"
        "        s.head.weight = s.emb.weight\n"  # tie
        "    # fusion: input = torch.randint(0, 10, (2, 5))\n"
        "    def forward(s, x):\n"
        "        return s.head(s.emb(x))\n",
    )
    res = tracer.module_summary(f)
    assert res["error"] is None, res["error"]
    # per-row params must reconcile with the (dedup'd) total -> <= 100% total share
    assert sum(r["params"] for r in res["rows"]) == res["totalParams"]


def test_paper_module_includes_augassign_residual(tmp_path):
    f = _write(
        tmp_path,
        "m.py",
        "import torch, torch.nn as nn\n"
        "class Res(nn.Module):\n"
        "    def __init__(s):\n"
        "        super().__init__(); s.fc = nn.Linear(8, 8)\n"
        "    # fusion: input = torch.randn(2, 8)\n"
        "    def forward(s, x):\n"
        "        x = x + s.fc(x)\n"
        "        x += s.fc(x)\n"  # AugAssign residual — must appear as a step
        "        return x\n",
    )
    res = tracer.paper_module(f)
    sec = next(s for s in res["sections"] if s["module"] == "Res.forward")
    assert any(st["lhs"] == "x" and st["line"] == 8 for st in sec["steps"]), sec["steps"]  # the `x += ...` line


def test_structure_shape_reqs_skips_short_receiver(tmp_path):
    # `def forward(s, x)` — `s` is the receiver and must NOT appear as a param/req.
    f = _write(tmp_path, "m.py", "class N:\n    def forward(s, x):\n        B, L, D = x.shape\n        return x\n")
    fns = structure.structure_file(f)["functions"]
    fwd = next(fn for fn in fns if fn["name"] == "forward")
    names = {r["name"] for r in fwd["shapeReqs"]}
    assert "s" not in names and "x" in names, names


def test_paper_module_skips_untraceable(tmp_path):
    f = _write(
        tmp_path,
        "m.py",
        "import torch, torch.nn as nn\n"
        "class Need(nn.Module):\n"
        "    def __init__(s, dim):\n"
        "        super().__init__(); s.fc = nn.Linear(dim, 4)\n"
        "    def forward(s, x):\n"
        "        return s.fc(x)\n",
    )
    res = tracer.paper_module(f)
    assert res["sections"] == []  # un-buildable -> no fabricated section


# --- trace_function (call directly, no __main__) --------------------------------
def test_trace_function_forward_no_main(tmp_path):
    # a pure library model: defines a class, never runs anything
    f = _write(
        tmp_path,
        "lib.py",
        "import torch, torch.nn as nn\n"          # 1
        "class E(nn.Module):\n"                    # 2
        "    def __init__(s):\n"                    # 3
        "        super().__init__()\n"              # 4
        "        s.a = nn.Linear(16, 64)\n"         # 5
        "        s.b = nn.Linear(64, 8)\n"          # 6
        "    def forward(s, x):\n"                  # 7
        "        h = s.a(x)\n"                       # 8
        "        return s.b(h)\n",                   # 9
    )
    records, err, crash, note = _tf(f, "forward", 7)
    assert err is None, err
    assert "E().forward(randn(2, 16))" == note
    assert any(any(v["shape"] == [2, 64] for v in rec.values()) for rec in records.values())
    assert any("return" in rec and rec["return"]["shape"] == [2, 8] for rec in records.values())


def test_trace_function_top_level_zero_arg(tmp_path):
    f = _write(tmp_path, "u.py", "import torch\ndef make():\n    return torch.randn(4, 3)\n")
    records, err, crash, note = _tf(f, "make", 2)
    assert err is None and note == "make()"
    assert any("return" in rec and rec["return"]["shape"] == [4, 3] for rec in records.values())


def test_trace_function_needs_args_reports(tmp_path):
    f = _write(tmp_path, "u.py", "def f(a, b):\n    return a + b\n")
    records, err, crash, note = _tf(f, "f", 1)
    assert records == {} and "needs 2 args" in note


def test_trace_function_init_does_not_inject_synth_tensor(tmp_path):
    # Regression: tracing __init__ used to pass a synth tensor as the first ctor arg,
    # so nn.Linear(d_in=<Tensor>, ...) raised TypeError("empty(): ... but got Tensor").
    # __init__ must just CONSTRUCT (note "E()"), never synthesize a tensor input.
    f = _write(
        tmp_path,
        "lib.py",
        "import torch, torch.nn as nn\n"
        "class E(nn.Module):\n"
        "    def __init__(self, d_in=16, d_hidden=64, d_out=8):\n"
        "        super().__init__()\n"
        "        self.proj = nn.Linear(d_in, d_hidden)\n"
        "        self.out = nn.Linear(d_hidden, d_out)\n"
        "    def forward(self, x):\n"
        "        return self.out(self.proj(x))\n",
    )
    records, err, crash, note = _tf(f, "__init__", 3)
    assert err is None, err
    assert note == "E()"


# --- trace_module (trace the whole file: build-check + synth forward, no __main__) ------
def test_trace_module_traces_library_no_main(tmp_path):
    # pure library (no __main__): trace_module must still fill in forward shapes (batch 2).
    f = _write(
        tmp_path,
        "lib.py",
        "import torch, torch.nn as nn\n"
        "class E(nn.Module):\n"
        "    def __init__(s):\n"
        "        super().__init__(); s.a = nn.Linear(16, 64); s.b = nn.Linear(64, 8)\n"
        "    def forward(s, x):\n"
        "        h = s.a(x)\n"
        "        return s.b(h)\n",
    )
    res = tracer.trace_module(f)
    assert res["problems"] == []
    assert any(n["note"] == "E().forward(randn(2, 16))" for n in res["notes"])
    assert any(any(v["shape"] == [2, 64] for v in rec.values()) for rec in res["records"].values())


def test_trace_module_catches_forward_shape_bug(tmp_path):
    f = _write(
        tmp_path,
        "bug.py",
        "import torch, torch.nn as nn\n"
        "class B(nn.Module):\n"
        "    def __init__(s):\n"
        "        super().__init__(); s.a = nn.Linear(16, 8); s.b = nn.Linear(99, 2)\n"
        "    def forward(s, x):\n"
        "        return s.b(s.a(x))\n",  # 8 vs 99 -> mismatch
    )
    res = tracer.trace_module(f)
    assert any("shapes cannot be multiplied" in p["message"] for p in res["problems"])


def test_trace_module_build_check_catches_init_shape_error(tmp_path):
    # A shape error in __init__ (build time) must surface as a "build" problem.
    f = _write(
        tmp_path,
        "buildbug.py",
        "import torch, torch.nn as nn\n"
        "class Bad(nn.Module):\n"
        "    def __init__(s):\n"
        "        super().__init__()\n"
        "        _ = torch.randn(2, 3) @ torch.randn(4, 5)\n",  # build-time mismatch
    )
    res = tracer.trace_module(f)
    assert any("build Bad()" in p["message"] and "multiplied" in p["message"] for p in res["problems"])


def test_trace_module_skips_ctor_arg_models(tmp_path):
    # A model needing constructor args can't be auto-built -> a note, not a crash.
    f = _write(
        tmp_path,
        "needs.py",
        "import torch, torch.nn as nn\n"
        "class Need(nn.Module):\n"
        "    def __init__(s, dim):\n"
        "        super().__init__(); s.fc = nn.Linear(dim, 4)\n"
        "    def forward(s, x):\n"
        "        return s.fc(x)\n",
    )
    res = tracer.trace_module(f)
    assert res["problems"] == []
    assert any("needs constructor args" in n["note"] for n in res["notes"])


def test_trace_module_directive_does_not_bleed_across_classes(tmp_path):
    # Regression for the transformer_body bug: a `# fusion: model` attached to one class's
    # method must NOT be mis-picked for the NEXT class (anchor-based binding). `Other` needs
    # a ctor arg and has no directive of its own, so it must report "needs constructor args"
    # — NOT get silently built from the stray `Inner()` directive above `Inner.extra`.
    f = _write(
        tmp_path,
        "bleed.py",
        "import torch, torch.nn as nn\n"
        "class Inner(nn.Module):\n"
        "    def __init__(s):\n"
        "        super().__init__(); s.fc = nn.Linear(4, 4)\n"
        "    def forward(s, x):\n"
        "        return s.fc(x)\n"
        "    # fusion: model = Inner()\n"
        "    # fusion: input = torch.randn(2, 4)\n"
        "    def extra(s, x):\n"
        "        return s.fc(x)\n"
        "class Other(nn.Module):\n"  # within the old 6-line window of the stray directive
        "    def __init__(s, cfg):\n"
        "        super().__init__(); s.fc = nn.Linear(4, 4)\n"
        "    def forward(s, x):\n"
        "        return s.fc(x)\n",
    )
    res = tracer.trace_module(f)
    assert res["problems"] == [], res["problems"]
    other = next(n for n in res["notes"] if n["label"] == "Other")
    assert "needs constructor args" in other["note"], res["notes"]


def test_trace_module_synth_forward_ndim_guess_is_soft_note(tmp_path):
    # forward needs a 4-D input (beyond the rank-3 synth hint) -> a NOTE ("couldn't
    # auto-trace"), NOT a red shape problem (the model may be fine; we guessed the rank).
    f = _write(
        tmp_path,
        "fourd.py",
        "import torch, torch.nn as nn\n"
        "class Block(nn.Module):\n"
        "    def __init__(s):\n"
        "        super().__init__(); s.fc = nn.Linear(8, 8)\n"
        "    def forward(s, x):\n"
        "        B, T, L, D = x.shape\n"
        "        return s.fc(x)\n",
    )
    res = tracer.trace_module(f)
    assert res["problems"] == [], res["problems"]
    assert any("couldn't auto-trace" in n["note"] for n in res["notes"]), res["notes"]


def test_trace_module_rank3_synth_from_own_shape_unpack(tmp_path):
    # `B, L, D = x.shape` statically pins forward's input to rank 3 -> the synth makes
    # randn(B, S, in_features) instead of randn(B, in_features), and the trace SUCCEEDS.
    f = _write(
        tmp_path,
        "rank3.py",
        "import torch, torch.nn as nn\n"
        "class Block(nn.Module):\n"
        "    def __init__(s):\n"
        "        super().__init__(); s.fc = nn.Linear(8, 8)\n"
        "    def forward(s, x):\n"
        "        B, L, D = x.shape\n"
        "        return s.fc(x)\n",
    )
    res = tracer.trace_module(f)
    assert res["problems"] == [], res["problems"]
    assert any("randn(2, 16, 8)" in n["note"] for n in res["notes"]), res["notes"]
    assert any(any(v["shape"] == [2, 16, 8] for v in rec.values()) for rec in res["records"].values())


def test_trace_module_rank3_synth_propagates_from_submodule(tmp_path):
    # Outer.forward(x) has NO shape ops of its own, but its Inner submodule (defined in
    # the same file) does `B, L, D = x.shape` -> Outer's synth input must be rank 3 too.
    f = _write(
        tmp_path,
        "outer.py",
        "import torch, torch.nn as nn\n"
        "class Inner(nn.Module):\n"
        "    def __init__(s):\n"
        "        super().__init__(); s.fc = nn.Linear(8, 8)\n"
        "    def forward(s, x):\n"
        "        B, L, D = x.shape\n"
        "        return s.fc(x)\n"
        "class Outer(nn.Module):\n"
        "    def __init__(s):\n"
        "        super().__init__(); s.inner = Inner()\n"
        "    def forward(s, x):\n"
        "        return s.inner(x)\n",
    )
    res = tracer.trace_module(f)
    assert res["problems"] == [], res["problems"]
    outer = next(n for n in res["notes"] if n["label"] == "Outer.forward")
    assert "randn(2, 16, 8)" in outer["note"], res["notes"]


def test_structure_shape_reqs(tmp_path):
    # exact via tuple-unpack, min via transpose, free when the param is never pinned.
    f = _write(
        tmp_path,
        "reqs.py",
        "import torch\n"
        "def fn(x, y, z):\n"
        "    B, L, D = x.shape\n"
        "    w = y.transpose(1, 2)\n"
        "    return x, w, z\n",
    )
    s = structure.structure_file(f)
    reqs = {r["name"]: r for r in s["functions"][0]["shapeReqs"]}
    assert reqs["x"]["kind"] == "exact" and reqs["x"]["rank"] == 3, reqs
    assert "x.shape" in reqs["x"]["via"]
    assert reqs["y"]["kind"] == "min" and reqs["y"]["rank"] == 3, reqs
    assert reqs["z"]["kind"] == "free", reqs


def test_trace_module_emits_dim_symbols(tmp_path):
    # The abstract view needs a value->symbol map derived from each forward's input dims.
    f = _write(
        tmp_path,
        "dims.py",
        "import torch, torch.nn as nn\n"
        "class Seq(nn.Module):\n"
        "    def __init__(s):\n"
        "        super().__init__(); s.fc = nn.Linear(8, 8)\n"
        "    # fusion: input = torch.randn(2, 5, 8)\n"
        "    def forward(s, x):\n"
        "        return s.fc(x)\n",
    )
    res = tracer.trace_module(f)
    assert res["problems"] == [], res["problems"]
    # input (2, 5, 8) is rank-3 -> B, L, D
    assert res["dims"] == {"2": "B", "5": "L", "8": "D"}, res["dims"]


def test_trace_module_dim_symbols_from_model_attrs(tmp_path):
    # n_heads / head_dim are MODEL attributes, not input dims — they must still get
    # symbols (H / dh) so qkv[2, 6, 3, 4, 32] reads as qkv[B, L, 3, H, dh].
    f = _write(
        tmp_path,
        "attn.py",
        "import torch, torch.nn as nn\n"
        "class A(nn.Module):\n"
        "    def __init__(s):\n"
        "        super().__init__()\n"
        "        s.n_heads = 4; s.head_dim = 32\n"
        "        s.qkv = nn.Linear(128, 3 * 128)\n"
        "    # fusion: input = torch.randn(2, 6, 128)\n"
        "    def forward(s, x):\n"
        "        B, L, D = x.shape\n"
        "        qkv = s.qkv(x).reshape(B, L, 3, s.n_heads, s.head_dim)\n"
        "        return qkv\n",
    )
    res = tracer.trace_module(f)
    assert res["problems"] == [], res["problems"]
    d = res["dims"]
    assert d.get("4") == "H" and d.get("32") == "dh", d
    assert d.get("2") == "B" and d.get("6") == "L" and d.get("128") == "D", d


def test_op_notes_cover_common_pytorch_functions(tmp_path):
    # reshape / permute / unbind / cat / softmax / sum / sdpa all get per-line notes.
    f = _write(
        tmp_path,
        "opszoo.py",
        "import torch, torch.nn as nn\n"
        "import torch.nn.functional as F\n"
        "class Z(nn.Module):\n"
        "    def __init__(s):\n"
        "        super().__init__(); s.fc = nn.Linear(8, 8)\n"
        "    # fusion: input = torch.randn(2, 6, 8)\n"
        "    def forward(s, x):\n"
        "        B, L, D = x.shape\n"
        "        qkv = x.reshape(B, L, 2, 4)\n"          # 9: reshape
        "        q = qkv.permute(0, 2, 1, 3)\n"           # 10: permute
        "        p, r = qkv.unbind(dim=2)\n"              # 11: unbind, tuple target
        "        o = torch.cat([p, r], dim=-1)\n"         # 12: cat
        "        a = F.softmax(o, dim=-1)\n"              # 13: softmax
        "        m = a.sum(dim=1)\n"                      # 14: reduce
        "        w = q.reshape(B, 2, L, 4)\n"             # 15
        "        s2 = torch.matmul(w, w.transpose(-2, -1))\n"  # 16: matmul
        "        out = F.scaled_dot_product_attention(w, w, w)\n"  # 17: sdpa
        "        return out\n",
    )
    res = tracer.trace_module(f)
    assert res["problems"] == [], res["problems"]
    ops = res["ops"]
    text = " | ".join(ops.values())
    assert ops.get(9, "").startswith("reshape "), ops
    assert "permute(0, 2, 1, 3)" in ops.get(10, ""), ops
    assert "unbind(dim=2)" in ops.get(11, "") and "2 ×" in ops.get(11, ""), ops
    assert "cat(dim=-1)" in ops.get(12, ""), ops
    assert "softmax(dim=-1)" in ops.get(13, ""), ops
    assert "sum(dim=1)" in ops.get(14, ""), ops
    assert "matmul" in ops.get(16, ""), ops
    assert "sdpa" in ops.get(17, ""), ops
    assert "→" in text


def test_trace_function_returns_ops_and_dims(tmp_path):
    # Per-function ▶ trace gets the same op notes + dim symbols as the file trace.
    f = _write(
        tmp_path,
        "tf.py",
        "import torch, torch.nn as nn\n"
        "class N(nn.Module):\n"
        "    def __init__(s):\n"
        "        super().__init__()\n"
        "        s.n_heads = 2\n"
        "        s.fc = nn.Linear(8, 8)\n"
        "    # fusion: input = torch.randn(2, 6, 8)\n"
        "    def forward(s, x):\n"
        "        y = x.permute(0, 2, 1)\n"
        "        return y\n",
    )
    r = tracer.trace_function(f, "forward", 0)
    assert r["error"] is None, r["error"]
    assert any("permute(0, 2, 1)" in o for o in r["ops"].values()), r["ops"]
    assert r["dims"].get("6") == "L" and r["dims"].get("8") == "D", r["dims"]
    assert r["dims"].get("2") == "B", r["dims"]  # input B wins over n_heads=2 -> H


def test_structure_qualifies_methods_with_class():
    # Methods carry their enclosing class so the cockpit can tell N identically-named
    # `forward`s apart; module-level functions have className None.
    s = structure.structure_file(os.path.join(DATA, "demo_model.py"))
    fwd = next(f for f in s["functions"] if f["name"] == "forward")
    assert fwd.get("className"), s["functions"]


def test_trace_function_multiarg_method_reports(tmp_path):
    # Can only auto-synthesize ONE tensor; a method needing 2+ args must report, not crash.
    f = _write(
        tmp_path,
        "lib.py",
        "import torch, torch.nn as nn\n"
        "class E(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__(); self.a = nn.Linear(8, 8)\n"
        "    def step(self, x, y):\n"
        "        return self.a(x) + y\n",
    )
    records, err, crash, note = _tf(f, "step", 5)
    assert records == {} and "needs 2 args" in note


# --- input synthesis & directives (completion) ---------------------------------
def test_synth_embedding_uses_long_indices(tmp_path):
    # NLP model: first layer is Embedding -> input must be a LongTensor of indices,
    # NOT randn floats (which would crash). dtype-aware synth handles this.
    f = _write(
        tmp_path,
        "nlp.py",
        "import torch, torch.nn as nn\n"
        "class Tagger(nn.Module):\n"
        "    def __init__(s):\n"
        "        super().__init__(); s.emb = nn.Embedding(1000, 32); s.fc = nn.Linear(32, 5)\n"
        "    def forward(s, x):\n"
        "        h = s.emb(x)\n"
        "        return s.fc(h)\n",
    )
    records, err, crash, note = _tf(f, "forward", 0)
    assert err is None, err
    assert "randint(0, 1000" in note
    # embedding output [2, 16, 32] (a named local) is captured; floats would have crashed
    assert any(any(v["shape"] == [2, 16, 32] for v in rec.values()) for rec in records.values())


def test_directive_input_override_multiarg(tmp_path):
    # forward(x, mask) needs 2 args -> a `# fusion: input =` tuple supplies them (*args).
    f = _write(
        tmp_path,
        "m.py",
        "import torch, torch.nn as nn\n"
        "class M(nn.Module):\n"
        "    def __init__(s):\n"
        "        super().__init__(); s.fc = nn.Linear(8, 4)\n"
        "    # fusion: input = (torch.randn(2, 8), torch.ones(2, 8))\n"
        "    def forward(s, x, mask):\n"
        "        return s.fc(x * mask)\n",
    )
    records, err, crash, note = _tf(f, "forward", 0)
    assert err is None, err
    assert "randn(2, 8)" in note and "ones(2, 8)" in note
    assert any(any(v["shape"] == [2, 4] for v in rec.values()) for rec in records.values())


def test_directive_model_override_ctor_args(tmp_path):
    # A model needing constructor args -> `# fusion: model =` builds it for trace_module.
    f = _write(
        tmp_path,
        "cfg.py",
        "import torch, torch.nn as nn\n"
        "# fusion: model = Net(16)\n"
        "class Net(nn.Module):\n"
        "    def __init__(s, dim):\n"
        "        super().__init__(); s.fc = nn.Linear(dim, 4)\n"
        "    def forward(s, x):\n"
        "        return s.fc(x)\n",
    )
    res = tracer.trace_module(f)
    assert res["problems"] == [], res["problems"]
    assert any(n["note"].startswith("Net(16)") for n in res["notes"])
    assert any(any(v["shape"] == [2, 4] for v in rec.values()) for rec in res["records"].values())


def test_trace_module_keeps_module_registered(tmp_path):
    # A model that looks up sys.modules[__name__] during __init__ (like a model registry).
    # The exec'd module must stay registered THROUGH construction — not popped right after
    # exec — else sys.modules.get(__name__) is None -> 'NoneType' has no attribute __dict__.
    f = _write(
        tmp_path,
        "reg.py",
        "import sys, torch, torch.nn as nn\n"
        "class M(nn.Module):\n"
        "    def __init__(s):\n"
        "        super().__init__()\n"
        "        _ = sys.modules.get(__name__).__dict__  # None.__dict__ if not registered\n"
        "        s.fc = nn.Linear(8, 4)\n"
        "    def forward(s, x):\n"
        "        return s.fc(x)\n",
    )
    res = tracer.trace_module(f)
    assert res["problems"] == [], res["problems"]
    assert any(any(v["shape"] == [2, 4] for v in rec.values()) for rec in res["records"].values())


def test_trace_module_notes_matmul_and_broadcast(tmp_path):
    f = _write(
        tmp_path,
        "ops.py",
        "import torch, torch.nn as nn\n"
        "class M(nn.Module):\n"
        "    def __init__(s):\n"
        "        super().__init__(); s.q = nn.Linear(8, 8); s.k = nn.Linear(8, 8)\n"
        "    def forward(s, x):\n"
        "        a = s.q(x)\n"           # [2, 8]
        "        bt = s.k(x).t()\n"      # [8, 2]
        "        scores = a @ bt\n"      # matmul -> [2, 2]
        "        bias = torch.zeros(8)\n"  # [8]
        "        out = a + bias\n"       # broadcast [2, 8] + [8] -> [2, 8]
        "        return out\n",
    )
    ops = list(tracer.trace_module(f)["ops"].values())
    assert any("matmul" in o and "[8, 2]" in o for o in ops), ops
    assert any("broadcast" in o for o in ops), ops


def test_directive_input_in_trace_module(tmp_path):
    # trace_module honors a forward input directive (custom shape/dtype).
    f = _write(
        tmp_path,
        "d.py",
        "import torch, torch.nn as nn\n"
        "class E(nn.Module):\n"
        "    def __init__(s):\n"
        "        super().__init__(); s.emb = nn.Embedding(50, 8)\n"
        "    # fusion: input = torch.randint(0, 50, (3, 7))\n"
        "    def forward(s, ids):\n"
        "        return s.emb(ids)\n",
    )
    res = tracer.trace_module(f)
    assert res["problems"] == [], res["problems"]
    assert any("randint(0, 50, (3, 7))" in n["note"] for n in res["notes"])
    assert any(any(v["shape"] == [3, 7, 8] for v in rec.values()) for rec in res["records"].values())
