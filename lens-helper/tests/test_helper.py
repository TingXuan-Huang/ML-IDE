"""Unit tests for lens-helper. Run: cd lens-helper && python -m pytest tests/ -q"""
import os

import torch
import torch.nn as nn

from lens_helper import callgraph, loaders, structure, tracer

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "spike", "sampledata")


def _write(tmp_path, name, src):
    p = tmp_path / name
    p.write_text(src)
    return str(p)


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
    records, err, crash, note = tracer.trace_function(f, "forward", 7)
    assert err is None, err
    assert "E().forward(randn(2, 16))" == note
    assert any(any(v["shape"] == [2, 64] for v in rec.values()) for rec in records.values())
    assert any("return" in rec and rec["return"]["shape"] == [2, 8] for rec in records.values())


def test_trace_function_top_level_zero_arg(tmp_path):
    f = _write(tmp_path, "u.py", "import torch\ndef make():\n    return torch.randn(4, 3)\n")
    records, err, crash, note = tracer.trace_function(f, "make", 2)
    assert err is None and note == "make()"
    assert any("return" in rec and rec["return"]["shape"] == [4, 3] for rec in records.values())


def test_trace_function_needs_args_reports(tmp_path):
    f = _write(tmp_path, "u.py", "def f(a, b):\n    return a + b\n")
    records, err, crash, note = tracer.trace_function(f, "f", 1)
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
    records, err, crash, note = tracer.trace_function(f, "__init__", 3)
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
    records, err, crash, note = tracer.trace_function(f, "step", 5)
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
    records, err, crash, note = tracer.trace_function(f, "forward", 0)
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
    records, err, crash, note = tracer.trace_function(f, "forward", 0)
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
