"""lens-helper — the warm Python sidecar for the Fusion IDE cockpit.

Public API:
  loaders.load_file(path)            -> Mode 2 data preview (bounded)
  tracer.trace_callable(fn, file)    -> Mode 1 per-line tensor shapes
  rpc.main()                         -> JSON-RPC over stdio (python -m lens_helper)

Built to be EXTRACTABLE (CEO cherry-pick 1): depends only on stdlib + numpy/pandas/pyarrow,
never on the cockpit/extension. Can ship as its own pip package later.
"""
from . import loaders, rpc, tracer

__all__ = ["loaders", "rpc", "tracer"]
__version__ = "0.0.1"
