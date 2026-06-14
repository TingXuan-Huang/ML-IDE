"""Data-file loaders for the cockpit's Mode 2 (Data Viz).

Each returns a JSON-serializable meta dict with a BOUNDED row sample, so the helper
never streams a multi-GB file to the editor. Stats are computed lazily and capped.
Degrades gracefully: any failure returns kind="unknown" with a note, never raises.
"""
import os
from typing import Any, Dict

MAX_ROWS = 200          # rows transported for display
MAX_CELLS = 2_000_000   # guard against materializing a huge array for stats
MAX_TENSOR_BYTES = 256_000_000  # real-input trace: refuse to load a file bigger than this


def _to_f32(arr):
    """Cast float64 -> float32 (most layers expect float32); leave int/bool as-is so
    Embedding index inputs keep their long/int dtype."""
    import numpy as np

    return arr.astype("float32") if np.issubdtype(arr.dtype, np.floating) else arr


def load_tensor(path: str, root: str = "") -> Any:
    """Load a real data file into a torch tensor (or dict/tuple/object) for REAL-INPUT
    tracing — the `load("relpath")` callable exposed inside `# fusion:` directives.
    Resolves a relative path against `root` (the opened project folder, else the traced
    file's dir). Raises (FileNotFoundError / ValueError) on missing/too-big/unsupported —
    the caller surfaces it as a clean directive-failed note, never a crash."""
    p = path if os.path.isabs(path) else os.path.join(root or os.getcwd(), path)
    p = os.path.abspath(p)
    if not os.path.exists(p):
        raise FileNotFoundError(f"data file not found: {path}")
    size = os.path.getsize(p)
    if size > MAX_TENSOR_BYTES:  # checked BEFORE reading -> never OOMs the warm helper
        raise ValueError(f"file too big: {size} bytes > {MAX_TENSOR_BYTES} cap — slice it in the directive, e.g. load('{path}')[:4]")
    ext = os.path.splitext(p)[1].lower()
    import torch

    if ext == ".npy":
        import numpy as np

        return torch.from_numpy(_to_f32(np.load(p)))
    if ext == ".npz":
        import numpy as np

        # The on-disk getsize check above can't catch a COMPRESSED npz that decompresses to
        # GBs. Sum the uncompressed nbytes from each entry's .npy header (cheap — no full
        # decompress) and reject before materializing. Falls back to a permissive load if the
        # header API ever changes.
        try:
            import zipfile
            from numpy.lib import format as npfmt

            uncompressed = 0
            with zipfile.ZipFile(p) as zf:
                for nm in zf.namelist():
                    if not nm.endswith(".npy"):
                        continue
                    with zf.open(nm) as fh:
                        ver = npfmt.read_magic(fh)
                        shape, _fortran, dt = npfmt._read_array_header(fh, ver)
                        uncompressed += int(np.prod(shape)) * dt.itemsize
            if uncompressed > MAX_TENSOR_BYTES:
                raise ValueError(f"npz uncompresses to {uncompressed} bytes > {MAX_TENSOR_BYTES} cap — slice it in the directive")
        except (KeyError, AttributeError, OSError):
            pass  # header peek unavailable -> trust the getsize cap above
        z = np.load(p)
        return {k: torch.from_numpy(_to_f32(z[k])) for k in z.files}
    if ext in (".pt", ".pth"):
        # weights_only=True -> never unpickles arbitrary objects (no code execution from a
        # checkpoint). A full pickled model fails this and surfaces as a clean note. A
        # state_dict/dict is returned raw so the directive can index it: load("x.pt")["x"].
        obj = torch.load(p, map_location="cpu", weights_only=True)
        if isinstance(obj, torch.Tensor) and obj.dtype == torch.float64:
            return obj.float()
        return obj
    if ext in (".csv", ".tsv"):
        import pandas as pd

        df = pd.read_csv(p, sep="\t" if ext == ".tsv" else ",")
        num = df.select_dtypes(include="number")
        if num.shape[1] == 0:
            raise ValueError("csv has no numeric columns to make a tensor")
        return torch.tensor(num.values.astype("float32"))
    raise ValueError(f"unsupported data extension '{ext}' (use .npy / .npz / .pt / .pth / .csv)")


def load_file(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {"kind": "unknown", "path": path, "note": "file not found"}
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".npy":
            return _load_npy(path)
        if ext in (".csv", ".tsv"):
            return _load_table(path, sep="\t" if ext == ".tsv" else ",")
        if ext in (".parquet", ".pq"):
            return _load_parquet(path)
        return {"kind": "unknown", "path": path, "note": f"unsupported extension '{ext}'"}
    except Exception as e:  # degrade, never crash the helper
        return {"kind": "unknown", "path": path, "note": f"load error: {type(e).__name__}: {e}"}


def _load_npy(path: str) -> Dict[str, Any]:
    import numpy as np

    arr = np.load(path, mmap_mode="r")  # don't pull the whole array into RAM
    flat = np.asarray(arr).reshape(-1)
    n = int(min(flat.size, MAX_ROWS))
    stats: Dict[str, float] = {}
    if np.issubdtype(arr.dtype, np.number):
        sub = flat[: int(min(flat.size, MAX_CELLS))]
        stats = {"min": float(sub.min()), "max": float(sub.max()), "mean": float(sub.mean())}
    return {
        "kind": "ndarray",
        "path": path,
        "shape": [int(d) for d in arr.shape],
        "dtype": str(arr.dtype),
        "rowSample": n,
        "sample": [float(x) for x in flat[:n].tolist()],
        "stats": stats,
    }


def _load_table(path: str, sep: str = ",") -> Dict[str, Any]:
    import pandas as pd

    df = pd.read_csv(path, sep=sep, nrows=MAX_ROWS)
    cols = [
        {"name": str(c), "dtype": str(df[c].dtype), "nulls": int(df[c].isna().sum())}
        for c in df.columns
    ]
    sample = df.where(pd.notnull(df), None).values.tolist()
    return {
        "kind": "table",
        "path": path,
        "columns": cols,
        "header": [str(c) for c in df.columns],
        "rowSample": int(min(len(df), MAX_ROWS)),
        "sample": sample,
    }


def _load_parquet(path: str) -> Dict[str, Any]:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    schema = pf.schema_arrow
    cols = [
        {"name": schema.field(i).name, "dtype": str(schema.field(i).type), "nulls": 0}
        for i in range(len(schema))
    ]
    header = [c["name"] for c in cols]
    try:
        batch = next(pf.iter_batches(batch_size=MAX_ROWS))
        d = batch.to_pydict()
        rows = [list(r) for r in zip(*[d[h] for h in header])] if header else []
    except StopIteration:
        rows = []
    return {
        "kind": "table",
        "path": path,
        "columns": cols,
        "header": header,
        "rowSample": len(rows),
        "sample": rows,
    }
