"""Data-file loaders for the cockpit's Mode 2 (Data Viz).

Each returns a JSON-serializable meta dict with a BOUNDED row sample, so the helper
never streams a multi-GB file to the editor. Stats are computed lazily and capped.
Degrades gracefully: any failure returns kind="unknown" with a note, never raises.
"""
import os
from typing import Any, Dict

MAX_ROWS = 200          # rows transported for display
MAX_CELLS = 2_000_000   # guard against materializing a huge array for stats


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
