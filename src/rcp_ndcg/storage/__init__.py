"""Universal storage layer.

Every path in RCP-nDCG -- datasets, run artifacts, prompts, indexes, logs --
goes through here, so a local directory, a GCS bucket, an S3 bucket, an Azure
container, a HuggingFace dataset repo and an HTTP URL are interchangeable at
the call site::

    from rcp_ndcg import storage

    storage.read_text("runs/abc/manifest.json")
    storage.read_text("gs://bucket/runs/abc/manifest.json")
    storage.read_text("hf://datasets/org/runs/abc/manifest.json")

Backends come from fsspec.  ``gs://`` works out of the box (``gcsfs`` is a core
dependency); ``s3://``, ``az://`` and ``hf://`` need their extras, and asking
for one that is missing raises a :class:`~rcp_ndcg.errors.DependencyError` naming the extra.

Readers that need a real file on disk -- anything handing a path to a C library
-- call :func:`cache`, which downloads once and re-uses the copy until the
remote object changes.
"""

from rcp_ndcg.storage.cache import cache, cache_path_for
from rcp_ndcg.storage.core import (
    exists,
    filesystem,
    get,
    info,
    ls,
    makedirs,
    open_path,
    parent_of,
    read_bytes,
    read_text,
    relative,
    write_bytes,
    write_text,
)
from rcp_ndcg.storage.uri import (
    is_remote,
    join,
    local_dir,
    parent,
    protocol_of,
    split_protocol,
)

__all__ = [
    "cache",
    "cache_path_for",
    "exists",
    "filesystem",
    "get",
    "info",
    "is_remote",
    "join",
    "local_dir",
    "ls",
    "makedirs",
    "open_path",
    "parent",
    "parent_of",
    "protocol_of",
    "read_bytes",
    "read_text",
    "relative",
    "split_protocol",
    "write_bytes",
    "write_text",
]
