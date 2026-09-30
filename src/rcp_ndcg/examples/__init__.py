"""The examples that ship with the package: the tiny dataset and named run configs.

* ``tiny/``: a three-query dataset (``rows.jsonl``, eight documents per query with labels), two systems'
  rankings (``systems.jsonl``: ``bm25`` and ``reranker``) and a released-style calibration (``released.json``:
  item parameters and thetas). :func:`tiny` is its directory; ``rcp-ndcg data fetch --dataset tiny --out DIR``
  copies it.
* ``runs/<name>.yaml``: run configs, started by name (``rcp-ndcg run start tiny``, ``rcp.run("tiny")``):
  ``tiny`` (the whole pipeline on the tiny dataset with the offline fake judge), ``rejudge_nfcorpus`` (re-judge
  the released NanoNFCorpus pools with your own judge) and ``nano_nfcorpus_gpt5`` (BM25 retrieval and a hosted
  judge on NanoNFCorpus). Paths inside a run config are relative to the config file.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

from rcp_ndcg.errors import MissingInputError


def _root() -> Path:
    return Path(str(files(__name__)))


def tiny() -> Path:
    """The directory of the tiny example dataset (``rows.jsonl``, ``systems.jsonl``, ``released.json``)."""
    return _root() / "tiny"


def run_config_names() -> list[str]:
    """The names of the packaged run configs, sorted."""
    return sorted(path.stem for path in (_root() / "runs").glob("*.yaml"))


def run_config_path(name: str) -> Path:
    """The file of the packaged run config ``name``.

    Raises:
        MissingInputError: No packaged run config has that name.
    """
    path = _root() / "runs" / f"{name}.yaml"
    if not path.is_file():
        raise MissingInputError(
            f"{name!r} is neither a run config file nor a packaged run config",
            hint=f"pass a YAML path, or one of the packaged configs: {', '.join(run_config_names())}",
            details={"config": name, "packaged": run_config_names()},
        )
    return path


__all__ = ["run_config_names", "run_config_path", "tiny"]
