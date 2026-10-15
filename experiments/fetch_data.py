"""Download the public data behind the paper's tables from the Hugging Face Hub.

Every file comes from a public dataset repository, pinned to the revision the checks were run against, and is
fetched without authentication (no token is read or sent). Files land under ``<data-dir>/<repo>/<revision>/<path>``;
a file that is already there is not fetched again. The directory is keyed by the revision, so files fetched at
another revision (release 1, which carries arXiv v1's numbers), or by an earlier version of this script (which
stored ``<repo>/<path>``), are never read for these checks.

Usage::

    python experiments/fetch_data.py                   # everything the checks need (about 150 MB)
    python experiments/fetch_data.py leaderboards      # only one group
    python experiments/fetch_data.py --data-dir /tmp/rcp-data

The data directory defaults to ``experiments/data`` and can be set with ``RCP_EXPERIMENTS_DATA``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

__all__ = ["DATASETS", "GROUPS", "data_dir", "fetch", "fetch_group", "local_path"]

OWNER = "fabianschmidt-cohere"

# repo name -> revision (commit sha) the published checks were run against: each dataset's release 2 (arXiv v2).
DATASETS: dict[str, str] = {
    "rcp-ndcg-nanobeir": "cb6c981c80a70b6a46f8b7ba3a60206a9fab5d8e",
    "rcp-ndcg-bright": "b827417851d3cfe997a65203b54c73a0fc885568",
    "rcp-ndcg-vidore-v3": "c85f096ff9c2b66395100a9795274bbdf087a9b0",
    "rcp-ndcg-trecdl": "ee65bffbd2dda46c1204e767118d428f32c3cfaf",
    "rcp-ndcg-external-validation": "8c8dda6ece90906d490c31b03ce9a8bb56cbd871",
}

NANOBEIR_SUBSETS = [
    "NanoArguAnaRetrieval",
    "NanoClimateFeverRetrieval",
    "NanoDBPediaRetrieval",
    "NanoFEVERRetrieval",
    "NanoFiQA2018Retrieval",
    "NanoHotpotQARetrieval",
    "NanoMSMARCORetrieval",
    "NanoNFCorpusRetrieval",
    "NanoNQRetrieval",
    "NanoQuoraRetrieval",
    "NanoSCIDOCSRetrieval",
    "NanoSciFactRetrieval",
    "NanoTouche2020Retrieval",
]
BRIGHT_SUBSETS = [
    "aops",
    "biology",
    "earth_science",
    "economics",
    "leetcode",
    "pony",
    "psychology",
    "robotics",
    "stackoverflow",
    "sustainable_living",
    "theoremqa_questions",
    "theoremqa_theorems",
]
BRIGHT_WITH_EXCLUSIONS = ["aops", "leetcode", "theoremqa_questions"]
# ViDoRe v3 domain -> language its questions were written in (the paper scores each question once, natively).
VIDORE_NATIVE = {
    "computer_science": "english",
    "energy": "french",
    "finance_en": "english",
    "finance_fr": "french",
    "hr": "english",
    "industrial": "english",
    "pharmaceuticals": "english",
    "physics": "french",
}
TRECDL_SUBSETS = ["trec_dl_2019", "trec_dl_2020"]


def _leaderboard_files() -> dict[str, list[str]]:
    nano = [f"{s}/{f}.parquet" for s in NANOBEIR_SUBSETS for f in ("qrels", "top_ranked")]
    nano += [f"provenance/runs/{s}.parquet" for s in NANOBEIR_SUBSETS]
    bright = [f"{s}/qrels.parquet" for s in BRIGHT_SUBSETS]
    bright += [f"{s}/excluded.parquet" for s in BRIGHT_WITH_EXCLUSIONS]
    bright += [f"provenance/runs/{s}.parquet" for s in BRIGHT_SUBSETS]
    vidore = [f"{d}__{lang}/{f}.parquet" for d, lang in VIDORE_NATIVE.items() for f in ("qrels", "top_ranked")]
    vidore += [f"provenance/runs/{d}.parquet" for d in VIDORE_NATIVE]
    trec = [f"{s}/{f}.parquet" for s in TRECDL_SUBSETS for f in ("qrels", "top_ranked")]
    trec += [f"provenance/runs/{s}.parquet" for s in TRECDL_SUBSETS]
    return {
        "rcp-ndcg-nanobeir": nano,
        "rcp-ndcg-bright": bright,
        "rcp-ndcg-vidore-v3": vidore,
        "rcp-ndcg-trecdl": trec,
    }


_VALIDATION_FILES = {
    "rcp-ndcg-external-validation": [
        "human_study/contests.parquet",
        "human_study/reviews.parquet",
        "human_study/grades.parquet",
        "blind_comparisons/external_judges_cases.parquet",
        "blind_comparisons/external_judges_verdicts.parquet",
        "blind_comparisons/study1_cases.parquet",
        "blind_comparisons/study1_external_panel_verdicts.parquet",
    ]
}

# group name -> {repo: [paths]}
GROUPS: dict[str, dict[str, list[str]]] = {
    "leaderboards": _leaderboard_files(),
    "validation": _VALIDATION_FILES,
}


def data_dir(override: str | Path | None = None) -> Path:
    """Directory that holds the downloaded files (``--data-dir``, else ``RCP_EXPERIMENTS_DATA``, else
    ``experiments/data``)."""
    if override:
        return Path(override)
    env = os.environ.get("RCP_EXPERIMENTS_DATA")
    return Path(env) if env else Path(__file__).resolve().parent / "data"


def local_path(repo: str, path: str, root: Path | None = None) -> Path:
    """Local path of ``path`` from dataset ``repo`` at its pinned revision (it may not exist yet).

    The path is keyed by the revision (``<data-dir>/<repo>/<revision>/<path>``): a file fetched at another revision
    is never read in its place.
    """
    return (root or data_dir()) / repo / DATASETS[repo] / path


def fetch(repo: str, paths: list[str], root: Path | None = None) -> list[Path]:
    """Download ``paths`` of the public dataset ``repo`` at its pinned revision; return their local paths.

    Files already present are kept. Requires ``huggingface_hub``; the download needs no token.
    """
    root = root or data_dir()
    missing = [p for p in paths if not local_path(repo, p, root).is_file()]
    if missing:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise SystemExit(
                "huggingface_hub is needed to download the data: pip install -r experiments/requirements.txt"
            ) from exc
        for p in missing:
            print(f"  fetching {repo}/{p}", file=sys.stderr, flush=True)
            hf_hub_download(
                repo_id=f"{OWNER}/{repo}",
                filename=p,
                repo_type="dataset",
                revision=DATASETS[repo],
                local_dir=root / repo / DATASETS[repo],
                token=False,
            )
    return [local_path(repo, p, root) for p in paths]


def fetch_group(group: str, root: Path | None = None) -> None:
    """Download every file of one group (``leaderboards`` or ``validation``)."""
    for repo, paths in GROUPS[group].items():
        fetch(repo, paths, root)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("groups", nargs="*", help=f"any of {', '.join(GROUPS)} (default: all)")
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args(argv)
    unknown = sorted(set(args.groups) - set(GROUPS))
    if unknown:
        ap.error(f"unknown group(s): {', '.join(unknown)}")
    root = data_dir(args.data_dir)
    for g in args.groups or list(GROUPS):
        print(f"[{g}] -> {root}", file=sys.stderr)
        fetch_group(g, root)
    print("done", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
