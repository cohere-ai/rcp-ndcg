"""The engine pins: every tag citation resolves at ``v0.33.0``, and no stale ``0.31`` claim survives.

Two sweeps the vLLM 0.31 -> 0.33 migration ran by hand are pinned here, so a later edit cannot reintroduce a
stale claim silently.  Both re-implement the migration gate's own measurer, so their records are comparable with
the counts that gate reported.

**The citation sweep.**  A *logical line* joins a YAML ``sources:`` item with its continuation lines; a line is
*tag-named* when it names the tag (``v0.33.0``, ``at the tag``, ``the tag``, ``vLLM v0.33``, ``vllm tag``); a
*citation* is ``path:line[-line]`` plus every bare ``:line[-line]`` continuation of the same file; the *claim* is
the parenthetical that contains the citation, else the clause up to the next citation / ``;`` / `` -- ``; and the
claim's *identifiers* are its snake_case and CamelCase tokens, minus the cited path's own.  Every citation must
resolve at the tag, its range must lie inside the tag's file, and one of the claim's identifiers must occur
inside the cited lines.  A citation is judged only when its claim names an identifier: a prose claim has nothing
to check, and guessing one would invent the defect.

Two records carry the rows a previous pass read and judged, and each is exact in **both** directions: a row that
is not in the record fails, and a recorded row that heals also fails until the entry is deleted.  A record is
therefore a declaration, never a silent filter.

* :data:`UNRESOLVABLE` -- the cited path is a HuggingFace checkpoint card or config, a transformers or torch
  file, this repository's own path, an abbreviation (``.../dir/file.py``), or a bare basename whose directory is
  not on the line; ``git ls-tree`` cannot answer for any of them.  One entry is a range past the end of the
  tag's file (``requirements/common.txt:98-108``, a bare continuation of a transformers path).
* :data:`DRIFT` -- the claim's identifier belongs to a sibling clause of the same parenthetical, or sits within
  two lines of the cited range, or the line holds the construct while its prose names another one (recorded, not
  renumbered).

The mutation each test catches:

* :func:`test_every_tag_citation_resolves_at_the_tag` -- a citation edited to a line that no longer holds its
  construct (the campaign's recorded defect: ``renderers/params.py:183`` for the tag's ``:199``), a citation of
  a path no tag tree has, a range past the end of the file.
* :func:`test_no_stale_engine_version_survives` -- a ``0.31`` claim reintroduced into a live surface: the
  guard's ``[0.31, 0.32)`` range, a module docstring's "v0.31.x", a recipe's ``min_version``, a schema's prose.
  ``0.31.0`` alone is not enough: ``v0.31.x`` and ``[0.31, 0.32)`` carry no ``.0``.

The tag checkout is not in the repository (``.deps/vllm`` is gitignored).  ``RCP_NDCG_VLLM_CHECKOUT`` names it
and defaults to ``.deps/vllm``; the citation sweep skips with a named reason when it is absent, so the contract
suite runs offline without it.  The version sweep needs no checkout and always runs.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

from tests.contract.surface import REPO

TAG = "v0.33.0"
"""The engine tag the citations are read at: the shipped image's tag, ``vllm/vllm-openai:v0.33.0``."""

CLONE_ENV = "RCP_NDCG_VLLM_CHECKOUT"
"""The environment variable that names the tag checkout; ``.deps/vllm`` when unset."""

DEFAULT_CLONE = REPO / ".deps" / "vllm"

#: The swept surface: the three ``src`` trees (the recipes live under the serving package's), the docs, the
#: exported schemas, and the loose files -- the root NOTICE and the four distributions' NOTICEs, plus the
#: reproducibility record.
SWEPT_ROOTS = ("docs", "rcp/src", "rcp-ndcg-test/src", "rcp-ndcg-vllm/src", "schemas")
SWEPT_FILES = (
    "NOTICE",
    "rcp/NOTICE",
    "rcp-core/NOTICE",
    "rcp-ndcg-test/NOTICE",
    "rcp-ndcg-vllm/NOTICE",
    "REPRODUCIBILITY.md",
)

_CITE = re.compile(r"((?:vllm/)?[A-Za-z0-9_./-]+\.(?:py|md|json|jinja|txt|yaml)):(\d+)(?:-(\d+))?")
_BARE = re.compile(r"(?<![\w.]):(\d+)(?:-(\d+))?")
_TAGISH = re.compile(r"v0\.33\.0|at the tag|vllm tag|vLLM v0\.33|the tag", re.IGNORECASE)
_SNAKE = re.compile(r"\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b")
_CAMEL = re.compile(r"\b([A-Z][a-z0-9]+(?:[A-Z][a-z0-9]*)+)\b")

#: Tokens that are ordinary English or YAML keys in these claims, not engine symbols.
_STOP = frozenset(
    {
        "per_group",
        "text_config",
        "hf_config",
        "the_tag",
        "vllm_v",
        "add_argument",
        "max_model_len",
        "known_deviations",
        "anchor_drop_over_cap",
        "trust_remote_code",
        "auto_map",
    }
)

#: ``0.31`` as an engine version: not a digit or a dot before it (``110.312`` is a coordinate), not a digit
#: after it (``0.31906667`` is a score), not an arithmetic operand (``= 0.31``, ``+ 0.31 = 2.31``) and not a
#: quantity (``0.31 GiB``, ``0.31 ms``).  ``v0.31.x`` and ``[0.31, 0.32)`` both match; ``0.31.0`` alone misses
#: them, which is the defect this sweep exists for.
_STALE = re.compile(r"(?<![\d.])(?<![=+]\s)0\.31(?!\d|\s*(?:GiB|MiB|KiB|Gi|Mi|Ki|GB|MB|KB|B|ms|s|%)\b)")

#: The paths a tag citation may name without the tag resolving them, by cause: a HuggingFace checkpoint card or
#: config (``README.md``, ``config*.json``, ``modeling_*.py``, the checkpoint's own code), a transformers or
#: torch file, this repository's own path, an abbreviation (``.../``) or a bare basename whose directory is not
#: on the line.  ``requirements/common.txt`` is here because its row is a range past the tag file's end: the
#: ``:98-108`` continuation belongs to the transformers path cited before it.
UNRESOLVABLE = frozenset(
    {
        ".../seqwise/heads.py",
        "README.md",
        "config.json",
        "config_sentence_transformers.json",
        "configuration_pplx_contextual.py",
        "experiments/paper/README.md",
        "flashinfer/gdn_kernels/blackwell/gdn_prefill.py",
        "modeling_jina_embeddings_v5.py",
        "modeling_topk_embed.py",
        "modeling_zembed.py",
        "models/gemma4/image_processing_gemma4.py",
        "models/gemma4/processing_gemma4.py",
        "models/gemma4_unified/image_processing_gemma4_unified.py",
        "models/qwen2_vl/image_processing_qwen2_vl.py",
        "models/utils.py",
        "processing_utils.py",
        "processor_config.json",
        "reference/qwen3.py",
        "registry.py",
        "requirements/common.txt",
        "retrieval/cross_encoder.py",
        "retrieval/external_rerankers.py",
        "sentence_bert_config.json",
        "topk_embed_st.py",
        "torch/nn/attention/flex_attention.py",
    }
)

#: The tag citations whose claim identifier is not inside the cited lines, read and judged one by one, as
#: ``{file: {line: (row, ...)}}`` with each row ``<cited path>:<start>-<hi> <NEAR|MISSING>``.  ``NEAR`` is
#: within two lines (the anchor moved with a neighbouring edit); ``MISSING`` is nowhere in the cited range,
#: which is a sibling clause's identifier: the claim is one parenthetical over several citations, so a token
#: belonging to one clause is tested against every citation in it.  The two rows that were *real* defects
#: rather than artefacts -- ``equivalence/stages.py``'s ``io_processor`` range and ``harrier-oss-v1``'s
#: "decode-only prompt path" -- are fixed at the tag's lines and are deliberately not recorded here.
DRIFT = frozenset(
    f"{path}:{line} {row}"
    for path, rows_by_line in {
        "rcp-ndcg-vllm/src/rcp_ndcg_vllm/models/pplx/late.py": {32: ("mimo_audio.py:1262-1272 MISSING",)},
        "rcp-ndcg-vllm/src/rcp_ndcg_vllm/models/pplx/late_data.py": {
            18: ("model_loader/default_loader.py:230-230 MISSING",)
        },
        "rcp-ndcg-vllm/src/rcp_ndcg_vllm/models/topk/pooling.py": {8: ("activations.py:108-110 MISSING",)},
        "rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/ctxl-rerank-v2-instruct-multilingual/family.yaml": {
            146: ("pooling_params.py:182-182 NEAR",)
        },
        "rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/embeddinggemma-2/family.yaml": {
            275: ("vllm/model_executor/models/embedding_gemma2.py:707-721 MISSING",)
        },
        "rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/gemma-4-12b/family.yaml": {
            94: ("reasoning/__init__.py:55-58 MISSING",)
        },
        "rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/harrier-oss-v1/family.yaml": {
            134: (
                "vllm/model_executor/models/registry.py:1341-1360 MISSING",
                "vllm/model_executor/models/registry.py:1402-1402 MISSING",
                "vllm/model_executor/models/registry.py:1456-1456 MISSING",
            ),
            35: ("vllm/config/model.py:2356-2356 MISSING",),
        },
        "rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/pplx-embed-v1/family.yaml": {
            22: ("vllm/config/model.py:777-804 MISSING",)
        },
        "rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/pplx-embed-v2-context/family.yaml": {
            138: ("vllm/config/pooler.py:33-33 MISSING",)
        },
        "rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/qwen3-reranker/family.yaml": {
            117: ("vllm/renderers/hf.py:759-845 NEAR", "vllm/renderers/hf.py:847-914 MISSING"),
            132: ("vllm/entrypoints/pooling/scoring/io_processor.py:782-782 MISSING",),
            137: ("vllm/model_executor/models/adapters.py:569-571 MISSING",),
        },
        "rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/qwen3-vl-reranker/family.yaml": {
            12: (
                "config/model.py:2347-2347 MISSING",
                "config/pooler.py:60-60 MISSING",
                "vllm/model_executor/models/adapters.py:436-437 MISSING",
                "vllm/multimodal/processing/context.py:376-376 MISSING",
                "vllm/multimodal/processing/context.py:57-120 MISSING",
            ),
            139: ("scoring/io_processor.py:807-818 NEAR",),
            152: ("vllm/multimodal/processing/context.py:57-120 MISSING",),
        },
        "rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/qwen3.8-flash-next/family.yaml": {
            92: (
                "config/model.py:1382-1403 MISSING",
                "model_executor/layers/quantization/modelopt.py:780-780 MISSING",
            )
        },
        "rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/topk-embed-v1/family.yaml": {
            21: ("qwen3_next.py:345-350 MISSING",),
            203: ("vllm/model_executor/models/utils.py:441-448 MISSING",),
            209: ("vllm/config/pooler.py:66-66 MISSING",),
        },
        "rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/zembed-1/family.yaml": {
            95: ("vllm/transformers_utils/config.py:1012-1095 MISSING",)
        },
        "rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/zerank/family.yaml": {
            # 97 is the `variants:` logical line: the variant's status block now names the tag, so the notes'
            # citation is judged; `is_original_qwen3_reranker` sits two lines past the cited range.
            97: ("vllm/entrypoints/pooling/scoring/io_processor.py:460-476 NEAR",),
            129: ("vllm/model_executor/layers/pooler/activations.py:140-140 NEAR",),
        },
        "rcp/src/rcp/data/resolution.py": {1473: ("vidcom2.py:19-40 MISSING",)},
        "rcp/src/rcp/inference/adapters/embeddings.py": {320: ("vllm/renderers/hf.py:847-914 MISSING",)},
    }.items()
    for line, rows in rows_by_line.items()
    for row in rows
)

#: The files whose ``0.31`` is a record, not a claim: the release history, the GPU ledger (every row was
#: recorded on the 0.31 image), the corpora state (the committed recordings are still ``vllm-0.31.0``), the
#: 0.31 status table the shakedown recordings measured, the two prose mentions of the images the waves
#: actually ran, and the fold's history comments.  A live surface -- a guard's range, a module docstring's
#: target, a schema, a NOTICE -- is not here, and a recipe is handled by :func:`_recorded`.
RECORDED_0_31 = {
    "CHANGELOG.md": "the release history",
    "docs/how-to/serve-a-model.md": "the pooling-full-context backport, a fix a 0.31 engine needed",
    "docs/how-to/use-verified-fake-engines.md": "the committed corpora are still the vllm-0.31.0 recording",
    "docs/reference/gpu-validation.md": "the GPU ledger: every row was recorded on the 0.31 image",
    "docs/reference/recipes.md": "the embeddinggemma-2 backport, in prose",
    "rcp-ndcg-test/src/rcp_ndcg_test/changes.py": "the corpora state: still the vllm-0.31.0 recording",
    "rcp-ndcg-test/src/rcp_ndcg_test/corpus.py": "the 0.31 recordings predate the record schema's version stamp",
    "rcp-ndcg-test/src/rcp_ndcg_test/jobs/wave-cosines-topk-small.json": "the wave's own image record",
    "rcp-ndcg-test/src/rcp_ndcg_test/observe/corpus.py": "the v0.31.0 status table the shakedown measured",
    "rcp-ndcg-vllm/src/rcp_ndcg_vllm/models/embedding_gemma2/__init__.py": "the fold's history comment",
}


def _swept_files(extra: tuple[str, ...] = ()) -> list[Path]:
    """The swept files, repository-relative and sorted: the four trees' files plus the loose ones.

    Generated trees are not swept: ``__pycache__`` and an editable install's ``*.egg-info`` are build output,
    and an egg-info's ``PKG-INFO`` embeds a copy of a README that no one edits.
    """
    files = {REPO / name for name in (*SWEPT_FILES, *extra)}
    for root in SWEPT_ROOTS:
        files.update(
            path
            for path in (REPO / root).rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and not any(part.endswith(".egg-info") for part in path.parts)
        )
    return sorted(files)


def _relative(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


def _identifiers(claim: str) -> list[str]:
    """The claim's code-shaped identifiers, in order, without repeats: snake_case and CamelCase tokens."""
    found = [match for match in _SNAKE.findall(claim) if match not in _STOP]
    found += _CAMEL.findall(claim)
    return list(dict.fromkeys(found))


def _claim(line: str, start: int, end: int, following: int) -> str:
    """The text a citation claims: its containing parenthetical, else the clause to the next citation."""
    opened = line.rfind("(", 0, start)
    if opened != -1 and line.rfind(")", 0, start) < opened:
        closed = line.find(")", end)
        return line[opened + 1 : closed if closed != -1 else following]
    segment = line[end:following]
    for separator in (";", "),", " -- "):
        cut = segment.find(separator)
        if cut != -1:
            segment = segment[:cut]
    return segment


def _logical_lines(text: str) -> list[tuple[int, str]]:
    """Join a YAML ``sources:`` item with its continuation lines, so a tag named on the item covers them."""
    joined: list[tuple[int, str]] = []
    number, buffer = 0, []
    for current, line in enumerate(text.splitlines(), 1):
        stripped = line.lstrip()
        if stripped.startswith(('- "', "- '")) or not line.startswith((" ", "\t")):
            if buffer:
                joined.append((number, " ".join(buffer)))
            number, buffer = current, [line]
        else:
            buffer.append(line.strip())
    if buffer:
        joined.append((number, " ".join(buffer)))
    return joined


def _citations(line: str) -> list[tuple[int, int, str, int, int]]:
    """Every ``path:line[-line]`` on the line plus every bare ``:line[-line]`` continuing the same file."""
    explicit = list(_CITE.finditer(line))
    found = [(m.start(), m.end(), m.group(1), int(m.group(2)), int(m.group(3) or m.group(2))) for m in explicit]
    previous, cursor = None, 0
    for match in explicit:
        if previous is not None:
            for bare in _BARE.finditer(line, cursor, match.start()):
                found.append(_bare_row(bare, previous))
        previous, cursor = match.group(1), match.end()
    if previous is not None:
        for bare in _BARE.finditer(line, cursor):
            found.append(_bare_row(bare, previous))
    return sorted(found)


def _bare_row(match: re.Match[str], cited: str) -> tuple[int, int, str, int, int]:
    """A bare ``:line[-line]`` continuation of ``cited``: its end defaults to its start."""
    low = int(match.group(1))
    return match.start(), match.end(), cited, low, int(match.group(2) or low)


def _resolved(path: str, tag_files: set[str]) -> str | None:
    """The tag path a citation names: an exact path, a ``vllm/``-prefixed one, or a unique directory suffix."""
    if path.startswith("vllm/"):
        return path if path in tag_files else None
    if path in tag_files:
        return path
    if f"vllm/{path}" in tag_files:
        return f"vllm/{path}"
    hits = [name for name in tag_files if name.endswith("/" + path)]
    return hits[0] if len(hits) == 1 else None


def _git(clone: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    """A read-only git call in the tag checkout: no network, no writes."""
    return subprocess.run(["git", "-C", str(clone), *arguments], capture_output=True, text=True, check=False)


def _tag_files(clone: Path) -> set[str]:
    """Every path the tag carries, or the empty set when the checkout does not have the tag."""
    return set(_git(clone, "ls-tree", "-r", "--name-only", TAG).stdout.split())


def _recorded(path: str, in_engine: bool) -> bool:
    """Whether a ``0.31`` in ``path`` is a record rather than a claim about the engine this tree targets.

    A file in :data:`RECORDED_0_31` is a record whole.  In a recipe the ``engine:`` mapping is the live claim
    (the image it declares and its ``min_version`` floor) and everything else is the history of the waves that
    ran: the ``status:`` block, the ``notes:``/``sources:`` prose and the comments.  The mapping's values are
    also pinned semantically by the serving package's ``test_engine_pin.py``.
    """
    if path in RECORDED_0_31:
        return True
    if not path.startswith("rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/"):
        return False
    return not in_engine


def _swept_citations(clone: Path, tag_files: set[str]) -> tuple[set[str], set[str]]:
    """The citation sweep: ``(drift rows, unresolvable paths)``, in the measurer's row format."""
    shown: dict[str, list[str]] = {}
    drift: set[str] = set()
    unresolvable: set[str] = set()
    for path in _swept_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        by_base: dict[str, set[str]] = {}
        for match in _CITE.finditer(text):
            candidate = match.group(1)
            if "/" in candidate and not candidate.startswith("..."):
                full = _resolved(candidate, tag_files)
                if full:
                    by_base.setdefault(candidate.rsplit("/", 1)[-1], set()).add(full)
        for number, line in _logical_lines(text):
            if not _TAGISH.search(line):
                continue
            citations = _citations(line)
            for index, (start, end, cited, low, high) in enumerate(citations):
                following = citations[index + 1][0] if index + 1 < len(citations) else len(line)
                claim = _claim(line, start, end, following)
                claimed = [name for name in _identifiers(claim) if name.lower() not in cited.lower()]
                if "/" not in cited:
                    same = {
                        match.group(1)
                        for match in _CITE.finditer(line)
                        if "/" in match.group(1)
                        and not match.group(1).startswith("...")
                        and match.group(1).rsplit("/", 1)[-1] == cited
                        and _resolved(match.group(1), tag_files)
                    }
                    if len(same) == 1:
                        cited = same.pop()
                    elif len(by_base.get(cited, ())) == 1:
                        cited = next(iter(by_base[cited]))
                real = _resolved(cited, tag_files)
                if real is None:
                    unresolvable.add(cited)
                    continue
                if real not in shown:
                    shown[real] = _git(clone, "show", f"{TAG}:{real}").stdout.splitlines()
                source = shown[real]
                if high > len(source):
                    unresolvable.add(cited)
                    continue
                if not claimed:
                    continue
                window = "\n".join(source[low - 1 : high])
                if any(name in window for name in claimed):
                    continue
                near = "\n".join(source[max(0, low - 3) : min(len(source), high + 2)])
                verdict = "NEAR" if any(name in near for name in claimed) else "MISSING"
                drift.add(f"{_relative(path)}:{number} {cited}:{low}-{high} {verdict}")
    return drift, unresolvable


def test_every_tag_citation_resolves_at_the_tag() -> None:
    """Every tag-named citation resolves at ``v0.33.0`` and its claim's identifier is inside the cited lines.

    Catches a citation edited to a line that no longer holds its construct, a path no tag tree has, and a range
    past the end of the tag's file.  Skips when the checkout is absent: the contract suite runs offline.
    """
    clone = Path(os.environ.get(CLONE_ENV, DEFAULT_CLONE))
    if not clone.is_dir():
        pytest.skip(f"no vLLM {TAG} checkout at {clone}; set {CLONE_ENV} (the repository does not carry it)")
    tag_files = _tag_files(clone)
    if not tag_files:
        pytest.skip(f"the checkout at {clone} has no {TAG} tag")
    drift, unresolvable = _swept_citations(clone, tag_files)
    assert not unresolvable - UNRESOLVABLE, (
        f"tag citations whose path does not resolve at {TAG}: {sorted(unresolvable - UNRESOLVABLE)}"
    )
    assert not UNRESOLVABLE - unresolvable, (
        f"recorded paths that now resolve at {TAG} (delete them from UNRESOLVABLE): "
        f"{sorted(UNRESOLVABLE - unresolvable)}"
    )
    assert not drift - DRIFT, f"tag citations whose claim is not at the cited lines of {TAG}: {sorted(drift - DRIFT)}"
    assert not DRIFT - drift, (
        f"recorded rows that now hold their claim (delete them from DRIFT): {sorted(DRIFT - drift)}"
    )


def test_no_stale_engine_version_survives() -> None:
    """No ``0.31`` engine version outside :data:`RECORDED_0_31`, in the swept files and the CHANGELOG.

    Catches a stale claim reintroduced into a live surface -- the guard's range, a module docstring's target, a
    recipe's declared image or version floor, a schema's prose -- and catches a record that heals, which must
    leave the record.  ``0.31.0`` alone would miss ``v0.31.x`` and ``[0.31, 0.32)``.
    """
    stale: set[str] = set()
    live: dict[str, list[int]] = {}
    for path in _swept_files(extra=("CHANGELOG.md",)):
        name = _relative(path)
        in_engine = False
        for number, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            if not line.startswith((" ", "\t")):
                in_engine = line.split(":", 1)[0].strip() == "engine"
            for _ in _STALE.finditer(line):
                stale.add(name)
                if not _recorded(name, in_engine):
                    live.setdefault(name, []).append(number)
    assert not live, f"a 0.31 engine version outside the recorded history: {live}"
    unused = sorted(name for name in RECORDED_0_31 if name not in stale)
    assert not unused, f"recorded files that no longer carry a 0.31 line (delete them from RECORDED_0_31): {unused}"
