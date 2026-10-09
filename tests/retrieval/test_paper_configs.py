"""Every migrated paper config validates as a config (and none may build a client).

The in-process paper models are served now: their configs name the recipe that serves them, the
checkpoint's tokenizer, and the paper's budgets, and their ``base_url`` is the placeholder the
runner's engines overlay replaces. The clients still refuse ``max_tokens`` until the text-budget
mechanism wires the cut, so a config that declares it is validated as a config and never run here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from rcp_ndcg.retrieval import validate_reranker, validate_retriever
from rcp_ndcg.retrieval.config import (
    BM25Config,
    CohereReranker,
    DenseConfig,
    LateInteractionConfig,
    ServedEmbedding,
    ServedReranker,
    VoyageReranker,
)
from tests import REPO_ROOT

PAPER = REPO_ROOT / "experiments" / "paper"


def _configs(directory: Path) -> list[tuple[Path, dict[str, Any]]]:
    paths = sorted(directory.glob("*.yaml"))
    assert paths, directory
    return [(path, yaml.safe_load(path.read_text(encoding="utf-8"))) for path in paths]


def test_every_paper_retrieval_config_validates() -> None:
    for path, data in _configs(PAPER / "retrieval"):
        assert isinstance(validate_retriever(data), BM25Config | DenseConfig | LateInteractionConfig), path


def test_every_paper_reranker_config_validates() -> None:
    for path, data in _configs(PAPER / "rerankers"):
        assert isinstance(validate_reranker(data), ServedReranker | CohereReranker | VoyageReranker), path


@pytest.mark.parametrize(
    "path",
    sorted((PAPER / "rerankers").glob("*.yaml")),
    ids=lambda path: path.name,
)
def test_each_served_paper_reranker_names_its_recipe_tokenizer_and_budgets(path: Path) -> None:
    """Every in-process paper model is served: its tokenizer and the paper's budgets are explicit (the configs
    keep their own content where it disagrees with the recipe's -- docs-firstcontact Q1's strict mapping form
    refuses a CONTENT field that disagrees with the recipe's, naming both values; the mapping-form configs --
    ctxl and octen -- take the recipe's block instead)."""
    config = validate_reranker(yaml.safe_load(path.read_text(encoding="utf-8")))
    if not isinstance(config, ServedReranker):
        return
    assert "@" in (config.tokenizer or ""), f"{path.name}: tokenizer declared"
    assert config.max_tokens == 8192 and config.query_max_tokens == 4096
    assert config.instruction == "none", "the release's in-process path passed the bare query"


def test_every_recipe_id_is_the_lowercased_hub_repo_name_of_its_tokenizer() -> None:
    """One recipe-id rule: the id is the checkpoint's lowercased Hub repo name, never a short Hub redirect
    (``zerank-1-reranker``, not ``zerank-1``). The rule's home is the recipe data itself (the shipped recipes):
    every shipped recipe is checked here, raw (no validation: a recipe's template refusal must not hide the
    canon), and every paper config that keeps a pointer names its shipped recipe."""
    import yaml as yaml_module
    from rcp_ndcg_vllm.recipe import default_recipes_root

    checked = 0
    for directory in sorted(default_recipes_root().iterdir()):
        if not (directory / "family.yaml").is_file():
            continue
        data = yaml_module.safe_load((directory / "family.yaml").read_text(encoding="utf-8"))
        for variant in data["variants"]:
            recipe = str(variant["id"])
            tokenizer = f"{variant['model']}@{variant['revision']}"
            repo = tokenizer.rsplit("@", 1)[0].split("/")[-1]
            assert recipe == repo.lower(), f"{directory.name}: recipe {recipe!r} != lowercased repo {repo.lower()!r}"
            checked += 1
    assert checked == 31, f"every shipped recipe names its checkpoint (checked {checked})"
    # every paper config that keeps a `recipe:` pointer names a shipped recipe (the mapping form resolved
    # it above; the pointer's value is the shipped id). Family ids are never pointers (decision 34).
    variant_ids = set()
    for directory in sorted(default_recipes_root().iterdir()):
        if not (directory / "family.yaml").is_file():
            continue
        data = yaml_module.safe_load((directory / "family.yaml").read_text(encoding="utf-8"))
        variant_ids.update(str(variant["id"]) for variant in data["variants"])
    for directory in ("retrieval", "rerankers"):
        for path, data in _configs(PAPER / directory):
            pointers = [data.get("recipe")] + (
                [data.get("encoder", {}).get("recipe")] if isinstance(data.get("encoder"), dict) else []
            )
            for pointer in pointers:
                if pointer is None:
                    continue
                assert str(pointer) in variant_ids, f"{path}: recipe pointer {pointer!r} names no shipped recipe"


def test_the_jina_paper_config_is_listwise_and_the_octen_one_takes_its_recipe_frame() -> None:
    jina = validate_reranker(yaml.safe_load((PAPER / "rerankers" / "jina_v3.yaml").read_text(encoding="utf-8")))
    assert jina.listwise is True

    octen = validate_retriever(yaml.safe_load((PAPER / "retrieval" / "octen.yaml").read_text(encoding="utf-8")))
    assert isinstance(octen.encoder, ServedEmbedding)
    # the mapping form: the recipe's frame is the paper's ("- " + document, queries as they are)
    assert octen.encoder.max_tokens == 8192
    template = octen.encoder.template
    assert template is not None and [segment.fixed for segment in template.segments("document")] == ["- ", None]
    assert [segment.fixed for segment in template.segments("query")] == [None], "queries encode as they are"
    assert octen.encoder.batch_size == 32, "the request-packing runtime field stays on the config"


def test_the_hosted_paper_configs_omit_base_url() -> None:
    for path in ("rerankers/cohere_rerank_v4_fast.yaml", "rerankers/voyage_rerank_2_5.yaml"):  # noqa: PTH118
        config = validate_reranker(yaml.safe_load((PAPER / path).read_text(encoding="utf-8")))
        assert config.base_url is None, path
    cohere = validate_retriever(
        yaml.safe_load((PAPER / "retrieval" / "cohere_embed_v4.yaml").read_text(encoding="utf-8"))
    )
    assert cohere.encoder.base_url is None


def test_every_served_paper_config_builds_its_client(tokenizer_json: str) -> None:
    """The budget is wired: every served paper config builds its client.

    The configs carry their real Hub tokenizers (the recipe serves the checkpoint); the client only loads
    the tokenizer *file* to count the budget, so the build test runs against the saved offline test
    tokenizer -- same field, different file, the paper config itself untouched."""
    from rcp_ndcg.inference.clients import RerankClient
    from tests.retrieval.test_api import _recording_sender

    for data_path, data in _configs(PAPER / "rerankers"):
        config = validate_reranker(data)
        if not isinstance(config, ServedReranker):
            continue
        assert data_path.name, "the paper config is a file"
        built = RerankClient(config.model_copy(update={"tokenizer": tokenizer_json}), sender=_recording_sender())
        assert built.config.max_tokens == 8192 and built.config.query_max_tokens == 4096
        built.close()
