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

from rcp_ndcg.errors import ConfigError
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
    """Every in-process paper model is served: recipe, tokenizer and the paper's budgets, per the brief."""
    config = validate_reranker(yaml.safe_load(path.read_text(encoding="utf-8")))
    if not isinstance(config, ServedReranker):
        return
    assert config.recipe and "@" in (config.tokenizer or ""), f"{path.name}: recipe and tokenizer declared"
    assert config.max_tokens == 8192 and config.query_max_tokens == 4096
    assert config.instruction == "none", "the release's in-process path passed the bare query"


def test_the_jina_paper_config_is_listwise_and_the_octen_one_carries_its_prefix() -> None:
    jina = validate_reranker(yaml.safe_load((PAPER / "rerankers" / "jina_v3.yaml").read_text(encoding="utf-8")))
    assert jina.listwise is True

    octen = validate_retriever(yaml.safe_load((PAPER / "retrieval" / "octen.yaml").read_text(encoding="utf-8")))
    assert isinstance(octen.encoder, ServedEmbedding)
    assert octen.encoder.doc_prompt == "- " and octen.encoder.max_tokens == 8192
    assert octen.encoder.query_prompt == "", "the paper encodes queries as they are"


def test_the_hosted_paper_configs_omit_base_url() -> None:
    for path in ("rerankers/cohere_rerank_v4_fast.yaml", "rerankers/voyage_rerank_2_5.yaml"):  # noqa: PTH118
        config = validate_reranker(yaml.safe_load((PAPER / path).read_text(encoding="utf-8")))
        assert config.base_url is None, path
    cohere = validate_retriever(
        yaml.safe_load((PAPER / "retrieval" / "cohere_embed_v4.yaml").read_text(encoding="utf-8"))
    )
    assert cohere.encoder.base_url is None


def test_no_paper_config_builds_a_client() -> None:
    """A served paper config declares ``max_tokens``, and the client refuses it: configs only, never clients."""
    from rcp_ndcg.inference.clients import RerankClient

    config = validate_reranker(
        yaml.safe_load((PAPER / "rerankers" / "qwen3_reranker_8b.yaml").read_text(encoding="utf-8"))
    )
    assert config.max_tokens == 8192
    with pytest.raises(ConfigError, match="max_tokens"):
        RerankClient(config)
