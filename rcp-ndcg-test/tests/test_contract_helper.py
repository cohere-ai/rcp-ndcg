"""The shared contract helper has teeth: two mutants on one recipe red it.

Sweep finding #9 (its shared half): the "loads and declares the contract" recipe tests pinned
neither ``serve.max_model_len`` nor ``reference.kind`` -- three of five reviewer mutants stayed
green.  ``tests/recipes/_contract.py`` pins every field of the three blocks; this test proves the
helper rejects both drift directions on the qwen3-reranker-8b recipe (the same two mutants the
reviewer ran), and pins its frozen expectation of that recipe.  The family lanes call
``assert_recipe_contract`` from their own recipe tests.
"""

from __future__ import annotations

import pytest
from rcp_ndcg_vllm.recipe import default_recipes_root, load_recipe

from tests.recipes._contract import assert_recipe_contract

RECIPE = default_recipes_root() / "qwen3-reranker-8b"

EXPECTED_SERVE = {
    "chat_template": "template.jinja",
    "convert": None,
    "dtype": "bfloat16",
    "extra_args": [],
    "hf_overrides": {
        "architectures": ["Qwen3ForSequenceClassification"],
        "classifier_from_token": ["no", "yes"],
        "is_original_qwen3_reranker": True,
    },
    "io_processor_plugin": None,
    "limit_mm_per_prompt": None,
    "max_model_len": 10000,
    "mm_processor_kwargs": {},
    "plugin": None,
    "pooler_config": {"use_activation": True},
    "runner": "pooling",
    "trust_remote_code": False,
}

EXPECTED_CLIENT = {
    "api": "rerank",
    "recipe": "vLLM v0.31.0 pooling runner; Qwen3ForCausalLM converted to Qwen3ForSequenceClassification "
    "in-engine (hf_overrides: classifier_from_token [no, yes], is_original_qwen3_reranker); the paper "
    "chat template (template.jinja); sigmoid-activated 1-label score head; client-side pair cut at "
    "8192 tokens",
    "tokenizer": "Qwen/Qwen3-Reranker-8B@77d193c791ed757ca307ee72715aa132723da912",
    "max_tokens": 8192,
    "query_max_tokens": 4096,
    "template": {
        "pair": [
            {
                "fixed": "{special:im_start}system\n"
                "Judge whether the Document meets the requirements based on the Query and "
                'the Instruct provided. Note that the answer can only be "yes" or '
                '"no".{special:im_end}\n'
                "{special:im_start}user\n"
                "<Instruct>: Given a web search query, retrieve relevant passages that "
                "answer the query\n"
                "<Query>: "
            },
            {"content": "query"},
            {"fixed": "\n<Document>: "},
            {"content": "document"},
            {"fixed": "{special:im_end}\n{special:im_start}assistant\n{special:<think>}\n\n{special:</think>}\n\n"},
        ],
        "anchor": "last",
        "add_special_tokens": True,
    },
    "instruction": "none",
    "use_activation": True,
    "on_overflow": "cut",
    "empty_doc": "send",
    "empty_query": "send",
    "request_shape": "text",
    "listwise": False,
    "model": "qwen3-reranker-8b",
    "revision": "77d193c791ed757ca307ee72715aa132723da912",
}

EXPECTED_REFERENCE = {
    "entry": "reference.py",
    "kind": "transformers",
    "known_deviations": ["over_cap_cut_differs"],
    "score_scale": "probability",
}

EXPECTED_TOP = {
    "id": "qwen3-reranker-8b",
    "licence": "apache-2.0",
    "revision": "77d193c791ed757ca307ee72715aa132723da912",
    "role": "rerank",
    "scoring": "pointwise",
}


def test_the_shipped_recipe_passes_the_full_contract() -> None:
    """Green on the recipe as shipped: the frozen mapping IS the recipe's resolved contract."""
    assert_recipe_contract(
        load_recipe(RECIPE),
        serve=EXPECTED_SERVE,
        client=EXPECTED_CLIENT,
        reference=EXPECTED_REFERENCE,
        top=EXPECTED_TOP,
    )


def test_a_drifted_serve_field_fails_naming_the_field() -> None:
    """Mutant 1 (the reviewer's): serve.max_model_len 10000 -> 16384 must red, naming the field."""
    drifted = {**EXPECTED_SERVE, "max_model_len": 16384}
    with pytest.raises(AssertionError, match=r"serve\.max_model_len"):
        assert_recipe_contract(
            load_recipe(RECIPE), serve=drifted, client=EXPECTED_CLIENT, reference=EXPECTED_REFERENCE, top=EXPECTED_TOP
        )


def test_a_drifted_reference_field_fails_naming_the_field() -> None:
    """Mutant 2 (the reviewer's): reference.kind transformers -> remote_code must red, naming the field."""
    drifted = {**EXPECTED_REFERENCE, "kind": "remote_code"}
    with pytest.raises(AssertionError, match=r"reference\.kind"):
        assert_recipe_contract(
            load_recipe(RECIPE), serve=EXPECTED_SERVE, client=EXPECTED_CLIENT, reference=drifted, top=EXPECTED_TOP
        )


def test_a_field_the_mapping_omits_is_refused_too() -> None:
    """Nothing rides unpinned: an expected mapping short one field fails, naming the missing field."""
    with pytest.raises(AssertionError, match=r"unpinned field"):
        assert_recipe_contract(
            load_recipe(RECIPE),
            serve=EXPECTED_SERVE,
            client=EXPECTED_CLIENT,
            reference={"kind": "transformers"},
            top=EXPECTED_TOP,
        )
