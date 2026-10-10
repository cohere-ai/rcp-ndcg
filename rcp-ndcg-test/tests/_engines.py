"""The verified fake engines' test wiring: one recipe id -> its registered emulator.

The emulators (``rcp_ndcg_test.engines``) are product code and know nothing of the harness; the
recipes (``rcp-ndcg-vllm``) name the tokenizer, the template and the listwise prompt builder. This is
the one adapter between them (test-side): it loads the recipe through the harness package, resolves the
recipe's real tokenizer from the corpora's vendored store, picks the prompt derivation the role and
scoring name, builds the emulator over the recipe's corpus and registers it for
``fake://vllm-0.31.0/<recipe>``. The wiring mirrors what a golden-replay or conformance run needs --
and what an out-of-tree ``rcp_ndcg.emulators`` entry point would construct for its own models.
"""

from __future__ import annotations

import functools
import importlib.util
import sys
from pathlib import Path
from typing import Any

from rcp_ndcg_vllm.recipe import default_recipes_root

ROOT = Path(__file__).resolve().parents[1]
ENGINES_ROOT = ROOT / "corpora"
RECIPES_ROOT = default_recipes_root()
_VLLM_SRC = str((ROOT / ".." / "rcp-ndcg-vllm" / "src").resolve())
STALE_FILE = ROOT / "tests" / "conformance" / "stale.json"
"""The corpora declared stale for re-recording: each entry names the recipe, the recorded fingerprint, the
fingerprint inputs that moved, why and when it was decided (the release checklist requires it empty)."""


def stale_corpora() -> dict[str, dict[str, Any]]:
    """The declared stale corpora by recipe id (``tests/conformance/stale.json``)."""
    import json

    return {str(entry["recipe_id"]): entry for entry in json.loads(STALE_FILE.read_text(encoding="utf-8"))}


def current_corpora() -> list[Path]:
    """Every committed corpus a conformance replay runs on: the ones not declared stale (found by scanning
    manifests, each read through the one corpus reader, ``rcp_ndcg_test.corpus.load_corpus``). A
    declared-stale corpus is checked to fail the staleness gate instead."""
    from rcp_ndcg_test.corpus import load_corpus
    from rcp_ndcg_test.engines import find_corpora

    stale = stale_corpora()
    return [
        directory
        for directory in find_corpora(ENGINES_ROOT)
        if load_corpus(directory).manifest["recipe"]["id"] not in stale
    ]


def harness() -> Any:
    """Import ``rcp_ndcg_vllm`` (outside the uv workspace: its ``src`` goes on the path) and register the
    vendored tokenizer stores, once."""
    if _VLLM_SRC not in sys.path:
        sys.path.insert(0, _VLLM_SRC)
    from rcp_ndcg_test.fingerprint import use_tokenizer_store

    for store in ENGINES_ROOT.glob("*/_tokenizers"):
        use_tokenizer_store(store)
    return __import__("rcp_ndcg_vllm.recipe", fromlist=["load_recipe"])


def load_recipe(recipe_id: str, root: Path | None = None) -> Any:
    """The recipe, loaded through the harness's own schema and the product's validation.

    ``recipe_id`` is a variant id (decision 34: the families' variants are the recipes); ``root``
    names the recipes root to resolve it in (default: the shipped one)."""
    module = harness()
    return module.load_recipe(recipe_id, root=root if root is not None else RECIPES_ROOT)


def corpus_of(recipe: Any) -> Any:
    """The committed corpus of the recipe's **current** behaviour fingerprint, resolved by scanning the
    manifests (``rcp_ndcg_test.changes.resolve_corpus``); a stale recipe raises ``StaleCorpusError``
    naming the fingerprint inputs that moved -- never another fingerprint's corpus, never a missing
    directory."""
    from rcp_ndcg_test.corpus import load_corpus

    harness()
    from rcp_ndcg_test.changes import resolve_corpus

    return load_corpus(resolve_corpus(recipe, ENGINES_ROOT / "vllm-0.31.0"))


def prompt_strategy(recipe: Any, tokenizer: Any) -> Any:
    """The recipe's engine-prompt derivation: input strings for the pooling routes (the route tokenizes
    the request's strings), the recipe's pair template for a pointwise reranker, the recipe reference's
    N-passage builder for a listwise one, and the chat route's render for a recipe whose requests are
    chat-shaped (``messages``) or whose media items ride one (the engine frames the content there).

    Media parts are keyed by content identity: :func:`media_model` hashes the sent bytes and counts what
    the engine adds for them under the recipe's declared processing, so a media request replays only for
    the same image, clip and policy.
    """
    from rcp_ndcg_test.engines import ChatPrompts, EnginePrompts, PairPrompts, RequestPrompts, StringsPrompts
    from rcp_ndcg_test.equivalence import fitting

    media = media_model(recipe, tokenizer)
    slot = "token_vector" if recipe.role == "multi_vector" else "vector"
    chat = ChatPrompts(render=chat_render(recipe), slot=slot, media=media)
    if recipe.role in ("embed", "multi_vector"):
        return RequestPrompts(StringsPrompts(slot=slot), chat)
    if recipe.client.get("listwise", False):
        return EnginePrompts(builder=_listwise_builder(recipe))
    return PairPrompts(template=fitting.client_template(recipe), tokenizer=tokenizer, media=media)


def chat_render(recipe: Any) -> Any:
    """The engine's render of one conversation's text parts: the served chat template (the recipe's file,
    else the checkpoint's own at the pinned revision), rendered as the engine renders it with the media
    parts dropped -- the engine expands each media part into its own vision block, which the media model
    counts separately.

    The template resolves lazily, on the first chat-shaped request: a text-only corpus never reads the
    checkpoint's template (and never needs the Hub for one).
    """
    resolved: list[str] = []

    def render(conversation: list[Any], add_generation_prompt: bool) -> str:
        from rcp_ndcg_test.equivalence.stages import render_chat, served_chat_template, text_only_conversation

        if not resolved:
            resolved.append(served_chat_template(recipe)[1])
        return render_chat(
            resolved[0], text_only_conversation(conversation), add_generation_prompt=add_generation_prompt
        )

    return render


def media_model(recipe: Any, tokenizer: Any) -> Any:
    """The recipe's declared media processing as a callable: one sent media part -> its content identity
    and the tokens the engine adds for it.

    The identity is the sent bytes' SHA-256 beside the declared processing (the recipe's image and video
    policies and its processor family, canonically), so a media request replays only for the same content
    under the same declared processing.  The token count is the product's own media count under those
    policies, with the client's tokenizer (the exact count the engine's ``usage`` adds).
    """
    import hashlib
    import json

    from rcp_ndcg_test.equivalence.wire import recipe_config

    from rcp_ndcg.data.prepare import media_policies_for
    from rcp_ndcg.data.resolution import ImagePolicy, content_media_tokens

    config = recipe_config(recipe)
    image_policy, video_policy = media_policies_for(config)
    processing = json.dumps(
        {
            "image_processor": recipe.client.get("image_processor"),
            "image_policy": recipe.client.get("image_policy"),
            "video_policy": recipe.client.get("video_policy"),
        },
        sort_keys=True,
        separators=(",", ":"),
    )

    def media(part: dict[str, Any]) -> tuple[Any, int]:
        from rcp_ndcg_test.engines import MediaIdentity
        from rcp_ndcg_test.equivalence.media import sent_media_content
        from rcp_ndcg_test.errors import EmulatorUnmodelledError, HarnessError

        try:
            kind, content = sent_media_content(part)
            payload = _inline_bytes(part)
        except HarnessError as error:
            # A part the harness cannot read (not inline, a header that states no geometry) is unmodelled:
            # typed, so `from_corpus` skips and names the record instead of failing the whole corpus.
            raise EmulatorUnmodelledError(str(error)) from error
        tokens = content_media_tokens(
            content, image_policy or ImagePolicy.native(), video_policy, tokenizer=tokenizer
        ).tokens
        return MediaIdentity(kind=kind, sha256=hashlib.sha256(payload).hexdigest(), processing=processing), tokens

    return media


def _inline_bytes(part: dict[str, Any]) -> bytes:
    """The bytes of one sent inline media part (the media lowering inlines every item; the product's resolver
    decodes the ``data:`` URI, one home with :func:`sent_media_content`)."""
    from rcp_ndcg_core.content import MediaRef
    from rcp_ndcg_test.errors import HarnessError

    from rcp_ndcg.data.media import default_resolver

    url = str(
        (part.get("image_url") or {}).get("url")
        if part.get("type") == "image_url"
        else (part.get("video_url") or {}).get("url")
    )
    if not url.startswith("data:"):
        raise HarnessError(f"the media model reads inline media only, got {url[:40]!r}")
    return default_resolver().bytes_of(MediaRef(uri=url))


def _listwise_builder(recipe: Any):
    """The listwise prompt builder the recipe itself declares (its ``reference.py``'s verbatim port of
    the checkpoint's prompt function): one home for the prompt shape, the emulator reads it from there."""
    directory = Path(str(recipe._dir)) if recipe._dir is not None else RECIPES_ROOT / recipe.id
    path = directory / (recipe.reference.entry or "reference.py")
    spec = importlib.util.spec_from_file_location(f"rcp_recipe_reference_{recipe.id.replace('-', '_')}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    builder = getattr(module, "format_docs_prompt", None)
    if builder is None:
        raise AssertionError(f"{path} declares no format_docs_prompt; a listwise recipe's reference does")

    def build(query: str, documents: list[str]) -> str:
        return builder(query, documents)

    return build


@functools.cache
def _build_emulator(recipe_id: str) -> Any:
    """The one emulator of ``recipe_id``, built once per process (facts from the recipe and its corpus)."""
    harness()
    from rcp_ndcg_test.engines import EngineFacts, VllmEmulator, registry
    from rcp_ndcg_test.fingerprint import behaviour_fingerprint, load_recipe_tokenizer

    recipe = load_recipe(recipe_id)
    tokenizer = load_recipe_tokenizer(recipe)
    corpus = corpus_of(recipe)
    # The engine-side Matryoshka gate is the serve block's own declaration (a checkpoint's config.json
    # carries neither key; the recipe enables the gate through serve.hf_overrides, which is also what the
    # engine is served with).
    overrides = dict(recipe.serve.hf_overrides or {})
    declared_dims = tuple(int(dimension) for dimension in overrides.get("matryoshka_dimensions") or ())
    facts = EngineFacts(
        engine_name=str(corpus.manifest["engine"]["name"]),
        engine_version=str(corpus.manifest["engine"]["version"]),
        served_name=recipe.id,
        model_root=recipe.model,
        max_model_len=recipe.serve.max_model_len,
        is_matryoshka=bool(declared_dims or overrides.get("is_matryoshka")),
        matryoshka_dimensions=declared_dims or None,
    )
    emulator = VllmEmulator.from_corpus(corpus, prompt_strategy(recipe, tokenizer), tokenizer, facts)
    assert emulator.verified is not None
    emulator.require_verified_for(
        recipe.id, recipe.revision, behaviour_fingerprint(recipe), str(corpus.manifest["engine"]["version"])
    )
    registry.register(emulator)
    return emulator


def emulator_for(recipe_id: str) -> Any:
    """The registered emulator of ``recipe_id``: one per recipe per process, and re-registered on every
    call (the registry may have been reset between tests)."""
    from rcp_ndcg_test.engines import registry

    emulator = _build_emulator(recipe_id)
    registry.register(emulator)
    return emulator


def observation_corpus(
    directory: Path,
    exchanges: Any,
    *,
    recipe_id: str = "tiny",
    fingerprint: str = "e" * 64,
    tolerance: tuple[float, float] | None = None,
) -> Any:
    """An in-memory corpus in the observation-corpus format (``rcp_ndcg_test.corpus``) for synthetic
    tests: each :class:`~rcp_ndcg_test.engines.Exchange` becomes one ``RECORD_SCHEMA`` record (its
    ``body_raw`` declared reconstructed unless the exchange carries raw bytes), and ``nondeterminism.json``
    declares ``tolerance`` measured, or nothing measured."""
    import base64
    import hashlib
    import json

    from rcp_ndcg_test.corpus import CORPUS_SCHEMA, NONDETERMINISM_FILE, RECORD_SCHEMA, ObservationCorpus

    records = []
    for exchange in exchanges:
        request_raw = json.dumps(exchange.request_body) if exchange.request_body is not None else ""
        if exchange.response_raw is not None and exchange.response is None:
            response_raw: Any = {"base64": base64.b64encode(exchange.response_raw).decode("ascii")}
        else:
            response_raw = (exchange.response_raw or json.dumps(exchange.response).encode()).decode("utf-8")
        canonical = json.dumps([exchange.method, exchange.path, request_raw])
        records.append(
            {
                "record_schema": RECORD_SCHEMA,
                "exchange_id": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
                "sequence": exchange.sequence,
                "repetition": "same_process_1",
                "request": {
                    "method": exchange.method,
                    "path": exchange.path,
                    "headers": {},
                    "body_raw": request_raw,
                    "body_parsed": exchange.request_body,
                },
                "response": {
                    "status": exchange.status,
                    "headers": dict(exchange.response_headers),
                    "body_raw": response_raw,
                    "body_parsed": exchange.response,
                },
                "inputs": {} if exchange.response_raw is not None else {"body_raw": "reconstructed: a test fixture"},
            }
        )
    derived = {"measured": False}
    if tolerance is not None:
        derived = {"measured": True, "abs_tolerance": tolerance[0], "rel_tolerance": tolerance[1]}
    directory.mkdir(parents=True, exist_ok=True)
    (directory / NONDETERMINISM_FILE).write_text(json.dumps({"derived": derived}), encoding="utf-8")
    manifest = {
        "schema": CORPUS_SCHEMA,
        "engine": {"name": "vllm", "version": "0.31.0"},
        "model": {"id": "fixtures/Tiny", "revision": "f" * 40},
        "recipe": {"id": recipe_id, "behaviour_fingerprint": fingerprint},
    }
    return ObservationCorpus(
        directory=directory,
        manifest=manifest,
        records=records,
        records_file="records.jsonl",
        nondeterminism={"derived": derived},
    )
