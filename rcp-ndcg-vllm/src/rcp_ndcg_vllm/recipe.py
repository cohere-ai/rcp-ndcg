"""The recipe schema: one declarative description of how a model is served with vLLM and read back by
``rcp-ndcg``.

The shipped recipes are package data (``rcp_ndcg_vllm/recipes/<family>/``), read through
:mod:`importlib.resources`. A **family directory** carries (decision 34: one family, many sizes,
every size its own tested recipe id):

- ``family.yaml`` — the family: the shared blocks (role, engine, serve, client, reference, gates,
  status) and a ``variants`` table carrying only the per-size facts (id, model, revision, notes,
  sources, status and the whitelisted overrides);
- ``template.jinja`` — the family's one chat template, given to ``vllm serve --chat-template``,
  when the model needs one;
- ``reference.py`` — the ONE reference implementation for the family, parameterised by the variant
  (the harness passes the resolved recipe through ``--recipe``; see ``reference.entry``);
- ``requirements-reference.txt`` — the reference environment, shared by the family's variants.

Every variant resolves to a full :class:`Recipe` — exactly what a standalone recipe described before
the families (the resolved recipe's JSON Schema is unchanged) — and every consumer (``serve``,
``recipe: <id>`` in rcp-ndcg, the harness, the wave lists, the catalog) works on variant ids; a
family id is never served. A single-size model is a family with one variant: there is no second,
standalone loading path.

The recipe's ``client`` block **is** the product's endpoint config for the role
(:class:`~rcp_ndcg.inference.config.EmbeddingEndpoint`, ``PoolingEndpoint`` or ``RerankEndpoint``): it stays
**plain data here** — this package depends on pydantic and PyYAML only, so ``pip install --no-deps`` works in a
stock engine image — and ``rcp-ndcg`` validates it with the product's own models when it reads it (a recipe
resolution through ``recipe: <id>``). The load-time checks below are the ones readable without the product: the
role/wire matrix, the injected ids and the budget arithmetic.

The schema is deliberately closed and role-aware: ``client.model``, ``client.revision`` and
``client.tokenizer`` are the recipe's own (injected at load, refused in the YAML). Recipe-level fields are the
ones the product cannot know: ``serve`` (the engine argv), ``reference`` (the subprocess reference), ``gates``,
``status``, ids and revisions.

Public names (pinned by ``tests/contract``):

- :func:`load_recipe` — the resolved recipe of a variant id (or of a single-variant family directory).
- :func:`resolve_recipe` — the resolved recipe of a variant id under a recipes root (the explicit form).
- :func:`load_family` — one family directory or ``family.yaml`` file.
- :func:`iter_families` — every family under a root (default: the shipped ones).
- :func:`iter_recipes` — every variant of every family under a root, as resolved :class:`Recipe` objects.
- :func:`serve_argv` — the ``vllm serve`` argv a recipe renders to.

Everything else in this module is internal.
"""

from __future__ import annotations

import json
import re
from importlib import resources
from pathlib import Path
from typing import Any, Literal

import yaml  # pyright: ignore[reportMissingModuleSource]
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator

from .errors import RecipeError

__all__ = [
    "Family",
    "Recipe",
    "Variant",
    "iter_families",
    "iter_recipes",
    "load_family",
    "load_recipe",
    "resolve_recipe",
    "serve_argv",
]

_PINNED_VLLM_REF = "d0d6e5f3a"
"""The upstream vLLM commit the PoolerConfig field list below was read at
(the ``.refs/vllm`` checkout of the rcp-ndcg working copy; image ``vllm/vllm-openai:v0.31.0``)."""

PINNED_POOLER_CONFIG_FIELDS = (
    "task",
    "pooling_type",
    "seq_pooling_type",
    "tok_pooling_type",
    "use_activation",
    "enable_flash_late_interaction",
    "dimensions",
    "enable_chunked_processing",
    "max_embed_len",
    "logit_mean",
    "logit_sigma",
    "step_tag_id",
    "returned_token_ids",
)
"""Every field of vLLM's ``PoolerConfig`` at the pinned ref (read from
``vllm/config/pooler.py`` at :data:`_PINNED_VLLM_REF`); ``serve.pooler_config`` keys are validated
against this tuple, because the engine rejects unknown keys (r-vllm-drift risk 2)."""

_ID_PATTERN = r"^[a-z0-9][a-z0-9.-]*$"
_REVISION_PATTERN = r"^[0-9a-f]{40}$"

Role = Literal["embed", "multi_vector", "rerank"]
ScoreScale = Literal["probability", "logit", "cosine"]

#: The per-variant ``serve`` fields a family may override (decision 34: only per-size facts): the engine's
#: context limit, the score-head construction (a family whose sizes build their head from different base
#: architectures — ctxl's mistral-based 6b beside its qwen3-based 1b/2b), and the per-size media pins and caps.
#: Anything else that a size would need differently is a modelling error the loader refuses: the family shares
#: the block, and a size that genuinely behaves differently is its own family.
PER_VARIANT_SERVE_FIELDS: tuple[str, ...] = (
    "max_model_len",
    "hf_overrides",
    "mm_processor_kwargs",
    "limit_mm_per_prompt",
)

#: The per-variant ``client`` fields a family may override: the max token lengths, the dimension knobs, the
#: paper's per-model batch (a runtime field), and the per-size media caps. Content shapes (template,
#: instruction mode, overflow rule, normalisation, media policies) are shared: a size that cuts differently
#: is not the same model family.
PER_VARIANT_CLIENT_FIELDS: tuple[str, ...] = (
    "max_tokens",
    "query_max_tokens",
    "document_max_tokens",
    "dim",
    "dimensions",
    "batch_size",
    "max_images",
    "max_videos",
)

_ROLE_WIRE = {"embed": "openai_embeddings", "multi_vector": "vllm_pooling", "rerank": "rerank"}
"""The wire each role speaks (the product refuses a config whose role and wire disagree)."""

_ENGINE_SPECIFIC_CLIENT_KEYS = frozenset(
    {
        # the serve block's keys (the vllm serve argv and the pre-install specs)
        "runner",
        "convert",
        "hf_overrides",
        "pooler_config",
        "trust_remote_code",
        "max_model_len",
        "dtype",
        "plugin",
        "io_processor_plugin",
        "mm_processor_kwargs",
        "limit_mm_per_prompt",
        "extra_args",
        # the engine block's
        "name",
        "image",
        "min_version",
        "startup_timeout_s",
        # resources
        "gpus",
    }
)
"""The keys that name engine-side knobs (decision 19): the engine-neutral ``client`` block refuses them.
The product's endpoint config has no field of any of these names (its media ``max_images``/``max_videos``
and its ``image_policy`` are the client's own gate and budget), so a key from this set in a client block is
always a misplaced engine setting -- silently ignored by the product's endpoint model, worse mis-read by
an engine of another family."""


def _no_extra() -> dict[str, Any]:
    """The common model config: frozen, unknown fields refused."""
    return {"extra": "forbid", "frozen": True}


class EngineSpec(BaseModel):
    """The engine image a recipe is served with.

    Attributes:
        name: The engine family; this package serves ``vllm``.
        image: The container image, ``repository:tag`` (public names only).
        min_version: The engine version the recipe is known to work with, ``MAJOR.MINOR.PATCH``.
        startup_timeout_s: How long :mod:`rcp_ndcg_test.jobs.run_wave` waits for ``GET /v1/models`` before it
            declares the recipe failed (seconds).  Large models override this per recipe.
    """

    model_config = ConfigDict(**_no_extra())

    name: Literal["vllm"]
    image: str = Field(min_length=1, description="repository:tag of the engine image")
    min_version: str = Field(
        pattern=r"^\d+\.\d+\.\d+(rc\d+)?$", description="known-good engine version (a release candidate counts)"
    )
    startup_timeout_s: int = Field(default=1800, gt=0)


class Resources(BaseModel):
    """The GPUs one engine instance of the recipe occupies.

    Attributes:
        gpus: How many GPUs ``vllm serve --tensor-parallel-size`` gets; the wave runner packs recipes onto the
            node's GPUs by this count.
    """

    model_config = ConfigDict(**_no_extra())

    gpus: int = Field(ge=1)


class ServeConfig(BaseModel):
    """Everything rendered into the ``vllm serve`` argv — nothing implicit.

    See :func:`serve_argv` for the exact argv.  ``plugin`` and ``io_processor_plugin`` never reach the argv: they
    name packages that must be installed into the image before the engine starts (a ``vllm.general_plugins``
    package and the checkpoint's IO-processor plugin, respectively; the node's bootstrap collects a wave's
    ``serve.plugin`` wheels with ``python -m rcp_ndcg_test.jobs.plugins`` and installs them with ``--no-deps``,
    under the freeze-diff guard).
    """

    model_config = ConfigDict(**_no_extra())

    runner: Literal["pooling", "generate"] = Field(description="vLLM --runner; the rcp-ndcg routes need pooling")
    convert: Literal["embed", "classify"] | None = Field(
        default=None, description="vLLM --convert, for a decoder checkpoint turned into an embed or classify model"
    )
    hf_overrides: dict[str, Any] = Field(
        default_factory=dict, description="vLLM --hf-overrides, a JSON object sent verbatim (keys sorted)"
    )
    chat_template: str | None = Field(
        default=None, description="chat template file in the recipe directory, given to vLLM --chat-template"
    )
    pooler_config: dict[str, Any] = Field(
        default_factory=dict,
        description="vLLM --pooler-config, a JSON object; keys must be PoolerConfig fields at the pinned engine",
    )
    trust_remote_code: bool = Field(default=False, description="vLLM --trust-remote-code flag")
    max_model_len: int = Field(gt=0, description="vLLM --max-model-len; at least every client budget")
    dtype: Literal["auto", "float16", "bfloat16", "float32", "float8", "half"] = Field(description="vLLM --dtype")
    plugin: str | None = Field(
        default=None, description="pip spec of a vllm.general_plugins package, installed before the engine starts"
    )
    io_processor_plugin: str | None = Field(
        default=None,
        description="name of the checkpoint's io-processor plugin, resolved through its config (the sanctioned "
        "seam when the stock embed IO-processor's prompt construction cannot express the model)",
    )
    mm_processor_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="vLLM --mm-processor-kwargs (min/max_pixels etc.): changes image-token counts, so content",
    )
    limit_mm_per_prompt: dict[str, int] | None = Field(
        default=None, description="vLLM --limit-mm-per-prompt, the per-request media caps"
    )
    extra_args: list[str] = Field(
        default_factory=list, description="further vllm serve flags, verbatim (one argv element per item)"
    )

    @field_validator("chat_template")
    @classmethod
    def _chat_template_is_a_bare_file(cls, value: str | None) -> str | None:
        if value is not None and (Path(value).name != value or value in (".", "..")):
            raise ValueError("chat_template must be a bare file name inside the recipe directory, e.g. template.jinja")
        return value

    @field_validator("pooler_config")
    @classmethod
    def _pooler_keys_exist_at_the_pinned_engine(cls, value: dict[str, Any]) -> dict[str, Any]:
        unknown = sorted(set(value) - set(PINNED_POOLER_CONFIG_FIELDS))
        if unknown:
            raise ValueError(
                f"pooler_config keys {unknown} are not PoolerConfig fields at vLLM {_PINNED_VLLM_REF} "
                f"(known: {', '.join(PINNED_POOLER_CONFIG_FIELDS)}); the engine rejects unknown keys"
            )
        return value


class ReferenceSpec(BaseModel):
    """The reference implementation the equivalence harness runs as a subprocess.

    Attributes:
        kind: Where the reference comes from: ``transformers`` (an HF checkpoint run in the reference
            environment), ``sentence_transformers`` (a sentence-transformers checkpoint), ``remote_code`` (the
            checkpoint's trusted code) or ``stored_scores`` (scores stored with the paper's runs).
        score_scale: The scale the reference's scores come back on, which selects the stage-2 gates:
            ``probability``, ``logit`` or ``cosine``.  Embedding recipes always compare vectors (cosine per
            vector or per token).
        entry: The reference module file in the recipe directory; the harness runs it as
            ``<reference-python> <recipe-dir>/<entry> --mode <mode> --pairs <file> --out <file>`` (see
            :mod:`rcp_ndcg_test.equivalence.reference` for the modes).  Not needed for ``stored_scores``.
        known_deviations: Deliberate reference deviations on over-cap inputs: ``anchor_drop_over_cap`` (the
            reference's whole-prompt right cut drops tail anchors) or ``over_cap_cut_differs`` (the reference keeps
            the anchors but cuts the content its own way, e.g. a joint ``longest_first`` truncation where the
            client settles the query at its share). Either way the harness gates only the inputs under the cap
            and reports the client's over-cap cuts in a separate, non-gating table; the reference stays the
            paper's or the model card's -- it never copies the client's cut to make an over-cap row pass.
    """

    model_config = ConfigDict(**_no_extra())

    kind: Literal["transformers", "sentence_transformers", "remote_code", "stored_scores"]
    score_scale: ScoreScale
    entry: str = Field(default="reference.py", description="reference module file inside the recipe directory")
    known_deviations: list[Literal["anchor_drop_over_cap", "over_cap_cut_differs"]] = Field(default_factory=list)

    @property
    def over_cap_deviation(self) -> str | None:
        """The declared over-cap deviation (over-cap inputs are reported, not gated), or ``None``."""
        return next(iter(self.known_deviations), None)


class Gates(BaseModel):
    """Overrides of the stage-2 gate defaults for one recipe's ``reference.score_scale``.

    Every field defaults to ``None`` (= the published default for the scale).  The defaults, and what each field
    means, are in :mod:`rcp_ndcg_test.equivalence.gates`; ``tau_min`` applies to every scored scale,
    ``metrics_max_abs`` to stage 3.
    """

    model_config = ConfigDict(**_no_extra())

    prob_p99_abs: float | None = Field(default=None, ge=0, le=1)
    prob_max_abs: float | None = Field(default=None, ge=0, le=1)
    logit_rel_abs: float | None = Field(default=None, ge=0)
    cos_max_abs: float | None = Field(default=None, ge=0, le=2)
    vec_min_cosine: float | None = Field(default=None, ge=-1, le=1)
    tau_min: float | None = Field(default=None, ge=0, le=1)
    metrics_max_abs: float | None = Field(default=None, ge=0, le=1)


class StatusSpec(BaseModel):
    """Where the recipe stands in the verification workflow.

    Attributes:
        state: ``unverified`` (written, not yet checked), ``verified`` (the harness passed every gate) or
            ``failed`` (a gate failed; the report says which).
        image: The engine image the recipe was last verified against.
        date: ISO date of the last verification.
        report: Path or URL of the equivalence report.
    """

    model_config = ConfigDict(**_no_extra())

    state: Literal["unverified", "verified", "failed"] = "unverified"
    image: str | None = None
    date: str | None = None
    report: str | None = None


class Recipe(BaseModel):
    """One served model: how vLLM serves it, how ``rcp-ndcg`` reads it, and what it is checked against.

    Construct it through :func:`load_recipe` (which also checks the referenced files exist, that ``id`` equals
    the directory name, and that the ``client`` block constructs the product's endpoint model); the model itself
    is frozen and refuses unknown fields.

    Attributes:
        id: The recipe identifier, ``^[a-z0-9][a-z0-9.-]*$``, equal to the directory name; also the
            ``--served-model-name`` the engine serves and the client config's ``model``.
        schema_version: The recipe file format's version; the versioned contract between this
            package and rcp-ndcg (decision 18).
        model: The Hugging Face repo id to serve.
        revision: The exact commit of ``model`` (40 hex); serving and client cutting pin it.
        role: What the model produces: ``embed`` (one dense vector), ``multi_vector`` (one vector per token) or
            ``rerank`` (query-document scores).
        input: The input modalities the model accepts, a non-empty subset of ``[text, image, video]``.
        scoring: Rerank only: ``pointwise`` (documents scored independently) or ``listwise`` (the whole
            candidate set in one prompt; the product endpoint's ``listwise`` flag).
        licence: SPDX identifier of the model's licence, or ``see-model-card``.
        engine: The engine image and the startup timeout.
        resources: The GPUs the engine occupies (``--tensor-parallel-size``).
        serve: Everything rendered into ``vllm serve`` argv.
        client: The product's endpoint config for the role, as plain data (the ``model`` and ``revision`` keys
            are injected here and refused in the YAML); ``rcp-ndcg`` validates the whole block with the
            product's endpoint class when it resolves the recipe.
        reference: The subprocess reference the harness compares against.
        gates: Overrides of the stage-2 gate defaults.
        status: Where the recipe stands in the verification workflow.
        sources: URLs and ``path:line`` references the recipe rests on.
        notes: Free-form notes.
    """

    model_config = ConfigDict(**_no_extra())

    id: str = Field(pattern=_ID_PATTERN)
    schema_version: str = Field(
        pattern=r"^\d+$",
        description="the recipe file format's version (decision 18: the file format is the versioned contract "
        "between rcp-ndcg and rcp-ndcg-vllm; the reader checks the versions it understands, so the two "
        "packages need no lockstep version pin)",
    )
    model: str = Field(min_length=1, description="Hugging Face repo id")
    revision: str = Field(pattern=_REVISION_PATTERN, description="40-hex commit of model")
    role: Role
    input: list[Literal["text", "image", "video"]] = Field(min_length=1)
    scoring: Literal["pointwise", "listwise"] | None = None
    licence: str = Field(min_length=1)
    engine: EngineSpec
    resources: Resources
    serve: ServeConfig
    client: dict[str, Any]
    reference: ReferenceSpec
    gates: Gates = Field(default_factory=Gates)
    status: StatusSpec = Field(default_factory=StatusSpec)
    sources: list[str] = Field(default_factory=list)
    notes: str = ""

    _dir: Path | None = PrivateAttr(default=None)
    """The directory the recipe was loaded from (set by :func:`load_recipe`; ``serve_argv`` resolves
    ``serve.chat_template`` against it)."""

    @field_validator("sources")
    @classmethod
    def _sources_are_strings(cls, value: list[str]) -> list[str]:
        if any(not source.strip() for source in value):
            raise ValueError("sources entries must be non-empty URLs or path:line references")
        return value

    @model_validator(mode="after")
    def _recipe_rules(self) -> Recipe:
        """The rules readable without the product: the role/wire matrix and the budget arithmetic. The product's
        own endpoint validations run when ``rcp-ndcg`` reads the client block."""
        rerank = self.role == "rerank"
        if rerank and self.scoring is None:
            raise ValueError("a rerank recipe must set scoring: pointwise or listwise")
        if not rerank and self.scoring is not None:
            raise ValueError(f"scoring is only valid for role=rerank, not role={self.role}")
        client = self.client
        misplaced = sorted(set(client) & _ENGINE_SPECIFIC_CLIENT_KEYS)
        if misplaced:
            raise ValueError(
                f"client carries the engine-specific key(s) {misplaced}: they belong in serve/engine/resources "
                "(decision 19: the engine-neutral client block stays strictly apart from the engine-specific "
                "blocks, so a config the product reads can never carry a knob the product silently ignores)"
            )
        api = client.get("api")
        if api != _ROLE_WIRE[self.role]:
            raise ValueError(
                f"role={self.role} speaks api: {_ROLE_WIRE[self.role]}, got client.api={api!r} -- the config "
                "would load and only fail at client construction; name the role's wire"
            )
        if rerank and self.serve.convert is not None:
            raise ValueError(
                f"serve.convert ({self.serve.convert}) serves an embed or classify endpoint, not a reranker; "
                "a rerank recipe declares the checkpoint's scorer through engine.hf_overrides instead"
            )
        if rerank and self.scoring == "listwise" and not client.get("listwise"):
            raise ValueError("scoring: listwise must set the endpoint's listwise flag")
        if rerank and self.scoring == "pointwise" and client.get("listwise"):
            raise ValueError("scoring: pointwise conflicts with the endpoint's listwise flag")
        if self.role == "multi_vector" and client.get("template", {}).get("pair") is not None:
            raise ValueError("a multi_vector recipe's template declares query and document shapes, not a pair")
        max_tokens = client.get("max_tokens")
        if isinstance(max_tokens, int) and max_tokens > self.serve.max_model_len:
            raise ValueError(
                f"client.max_tokens ({max_tokens}) must not exceed engine.max_model_len "
                f"({self.serve.max_model_len}): the engine would 400 the rendered prompt"
            )
        if self.role in ("embed", "multi_vector") and client.get("template") is not None:
            # The embed roles' clients fill no instruction span (their encode carries no instruction): a recipe
            # declaring one would render it empty -- silently, so it is refused at load.
            template = client.get("template") or {}
            for shape in ("query", "document"):
                segments = template.get(shape) or ()
                if any(segment.get("content") == "instruction" for segment in segments):
                    raise ValueError(
                        f"an {self.role} recipe's {shape!r} template declares an {{content: instruction}} span, "
                        "but the role's client cannot fill one (its encode carries no instruction); fold the "
                        "instruction into the query text, or serve the model as role=rerank"
                    )
        if "image" in self.input and not client.get("max_images"):
            raise ValueError(
                "recipe.input declares image but the client config carries max_images: 0 -- the client would "
                "refuse every image before the engine saw one; declare max_images (or drop the modality)"
            )
        if "video" in self.input and not client.get("max_videos"):
            raise ValueError(
                "recipe.input declares video but the client config carries max_videos: 0; declare max_videos "
                "(or drop the modality)"
            )
        _pixel_budgets_agree(self)
        return self


_PIXEL_KEYS = {"min_pixels": "min_px", "max_pixels": "max_px"}
"""The engine's pixel-budget keys and the client image policy's fields they mirror."""

_SIZE_KEYS = {"shortest_edge": "min_px", "longest_edge": "max_px"}
"""The HF Qwen-VL image processor's ``size`` keys (pixel counts despite their names) and the fields they mirror."""


def _pixel_pins(kwargs: dict[str, Any], prefix: str) -> list[tuple[str, str, Any]]:
    """Every pixel number one ``mm_processor_kwargs`` scope pins: ``(where, client field, value)``."""
    pins = [(f"{prefix}.{key}", field, kwargs[key]) for key, field in _PIXEL_KEYS.items() if key in kwargs]
    size = kwargs.get("size")
    if isinstance(size, dict):
        pins += [(f"{prefix}.size.{key}", field, size[key]) for key, field in _SIZE_KEYS.items() if key in size]
    return pins


def _pixel_budgets_agree(recipe: Recipe) -> None:
    """The client's image pixel budget and the engine's pinned one are the same numbers (R20, H4).

    The client resizes and counts every image under ``client.image_policy``; the engine resizes it again under
    its own budget -- the processor family's stock range, or what ``serve.mm_processor_kwargs`` pins: the
    nested ``images_kwargs`` (the one pixel-pin shape, read from the vLLM v0.31.0 source; it reaches the HF
    image processor and the vLLM-side image token budget) or the flat keys (which also reach every image),
    each as ``min_pixels``/``max_pixels`` or the HF processor's ``size: {shortest_edge, longest_edge}``.  A
    client policy that declares ``engine_pixel_pinning`` -- the only way to declare a budget outside the
    stock range -- needs the nested pin on serve; every pixel number serve pins must equal the client's
    declared budget, and a serve pin needs a client budget to agree with, or the counted tokens describe a
    size the engine never keeps.

    Raises:
        ValueError: a pinned client policy without the nested serve pin, a serve pin beside a client that
            declares no pixel budget, or a serve pin that differs from the client's declared budget.
    """
    policy = recipe.client.get("image_policy")
    policy = policy if isinstance(policy, dict) else None
    kwargs = recipe.serve.mm_processor_kwargs
    nested = kwargs.get("images_kwargs")
    nested = nested if isinstance(nested, dict) else {}
    if policy is not None and policy.get("engine_pixel_pinning"):
        missing = [key for key in _PIXEL_KEYS if key not in nested]
        if missing:
            raise ValueError(
                "client.image_policy declares engine_pixel_pinning, but serve.mm_processor_kwargs pins no "
                f"images_kwargs {' and '.join(missing)}: a stock engine would resize the prepared image again "
                f"(declare serve.mm_processor_kwargs: {{images_kwargs: {{min_pixels: {policy.get('min_px')}, "
                f"max_pixels: {policy.get('max_px')}}}}})"
            )
    pins = _pixel_pins(nested, "serve.mm_processor_kwargs.images_kwargs") + _pixel_pins(
        kwargs, "serve.mm_processor_kwargs"
    )
    if not pins:
        return
    if policy is None or policy.get("max_px") is None:
        raise ValueError(
            f"{pins[0][0]} pins the engine's image pixel budget, but client.image_policy declares none: the client "
            "would send images it cannot count under the budget the engine resizes them to (declare "
            "client.image_policy with the same numbers)"
        )
    for where, field, value in pins:
        declared = policy.get(field)
        if value != declared:
            raise ValueError(
                f"{where} ({value}) differs from client.image_policy's {field} ({declared}): the client counts "
                "every image under its declared budget and the engine resizes it under the pinned one, so both "
                "sides carry the same numbers"
            )


class VariantOverrides(BaseModel):
    """The per-variant overrides one variant applies to the family's shared blocks.

    Only the declared per-size fields (decision 34) may be overridden: :data:`PER_VARIANT_SERVE_FIELDS` under
    ``serve``, :data:`PER_VARIANT_CLIENT_FIELDS` under ``client``, and the whole ``resources`` block (the GPU
    count). A key outside the whitelist is refused with :class:`RecipeError` naming the field — nothing about a
    size's behaviour may differ silently. Each present key REPLACES the family's value at that path (no deep
    merge: a half-merged mapping is exactly the silent difference this refuses).
    """

    model_config = ConfigDict(**_no_extra())

    resources: Resources | None = None
    serve: dict[str, Any] = Field(default_factory=dict)
    client: dict[str, Any] = Field(default_factory=dict)


class Variant(BaseModel):
    """One size of a family: the per-size facts and the whitelisted overrides.

    Attributes:
        id: The variant's recipe id — the lowercased canonical Hub repo name, what ``serve``,
            ``recipe: <id>`` and the catalog all take; a family id is never served.
        model: The Hugging Face repo id to serve.
        revision: The exact commit of ``model`` (40 hex).
        notes: The variant's own notes, appended to the family's.
        sources: The variant's own citations (its model card at ``revision``), appended to the family's.
        status: The variant's verification status; the family's until a variant declares its own.
        overrides: The whitelisted per-size overrides (see :class:`VariantOverrides`).
    """

    model_config = ConfigDict(**_no_extra())

    id: str = Field(pattern=_ID_PATTERN)
    model: str = Field(min_length=1, description="Hugging Face repo id")
    revision: str = Field(pattern=_REVISION_PATTERN, description="40-hex commit of model")
    notes: str = ""
    sources: list[str] = Field(default_factory=list)
    status: StatusSpec | None = None
    overrides: VariantOverrides = Field(default_factory=VariantOverrides)

    @field_validator("sources")
    @classmethod
    def _sources_are_strings(cls, value: list[str]) -> list[str]:
        if any(not source.strip() for source in value):
            raise ValueError("sources entries must be non-empty URLs or path:line references")
        return value


class Family(BaseModel):
    """One model family and its sizes (decision 34): the shared serving contract plus a ``variants`` table.

    The family carries every block the variants share — role, input, scoring, licence, engine,
    resources, serve, client, reference, gates, status, sources, notes — and a ``variants`` table
    whose rows hold only the per-size facts. :func:`load_family` validates the file;
    :func:`resolve_recipe` / :func:`iter_recipes` expand each row into a full :class:`Recipe`
    (shared blocks + the variant's overrides, validated by the ``Recipe`` schema — the resolved
    recipe is exactly what a standalone recipe described). Construct through the loading functions.

    Attributes:
        id: The family's identifier, equal to the directory name; never a served id.
        schema_version: The family file format's version (the recipe contract's version, decision 18:
            every resolved recipe carries the family's value).
        role, input, scoring, licence: Shared across the family (the ``Recipe`` schema's meaning).
        engine: The engine image and the startup timeout.
        resources: The default GPUs the engine occupies; a variant's ``overrides.resources`` replaces it.
        serve: The shared ``vllm serve`` block; a variant's ``overrides.serve`` replaces whitelisted keys.
        client: The shared product endpoint config, plain data; ``model``/``revision``/``tokenizer`` are
            injected per variant and refused here. A variant's ``overrides.client`` replaces whitelisted keys.
        reference: The ONE subprocess reference the family's variants share (the harness passes the
            resolved recipe to it through ``--recipe``).
        gates: Shared stage-2 gate overrides.
        status: The default status; a variant's ``status`` replaces it.
        sources: The shared citations (engine behaviour, the paper), prepended to each variant's.
        notes: Shared notes, prepended to each variant's.
        variants: The sizes, in file order; every variant id is a full recipe id.
    """

    model_config = ConfigDict(**_no_extra())

    id: str = Field(pattern=_ID_PATTERN)
    schema_version: str = Field(
        pattern=r"^\d+$",
        description="the family/recipe file format's version (decision 18); every resolved recipe carries it",
    )
    role: Role
    input: list[Literal["text", "image", "video"]] = Field(min_length=1)
    scoring: Literal["pointwise", "listwise"] | None = None
    licence: str = Field(min_length=1)
    engine: EngineSpec
    resources: Resources
    serve: ServeConfig
    client: dict[str, Any]
    reference: ReferenceSpec
    gates: Gates = Field(default_factory=Gates)
    status: StatusSpec = Field(default_factory=StatusSpec)
    sources: list[str] = Field(default_factory=list)
    notes: str = ""
    variants: list[Variant] = Field(min_length=1)

    @field_validator("sources")
    @classmethod
    def _sources_are_strings(cls, value: list[str]) -> list[str]:
        if any(not source.strip() for source in value):
            raise ValueError("sources entries must be non-empty URLs or path:line references")
        return value

    @model_validator(mode="after")
    def _family_rules(self) -> Family:
        """The family-level rules: unique variant ids, and overrides restricted to the per-size fields."""
        ids = [variant.id for variant in self.variants]
        duplicates = sorted({variant_id for variant_id in ids if ids.count(variant_id) > 1})
        if duplicates:
            raise ValueError(f"variant ids must be unique within the family, got duplicates {duplicates}")
        if len(self.variants) > 1 and self.id in ids:
            raise ValueError(
                f"variant id {self.id!r} equals the family id: family ids are never served, and a multi-variant "
                "family whose one variant shares the family's name would hide it from the catalog"
            )
        for variant in self.variants:
            misplaced_serve = sorted(set(variant.overrides.serve) - set(PER_VARIANT_SERVE_FIELDS))
            if misplaced_serve:
                raise ValueError(
                    f"variant {variant.id!r}: overrides.serve carries {misplaced_serve}, which are not declared "
                    f"per-size fields (allowed: {', '.join(PER_VARIANT_SERVE_FIELDS)}): the family shares every "
                    "other serve field, so a size that needs another value differs in behaviour the family "
                    "cannot see -- split the family, or move the field into the shared block"
                )
            misplaced_client = sorted(set(variant.overrides.client) - set(PER_VARIANT_CLIENT_FIELDS))
            if misplaced_client:
                raise ValueError(
                    f"variant {variant.id!r}: overrides.client carries {misplaced_client}, which are not "
                    f"declared per-size fields (allowed: {', '.join(PER_VARIANT_CLIENT_FIELDS)}): the client "
                    "block is the family's shared contract (template, instruction mode, overflow rule, media "
                    "policies), so a size that differs there is a modelling error, not an override"
                )
        return self


def default_recipes_root() -> Path:
    """The package's shipped ``recipes/`` directory (package data, read through :mod:`importlib.resources`).

    Raises:
        RecipeError: the install is zipped (the recipes are not on the filesystem); install the wheel unpacked.
    """
    root = resources.files("rcp_ndcg_vllm").joinpath("recipes")
    if not isinstance(root, Path):
        raise RecipeError("the shipped recipes are not on the filesystem (zipped install); install the wheel unpacked")
    return root


class _DuplicateKeyError(ValueError):
    """A YAML mapping declares one key twice (YAML would keep the last silently)."""


class _UniqueKeyLoader(yaml.SafeLoader):
    """``yaml.SafeLoader`` that refuses a mapping declaring one key twice: YAML keeps the last of two equal keys
    silently, so a recipe that declares a field twice would serve whichever value came last."""


def _unique_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
    seen: set[Any] = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)  # pyright: ignore[reportUnknownMemberType]
        if key in seen:
            raise _DuplicateKeyError(f"duplicate key {key!r} at line {key_node.start_mark.line + 1}")
        seen.add(key)
    return loader.construct_mapping(node, deep=deep)


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def load_family(path: str | Path) -> Family:
    """Load and validate one family from a family directory or a ``family.yaml`` file.

    Inputs: ``path``, the family directory (containing ``family.yaml``) or the YAML file itself.  Outputs: the
    validated :class:`Family` (shared blocks + the ``variants`` table; no expansion).  Raises :class:`RecipeError`
    with the file path and the validator message when the YAML declares a key twice in one mapping (YAML would
    keep the last silently), does not satisfy the schema, or overrides a field the family schema does not declare
    per-size, or when ``family.id`` differs from the directory name.
    """
    path = Path(path)
    yaml_path = path / "family.yaml" if path.is_dir() else path
    if not yaml_path.is_file():
        raise RecipeError(f"no family at {path}: expected {yaml_path}")
    try:
        data = yaml.load(yaml_path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)  # noqa: S506 - safe loader
    except _DuplicateKeyError as error:
        raise RecipeError(f"{yaml_path}: {error}") from error
    except yaml.YAMLError as error:
        raise RecipeError(f"{yaml_path} is not valid YAML: {error}") from error
    if not isinstance(data, dict):
        raise RecipeError(f"{yaml_path} must contain a YAML mapping of the Family schema, got {type(data).__name__}")
    try:
        family = Family.model_validate(data)
    except Exception as error:
        raise RecipeError(f"{yaml_path}: {error}") from error
    if path.is_dir() and path.name != family.id:
        raise RecipeError(f"{yaml_path}: family id {family.id!r} must equal the directory name {path.name!r}")
    return family


def _family_dirs(root: Path) -> list[Path]:
    """The family directories under ``root``, sorted (a directory carrying ``family.yaml``)."""
    return sorted(p for p in root.iterdir() if p.is_dir() and (p / "family.yaml").is_file())


def _expand_variant(family: Family, variant: Variant, directory: Path, yaml_path: Path) -> Recipe:
    """One variant's resolved :class:`Recipe`: the family's shared blocks plus its whitelisted overrides.

    The merge is key-level replacement (``overrides`` keys replace the family's value; nothing deep-merges),
    the client block gains the injected ``model``/``revision``/``tokenizer``, and the result validates against
    the unchanged ``Recipe`` schema — the resolved recipe is exactly what a standalone recipe described. The
    referenced files (the family's template, the one ``reference.py``) are checked against the family directory.
    """
    serve = dict(family.serve.model_dump(mode="json"))
    serve.update(variant.overrides.serve)
    client = dict(family.client)
    if overridden := sorted(set(variant.overrides.client) & {"model", "revision", "tokenizer"}):
        raise RecipeError(
            f"{yaml_path}: variant {variant.id!r} overrides.client carries {overridden}: model and revision are "
            "the variant's own fields, and the tokenizer is the family's shared client field (or injected as "
            "model@revision) -- never a per-variant override"
        )
    client.update(variant.overrides.client)
    # client.model/revision are the variant's identity and client.tokenizer its checkpoint's tokenizer spec:
    # injected, refused in the family YAML (the Family schema's client block is plain data, so the refusal
    # rides here, where the family file is in hand).
    for injected in ("model", "revision"):
        if injected in client:
            raise RecipeError(
                f"{yaml_path}: client.{injected} is the variant's own {injected}; drop the field: the loader "
                "injects it from the variant row"
            )
    client.setdefault("model", variant.id)
    client.setdefault("revision", variant.revision)
    client.setdefault("tokenizer", f"{variant.model}@{variant.revision}")
    notes = family.notes
    if variant.notes:
        notes = f"{family.notes}\n\n{variant.notes}" if family.notes else variant.notes
    data = {
        "id": variant.id,
        "schema_version": family.schema_version,
        "model": variant.model,
        "revision": variant.revision,
        "role": family.role,
        "input": family.input,
        "scoring": family.scoring,
        "licence": family.licence,
        "engine": family.engine.model_dump(mode="json"),
        "resources": (variant.overrides.resources or family.resources).model_dump(mode="json"),
        "serve": serve,
        "client": client,
        "reference": family.reference.model_dump(mode="json"),
        "gates": family.gates.model_dump(mode="json"),
        "status": (variant.status or family.status).model_dump(mode="json"),
        "sources": [*family.sources, *variant.sources],
        "notes": notes,
    }
    try:
        recipe = Recipe.model_validate(data)
    except Exception as error:
        raise RecipeError(f"{yaml_path}: variant {variant.id!r} does not resolve to a valid recipe: {error}") from error
    recipe._dir = directory
    _check_referenced_files(recipe, directory)
    return recipe


def load_recipes_of(family: Family, directory: Path) -> list[Recipe]:
    """Every variant of ``family`` (loaded from ``directory``), expanded to resolved recipes in file order."""
    yaml_path = directory / "family.yaml"
    return [_expand_variant(family, variant, directory, yaml_path) for variant in family.variants]


def load_recipe(source: str | Path, *, root: str | Path | None = None) -> Recipe:
    """The resolved recipe: a variant id (resolved under ``root``, default the shipped ones) or a
    single-variant family path.

    Inputs: ``source`` — a variant id (never a family id: family ids are not served), or a path to a
    family directory / ``family.yaml`` whose family has exactly one variant; ``root`` names the recipes
    root for an id (default: the package's ``recipes/``).  Outputs: the frozen resolved :class:`Recipe`.
    Raises :class:`RecipeError` when the source names no recipe, names a multi-variant family (the error
    lists the variant ids and names :func:`resolve_recipe`), or fails validation.
    """
    candidate = Path(source)
    if candidate.exists():
        # a filesystem path: a family directory or a family.yaml file
        yaml_path = candidate / "family.yaml" if candidate.is_dir() else candidate
        if yaml_path.name == "recipe.yaml":
            raise RecipeError(
                f"{yaml_path}: the standalone recipe.yaml path is gone (decision 34: one family, many sizes): "
                "wrap the recipe in a family.yaml with one variant"
            )
        family = load_family(candidate)
        if len(family.variants) != 1:
            ids = ", ".join(variant.id for variant in family.variants)
            raise RecipeError(
                f"{candidate}: family {family.id!r} declares {len(family.variants)} variants ({ids}); "
                "load_recipe resolves one recipe: name a variant id (resolve_recipe), not the family"
            )
        directory = candidate if candidate.is_dir() else yaml_path.parent
        return load_recipes_of(family, directory)[0]
    if not isinstance(source, str) or not re.fullmatch(_ID_PATTERN, source):
        raise RecipeError(f"no recipe at {source}: name a variant id or a family directory")
    return resolve_recipe(source, root=root)


def resolve_recipe(variant_id: str, root: str | Path | None = None) -> Recipe:
    """The resolved :class:`Recipe` of the variant id ``variant_id`` under ``root`` (default: the shipped ones).

    Inputs: the recipe id (the variant id of the family that declares it) and the recipes root whose family
    directories carry it.  Outputs: the frozen resolved recipe, its ``_dir`` set to the family directory.
    Raises :class:`RecipeError`: an unknown id (the known ids named), or the first family that failed to load —
    a broken family must not read as an unknown id.
    """
    root = Path(root) if root is not None else default_recipes_root()
    if not root.is_dir():
        raise RecipeError(f"no recipe root at {root}")
    failure: RecipeError | None = None
    known: list[str] = []
    for directory in _family_dirs(root):
        try:
            family = load_family(directory)
        except RecipeError as error:
            failure = failure or error
            continue
        known.extend(variant.id for variant in family.variants)
        for variant in family.variants:
            if variant.id == variant_id:
                return _expand_variant(family, variant, directory, directory / "family.yaml")
    if failure is not None:
        raise failure
    known_text = ", ".join(sorted(known)) if known else "(none)"
    raise RecipeError(
        f"no recipe variant {variant_id!r} under {root}; the known recipe ids are: {known_text}. "
        "Family ids are never served: name a variant id"
    )


def iter_families(root: str | Path | None = None) -> list[Family]:
    """Load every family under ``root`` (default: the package's ``recipes/`` directory).

    Inputs: a root directory whose direct children are family directories (``family.yaml``).  Outputs: the
    families, sorted by id.  Raises :class:`RecipeError` naming the directory when any family fails to load —
    a broken family among thirteen must not pass silently.
    """
    root = Path(root) if root is not None else default_recipes_root()
    if not root.is_dir():
        raise RecipeError(f"no recipe root at {root}")
    return [load_family(directory) for directory in _family_dirs(root)]


def iter_recipes(root: str | Path | None = None) -> list[Recipe]:
    """Every variant of every family under ``root`` (default: the package's ``recipes/``), as resolved recipes.

    Inputs: a root of family directories.  Outputs: the resolved recipes, sorted by id — one :class:`Recipe`
    per variant id, exactly the ids the catalog, ``serve`` and the wave lists take.  Raises :class:`RecipeError`
    when a family fails to load or two families resolve the same variant id.
    """
    root = Path(root) if root is not None else default_recipes_root()
    if not root.is_dir():
        raise RecipeError(f"no recipe root at {root}")
    recipes: list[Recipe] = []
    seen: dict[str, Path] = {}
    for directory in _family_dirs(root):
        family_recipes = load_recipes_of(load_family(directory), directory)
        for recipe in family_recipes:
            if recipe.id in seen:
                raise RecipeError(
                    f"recipe id {recipe.id!r} resolves from two families ({seen[recipe.id].name} and "
                    f"{directory.name}): variant ids must be unique across the catalog"
                )
            seen[recipe.id] = directory
        recipes.extend(family_recipes)
    return sorted(recipes, key=lambda recipe: recipe.id)


def _check_referenced_files(recipe: Recipe, directory: Path) -> None:
    """Every file the recipe names must exist, or the recipe would serve and fail later."""
    if recipe.serve.chat_template is not None and not (directory / recipe.serve.chat_template).is_file():
        raise RecipeError(
            f"{directory / 'recipe.yaml'}: serve.chat_template {recipe.serve.chat_template!r} does not exist in "
            f"{directory}"
        )
    needs_reference = recipe.reference.kind != "stored_scores"
    if needs_reference and not (directory / recipe.reference.entry).is_file():
        raise RecipeError(
            f"{directory / 'recipe.yaml'}: reference.entry {recipe.reference.entry!r} does not exist in {directory}; "
            "a recipe needs reference.py unless reference.kind is stored_scores"
        )


def serve_argv(recipe: Recipe, *, port: int, served_model_name: str) -> list[str]:
    """Render the ``vllm serve`` argv a recipe stands for.

    Inputs: a loaded :class:`Recipe`, the port to serve on and the ``--served-model-name`` (the wave runner uses
    the recipe's ``id``).  Output: ``["vllm", "serve", <model>, "--revision", ..., ...]`` — the fixed head, then
    one flag per ``serve`` field in a deterministic order (JSON objects with ``json.dumps(sort_keys=True)``), then
    ``extra_args`` verbatim.  ``serve.plugin`` and ``serve.io_processor_plugin`` render nothing: they name pip
    packages installed before the engine starts.  Raises :class:`RecipeError` when the recipe sets
    ``serve.chat_template`` but was not loaded from a directory (the template's absolute path is needed).
    """
    argv = [
        "vllm",
        "serve",
        recipe.model,
        "--revision",
        recipe.revision,
        "--served-model-name",
        served_model_name,
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--tensor-parallel-size",
        str(recipe.resources.gpus),
        "--runner",
        recipe.serve.runner,
    ]
    if recipe.serve.convert is not None:
        argv += ["--convert", recipe.serve.convert]
    argv += ["--dtype", recipe.serve.dtype]
    argv += ["--max-model-len", str(recipe.serve.max_model_len)]
    if recipe.serve.trust_remote_code:
        argv.append("--trust-remote-code")
    argv += ["--hf-overrides", json.dumps(recipe.serve.hf_overrides, sort_keys=True)]
    if recipe.serve.chat_template is not None:
        directory = recipe._dir
        if directory is None:
            raise RecipeError(
                f"recipe {recipe.id}: serve.chat_template needs the recipe directory; load the recipe with "
                "load_recipe, or clear serve.chat_template"
            )
        argv += ["--chat-template", str(directory / recipe.serve.chat_template)]
    argv += ["--pooler-config", json.dumps(recipe.serve.pooler_config, sort_keys=True)]
    if recipe.serve.mm_processor_kwargs:
        argv += ["--mm-processor-kwargs", json.dumps(recipe.serve.mm_processor_kwargs, sort_keys=True)]
    if recipe.serve.limit_mm_per_prompt is not None:
        argv += ["--limit-mm-per-prompt", json.dumps(recipe.serve.limit_mm_per_prompt, sort_keys=True)]
    argv += list(recipe.serve.extra_args)
    return argv


def client_config(recipe: Recipe, *, base_url: str | None) -> dict[str, Any]:
    """The product's endpoint config dict this recipe's ``client`` block implies, with ``base_url`` filled.

    Inputs: a recipe and the endpoint's ``base_url`` (e.g. ``http://127.0.0.1:8100/v1``), or ``None`` for a
    serve-by-role run whose URLs arrive at runtime.  Output: a plain dict — the recipe's ``client`` block plus
    the ``base_url`` key — exactly what the product's config loader accepts; it validates the block with the
    product's endpoint model when it reads it.  A ``client.recipe`` the recipe declared itself is kept as
    declared (never overwritten with the recipe id).
    """
    client = dict(recipe.client)
    client.setdefault("recipe", recipe.id)
    client["base_url"] = base_url
    return client


def recipe_json_schema() -> dict[str, Any]:
    """The JSON Schema of :class:`Recipe`, exported to ``schema/recipe.schema.json`` and kept current by a test."""
    return Recipe.model_json_schema()


def family_json_schema() -> dict[str, Any]:
    """The JSON Schema of :class:`Family`, exported to ``schema/family.schema.json`` and kept current by a test."""
    return Family.model_json_schema()
