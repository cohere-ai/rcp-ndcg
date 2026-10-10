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

The **deployment surface** is declared once, in :data:`FIELD_ROLES`, beside the fields it names: every field of
``serve``, ``engine`` and ``resources`` carries a role -- ``CONTENT`` (it shapes what the model returns: a
different value is a different variant), ``RUNTIME`` (the run owns it) or ``DEPLOYMENT`` (the engine's
resource, scheduling and address knobs). ``rcp-ndcg-vllm serve <id> --set <path>=<value>`` may name exactly the
DEPLOYMENT paths and renders them into the argv; a CONTENT path is refused by name, with the hint that a
different revision or content is a different variant. ``resources.gpus`` and ``serve.max_model_len`` are recipe
fields whose declared value an override replaces -- the latter only at or above the client's largest token
budget; the other deployment fields are the engine's own flags, which the recipe does not carry (the engine's
defaults apply until an operator sets them).

A recipe may also be **a file of the operator's own**: ``rcp-ndcg-vllm serve ./family-dir/ [--variant <id>]``
(and ``recipe:./family-dir`` in an rcp-ndcg config) loads a family directory through this same schema, marks it
unshipped (:attr:`Recipe.shipped`) with ``status: unverified``, and identifies it by the content hash of its
resolved form (:attr:`Recipe.identity`) instead of a shipped id.

Public names (pinned by ``tests/contract``):

- :func:`load_recipe` -- the resolved recipe of a variant id, or of a family directory (``variant`` selects one
  of several).
- :func:`resolve_recipe` -- the resolved recipe of a variant id under a recipes root (the explicit form).
- :func:`load_family` -- one family directory or ``family.yaml`` file.
- :func:`iter_families` -- every family under a root (default: the shipped ones).
- :func:`iter_recipes` -- every variant of every family under a root, as resolved :class:`Recipe` objects.
- :func:`serve_argv` -- the ``vllm serve`` argv a recipe renders to, deployment overrides included.
- :func:`deployment_fields` -- the ``--set`` paths the schema declares DEPLOYMENT.
- :func:`parse_deployment_overrides` -- the ``--set <path>=<value>`` pairs, checked against that declaration.
- :func:`recipe_digest` -- the content hash of a recipe's resolved form (an unshipped recipe's identity).
- :class:`RecipeFieldRole` -- the role vocabulary the declaration uses (``CONTENT``/``RUNTIME``/``DEPLOYMENT``).
- :data:`FIELD_ROLES` -- the declaration itself: every ``--set`` path with its role and rendering.
- :class:`FieldSpec` -- one entry of that declaration (role, flag, value kind and range).

Everything else in this module is internal.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from importlib import resources
from pathlib import Path
from typing import Any, Literal

import yaml  # pyright: ignore[reportMissingModuleSource]
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator

from .errors import RecipeError
from .patches import PATCH_NAMES

__all__ = [
    "FIELD_ROLES",
    "Family",
    "FieldSpec",
    "Recipe",
    "RecipeFieldRole",
    "Variant",
    "deployment_fields",
    "iter_families",
    "iter_recipes",
    "load_family",
    "load_recipe",
    "parse_deployment_overrides",
    "recipe_digest",
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
    "patches",
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
        min_version: The engine version the recipe is known to work with, ``MAJOR.MINOR.PATCH`` (a release
            candidate or a setuptools-scm dev series counts; for a digest-pinned nightly the image digest is
            the real pin and the version is the floor).
        startup_timeout_s: How long :mod:`rcp_ndcg_test.jobs.run_wave` waits for ``GET /v1/models`` before it
            declares the recipe failed (seconds).  Large models override this per recipe.
        step_budget_s: The floor of every harness step's wall-clock budget, seconds (GPU-E1: one stuck
            request once held a node for hours).  The runner computes each step's budget from the
            recipe's request count; a recipe that legitimately needs longer declares the floor here and
            the runner only raises its formula to it.
    """

    model_config = ConfigDict(**_no_extra())

    name: Literal["vllm"]
    image: str = Field(min_length=1, description="repository:tag of the engine image")
    min_version: str = Field(
        pattern=r"^\d+\.\d+\.\d+(rc\d+)?(\.dev\d+)?$",
        description="known-good engine version (a release candidate or a setuptools-scm dev series counts)",
    )
    startup_timeout_s: int = Field(default=1800, gt=0)
    step_budget_s: int | None = Field(
        default=None,
        gt=0,
        description="floor of every harness step's wall-clock budget (seconds); the runner's formula "
        "from the recipe's request count can only raise it",
    )


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
    patches: list[str] = Field(
        default_factory=list,
        description="engine-side patch modules the engine process opts into through RCP_NDCG_VLLM_PATCHES; "
        "names must be shipped by this package (rcp_ndcg_vllm.patches.PATCH_NAMES). Content: the patched "
        "engine is a different serving environment.",
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

    @field_validator("patches")
    @classmethod
    def _patch_names_are_shipped(cls, value: list[str]) -> list[str]:
        unknown = sorted(set(value) - set(PATCH_NAMES))
        if unknown:
            raise ValueError(
                f"serve.patches names {unknown}, which this package does not ship; known patches: "
                f"{', '.join(PATCH_NAMES)}. An unknown name would be ignored by the engine silently."
            )
        return list(dict.fromkeys(value))

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
        device: The device the reference must run on, ``cpu`` or ``cuda`` (GPU-E1: the wave's references
            all ran on the pod's CPU, where the Qwen3.5-based references cannot run at all and the bf16
            engines' precision differs).  ``None`` (the default) leaves the choice to the runner;
            ``cuda`` requires a GPU of the reference's own beside the engine's (never the engine's GPU),
            and a CPU reference run for such a recipe is refused with that hint.
        attn_implementation: The attention implementation the reference loads its checkpoint with, a
            declared parameter rather than a silent ``torch.cuda.is_available()`` choice: the stock
            reference environment carries the image's torch and no compiled extras, so
            ``flash_attention_2`` cannot load there (GPU-E1: the six reranker references died on it).
            ``None`` (the default) leaves the choice to the reference's own code; the reranker families
            declare ``sdpa`` -- the GPU-E1 follow-up measured sdpa references: Kendall tau 1.0 and
            max |delta| <= 0.041 for qwen3-reranker, ctxl-6b verified -- and the recipe notes carry the
            evidence.
    """

    model_config = ConfigDict(**_no_extra())

    kind: Literal["transformers", "sentence_transformers", "remote_code", "stored_scores"]
    score_scale: ScoreScale
    entry: str = Field(default="reference.py", description="reference module file inside the recipe directory")
    known_deviations: list[Literal["anchor_drop_over_cap", "over_cap_cut_differs"]] = Field(default_factory=list)
    device: Literal["cpu", "cuda"] | None = Field(
        default=None,
        description='the device the reference must run on ("cuda": a GPU of its own is required; None: the '
        "runner decides)",
    )
    attn_implementation: Literal["sdpa", "flash_attention_2", "eager"] | None = Field(
        default=None,
        description="the attention implementation the reference loads with (declared, never chosen by "
        "torch.cuda.is_available(): the stock reference environment has no compiled extras); None leaves it "
        "to the reference's own code",
    )

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


class RecipeFieldRole(StrEnum):
    """What a field of the recipe schema may do to a served identity.

    The vocabulary is rcp-ndcg's CONTENT/RUNTIME split (``rcp_ndcg.support.identity``), re-declared here because
    this package must not import ``rcp-ndcg``: the engine image installs it with ``--no-deps`` and rcp-ndcg is
    not there. The role that is new is the one that makes a field an operator's knob:

    ``CONTENT``
        the field shapes what the model returns: two different values are two different variants, so a
        serve-time override is refused -- *a different revision or content is a different variant: add a variant
        row*.
    ``RUNTIME``
        the field is out of the served identity (the run owns it, e.g. ``engine.startup_timeout_s``) and it is
        not a serve-time knob either.
    ``DEPLOYMENT``
        a RUNTIME field an operator may set at serve time (``rcp-ndcg-vllm serve <id> --set <path>=<value>``):
        the engine's resource, scheduling and address knobs. ``resources.gpus`` and ``serve.max_model_len`` are
        recipe fields whose declared value an override replaces; the rest are the engine's own flags, which the
        recipe schema does not carry (the engine's defaults apply until an operator sets them).
    """

    CONTENT = "content"
    RUNTIME = "runtime"
    DEPLOYMENT = "deployment"


@dataclass(frozen=True)
class FieldSpec:
    """One entry of :data:`FIELD_ROLES`: a field's role and, for a DEPLOYMENT field, how it renders.

    Attributes:
        role: The field's role (see :class:`RecipeFieldRole`).
        flag: The ``vllm serve`` flag the value renders to (DEPLOYMENT fields only).
        kind: How ``--set`` parses the value's text: ``int``, ``float`` or ``str``.
        low: The smallest accepted value, when the field has a floor (exclusive when
            :attr:`low_exclusive`).  ``serve.port``'s floor is 0: the engine then binds an ephemeral port,
            which the wave runner's stub engines announce.
        high: The largest accepted value (inclusive), when the field has a ceiling.
        low_exclusive: Whether ``low`` itself is refused (``serve.gpu_memory_utilization``: the engine's flag
            is a fraction strictly above zero).
        position: Where the value renders in the argv: ``head`` (the address and resource block -- ``--host``,
            ``--port``, ``--tensor-parallel-size``), ``body`` (``--max-model-len``) or ``tail`` (the scheduling
            knobs, after the body).
        default: The value when neither the recipe nor the caller supplies one (``serve.host``: the interface
            every shipped recipe has always been served on).
    """

    role: RecipeFieldRole
    flag: str | None = None
    kind: type = str
    low: float | None = None
    high: float | None = None
    low_exclusive: bool = False
    position: Literal["head", "body", "tail"] = "tail"
    default: Any = None


FIELD_ROLES: Mapping[str, FieldSpec] = {
    # The deployment fields, in argv order. Each is a resource, scheduling or address knob of the engine: it
    # cannot change what the model returns, so an operator may set it at serve time.
    "serve.host": FieldSpec(RecipeFieldRole.DEPLOYMENT, "--host", str, position="head", default="0.0.0.0"),
    "serve.port": FieldSpec(RecipeFieldRole.DEPLOYMENT, "--port", int, low=0, high=65535, position="head"),
    "resources.gpus": FieldSpec(RecipeFieldRole.DEPLOYMENT, "--tensor-parallel-size", int, low=1, position="head"),
    "serve.max_model_len": FieldSpec(RecipeFieldRole.DEPLOYMENT, "--max-model-len", int, low=1, position="body"),
    "serve.gpu_memory_utilization": FieldSpec(
        RecipeFieldRole.DEPLOYMENT, "--gpu-memory-utilization", float, low=0, high=1, low_exclusive=True
    ),
    "serve.max_num_seqs": FieldSpec(RecipeFieldRole.DEPLOYMENT, "--max-num-seqs", int, low=1),
    "serve.max_num_batched_tokens": FieldSpec(RecipeFieldRole.DEPLOYMENT, "--max-num-batched-tokens", int, low=1),
    # The declared non-deployment fields: a --set naming one is refused with its role's reason, never silently.
    "engine.startup_timeout_s": FieldSpec(RecipeFieldRole.RUNTIME),
}
"""Every path ``--set`` may name, with its role (the one declaration: the CLI reads it and never lists fields
itself). Every other field of the recipe schema is CONTENT by default -- it shapes what the model returns, so a
serve-time override of it is a different variant. A DEPLOYMENT path that is also a schema field
(``resources.gpus``, ``serve.max_model_len``) takes the recipe's declared value as its default."""

_CONTENT_HINT = "a different revision or content is a different variant: add a variant row"
"""The one refusal hint for a CONTENT override (the brief's wording, verbatim: a variant row is the answer)."""


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

    Derived (never part of the file): :attr:`shipped` -- whether the recipe is a shipped one -- and
    :attr:`identity`, the shipped id or ``unshipped:sha256:<hex>``.
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

    _shipped: bool = PrivateAttr(default=True)
    """Whether the recipe is one of the package's shipped ones (set by the loading functions; ``False`` for a
    family directory loaded by path or under a root of the caller's own)."""

    _identity: str | None = PrivateAttr(default=None)
    """The unshipped identity, computed once at load (the recipe is a snapshot of the files it was read from);
    ``None`` until :attr:`identity` computes it for a recipe built by hand."""

    @property
    def shipped(self) -> bool:
        """Whether this recipe is a shipped one (package data).

        An unshipped recipe is a file of the operator's own: it is marked ``unverified`` in every record (its
        ``status`` is the loader's, never the file's claim) and its :attr:`identity` is a content hash, never a
        shipped id.
        """
        return self._shipped

    @property
    def identity(self) -> str:
        """The recipe's identity in every record and client config: the shipped id, or
        ``unshipped:sha256:<hex>`` for an unshipped recipe.

        The hash is :func:`recipe_digest` -- the content hash of the resolved form, template file included --
        computed once when the recipe is loaded, so two runs whose files differ never share an identity, two
        directories holding the same files are one recipe, and an identity never moves under a loaded recipe.
        """
        if self._shipped:
            return self.id
        if self._identity is None:
            self._identity = f"unshipped:sha256:{recipe_digest(self)}"
        return self._identity

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
            # An embed or multi-vector recipe's client CAN fill an instruction span (its encode takes the task
            # instruction and the fit renders the span): the client must declare the policy, or the span would
            # render empty -- and the messages route cannot carry one at all (it sends the content and leaves
            # the frame to the engine's chat template). The product's own config rules, restated here so a
            # recipe fails at load rather than at its first client read.
            template = client.get("template") or {}
            article = "an" if self.role == "embed" else "a"
            for shape in ("query", "document"):
                segments = template.get(shape) or ()
                if not any(segment.get("content") == "instruction" for segment in segments):
                    continue
                if client.get("request_shape") == "messages":
                    raise ValueError(
                        f"{article} {self.role} recipe's {shape!r} template declares an {{content: instruction}} span "
                        "and request_shape: messages sends the content only: the engine's chat template cannot "
                        "render the span, so the instruction would be dropped; declare request_shape: text, or "
                        "drop the template's instruction span"
                    )
                if client.get("instruction") != "fold":
                    raise ValueError(
                        f"{article} {self.role} recipe's {shape!r} template declares an {{content: instruction}} span, "
                        "but the client block declares no instruction policy (or none): the span would render "
                        "empty; declare instruction: fold (the fit fills the span with the task instruction)"
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
        _video_pixel_budgets_agree(self)
        _video_pruning_agrees(self)
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


def _video_pixel_budgets_agree(recipe: Recipe) -> None:
    """The client's video pixel budget and the engine's pinned one are the same numbers.

    The engine's per-clip video budget lives in ``serve.mm_processor_kwargs``'s ``videos_kwargs``
    scope (or the flat keys, which also reach the video processor).  The client counts a
    ``qwen3_vl`` clip under the processor family's stock ceiling unless
    ``client.video_policy.engine_video_max_pixels``/``engine_video_min_pixels`` declare the
    engine's pin -- so a serve pin without the client declaration (or a client declaration without
    the pin) would count tokens the engine never renders.  Both directions are refused.

    Raises:
        ValueError: a video pixel pin in one half only, or a pin that differs between the halves.
    """
    policy = recipe.client.get("video_policy")
    policy = policy if isinstance(policy, dict) else {}
    kwargs = recipe.serve.mm_processor_kwargs
    videos = kwargs.get("videos_kwargs")
    videos = videos if isinstance(videos, dict) else {}
    pairs = (
        ("min_pixels", "engine_video_min_pixels"),
        ("max_pixels", "engine_video_max_pixels"),
    )
    for serve_key, client_key in pairs:
        serve_value = videos.get(serve_key, kwargs.get(serve_key))
        client_value = policy.get(client_key)
        if serve_value is not None and client_value != serve_value:
            raise ValueError(
                f"serve.mm_processor_kwargs pins {serve_key} {serve_value}, but client.video_policy declares "
                f"{client_key} {client_value!r}: the engine's clip budget would differ from the one the "
                f"client counts (declare video_policy.{client_key}: {serve_value}, or drop the serve pin)"
            )
        if client_value is not None and serve_value is None:
            raise ValueError(
                f"client.video_policy declares {client_key} {client_value}, but serve.mm_processor_kwargs "
                f"pins no video {serve_key}: the engine's own default clip budget would decide, and the "
                f"client's count describes a different clip (pin serve.mm_processor_kwargs: "
                f"{{videos_kwargs: {{{serve_key}: {client_value}}}}})"
            )


def _flag_value(args: list[str], flag: str) -> str | None:
    """The last value of ``flag`` in ``args``: ``--flag value`` or ``--flag=value`` (argparse's last wins)."""
    value: str | None = None
    for index, arg in enumerate(args):
        if arg == flag:
            if index + 1 < len(args):
                value = args[index + 1]
        elif arg.startswith(f"{flag}="):
            value = arg.split("=", 1)[1]
    return value


def _video_pruning_agrees(recipe: Recipe) -> None:
    """The engine's video-token pruning changes the Qwen-VL video prompt layout: the client's ``video_policy``
    declares the same rate and method the serve args carry, and declares none when they carry none.

    vLLM ``--video-pruning-rate`` (and ``--video-pruning-method``, default ``evs``) retains a computed subset
    of the per-frame video tokens and renders them in the first temporal group; the client counts that layout
    only when the rate and method are declared (``VideoPolicy.engine_video_pruning`` and
    ``engine_video_pruning_method``). A flag the client has not declared -- or a declaration the serve args
    do not carry -- would make the counted tokens describe a prompt the engine never renders.

    Raises:
        ValueError: a nonzero serve rate the client does not declare (or declares differently), a declared
            method that differs from the serve method, a client rate without the flag, or an inert
            ``--video-pruning-method``.
    """
    args = list(recipe.serve.extra_args)
    rate_arg = _flag_value(args, "--video-pruning-rate")
    method_arg = _flag_value(args, "--video-pruning-method")
    policy = recipe.client.get("video_policy")
    policy = policy if isinstance(policy, dict) else {}
    declared_rate = policy.get("engine_video_pruning")
    declared_method = policy.get("engine_video_pruning_method")
    try:
        rate = float(rate_arg) if rate_arg is not None else 0.0
    except ValueError as error:
        raise ValueError(f"serve.extra_args --video-pruning-rate {rate_arg!r} is not a number") from error
    if method_arg is not None and rate <= 0:
        raise ValueError(
            "serve.extra_args carries --video-pruning-method with no nonzero --video-pruning-rate: the flag is inert"
        )
    if rate > 0:
        method = method_arg or "evs"
        if declared_rate != rate:
            raise ValueError(
                f"serve.extra_args pins --video-pruning-rate {rate:g}, but client.video_policy declares "
                f"engine_video_pruning {declared_rate!r}: the engine's video prompt layout is the pruned one, "
                "and the client's count describes a different prompt (declare the same rate and method, or "
                "serve without the flag)"
            )
        if declared_method != method:
            raise ValueError(
                f"serve.extra_args runs --video-pruning-method {method!r}, but client.video_policy declares "
                f"engine_video_pruning_method {declared_method!r}: both sides must name the same algorithm"
            )
    elif declared_rate not in (None, 0.0):
        raise ValueError(
            f"client.video_policy declares engine_video_pruning {declared_rate!r}, but serve.extra_args carries "
            "no nonzero --video-pruning-rate: the declared layout would never be the served one (add the flag, "
            "or drop the declaration)"
        )


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


def deployment_fields() -> dict[str, FieldSpec]:
    """The deployment surface: every path ``--set`` may name, with its flag, value kind and range.

    Inputs: none.  Outputs: :data:`FIELD_ROLES`' DEPLOYMENT entries, in argv order (the order
    :func:`serve_argv` renders them in).  The declaration is the only source of the allowed set: the CLI
    reads this, it never lists fields itself.
    """
    return {path: spec for path, spec in FIELD_ROLES.items() if spec.role is RecipeFieldRole.DEPLOYMENT}


def recipe_digest(recipe: Recipe) -> str:
    """The content hash of a recipe's **resolved form**: the identity of an unshipped recipe.

    Inputs: a loaded :class:`Recipe`.  Outputs: the lowercase hex SHA-256 of two parts -- its
    ``model_dump(mode="json")`` as canonical JSON (sorted keys, no insignificant whitespace) and, when the
    recipe names a chat template, the template file's own SHA-256 (the one referenced file whose bytes change
    what the engine renders).  The digest is therefore a pure function of the recipe's content: two
    directories holding the same files hash alike, and a directory whose resolved form or template changed
    does not.

    Raises:
        RecipeError: the recipe names a chat template but was not loaded from a directory, or the template
            file cannot be read (its bytes cannot enter the hash).
    """
    form = json.dumps(recipe.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    payload = f"{form}\n{_template_digest(recipe)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _template_digest(recipe: Recipe) -> str:
    """``sha256:<hex>`` of the chat template file the recipe names, or ``absent`` when it names none."""
    name = recipe.serve.chat_template
    if name is None:
        return "absent"
    directory = recipe._dir
    if directory is None:
        raise RecipeError(
            f"recipe {recipe.id}: serve.chat_template needs the recipe directory to hash the file's bytes; "
            "load the recipe with load_recipe"
        )
    try:
        data = (directory / name).read_bytes()
    except OSError as error:
        raise RecipeError(
            f"recipe {recipe.id}: serve.chat_template {name!r} cannot be read in {directory} ({error})"
        ) from error
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _names_a_recipe_field(path: str) -> bool:
    """Whether ``path`` names a CONTENT field of the recipe schema (``serve.dtype``, ``model``).

    Such a field is refused with the variant hint rather than as a typo.  A path *below* a content field
    (``serve.hf_overrides.architectures``) counts too: the field it reaches into is content, whatever the leaf
    is called.  A path below a DEPLOYMENT or RUNTIME field (``resources.gpus.extra``) does not: that is not a
    field of the declaration at all, and the unknown-path refusal says so.
    """
    block, _, field = path.partition(".")
    first = field.split(".", 1)[0]
    blocks: dict[str, type[BaseModel]] = {"serve": ServeConfig, "engine": EngineSpec, "resources": Resources}
    if block in blocks:
        if first not in blocks[block].model_fields:
            return False
        declared = FIELD_ROLES.get(f"{block}.{first}")
        return declared is None or declared.role is RecipeFieldRole.CONTENT
    return block in Recipe.model_fields  # a top-level field, or a block kept as plain data (the client)


def _deployment_spec(path: str) -> FieldSpec:
    """The declaration entry for the ``--set`` path ``path``, or the refusal that says why it is not settable.

    Raises:
        RecipeError: an unknown path (the deployment surface is listed), a CONTENT path (a different value is a
            different variant), or a RUNTIME path (the run owns it).
    """
    spec = FIELD_ROLES.get(path)
    if spec is None:
        if _names_a_recipe_field(path):
            raise RecipeError(f"{path} is a CONTENT field of the recipe schema: {_CONTENT_HINT}")
        known = ", ".join(sorted(deployment_fields()))
        raise RecipeError(f"{path}: no such recipe field; the deployment fields are: {known}")
    if spec.role is RecipeFieldRole.CONTENT:
        raise RecipeError(f"{path} is a CONTENT field of the recipe schema: {_CONTENT_HINT}")
    if spec.role is RecipeFieldRole.RUNTIME:
        known = ", ".join(sorted(deployment_fields()))
        raise RecipeError(f"{path} is a RUNTIME field (the run owns it, not the serve command): {known}")
    return spec


def _checked_value(path: str, spec: FieldSpec, value: Any, *, label: str | None = None) -> Any:
    """``value`` when it fits the declared kind and range of ``path``, else a refusal naming both sides.

    ``label`` is where the value came from (default ``--set <path>=<value>``), so a refused ``--port`` does not
    read as a refused ``--set``.  A float must be finite: ``nan`` compares false against both bounds and would
    reach the engine's flag unchecked.
    """
    where = f"--set {path}={value!r}" if label is None else label
    if spec.kind is str:
        if not isinstance(value, str) or not value:
            raise RecipeError(f"{where}: expected a non-empty string")
        return value
    if isinstance(value, bool) or not isinstance(value, spec.kind):
        raise RecipeError(f"{where}: expected {spec.kind.__name__}")
    if spec.kind is float and not math.isfinite(value):
        raise RecipeError(f"{where}: expected a finite number")
    if spec.low is not None and (value <= spec.low if spec.low_exclusive else value < spec.low):
        raise RecipeError(f"{where}: must be {'above' if spec.low_exclusive else 'at least'} {spec.low}")
    if spec.high is not None and value > spec.high:
        raise RecipeError(f"{where}: must be at most {spec.high}")
    return value


def parse_deployment_overrides(pairs: Iterable[str]) -> dict[str, Any]:
    """Parse and check ``--set <path>=<value>`` pairs against :data:`FIELD_ROLES`.

    Inputs: the console's raw ``--set`` strings, in the order given.  Outputs: ``{path: value}`` with each
    value parsed to its declared kind and checked against its declared range -- what :func:`serve_argv`
    renders.  Raises :class:`RecipeError`: a pair that is not ``PATH=VALUE``; a path that names no recipe
    field (the deployment surface is listed); a path declared CONTENT (a different value is a different
    variant) or RUNTIME (the run owns it); a value of the wrong kind or out of range.
    """
    values: dict[str, Any] = {}
    for pair in pairs:
        path, separator, raw = pair.partition("=")
        path = path.strip()
        if not separator or not path:
            raise RecipeError(f"--set {pair!r}: the form is --set PATH=VALUE, e.g. --set serve.max_num_seqs=64")
        spec = _deployment_spec(path)
        try:
            value = spec.kind(raw.strip())
        except (TypeError, ValueError):
            raise RecipeError(f"--set {path}={raw.strip()!r}: expected {spec.kind.__name__}") from None
        values[path] = _checked_value(path, spec, value)
    return values


def _recipe_field(recipe: Recipe, path: str) -> Any:
    """The recipe's own value for the declared deployment path ``path``, or ``None`` when it carries none."""
    block, _, field = path.partition(".")
    blocks: dict[str, BaseModel] = {"serve": recipe.serve, "engine": recipe.engine, "resources": recipe.resources}
    model = blocks.get(block)
    if model is None or field not in type(model).model_fields:
        return None
    return getattr(model, field)


def _client_budgets(recipe: Recipe) -> dict[str, int]:
    """The token budgets the recipe's client block declares, by field name (the largest is the engine's floor)."""
    budgets: dict[str, int] = {}
    for name in ("max_tokens", "query_max_tokens", "document_max_tokens"):
        value = recipe.client.get(name)
        if isinstance(value, int) and not isinstance(value, bool):
            budgets[name] = value
    return budgets


def _deployment_values(recipe: Recipe, *, port: int | None, deployment: Mapping[str, Any] | None) -> dict[str, Any]:
    """The effective deployment values: the recipe's own, the run's ``port``, the declared defaults, the overrides.

    Raises:
        RecipeError: a value that is not a DEPLOYMENT field or does not fit it, no port to serve on, or a
            ``serve.max_model_len`` below the client's largest token budget (the engine would reject admissible
            prompts, so both numbers are named).
    """
    values: dict[str, Any] = {}
    for path, spec in deployment_fields().items():
        declared = _recipe_field(recipe, path)
        values[path] = spec.default if declared is None else declared
    if port is not None:
        # The run's own spelling of serve.port gets the same check as the deployment field (and a message that
        # names the flag the operator actually used).
        values["serve.port"] = _checked_value(
            "serve.port", deployment_fields()["serve.port"], port, label=f"--port {port}"
        )
    for path, value in (deployment or {}).items():
        values[path] = _checked_value(path, _deployment_spec(path), value)
    if values["serve.port"] is None:
        raise RecipeError(f"recipe {recipe.id}: no port to serve on; pass --port or --set serve.port=<0-65535>")
    budgets = _client_budgets(recipe)
    if budgets:
        largest = max(budgets, key=lambda name: budgets[name])
        if values["serve.max_model_len"] < budgets[largest]:
            raise RecipeError(
                f"serve.max_model_len {values['serve.max_model_len']} is below the client's largest token budget "
                f"{budgets[largest]} (client.{largest}): the engine would reject admissible prompts"
            )
    return values


def _render(values: Mapping[str, Any], position: str) -> list[str]:
    """The argv elements of every DEPLOYMENT field that renders at ``position`` and carries a value."""
    argv: list[str] = []
    for path, spec in deployment_fields().items():
        if spec.position != position or values[path] is None:
            continue
        argv += [str(spec.flag), str(values[path])]
    return argv


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


def _is_shipped_directory(directory: Path) -> bool:
    """Whether ``directory`` is one of the package's shipped family directories.

    Shipped-ness is where the file came from, never how the caller spelled it: a family directory under the
    package's ``recipes/`` root is a shipped recipe (``resolve_recipe(id, root=default_recipes_root())``
    included), and anything else -- a path of the operator's own, or a root of theirs -- is not.
    """
    try:
        root = default_recipes_root()
    except RecipeError:  # a zipped install has no recipes directory to compare against
        return False
    try:
        directory.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _expand_variant(family: Family, variant: Variant, directory: Path, yaml_path: Path) -> Recipe:
    """One variant's resolved :class:`Recipe`: the family's shared blocks plus its whitelisted overrides.

    The merge is key-level replacement (``overrides`` keys replace the family's value; nothing deep-merges),
    the client block gains the injected ``model``/``revision``/``tokenizer``, and the result validates against
    the unchanged ``Recipe`` schema -- the resolved recipe is exactly what a standalone recipe described.  The
    referenced files (the family's template, the one ``reference.py``) are checked against the family directory.
    A directory outside the package's recipes root (:func:`_is_shipped_directory`) makes an unshipped recipe:
    its ``status`` is forced to ``unverified``, because the verification record belongs to a shipped recipe,
    never to a file the loader was handed.
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
    shipped = _is_shipped_directory(directory)
    status = (variant.status or family.status).model_dump(mode="json")
    if not shipped:
        status = StatusSpec().model_dump(mode="json")  # unverified, with no verification evidence beside it
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
        "status": status,
        "sources": [*family.sources, *variant.sources],
        "notes": notes,
    }
    try:
        recipe = Recipe.model_validate(data)
    except Exception as error:
        raise RecipeError(f"{yaml_path}: variant {variant.id!r} does not resolve to a valid recipe: {error}") from error
    recipe._dir = directory
    recipe._shipped = shipped
    _check_referenced_files(recipe, directory)
    if not shipped:
        # Computed once, from the files as they are now: a loaded recipe is a snapshot, so its identity never
        # moves under it (a file edited afterwards is a different recipe, loaded again).  After the referenced
        # files are checked, so a missing template is the check's refusal, not this one's.
        recipe._identity = f"unshipped:sha256:{recipe_digest(recipe)}"
    return recipe


def load_recipes_of(family: Family, directory: Path) -> list[Recipe]:
    """Every variant of ``family`` (loaded from ``directory``), expanded to resolved recipes in file order."""
    yaml_path = directory / "family.yaml"
    return [_expand_variant(family, variant, directory, yaml_path) for variant in family.variants]


def load_recipe(source: str | Path, *, root: str | Path | None = None, variant: str | None = None) -> Recipe:
    """The resolved recipe: a variant id (resolved under ``root``, default the shipped ones) or a family path.

    Inputs: ``source`` -- a variant id (never a family id: family ids are not served), or a path to a family
    directory / ``family.yaml``; ``root`` names the recipes root for an id (default: the package's
    ``recipes/``); ``variant`` selects one variant when the path's family declares more than one (a shipped id
    refuses it: the id already names one variant).  An **id-shaped** source is an id first: the catalog's
    variant of that name wins over a directory of the same name in the working directory, and the file is
    named as a path (``./name``).  Outputs: the frozen resolved :class:`Recipe` -- loaded from outside the
    package's recipes root it is unshipped (:attr:`Recipe.shipped`), its ``status`` is forced to
    ``unverified`` and its :attr:`Recipe.identity` is the content hash of its resolved form.  Raises
    :class:`RecipeError` when the source names no recipe, names a multi-variant family without ``variant`` (the
    variant ids are listed), or fails validation.
    """
    candidate = Path(source)
    source_text = source if isinstance(source, str) else None
    id_shaped = source_text is not None and re.fullmatch(_ID_PATTERN, source_text) is not None
    if id_shaped and source_text is not None:
        try:
            shipped = resolve_recipe(source_text, root=root)
        except RecipeError:
            shipped = None  # no variant of that name: it may still be a family directory of the caller's own
        if shipped is not None:
            if variant is not None:
                raise RecipeError(
                    f"--variant names a variant of a family directory, not of the shipped recipe id {source_text!r}"
                )
            return shipped
    if candidate.exists():
        # a filesystem path: a family directory or a family.yaml file
        yaml_path = candidate / "family.yaml" if candidate.is_dir() else candidate
        if yaml_path.name == "recipe.yaml":
            raise RecipeError(
                f"{yaml_path}: the standalone recipe.yaml path is gone (decision 34: one family, many sizes): "
                "wrap the recipe in a family.yaml with one variant"
            )
        family = load_family(candidate)
        directory = candidate if candidate.is_dir() else yaml_path.parent
        if variant is not None:
            chosen = next((row for row in family.variants if row.id == variant), None)
            if chosen is None:
                ids = ", ".join(row.id for row in family.variants)
                raise RecipeError(
                    f"{candidate}: family {family.id!r} declares no variant {variant!r}; it declares: {ids}"
                )
            return _expand_variant(family, chosen, directory, yaml_path)
        if len(family.variants) != 1:
            ids = ", ".join(row.id for row in family.variants)
            raise RecipeError(
                f"{candidate}: family {family.id!r} declares {len(family.variants)} variants ({ids}); "
                "name one with --variant <variant-id> (rcp-ndcg-vllm serve), or point at a family directory "
                "with exactly one variant"
            )
        return _expand_variant(family, family.variants[0], directory, yaml_path)
    if variant is not None:
        raise RecipeError(f"--variant names a variant of a family directory, not of the shipped recipe id {source!r}")
    if source_text is None or not id_shaped:
        raise RecipeError(f"no recipe at {source}: name a variant id or a family directory")
    return resolve_recipe(source_text, root=root)  # the refusal names the ids the root declares


def resolve_recipe(variant_id: str, root: str | Path | None = None) -> Recipe:
    """The resolved :class:`Recipe` of the variant id ``variant_id`` under ``root`` (default: the shipped ones).

    Inputs: the recipe id (the variant id of the family that declares it) and the recipes root whose family
    directories carry it.  Outputs: the frozen resolved recipe, its ``_dir`` set to the family directory --
    shipped when the directory is one of the package's own (any spelling of that root), unshipped otherwise.
    Raises :class:`RecipeError`: an unknown id (the known ids named), or the first family that failed to load --
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
    per variant id, exactly the ids the catalog, ``serve`` and the wave lists take (unshipped outside the
    package's own recipes root).  Raises :class:`RecipeError` when a family fails to load or two families
    resolve the same variant id.
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
            f"{directory / 'family.yaml'}: serve.chat_template {recipe.serve.chat_template!r} does not exist in "
            f"{directory}"
        )
    needs_reference = recipe.reference.kind != "stored_scores"
    if needs_reference and not (directory / recipe.reference.entry).is_file():
        raise RecipeError(
            f"{directory / 'family.yaml'}: reference.entry {recipe.reference.entry!r} does not exist in {directory}; "
            "a recipe needs reference.py unless reference.kind is stored_scores"
        )


def serve_argv(
    recipe: Recipe,
    *,
    port: int | None = None,
    served_model_name: str,
    deployment: Mapping[str, Any] | None = None,
) -> list[str]:
    """Render the ``vllm serve`` argv a recipe stands for, with its deployment values.

    Inputs: a loaded :class:`Recipe`; ``port``, the port the run serves on (``None``: the deployment value --
    an override or the declaration's default -- is used); the ``--served-model-name`` (the wave runner uses the
    recipe's ``id``); and ``deployment``, the serve-time overrides as :func:`parse_deployment_overrides`
    returns them (an override wins over the recipe's declared value and over ``port``).  Output:
    ``["vllm", "serve", <model>, "--revision", ..., ...]`` -- the fixed head, then one flag per ``serve`` field
    in a deterministic order (JSON objects with ``json.dumps(sort_keys=True)``), then ``extra_args`` verbatim.
    ``serve.plugin`` and ``serve.io_processor_plugin`` render nothing: they name pip packages installed before
    the engine starts.  Raises :class:`RecipeError` when the recipe sets ``serve.chat_template`` but was not
    loaded from a directory (the template's absolute path is needed), when a deployment value is not a
    DEPLOYMENT field or does not fit it, when no port is resolvable, or when ``serve.max_model_len`` falls below
    the client's largest token budget.
    """
    values = _deployment_values(recipe, port=port, deployment=deployment)
    argv = [
        "vllm",
        "serve",
        recipe.model,
        "--revision",
        recipe.revision,
        "--served-model-name",
        served_model_name,
        *_render(values, "head"),
        "--runner",
        recipe.serve.runner,
    ]
    if recipe.serve.convert is not None:
        argv += ["--convert", recipe.serve.convert]
    argv += ["--dtype", recipe.serve.dtype]
    argv += _render(values, "body")
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
    argv += _render(values, "tail")
    argv += list(recipe.serve.extra_args)
    return argv


def client_config(recipe: Recipe, *, base_url: str | None) -> dict[str, Any]:
    """The product's endpoint config dict this recipe's ``client`` block implies, with ``base_url`` filled.

    Inputs: a recipe and the endpoint's ``base_url`` (e.g. ``http://127.0.0.1:8100/v1``), or ``None`` for a
    serve-by-role run whose URLs arrive at runtime.  Output: a plain dict — the recipe's ``client`` block plus
    the ``base_url`` key — exactly what the product's config loader accepts; it validates the block with the
    product's endpoint model when it reads it.  The ``recipe`` pointer defaults to the recipe's
    :attr:`Recipe.identity` (its shipped id, or ``unshipped:sha256:<hex>`` for a file of the operator's own),
    and a ``client.recipe`` the recipe declared itself is kept as declared.
    """
    client = dict(recipe.client)
    client.setdefault("recipe", recipe.identity)
    client["base_url"] = base_url
    return client


def recipe_json_schema() -> dict[str, Any]:
    """The JSON Schema of :class:`Recipe`, exported to ``schema/recipe.schema.json`` and kept current by a test."""
    return Recipe.model_json_schema()


def family_json_schema() -> dict[str, Any]:
    """The JSON Schema of :class:`Family`, exported to ``schema/family.schema.json`` and kept current by a test."""
    return Family.model_json_schema()
