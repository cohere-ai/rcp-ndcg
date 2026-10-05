"""The recipe schema: one declarative description of how a model is served with vLLM and read back by
``rcp-ndcg``.

A recipe lives in one directory, ``recipes/<id>/``, with:

- ``recipe.yaml`` — the recipe itself (the :class:`Recipe` schema below);
- ``template.jinja`` — the chat template given to ``vllm serve --chat-template``, when the model needs one;
- ``reference.py`` — the in-process reference implementation the equivalence harness compares against.

The schema is deliberately closed (``extra="forbid"``) and role-aware: a field that only makes sense for one role
is rejected for the others, so a typo cannot silently change what is served.  Everything the engine is told is in
the file — nothing is defaulted implicitly: ``serve_argv`` renders exactly the fields present.

Public helpers:

- :func:`load_recipe` — load and validate one recipe directory or ``recipe.yaml`` file.
- :func:`iter_recipes` — every recipe under a root of recipe directories.
- :func:`serve_argv` — the ``vllm serve`` argv a recipe renders to.
- :func:`client_config` — the ``rcp-ndcg`` encoder/reranker config dict a recipe implies.
- :func:`recipe_json_schema` — the JSON Schema of :class:`Recipe` (exported to ``schema/recipe.schema.json``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator

from .errors import RecipeError

__all__ = [
    "ClientConfig",
    "EngineSpec",
    "Gates",
    "ReferenceSpec",
    "Recipe",
    "Resources",
    "ServeConfig",
    "StatusSpec",
    "client_config",
    "default_recipes_root",
    "effective_embed_dtype",
    "iter_recipes",
    "load_recipe",
    "recipe_json_schema",
    "serve_argv",
]

_PINNED_VLLM_REF = "d0d6e5f3a"
"""The vLLM source checkout the PoolerConfig field list below was read at
(``/refs/vllm`` in the rcp-ndcg repository; image ``vllm/vllm-openai:v0.31.0``)."""

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
_TOKENIZER_PATTERN = r"^[^@\s]+/[^@\s]+@[0-9a-f]{40}$"

JSONValue = str | int | float | bool | None | list[Any] | dict[str, Any]
"""Any JSON-representable value (the value type of ``serve.hf_overrides`` and ``serve.pooler_config``)."""

Role = Literal["embed", "multi_vector", "rerank"]
ClientApi = Literal["openai_embeddings", "vllm_pooling", "rerank"]
ScoreScale = Literal["probability", "logit", "cosine"]
InstructionMode = Literal["none", "field", "fold", "system"]
EmbedDType = Literal["float16", "float32"]

_API_FOR_ROLE: dict[Role, ClientApi] = {
    "embed": "openai_embeddings",
    "multi_vector": "vllm_pooling",
    "rerank": "rerank",
}


def _no_extra() -> dict[str, Any]:
    """The common model config: frozen, unknown fields refused."""
    return {"extra": "forbid", "frozen": True}


class EngineSpec(BaseModel):
    """The engine image a recipe is served with.

    Attributes:
        name: The engine family; this package serves ``vllm``.
        image: The container image, ``repository:tag`` (public names only).
        min_version: The engine version the recipe is known to work with, ``MAJOR.MINOR.PATCH``.
        startup_timeout_s: How long :mod:`rcp_ndcg_vllm.jobs.run_wave` waits for ``GET /v1/models`` before it
            declares the recipe failed (seconds).  Large models override this per recipe.
    """

    model_config = ConfigDict(**_no_extra())

    name: Literal["vllm"]
    image: str = Field(min_length=1, description="repository:tag of the engine image")
    min_version: str = Field(pattern=r"^\d+\.\d+\.\d+$", description="known-good engine version")
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
    package and the checkpoint's IO-processor plugin, respectively; bootstrap installs the directories a wave's
    ``plugins.txt`` lists).
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


class TemplateSegment(BaseModel):
    """One segment of a declared request shape: exactly one of ``special``, ``text``, ``ids`` or ``content``.

    ``special`` names an added/special token of the recipe's tokenizer (resolved by the harness by name, never
    typed literally in the recipe); ``text`` is ordinary template text; ``ids`` is a measured token-id sequence;
    ``content`` names the cuttable span (``query`` or ``document``).  Fixed segments (everything but ``content``)
    are reserved in the text budget and re-attached after a cut — they carry the model's anchors.
    """

    model_config = ConfigDict(**_no_extra())

    special: str | None = Field(default=None, min_length=1, description="added/special token NAME of the tokenizer")
    text: str | None = Field(default=None, description="ordinary template text (never a special-token literal)")
    ids: list[int] | None = Field(default=None, description="explicit token ids of a measured fixed segment")
    content: Literal["query", "document"] | None = Field(
        default=None, description="the cuttable span this segment stands for"
    )

    @model_validator(mode="after")
    def _exactly_one_kind(self) -> TemplateSegment:
        kinds = [name for name in ("special", "text", "ids", "content") if getattr(self, name) is not None]
        if len(kinds) != 1:
            raise ValueError(f"a template segment sets exactly one of special/text/ids/content, got {kinds}")
        if self.ids is not None and not self.ids:
            raise ValueError("a template segment's ids must list at least one token id")
        return self


class TemplateSpec(BaseModel):
    """The request shapes as data: ordered segments per shape, the anchors, and the pair's query budget.

    Shapes are ``query`` and ``document`` (embedding roles) and ``pair`` (rerank, the full scored prompt).  Every
    declared shape must contain at least one content span; with ``anchor: last`` every declared shape must end
    with a fixed segment (or the recipe pins ``add_special_tokens: true``, declaring the tokenizer's end token as
    the anchor).  ``anchor: marker`` requires ``anchor_markers``.  ``query_max_tokens`` splits the pair budget:
    the document span gets the rest.
    """

    model_config = ConfigDict(**_no_extra())

    query: list[TemplateSegment] | None = None
    document: list[TemplateSegment] | None = None
    pair: list[TemplateSegment] | None = None
    anchor: Literal["last", "first", "mean", "marker"]
    anchor_markers: list[int] = Field(default_factory=list)
    query_max_tokens: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _shape_rules(self) -> TemplateSpec:
        names = ("query", "document", "pair")
        shapes = {name: getattr(self, name) for name in names if getattr(self, name) is not None}
        if not shapes:
            raise ValueError("a template declares at least one shape: query, document or pair")
        for name, segments in shapes.items():
            if not any(segment.content is not None for segment in segments):
                raise ValueError(f"template.{name} needs at least one content span (the cuttable part)")
            fixed = lambda segment: segment.content is None  # noqa: E731
            if self.anchor == "last" and not fixed(segments[-1]):
                if not (self.anchor_markers and segments[-1].content is not None):
                    raise ValueError(
                        f"template.{name}: anchor=last requires the shape to end with a fixed segment (or pin "
                        "client.add_special_tokens: true for a post-processor end token)"
                    )
            if self.anchor == "first" and not fixed(segments[0]):
                raise ValueError(f"template.{name}: anchor=first requires the shape to start with a fixed segment")
        if self.anchor == "marker" and not self.anchor_markers:
            raise ValueError("anchor=marker needs template.anchor_markers (the token ids the model reads)")
        if self.anchor != "marker" and self.anchor_markers:
            raise ValueError("template.anchor_markers is only used with anchor=marker")
        if self.pair is not None and self.query_max_tokens is None:
            raise ValueError("template.pair needs template.query_max_tokens: the pair budget's query share")
        if self.pair is None and self.query_max_tokens is not None:
            raise ValueError("template.query_max_tokens is only used with a pair shape")
        return self


class BlockingSpec(BaseModel):
    """The block loop of a listwise reranker (jina-v3 style): the client renders blocks, the engine scores them.

    Attributes:
        block_size: Documents per block.
        capacity_formula: How the block budget is computed, as data (e.g. ``"max_tokens - query - separator"``).
        weighting: How block scores combine into document scores (e.g. ``max``).
    """

    model_config = ConfigDict(**_no_extra())

    block_size: int = Field(gt=0)
    capacity_formula: str = Field(min_length=1)
    weighting: str = Field(min_length=1)


class ClientConfig(BaseModel):
    """The ``rcp-ndcg`` endpoint fields this recipe implies.

    The fields mirror the design's configuration section for a served encoder or reranker.  Role-dependent fields
    are validated in :class:`Recipe`: ``instruction`` and ``use_activation`` exist only for a reranker,
    ``normalize`` only for ``embed`` and ``multi_vector``, ``embed_dtype`` only for ``multi_vector``, and a
    ``listwise`` reranker has no ``batch_size`` (the whole candidate set is one request).  Budgets are explicit:
    ``tokenizer`` and ``max_tokens`` are required, and ``on_overflow`` names what happens past the budget (the
    client owns every cut — there is no engine-side truncation field, and a recipe that needs one to reproduce its
    reference declares ``reference.known_deviations`` instead).

    Attributes:
        api: Which route the client speaks: ``openai_embeddings`` (dense), ``vllm_pooling`` (late interaction,
            ``POST /pooling`` with ``task: token_embed``) or ``rerank`` (Cohere shape).
        request_shape: What the client sends: ``text`` (the rendered prompt string), ``messages`` (chat messages)
            or ``token_ids`` (the client's own ids).
        add_special_tokens: Pins the request form's special-token flag; ``None`` sends nothing and the engine's
            per-form default applies.  ``true`` declares the tokenizer's end token as the shape's declared anchor.
        template: The request shapes as data: ordered segments, the anchors and the pair's query budget.
        query_prompt: Text prepended to every query client-side (embedding models with a query prefix).
        doc_prompt: Text prepended to every document client-side (Octen-style ``"- "``).
        instruction: How the client passes a reranker instruction: ``none`` (drop it), ``field`` (the endpoint's
            ``instruction`` field), ``fold`` (appended to the query text) or ``system`` (a system message).
        default_instruction: The instruction used when the caller passes none; required when ``instruction`` is
            ``field``, ``fold`` or ``system``, because those move the instruction's text to the client.  An empty
            string is refused: the engine would silently serve its template default where the reference renders
            the empty instruction.
        tokenizer: The tokenizer the client cuts text with, ``repo@revision`` (40-hex revision).  Required —
            there is no implicit tokenizer.
        max_tokens: The pair (rerank) or text (embed) budget in the declared tokenizer's tokens.  Required —
            there is no implicit budget.
        on_overflow: What the client does when content exceeds its share of the budget: ``cut`` (default, at
            token boundaries, anchors reserved), ``chunk`` (chunks each carrying the full template) or ``fail``.
        aggregation: How chunk scores collapse into a document score; ``max`` is the only supported
            aggregation (the cobble/max-over-chunks families; the engine's own chunked processing is a different
            mechanism and must never be combined with a client one).
        empty_doc: What an empty document becomes: ``omit_zero`` (filtered, scored 0.0 without a request),
            ``send`` (the empty string is sent) or ``send_text`` (a literal placeholder text is sent).
        normalize: L2-normalise returned vectors (``embed`` and ``multi_vector`` only).
        dimensions: Matryoshka cut requested through the endpoint's ``dimensions`` field.
        embed_dtype: The transfer precision of ``/pooling`` vectors; ``multi_vector`` only, **float16 by owner
            decision**.  The engine's own default is float32, so the client always sends this field explicitly
            (the effective value is emitted in the client config for every multi_vector recipe).
        use_activation: Rerank request field for raw-logit rerankers (``false``) or activated scores (``true``);
            ``None`` sends nothing and the server's default applies.
        batch_size: Texts per request, where a batch is meaningful; a listwise reranker must leave it unset.
        blocking: The listwise block loop (jina-v3 style), as data; ``listwise`` only.
    """

    model_config = ConfigDict(**_no_extra())

    api: ClientApi
    request_shape: Literal["text", "messages", "token_ids"] = "text"
    add_special_tokens: bool | None = None
    template: TemplateSpec | None = None
    query_prompt: str = ""
    doc_prompt: str = ""
    instruction: Literal["none", "field", "fold", "system"] | None = None
    default_instruction: str | None = None
    tokenizer: str = Field(
        pattern=_TOKENIZER_PATTERN, description="repo@revision of the tokenizer the client cuts with"
    )
    max_tokens: int = Field(gt=0, description="the pair (rerank) or text (embed) budget, in tokenizer tokens")
    on_overflow: Literal["cut", "chunk", "fail"] = "cut"
    aggregation: Literal["max"] | None = None
    empty_doc: Literal["omit_zero", "send", "send_text"] = "omit_zero"
    normalize: bool | None = None
    dimensions: int | None = Field(default=None, gt=0)
    embed_dtype: EmbedDType | None = None
    use_activation: bool | None = None
    batch_size: int | None = Field(default=None, gt=0)
    blocking: BlockingSpec | None = None


class ReferenceSpec(BaseModel):
    """The in-process reference implementation the equivalence harness compares the served engine against.

    Attributes:
        kind: Where the reference comes from: ``transformers`` (an HF checkpoint run in process),
            ``sentence_transformers`` (a sentence-transformers checkpoint), ``remote_code`` (the checkpoint's
            trusted code) or ``stored_scores`` (scores stored with the paper's runs; no ``reference.py``).
        score_scale: The scale ``reference.score`` returns, which selects the stage-2 gates: ``probability``,
            ``logit`` or ``cosine``.  Embedding recipes always compare vectors (cosine per vector or per token).
        entry: The reference module file in the recipe directory; it defines ``load(device)``,
            ``embed(texts, role)``, ``score(query, documents, instruction)`` and ``render(query, document,
            instruction)``.  Not needed for ``stored_scores``.
    """

    model_config = ConfigDict(**_no_extra())

    kind: Literal["transformers", "sentence_transformers", "remote_code", "stored_scores"]
    score_scale: ScoreScale
    entry: str = Field(default="reference.py", description="reference module file inside the recipe directory")
    known_deviations: list[Literal["anchor_drop_over_cap"]] = Field(
        default_factory=list,
        description="deliberate reference deviations the paper code carries (e.g. a whole-prompt right cut that "
        "drops tail anchors on over-cap pairs); stage 2 gates only pairs under the cap and reports the rest "
        "in a separate, non-gating table",
    )


class Gates(BaseModel):
    """Overrides of the stage-2 gate defaults for one recipe's ``reference.score_scale``.

    Every field defaults to ``None`` (= the published default for the scale).  The defaults, and what each field
    means, are in :mod:`rcp_ndcg_vllm.equivalence.gates`; ``tau_min`` applies to every scored scale,
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

    Construct it through :func:`load_recipe` (which also checks the referenced files exist and that ``id`` equals
    the directory name); the model itself is frozen and refuses unknown fields.

    Attributes:
        id: The recipe identifier, ``^[a-z0-9][a-z0-9.-]*$``, equal to the directory name; also the
            ``--served-model-name`` the engine serves and the ``model`` field of the client config.
        model: The Hugging Face repo id to serve.
        revision: The exact commit of ``model`` (40 hex); serving and client cutting pin it.
        role: What the model produces: ``embed`` (one dense vector), ``multi_vector`` (one vector per token) or
            ``rerank`` (query-document scores).
        input: The input modalities the model accepts, a non-empty subset of ``[text, image, video]``.
        scoring: Rerank only: ``pointwise`` (documents scored independently) or ``listwise`` (one request carries
            the whole candidate set).
        licence: SPDX identifier of the model's licence, or ``see-model-card``.
        engine: The engine image and the startup timeout.
        resources: The GPUs the engine occupies (``--tensor-parallel-size``).
        serve: Everything rendered into ``vllm serve`` argv.
        client: The ``rcp-ndcg`` endpoint fields the recipe implies.
        reference: The in-process reference the harness compares against.
        gates: Overrides of the stage-2 gate defaults.
        status: Where the recipe stands in the verification workflow.
        sources: URLs and ``path:line`` references the recipe rests on.
        notes: Free-form notes.
    """

    model_config = ConfigDict(**_no_extra())

    _dir: Path | None = PrivateAttr(default=None)
    """The directory the recipe was loaded from (set by :func:`load_recipe`; ``serve_argv`` resolves
    ``serve.chat_template`` against it)."""

    id: str = Field(pattern=_ID_PATTERN)
    model: str = Field(min_length=1, description="Hugging Face repo id")
    revision: str = Field(pattern=_REVISION_PATTERN, description="40-hex commit of model")
    role: Role
    input: list[Literal["text", "image", "video"]] = Field(min_length=1)
    scoring: Literal["pointwise", "listwise"] | None = None
    licence: str = Field(min_length=1)
    engine: EngineSpec
    resources: Resources
    serve: ServeConfig
    client: ClientConfig
    reference: ReferenceSpec
    gates: Gates = Field(default_factory=Gates)
    status: StatusSpec = Field(default_factory=StatusSpec)
    sources: list[str] = Field(default_factory=list)
    notes: str = ""

    @field_validator("sources")
    @classmethod
    def _sources_are_strings(cls, value: list[str]) -> list[str]:
        if any(not source.strip() for source in value):
            raise ValueError("sources entries must be non-empty URLs or path:line references")
        return value

    @model_validator(mode="after")
    def _role_consistency(self) -> Recipe:
        rerank = self.role == "rerank"
        if rerank and self.scoring is None:
            raise ValueError("a rerank recipe must set scoring: pointwise or listwise")
        if not rerank and self.scoring is not None:
            raise ValueError(f"scoring is only valid for role=rerank, not role={self.role}")
        if rerank and self.client.instruction is None:
            raise ValueError("a rerank recipe must set client.instruction: none, field or fold")
        if not rerank and self.client.instruction is not None:
            raise ValueError(f"client.instruction is only valid for role=rerank, not role={self.role}")
        if not rerank and self.client.use_activation is not None:
            raise ValueError(f"client.use_activation is only valid for role=rerank, not role={self.role}")
        if (
            rerank
            and self.client.instruction in ("field", "fold", "system")
            and self.client.default_instruction is None
        ):
            raise ValueError(
                f"client.instruction={self.client.instruction} moves the instruction text to the client: "
                "set client.default_instruction"
            )
        if self.client.default_instruction == "":
            raise ValueError(
                "client.default_instruction must not be an empty string: the engine would silently serve its "
                "template default while the reference renders the empty instruction"
            )
        if rerank and self.client.instruction == "none" and self.client.default_instruction is not None:
            raise ValueError("client.default_instruction is only used when client.instruction is field or fold")
        if not rerank and self.client.normalize is None:
            raise ValueError(f"an embedding recipe must set client.normalize, not role={self.role}")
        if rerank and self.client.normalize is not None:
            raise ValueError("client.normalize is only valid for embed and multi_vector, not role=rerank")
        if self.role != "multi_vector" and self.client.embed_dtype is not None:
            raise ValueError(f"client.embed_dtype is only valid for role=multi_vector, not role={self.role}")
        if self.role == "multi_vector" and self.serve.pooler_config.get("task") not in (None, "token_embed"):
            raise ValueError("a multi_vector recipe serves --pooler-config.task token_embed")
        if self.client.api != _API_FOR_ROLE[self.role]:
            raise ValueError(
                f"role={self.role} implies client.api={_API_FOR_ROLE[self.role]}, got client.api={self.client.api}"
            )
        if self.scoring == "listwise" and self.client.batch_size is not None:
            raise ValueError("a listwise reranker has no batch size: the whole candidate set is one request")
        if self.client.template is not None:
            if rerank and self.client.template.pair is None:
                raise ValueError("a rerank recipe's template declares a pair shape (the full scored prompt)")
            if not rerank and self.client.template.document is None:
                raise ValueError(f"a {self.role} recipe's template declares at least a document shape")
        if self.client.blocking is not None and not rerank:
            raise ValueError(f"client.blocking is only valid for role=rerank, not role={self.role}")
        if self.client.blocking is not None and self.scoring != "listwise":
            raise ValueError("client.blocking is only valid for a listwise reranker")
        if self.client.aggregation is not None and self.client.on_overflow != "chunk":
            raise ValueError("client.aggregation only applies to chunked inputs: set client.on_overflow: chunk")
        if self.client.max_tokens > self.serve.max_model_len:
            raise ValueError(
                f"client.max_tokens ({self.client.max_tokens}) must not exceed engine.max_model_len "
                f"({self.serve.max_model_len}): the engine would 400 the rendered prompt"
            )
        return self


def default_recipes_root() -> Path:
    """The package's own ``recipes/`` directory (where the recipe lanes write)."""
    return Path(__file__).resolve().parents[2] / "recipes"


def load_recipe(path: str | Path) -> Recipe:
    """Load and validate one recipe from a recipe directory or a ``recipe.yaml`` file.

    Inputs: ``path``, the recipe directory (containing ``recipe.yaml``) or the YAML file itself.  Outputs: a
    frozen :class:`Recipe` whose ``_dir`` records where it came from.  Raises :class:`RecipeError` with the file
    path and the validator message when the YAML does not satisfy the schema, when ``id`` differs from the
    directory name, or when a referenced file (``serve.chat_template``, ``reference.entry``) does not exist.
    """
    path = Path(path)
    yaml_path = path / "recipe.yaml" if path.is_dir() else path
    if not yaml_path.is_file():
        raise RecipeError(f"no recipe at {path}: expected {yaml_path}")
    try:
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise RecipeError(f"{yaml_path} is not valid YAML: {error}") from error
    if not isinstance(data, dict):
        raise RecipeError(f"{yaml_path} must contain a YAML mapping of the Recipe schema, got {type(data).__name__}")
    try:
        recipe = Recipe.model_validate(data)
    except Exception as error:
        raise RecipeError(f"{yaml_path}: {error}") from error
    directory = path if path.is_dir() else yaml_path.parent
    recipe._dir = directory
    if path.is_dir() and directory.name != recipe.id:
        raise RecipeError(
            f"{directory / 'recipe.yaml'}: id {recipe.id!r} must equal the directory name {directory.name!r}"
        )
    _check_referenced_files(recipe, directory)
    return recipe


def iter_recipes(root: str | Path | None = None) -> list[Recipe]:
    """Load every recipe under ``root`` (default: the package's ``recipes/`` directory).

    Inputs: a root directory whose direct children are recipe directories.  Outputs: the recipes, sorted by id.
    Raises :class:`RecipeError` naming the directory when any recipe fails to load — a broken recipe among fifteen
    must not pass silently.
    """
    root = Path(root) if root is not None else default_recipes_root()
    if not root.is_dir():
        raise RecipeError(f"no recipe root at {root}")
    recipes: list[Recipe] = []
    for directory in sorted(p for p in root.iterdir() if p.is_dir() and (p / "recipe.yaml").is_file()):
        recipes.append(load_recipe(directory))
    return recipes


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
    ``extra_args`` verbatim.  ``serve.plugin`` renders nothing: it names a pip package installed before the engine
    starts.  Raises :class:`RecipeError` when the recipe sets ``serve.chat_template`` but was not loaded from a
    directory (the template's absolute path is needed).
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


def client_config(recipe: Recipe, *, base_url: str) -> dict[str, Any]:
    """The ``rcp-ndcg`` encoder or reranker config dict this recipe implies, per the design's configuration
    section.

    Inputs: a recipe and the endpoint's ``base_url`` (e.g. ``http://127.0.0.1:8100/v1``).  Output: for a reranker,
    the flat reranker config; for an embedding or late-interaction model, ``{"kind": "dense", "encoder": {...}}``.
    ``model`` is the recipe ``id`` — the name :func:`serve_argv` serves, so the config and the engine agree by
    construction.  Fields the recipe leaves unset are omitted; a listwise reranker never carries a batch size.
    """
    config: dict[str, Any] = {
        "api": recipe.client.api,
        "base_url": base_url,
        "model": recipe.id,
        "revision": recipe.revision,
        "recipe": recipe.id,
        "tokenizer": recipe.client.tokenizer,
        "max_tokens": recipe.client.max_tokens,
    }
    if recipe.client.request_shape != "text":
        config["request_shape"] = recipe.client.request_shape
    if recipe.client.add_special_tokens is not None:
        config["add_special_tokens"] = recipe.client.add_special_tokens
    if recipe.client.template is not None:
        config["template"] = recipe.client.template.model_dump(exclude_none=True)
    if recipe.client.on_overflow != "cut":
        config["on_overflow"] = recipe.client.on_overflow
    if recipe.client.aggregation is not None:
        config["aggregation"] = recipe.client.aggregation
    if recipe.client.empty_doc != "omit_zero":
        config["empty_doc"] = recipe.client.empty_doc
    if recipe.role == "rerank":
        assert recipe.client.instruction is not None  # guaranteed by the schema's validators
        config["instruction"] = recipe.client.instruction
        if recipe.client.default_instruction is not None:
            config["default_instruction"] = recipe.client.default_instruction
        if recipe.client.use_activation is not None:
            config["use_activation"] = recipe.client.use_activation
        if recipe.client.batch_size is not None:
            config["batch_size"] = recipe.client.batch_size
        if recipe.client.blocking is not None:
            config["blocking"] = recipe.client.blocking.model_dump()
        return config
    encoder: dict[str, Any] = dict(config)
    if recipe.client.query_prompt:
        encoder["query_prompt"] = recipe.client.query_prompt
    if recipe.client.doc_prompt:
        encoder["doc_prompt"] = recipe.client.doc_prompt
    if recipe.client.normalize is not None:
        encoder["normalize"] = recipe.client.normalize
    if recipe.client.dimensions is not None:
        encoder["dimensions"] = recipe.client.dimensions
    if recipe.client.batch_size is not None:
        encoder["batch_size"] = recipe.client.batch_size
    if recipe.role == "multi_vector":
        encoder["embed_dtype"] = effective_embed_dtype(recipe)
    return {"kind": "dense", "encoder": encoder}


def effective_embed_dtype(recipe: Recipe) -> EmbedDType:
    """The transfer precision of ``/pooling`` vectors: the recipe's ``embed_dtype``, else the ``float16`` default."""
    return recipe.client.embed_dtype or "float16"


def recipe_json_schema() -> dict[str, Any]:
    """The JSON Schema of :class:`Recipe`, exported to ``schema/recipe.schema.json`` and kept current by a test."""
    return Recipe.model_json_schema()
