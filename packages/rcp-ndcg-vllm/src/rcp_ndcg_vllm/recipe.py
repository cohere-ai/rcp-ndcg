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

_ID_PATTERN = r"^[a-z0-9][a-z0-9.-]*$"
_REVISION_PATTERN = r"^[0-9a-f]{40}$"
_TOKENIZER_PATTERN = r"^[^@\s]+/[^@\s]+@[0-9a-f]{40}$"

JSONValue = str | int | float | bool | None | list[Any] | dict[str, Any]
"""Any JSON-representable value (the value type of ``serve.hf_overrides`` and ``serve.pooler_config``)."""

Role = Literal["embed", "multi_vector", "rerank"]
ClientApi = Literal["openai_embeddings", "vllm_pooling", "rerank"]
ScoreScale = Literal["probability", "logit", "cosine"]
InstructionMode = Literal["none", "field", "fold"]
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

    See :func:`serve_argv` for the exact argv.  ``plugin`` is the one field that never reaches the argv: it names
    a ``vllm.general_plugins`` pip package that must be installed into the image before the engine starts
    (bootstrap installs the directories a wave's ``plugins.txt`` lists).
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
        default_factory=dict, description="vLLM --pooler-config, a JSON object sent verbatim (keys sorted)"
    )
    trust_remote_code: bool = Field(default=False, description="vLLM --trust-remote-code flag")
    max_model_len: int = Field(gt=0, description="vLLM --max-model-len")
    dtype: Literal["auto", "float16", "bfloat16", "float32", "float8", "half"] = Field(description="vLLM --dtype")
    plugin: str | None = Field(
        default=None, description="pip spec of a vllm.general_plugins package, installed before the engine starts"
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


class ClientConfig(BaseModel):
    """The ``rcp-ndcg`` endpoint fields this recipe implies.

    The fields mirror the design's configuration section for a served encoder or reranker.  Role-dependent fields
    are validated in :class:`Recipe`: ``instruction`` and ``use_activation`` exist only for a reranker,
    ``normalize`` only for ``embed`` and ``multi_vector``, ``embed_dtype`` only for ``multi_vector``, and a
    ``listwise`` reranker has no ``batch_size`` (the whole candidate set is one request).

    Attributes:
        api: Which route the client speaks: ``openai_embeddings`` (dense), ``vllm_pooling`` (late interaction,
            ``POST /pooling`` with ``task: token_embed``) or ``rerank`` (Cohere shape).
        query_prompt: Text prepended to every query client-side (embedding models with a query prefix).
        doc_prompt: Text prepended to every document client-side (Octen-style ``"- "``).
        instruction: How the client passes a reranker instruction: ``none`` (drop it), ``field`` (the endpoint's
            ``instruction`` field) or ``fold`` (appended to the query text).
        default_instruction: The instruction used when the caller passes none; required when ``instruction`` is
            ``field`` or ``fold``, because those move the instruction's text to the client.
        tokenizer: The tokenizer the client cuts text with, ``repo@revision`` (40-hex revision).
        max_tokens: The pair (or text) budget in tokens; the client cuts at token boundaries with the tokenizer.
        normalize: L2-normalise returned vectors (``embed`` and ``multi_vector`` only).
        dimensions: Matryoshka cut requested through the endpoint's ``dimensions`` field.
        embed_dtype: The transfer precision of ``/pooling`` vectors; ``multi_vector`` only, ``float16`` by default
            (set ``float32`` to opt out; the field must stay unset for other roles).
        use_activation: Rerank request field for raw-logit rerankers (``false``) or activated scores (``true``);
            ``None`` sends nothing and the server's default applies.
        batch_size: Texts per request, where a batch is meaningful; a listwise reranker must leave it unset.
    """

    model_config = ConfigDict(**_no_extra())

    api: ClientApi
    query_prompt: str = ""
    doc_prompt: str = ""
    instruction: InstructionMode | None = None
    default_instruction: str | None = None
    tokenizer: str = Field(
        pattern=_TOKENIZER_PATTERN, description="repo@revision of the tokenizer the client cuts with"
    )
    max_tokens: int = Field(gt=0)
    normalize: bool | None = None
    dimensions: int | None = Field(default=None, gt=0)
    embed_dtype: EmbedDType | None = None
    use_activation: bool | None = None
    batch_size: int | None = Field(default=None, gt=0)


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


class Gates(BaseModel):
    """Overrides of the stage-2 gate defaults for one recipe's ``reference.score_scale``.

    Every field defaults to ``None`` (= the published default for the scale).  The defaults, and what each field
    means, are in :mod:`rcp_ndcg_vllm.equivalence.gates`; ``kendall_tau_min`` applies to every scored scale,
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
        if rerank and self.client.instruction in ("field", "fold") and self.client.default_instruction is None:
            raise ValueError(
                f"client.instruction={self.client.instruction} moves the instruction text to the client: "
                "set client.default_instruction"
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
    if directory.name != recipe.id:
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
    if recipe.role == "rerank":
        assert recipe.client.instruction is not None  # guaranteed by the schema's validators
        config["instruction"] = recipe.client.instruction
        if recipe.client.default_instruction is not None:
            config["default_instruction"] = recipe.client.default_instruction
        if recipe.client.use_activation is not None:
            config["use_activation"] = recipe.client.use_activation
        if recipe.client.batch_size is not None:
            config["batch_size"] = recipe.client.batch_size
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
