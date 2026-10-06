"""The recipe schema: one declarative description of how a model is served with vLLM and read back by
``rcp-ndcg``.

A recipe lives in one directory, ``recipes/<id>/``, with:

- ``recipe.yaml`` — the recipe itself (the :class:`Recipe` schema below);
- ``template.jinja`` — the chat template given to ``vllm serve --chat-template``, when the model needs one;
- ``reference.py`` — the reference implementation, run as a subprocess (its own python via
  ``--reference-python``; the harness imports no torch).

The recipe's ``client`` block **is** the product's endpoint config for the role
(:class:`~rcp_ndcg.inference.config.EmbeddingEndpoint`, :class:`~rcp_ndcg.inference.config.PoolingEndpoint` or
:class:`~rcp_ndcg.inference.config.RerankEndpoint`): ``load_recipe`` validates it by constructing the product
model, with the product's :class:`~rcp_ndcg.data.templates.TemplateSpec` and text budget, so a recipe the
product would refuse is refused at load, with the product's message. The harness declares no parallel schema.

The schema is deliberately closed and role-aware: ``client.model`` and ``client.revision`` are the recipe's
``id`` and ``revision`` (injected at load, refused in the YAML), and ``client.api`` defaults to the role's wire
(:data:`~rcp_ndcg.inference.config.SELF_HOSTED_APIS`). Recipe-level fields are the ones the product cannot
know: ``serve`` (the engine argv), ``reference`` (the subprocess reference), ``gates``, ``status``, ids and
revisions.

Public helpers:

- :func:`load_recipe` — load and validate one recipe directory or ``recipe.yaml`` file.
- :func:`iter_recipes` — every recipe under a root of recipe directories.
- :func:`serve_argv` — the ``vllm serve`` argv a recipe renders to.
- :func:`client_config` — the product endpoint config dict a recipe implies (base URL filled).
- :func:`recipe_json_schema` — the JSON Schema of :class:`Recipe` (exported to ``schema/recipe.schema.json``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import yaml  # pyright: ignore[reportMissingModuleSource]
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator

from rcp_ndcg.inference.config import EmbeddingEndpoint, PoolingEndpoint, RerankEndpoint

from .errors import RecipeError

__all__ = [
    "ClientEndpoint",
    "EngineSpec",
    "Gates",
    "ReferenceSpec",
    "Recipe",
    "Resources",
    "ServeConfig",
    "StatusSpec",
    "client_config",
    "default_recipes_root",
    "iter_recipes",
    "load_recipe",
    "recipe_json_schema",
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

type ClientEndpoint = EmbeddingEndpoint | PoolingEndpoint | RerankEndpoint
"""The product endpoint config a recipe's ``client`` block constructs, by the recipe's role."""


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
    package and the checkpoint's IO-processor plugin, respectively; the node's own bootstrap — owned by the
    ``rc-build`` lane — installs what a wave's ``plugins.txt`` lists).
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
            :mod:`rcp_ndcg_vllm.equivalence.reference` for the modes).  Not needed for ``stored_scores``.
        known_deviations: Deliberate reference deviations the paper code carries (e.g. a whole-prompt right
            cut that drops tail anchors on over-cap pairs); stage 2 gates only pairs under the cap and reports
            the rest in a separate, non-gating table.
    """

    model_config = ConfigDict(**_no_extra())

    kind: Literal["transformers", "sentence_transformers", "remote_code", "stored_scores"]
    score_scale: ScoreScale
    entry: str = Field(default="reference.py", description="reference module file inside the recipe directory")
    known_deviations: list[Literal["anchor_drop_over_cap"]] = Field(default_factory=list)


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

    Construct it through :func:`load_recipe` (which also checks the referenced files exist, that ``id`` equals
    the directory name, and that the ``client`` block constructs the product's endpoint model); the model itself
    is frozen and refuses unknown fields.

    Attributes:
        id: The recipe identifier, ``^[a-z0-9][a-z0-9.-]*$``, equal to the directory name; also the
            ``--served-model-name`` the engine serves and the client config's ``model``.
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
        client: The product's endpoint config for the role (validated by constructing it at load).
        reference: The subprocess reference the harness compares against.
        gates: Overrides of the stage-2 gate defaults.
        status: Where the recipe stands in the verification workflow.
        sources: URLs and ``path:line`` references the recipe rests on.
        notes: Free-form notes.
    """

    model_config = ConfigDict(**_no_extra())

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
    client: ClientEndpoint
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
        rerank = self.role == "rerank"
        if rerank and self.scoring is None:
            raise ValueError("a rerank recipe must set scoring: pointwise or listwise")
        if not rerank and self.scoring is not None:
            raise ValueError(f"scoring is only valid for role=rerank, not role={self.role}")
        client = self.client
        if isinstance(client, RerankEndpoint):
            if not rerank:
                raise ValueError(f"a RerankEndpoint config belongs to role=rerank, not role={self.role}")
            if self.scoring == "listwise" and not client.listwise:
                raise ValueError("scoring: listwise must set the endpoint's listwise flag")
            if self.scoring == "pointwise" and client.listwise:
                raise ValueError("scoring: pointwise conflicts with the endpoint's listwise flag")
        else:
            if rerank:
                raise ValueError(f"role=rerank needs a RerankEndpoint client config, got {type(client).__name__}")
        if self.role == "embed" and client.api != "openai_embeddings":
            raise ValueError(f"role=embed speaks api: openai_embeddings, got client.api={client.api!r}")
        if self.role == "multi_vector" and client.api != "vllm_pooling":
            raise ValueError(f"role=multi_vector speaks api: vllm_pooling, got client.api={client.api!r}")
        if isinstance(client, PoolingEndpoint) and client.template is not None and client.template.pair is not None:
            raise ValueError("a multi_vector recipe's template declares query and document shapes, not a pair")
        if self.role == "multi_vector" and not isinstance(client, PoolingEndpoint):
            raise ValueError(f"role=multi_vector needs a PoolingEndpoint client, got {type(client).__name__}")
        if self.role == "embed" and not isinstance(client, EmbeddingEndpoint):
            raise ValueError(f"role=embed needs an EmbeddingEndpoint client, got {type(client).__name__}")
        if self.client.max_tokens is not None and self.client.max_tokens > self.serve.max_model_len:
            raise ValueError(
                f"client.max_tokens ({self.client.max_tokens}) must not exceed engine.max_model_len "
                f"({self.serve.max_model_len}): the engine would 400 the rendered prompt"
            )
        return self


def default_recipes_root() -> Path:
    """The package's own ``recipes/`` directory (where the recipe lanes write)."""
    return Path(__file__).resolve().parents[2] / "recipes"


def _build_client(role: str, client_data: dict[str, Any], recipe_id: str, revision: str) -> ClientEndpoint:
    """The product's endpoint config for the role, from the recipe's ``client`` block.

    ``model`` and ``revision`` are the recipe's id and revision (the served name is the recipe id); the YAML
    may not carry them. The product model validates the rest, with the product's messages.
    """
    for injected in ("model", "revision"):
        if injected in client_data:
            raise ValueError(
                f"client.{injected} is the recipe's own {injected}; drop the field: the harness injects both "
                "from the recipe's id and revision"
            )
    client_data = {**client_data, "model": recipe_id, "revision": revision}
    classes: dict[str, type] = {
        "embed": EmbeddingEndpoint,
        "multi_vector": PoolingEndpoint,
        "rerank": RerankEndpoint,
    }
    endpoint_cls = classes[role]
    try:
        return endpoint_cls(**client_data)
    except Exception as error:
        raise RecipeError(f"the client block is not a valid {endpoint_cls.__name__}: {error}") from error


def load_recipe(path: str | Path) -> Recipe:
    """Load and validate one recipe from a recipe directory or a ``recipe.yaml`` file.

    Inputs: ``path``, the recipe directory (containing ``recipe.yaml``) or the YAML file itself.  Outputs: a
    frozen :class:`Recipe` whose ``client`` is the product's endpoint model (validated at load, with the
    product's messages) and whose ``_dir`` records where it came from.  Raises :class:`RecipeError` with the
    file path and the validator message when the YAML does not satisfy the schema, when ``id`` differs from the
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
    directory = path if path.is_dir() else yaml_path.parent
    try:
        client = _build_client(
            _role_of(data), dict(data.get("client") or {}), str(data.get("id")), str(data.get("revision") or "")
        )
        recipe = Recipe.model_validate({**data, "client": client})
    except RecipeError:
        raise
    except Exception as error:
        raise RecipeError(f"{yaml_path}: {error}") from error
    recipe._dir = directory
    if path.is_dir() and directory.name != recipe.id:
        raise RecipeError(
            f"{directory / 'recipe.yaml'}: id {recipe.id!r} must equal the directory name {directory.name!r}"
        )
    _check_referenced_files(recipe, directory)
    return recipe


def _role_of(data: dict[str, Any]) -> str:
    """The recipe's role, for the endpoint class; validated in full by the ``Recipe`` model afterwards."""
    role = data.get("role")
    if role not in ("embed", "multi_vector", "rerank"):
        raise RecipeError(f"recipe role {role!r} must be one of embed, multi_vector, rerank")
    return str(role)


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


def client_config(recipe: Recipe, *, base_url: str) -> dict[str, Any]:
    """The product's endpoint config dict this recipe implies, with the engine's ``base_url`` filled.

    Inputs: a recipe and the endpoint's ``base_url`` (e.g. ``http://127.0.0.1:8100/v1``).  Output: the product
    model's dump (:class:`~rcp_ndcg.inference.config.EmbeddingEndpoint`,
    :class:`~rcp_ndcg.inference.config.PoolingEndpoint` or
    :class:`~rcp_ndcg.inference.config.RerankEndpoint`), with ``base_url`` set — the dict the product's own
    config loader accepts unchanged, and what :func:`load_recipe` validated at load.
    """
    endpoint = recipe.client.model_copy(update={"base_url": base_url, "recipe": recipe.id})
    return endpoint.model_dump()


def recipe_json_schema() -> dict[str, Any]:
    """The JSON Schema of :class:`Recipe`, exported to ``schema/recipe.schema.json`` and kept current by a test."""
    return Recipe.model_json_schema()
