"""The case format and its validation: one reference case per YAML file under ``cases/<recipe-id>/``.

A case is one declared exercise of a served recipe: its inputs (the card's verbatim example, or a
generated stratum), what the recipe's engine is expected to return under a tolerance, and where those
numbers come from. The format is fixed by the operator (`drafts/case-format.md` in the lanes workspace);
this module implements it and refuses everything else -- a case the format would reject never loads.

Two validation levels:

- **File level** (no recipe needed): the schema, the ``id``/``recipe``/path agreement, the
  ``model_card`` provenance rules (a Hub URL, the verbatim quote, a 40-hex README revision), the media
  files' existence, the strata labels against the case's own inputs (a ``mixed`` length needs a
  ``mixed_length`` batch, a ``modality: image`` case needs an image document), and the ``expected``
  block's shape rules. Every load applies it.
- **Recipe level** (a loaded :class:`rcp_ndcg_vllm.recipe.Recipe` given or found): the case's role and
  modality against the recipe, the template shapes the case's sides need, the strata grid coverage of a
  recipe's case directory, and the long inputs' measured token lengths against the recipe's
  ``client.max_tokens`` with the product's tokenizer
  (:func:`rcp_ndcg.data.tokenizer.load_tokenizer`, the recipe's own spec).

The token lengths and the strata grid are what the GPU waves and CI lean on: a ``long_under`` case that
measures far under the budget, or a recipe missing its ``long_over`` cell, is a case-format violation,
not a weaker exercise.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml  # pyright: ignore[reportMissingModuleSource]
from pydantic import BaseModel, ConfigDict, Field, model_validator
from rcp_ndcg_vllm.recipe import Recipe, load_recipe

from .errors import CaseError

__all__ = [
    "Case",
    "CaseBundle",
    "CaseDocument",
    "CaseExpected",
    "CaseInputs",
    "CaseQuery",
    "CaseSource",
    "CaseStrata",
    "CaseTolerance",
    "default_cases_root",
    "load_case",
    "load_cases",
]

_ID_PATTERN = r"^[a-z0-9][a-z0-9.-]*$"
"""A recipe id, as the ``Recipe`` schema declares it (``rcp_ndcg_vllm.recipe``); a case's ``recipe``
field and the case directory name must match it."""

_REVISION_PATTERN = r"^[0-9a-f]{40}$"
"""A model-card revision: the 40-hex commit of the README the case was copied from."""

_HUB_URL_PREFIX = "https://huggingface.co/"
"""Where a model_card case's url must point (the format pins the card's provenance to the Hub)."""

_MEDIA_PREFIX = "media/"
"""Where a case's media files live, relative to the recipe's case directory."""

_LONG_UNDER_SHARE = 0.95
"""A ``long_under`` input measures at least this share of ``max_tokens`` ("within 5% under")."""


def _closed() -> dict[str, Any]:
    """The common model config: frozen, unknown fields refused (a typo must never load as data)."""
    return {"extra": "forbid", "frozen": True}


class CaseSource(BaseModel):
    """Where a case's inputs and printed outputs come from.

    A ``model_card`` case pins the Hub README it was copied from: ``url`` (a ``https://huggingface.co/``
    URL, as the format pins it), ``revision`` (40-hex), ``section`` and the verbatim ``quote`` are all
    required. A ``generated`` case carries none of them: it was built for a stratum, and naming a card
    would fake provenance.
    """

    model_config = ConfigDict(**_closed())

    kind: Literal["model_card", "generated"]
    url: str | None = Field(default=None, min_length=1)
    revision: str | None = Field(default=None, pattern=_REVISION_PATTERN)
    section: str | None = Field(default=None, min_length=1)
    quote: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _card_url_is_the_hub(self) -> CaseSource:
        """A model_card case's url names the Hub README the quote was copied from."""
        if self.kind == "model_card" and self.url is not None and not self.url.startswith(_HUB_URL_PREFIX):
            raise ValueError(
                f"a model_card case's url is the card's Hub README, {self.url!r} does not name "
                f"{_HUB_URL_PREFIX}<org>/<repo>"
            )
        return self


class CaseStrata(BaseModel):
    """Which cell of the modality x length x batch grid the case fills.

    ``length``: ``short`` (well under the budget), ``long_under`` (within 5% under
    ``client.max_tokens``, never cut), ``long_over`` (over it; the client cuts, the anchors must
    survive) or ``mixed`` (a batch that mixes lengths).
    """

    model_config = ConfigDict(**_closed())

    modality: Literal["text", "image", "video", "mixed"]
    length: Literal["short", "long_under", "long_over", "mixed"]
    batch: Literal["single", "uniform", "mixed_length", "mixed_modality"]


class CaseQuery(BaseModel):
    """One query of a case: its id and its text (the case format has text-only queries)."""

    model_config = ConfigDict(**_closed())

    id: str = Field(min_length=1)
    text: str


class CaseDocument(BaseModel):
    """One candidate of a case: its id and at least one of text, image or video.

    ``image`` / ``video`` name files under the recipe's ``media/`` directory, relative to the recipe's
    case directory (``media/<file>``); the files must exist (no network at test time).
    """

    model_config = ConfigDict(**_closed())

    id: str = Field(min_length=1)
    text: str | None = None
    image: str | None = Field(default=None, min_length=1)
    video: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _one_part_at_least(self) -> CaseDocument:
        """A document carries text, an image or a video; an image and a video never share one document."""
        if self.text is None and self.image is None and self.video is None:
            raise ValueError(f"document {self.id!r} carries no text, image or video")
        if self.image is not None and self.video is not None:
            raise ValueError(f"document {self.id!r} carries both an image and a video; one media kind per document")
        for field in ("image", "video"):
            value = getattr(self, field)
            if value is not None and not value.startswith(_MEDIA_PREFIX):
                raise ValueError(
                    f"document {self.id!r}: {field} paths live under the recipe's {_MEDIA_PREFIX} directory "
                    f"(relative to the case's recipe directory), got {value!r}"
                )
        return self


class CaseInputs(BaseModel):
    """What a case sends: the run-level instruction, the queries and the candidate documents."""

    model_config = ConfigDict(**_closed())

    instruction: str | None = None
    queries: list[CaseQuery] = Field(min_length=1)
    documents: list[CaseDocument] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_ids(self) -> CaseInputs:
        """The queries (and the documents) are identified by their ids; a duplicate would make an
        alignment error look like a result."""
        for kind, items in (("query", self.queries), ("document", self.documents)):
            ids = [item.id for item in items]
            duplicated = sorted({name for name in ids if ids.count(name) > 1})
            if duplicated:
                raise ValueError(f"duplicate {kind} ids: {duplicated}")
        return self


class CaseTolerance(BaseModel):
    """How far the engine's answer may sit from ``expected.values``: at most one rule per case.

    ``abs`` bounds every cell's absolute difference (the card's rounding plus a declared margin);
    ``rank_exact`` requires the derived ranking to equal ``expected.values`` exactly (ties broken toward
    the lower document index, as everywhere in the package); ``spearman_min`` bounds the rank correlation
    from below (an all-tie row carries no ranking information and is refused at load).
    """

    model_config = ConfigDict(**_closed())

    abs: float | None = Field(default=None, gt=0)
    rank_exact: bool | None = None
    spearman_min: float | None = Field(default=None, ge=-1, le=1)

    @property
    def rules(self) -> tuple[str, ...]:
        """The names of the rules this tolerance declares (at most one of them)."""
        return tuple(name for name in ("abs", "rank_exact", "spearman_min") if getattr(self, name) is not None)


class CaseExpected(BaseModel):
    """What the engine must return, under the tolerance, and where the numbers came from.

    ``values: null`` (every freshly generated case: ``status: pending_gpu``) is a *skip* for the
    conformance runner, never a pass -- the GPU waves fill it from the reference implementation and the
    engine. ``kind: none`` declares a case that only exercises a path; it carries no values and no
    tolerance.
    """

    model_config = ConfigDict(**_closed())

    kind: Literal["similarity_matrix", "scores", "ranking", "none"]
    values: Any = None
    tolerance: CaseTolerance | None = None
    origin: Literal["published", "reference", "engine"]
    status: Literal["published_unverified", "reproduced", "pending_gpu"]


class Case(BaseModel):
    """One reference case, validated at file level.

    Attributes:
        id: ``<recipe-id>/<case-slug>``; equal to the file's location (``cases/<recipe-id>/<slug>.yaml``).
        recipe: The recipe id the case exercises (the canonical id of the recipe directory).
        role: What the recipe produces; a case runs through the role client of its recipe's role.
        source: Where the inputs and printed outputs come from.
        strata: The grid cell the case covers.
        inputs: The instruction, the queries and the documents.
        expected: What the engine must return, under the tolerance.
        notes: Anything a reader needs (rounding in the card, fp32 vs bf16, a seed, a source).
    """

    model_config = ConfigDict(**_closed())

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9.-]*/[a-z0-9][a-z0-9._-]*$")
    recipe: str = Field(pattern=_ID_PATTERN)
    role: Literal["embed", "multi_vector", "rerank"]
    source: CaseSource
    strata: CaseStrata
    inputs: CaseInputs
    expected: CaseExpected
    notes: str = ""

    def _check_strata_labels(self) -> None:
        """The strata labels describe the case's own inputs: a mislabel cannot satisfy the grid."""
        documents = self.inputs.documents
        with_image = [document for document in documents if document.image is not None]
        with_video = [document for document in documents if document.video is not None]
        with_media = [*with_image, *with_video]
        text_only = [document for document in documents if document.image is None and document.video is None]
        modality = self.strata.modality
        if modality == "text" and with_media:
            raise ValueError(f"strata.modality 'text', but {len(with_media)} document(s) carry media")
        if modality == "image" and not with_image:
            raise ValueError("strata.modality 'image', but no document carries an image")
        if modality == "video" and not with_video:
            raise ValueError("strata.modality 'video', but no document carries a video")
        if modality == "mixed" and not (with_media and text_only):
            raise ValueError("strata.modality 'mixed' needs both media-bearing and text-only documents")
        if self.strata.batch == "mixed_modality" and not (with_media and text_only):
            raise ValueError(
                "strata.batch 'mixed_modality' means the batch mixes text-only and media-bearing documents"
            )

    @model_validator(mode="after")
    def _case_rules(self) -> Case:
        """The rules one case must satisfy: identity, provenance, tolerance and the values' shape."""
        head, _, slug = self.id.partition("/")
        if head != self.recipe or not slug:
            raise ValueError(f"id {self.id!r} must be <recipe>/<case-slug> with recipe {self.recipe!r}")
        if self.strata.length == "mixed" and self.strata.batch != "mixed_length":
            raise ValueError(
                f"length 'mixed' means the batch mixes lengths; declare batch: mixed_length, not {self.strata.batch!r}"
            )
        self._check_strata_labels()
        source = self.source
        if source.kind == "model_card":
            missing = [name for name in ("url", "revision", "section", "quote") if getattr(source, name) is None]
            if missing:
                raise ValueError(f"a model_card case must name its card: missing {', '.join(missing)}")
        elif any(getattr(source, name) is not None for name in ("url", "revision", "section", "quote")):
            raise ValueError("a generated case carries no card provenance: drop url, revision, section and quote")
        expected = self.expected
        if self.expected.origin == "published" and source.kind != "model_card":
            raise ValueError("origin 'published' means the card printed it, so source.kind must be model_card")
        n_queries, n_documents = len(self.inputs.queries), len(self.inputs.documents)
        if expected.kind == "none":
            if expected.values is not None or expected.tolerance is not None:
                raise ValueError("expected.kind 'none' exercises a path only: no values and no tolerance")
            return self
        if expected.tolerance is None:
            raise ValueError(f"expected.kind {expected.kind!r} needs a tolerance (abs, rank_exact or spearman_min)")
        if len(expected.tolerance.rules) != 1:
            raise ValueError(f"exactly one tolerance rule per case, got {expected.tolerance.rules}")
        rule = expected.tolerance.rules[0]
        if rule == "rank_exact" and expected.kind != "ranking":
            raise ValueError(f"rank_exact applies to kind 'ranking', not {expected.kind!r}")
        if expected.kind == "ranking" and rule == "abs":
            raise ValueError("kind 'ranking' compares orders: tolerance is rank_exact or spearman_min")
        if rule == "spearman_min" and n_documents < 2:
            raise ValueError("a spearman_min tolerance needs at least two documents to correlate")
        if rule == "spearman_min" and expected.values is not None:
            for row in expected.values:
                if isinstance(row, list) and len(row) >= 2 and all(value == row[0] for value in row):
                    raise ValueError(
                        "an all-tie expected row carries no ranking information; a spearman_min tolerance "
                        "refuses it at load, and the runner would fail it with a nan correlation"
                    )
        if expected.values is None:
            if expected.status != "pending_gpu":
                raise ValueError(f"expected.values is null, so status must be pending_gpu, not {expected.status!r}")
            return self
        if expected.status == "pending_gpu":
            raise ValueError("status pending_gpu means no values yet: the GPU wave fills them")
        _check_values(expected, n_queries, n_documents, [document.id for document in self.inputs.documents])
        return self


def _check_values(expected: CaseExpected, n_queries: int, n_documents: int, doc_ids: list[str]) -> None:
    """The values' shape per kind: a query x document rectangle, or ranked document ids per query."""
    values = expected.values
    if not isinstance(values, list) or any(not isinstance(row, list) for row in values):
        raise ValueError("expected.values is a list of rows (one per query)")
    if len(values) != n_queries:
        raise ValueError(f"expected.values holds {len(values)} row(s) for {n_queries} quer(y/ies)")
    if expected.kind == "ranking":
        for row in values:
            if not all(isinstance(entry, str) for entry in row):
                raise ValueError("a ranking row is a list of document ids")
            if len(set(row)) != len(row):
                raise ValueError(f"a ranking row names a document twice: {row}")
            unknown = sorted(set(row) - set(doc_ids))
            if unknown:
                raise ValueError(f"a ranking row names unknown documents: {unknown}")
            if expected.tolerance is not None and expected.tolerance.rank_exact and len(row) != n_documents:
                raise ValueError(f"rank_exact ranks every document: the row names {len(row)} of {n_documents}")
        return
    for index, row in enumerate(values):
        if len(row) != n_documents:
            raise ValueError(f"expected.values row {index} holds {len(row)} value(s) for {n_documents} document(s)")
        for value in row:
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"expected.values row {index} holds a non-numeric value: {value!r}")
            if value != value or value in (float("inf"), float("-inf")):
                raise ValueError(f"expected.values row {index} holds a non-finite value")


def _check_strata_labels(self) -> None:
    """The strata labels describe the case's own inputs, so a mislabel cannot satisfy the grid."""
    documents = self.inputs.documents
    with_image = [document for document in documents if document.image is not None]
    with_video = [document for document in documents if document.video is not None]
    with_media = [*with_image, *with_video]
    modality = self.strata.modality
    if modality == "text" and with_media:
        raise ValueError(f"strata.modality 'text', but {len(with_media)} document(s) carry media")
    if modality == "image" and not with_image:
        raise ValueError("strata.modality 'image', but no document carries an image")
    if modality == "video" and not with_video:
        raise ValueError("strata.modality 'video', but no document carries a video")
    if modality == "mixed" and not (
        with_media and any(document.image is None and document.video is None for document in documents)
    ):
        raise ValueError("strata.modality 'mixed' needs both media-bearing and text-only documents")
    if self.strata.batch == "mixed_modality" and not (
        with_media and any(document.image is None and document.video is None for document in documents)
    ):
        raise ValueError("strata.batch 'mixed_modality' means the batch mixes text-only and media-bearing documents")


class CaseBundle:
    """What :func:`load_cases` loaded: the cases, the recipes that backed them, and what did not run.

    Attributes:
        root: The cases root the bundle was read from.
        cases: Every loaded case, sorted by id.
        recipes: The recipe id -> the loaded :class:`rcp_ndcg_vllm.recipe.Recipe` for every case
            directory whose recipe exists at the recipes root (the recipe-backed checks ran on those).
        recipes_missing: The recipe ids with a case directory but no recipe at the recipes root. Their
            cases are file-validated only; the recipe-backed checks are recorded here, never silently
            dropped.
        skipped_checks: The named checks that did not run (``lengths:<recipe>`` when ``check_lengths``
            is false, or when a recipe declares no ``max_tokens`` to measure against).
    """

    def __init__(
        self,
        root: Path,
        cases: tuple[Case, ...],
        recipes: dict[str, Recipe],
        recipes_missing: tuple[str, ...],
        skipped_checks: tuple[str, ...],
    ) -> None:
        self.root = root
        self.cases = cases
        self.recipes = recipes
        self.recipes_missing = recipes_missing
        self.skipped_checks = skipped_checks

    def __repr__(self) -> str:
        return (
            f"CaseBundle(root={str(self.root)!r}, cases={len(self.cases)}, recipes={len(self.recipes)}, "
            f"recipes_missing={list(self.recipes_missing)}, skipped_checks={list(self.skipped_checks)})"
        )


def default_cases_root() -> Path:
    """The package's own ``cases/`` directory (where the cases lanes write)."""
    return Path(__file__).resolve().parents[2] / "cases"


def load_case(path: str | Path) -> Case:
    """Load and validate one case file.

    Inputs: the path of a ``cases/<recipe-id>/<case-slug>.yaml`` file. Outputs: a frozen :class:`Case`.
    The file-level rules apply (schema, id/path agreement, card provenance, media files' existence,
    expected shapes); the recipe-level rules need the recipe and run in :func:`load_cases`.

    Raises:
        CaseError: the file is not valid YAML, not a mapping of the schema, its ``id`` disagrees with
            its path, or a rule of the case format is broken. The message names the file, the field and
            the value.
    """
    path = Path(path)
    if not path.is_file():
        raise CaseError(f"no case file at {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise CaseError(f"{path} is not valid YAML: {error}") from error
    if not isinstance(data, dict):
        raise CaseError(f"{path} must contain a mapping of the case format, got {type(data).__name__}")
    recipe_id, _, slug = str(data.get("id", "")).partition("/")
    if path.stem != slug or path.parent.name != recipe_id:
        raise CaseError(
            f"{path}: id {data.get('id')!r} must equal the file's location "
            f"(<recipe-id>/{path.stem!r}, in a directory named for the recipe id)"
        )
    try:
        case = Case.model_validate(data)
    except Exception as error:
        raise CaseError(f"{path}: {error}") from error
    _check_media(case, path.parent)
    return case


def _check_media(case: Case, recipe_cases_dir: Path) -> None:
    """Every media file a case names exists, under the recipe's ``media/`` directory.

    No network at test time: the file is the case's asset. A media path that escapes the recipe's
    directory (``..``, an absolute path) never matches a file and is refused by the existence check.
    """
    for document in case.inputs.documents:
        for field in ("image", "video"):
            value = getattr(document, field)
            if value is None:
                continue
            media = recipe_cases_dir / value
            if not media.is_file():
                raise CaseError(
                    f"{recipe_cases_dir.name}/{case.id.partition('/')[2]}.yaml: document {document.id!r} names "
                    f"{field} {value!r}, which does not exist (expected {media})"
                )


def load_cases(
    root: str | Path | None = None,
    recipe: Recipe | str | None = None,
    *,
    recipes_root: str | Path | None = None,
    check_lengths: bool = True,
) -> CaseBundle:
    """Load a root of case directories, validating every rule of the case format that holds without a GPU.

    Inputs: ``root`` (the cases root; the package's ``cases/`` when ``None``), an optional ``recipe``
    (a loaded :class:`rcp_ndcg_vllm.recipe.Recipe`, or its id: only that recipe's directory is loaded
    and the full recipe-backed validation runs against it), ``recipes_root`` (where recipe directories
    live; the serving-recipes package's ``recipes/`` when ``None``) and ``check_lengths``.

    Outputs: a :class:`CaseBundle`. Every case is file-validated. For a case directory whose recipe
    exists at the recipes root, the recipe-backed rules run: the role and modality against the recipe,
    the template shapes the case's sides need, the strata grid coverage of the directory, and -- with
    ``check_lengths`` (the default) -- the long inputs' measured token lengths against
    ``client.max_tokens``, measured with the product tokenizer the recipe names. A directory whose
    recipe is missing is recorded in ``recipes_missing`` (its cases stay file-validated); a check
    skipped by request or for lack of a measurable budget is recorded in ``skipped_checks``.

    Raises:
        CaseError: any file-level or recipe-level rule is broken. A malformed case never loads.
    """
    from rcp_ndcg_vllm.recipe import default_recipes_root

    root = Path(root) if root is not None else default_cases_root()
    recipes_dir = Path(recipes_root) if recipes_root is not None else default_recipes_root()
    cases: list[Case] = []
    backing: dict[str, Recipe] = {}
    missing: list[str] = []
    skipped: list[str] = []
    for directory in _case_directories(root, recipe):
        case_files = sorted(directory.glob("*.yaml"))
        if not case_files:
            raise CaseError(f"{directory} holds no case files (one case per <case-slug>.yaml)")
        dir_cases = [load_case(path) for path in case_files]
        cases.extend(dir_cases)
        loaded = _recipe_of(directory.name, recipe, recipes_dir)
        if loaded is None:
            missing.append(directory.name)
            continue
        backing[loaded.id] = loaded
        _validate_against_recipe(loaded, dir_cases, check_lengths=check_lengths, skipped=skipped)
        _check_coverage(loaded, dir_cases)
    cases.sort(key=lambda case: case.id)
    return CaseBundle(
        root=root,
        cases=tuple(cases),
        recipes=backing,
        recipes_missing=tuple(missing),
        skipped_checks=tuple(dict.fromkeys(skipped)),
    )


def _case_directories(root: Path, recipe: Recipe | str | None) -> list[Path]:
    """The case directories to load: the one ``recipe``'s, or every child of ``root`` holding YAML files."""
    if not root.is_dir():
        raise CaseError(f"no cases root at {root}")
    if recipe is not None:
        recipe_id = recipe.id if isinstance(recipe, Recipe) else recipe
        directory = root / recipe_id
        if not directory.is_dir():
            raise CaseError(f"no case directory for recipe {recipe_id!r} at {directory}")
        return [directory]
    return sorted(entry for entry in root.iterdir() if entry.is_dir() and any(entry.glob("*.yaml")))


def _recipe_of(recipe_id: str, recipe: Recipe | str | None, recipes_dir: Path) -> Recipe | None:
    """The recipe that backs a case directory, loaded from the recipes root; ``None`` when it is absent."""
    if isinstance(recipe, Recipe):
        return recipe
    path = recipes_dir / recipe_id / "recipe.yaml"
    if not path.is_file():
        return None
    try:
        return load_recipe(path)
    except Exception as error:
        raise CaseError(f"the recipe backing the cases at {recipes_dir / recipe_id} does not load: {error}") from error


def _validate_against_recipe(recipe: Recipe, cases: list[Case], *, check_lengths: bool, skipped: list[str]) -> None:
    """The recipe-backed rules for one recipe's cases: role, modality, template shapes, lengths."""
    for case in cases:
        if case.recipe != recipe.id:
            raise CaseError(f"case {case.id!r} declares recipe {case.recipe!r}, loaded against {recipe.id!r}")
        if case.role != recipe.role:
            raise CaseError(
                f"case {case.id!r} declares role {case.role!r} but recipe {recipe.id} serves role {recipe.role!r}"
            )
        _check_media_kinds(recipe, case)
        _check_template_shapes(recipe, case)
    if check_lengths:
        _check_lengths(recipe, cases)
    elif recipe.client.max_tokens is not None:
        skipped.append(f"lengths:{recipe.id} (check_lengths=False)")
    if recipe.client.max_tokens is None:
        skipped.append(f"lengths:{recipe.id} (no max_tokens declared)")


def _check_media_kinds(recipe: Recipe, case: Case) -> None:
    """A case only names media the recipe's model accepts (an ``image`` cell needs an image input)."""
    for document in case.inputs.documents:
        if document.image is not None and "image" not in recipe.input:
            raise CaseError(f"case {case.id!r} names an image, but recipe {recipe.id} accepts input {recipe.input}")
        if document.video is not None and "video" not in recipe.input:
            raise CaseError(f"case {case.id!r} names a video, but recipe {recipe.id} accepts input {recipe.input}")
    modality = case.strata.modality
    allowed = {kind for kind in ("text", "image", "video") if kind in recipe.input}
    if modality == "mixed":
        if len(allowed) < 2:
            raise CaseError(
                f"case {case.id!r} declares modality 'mixed', but recipe {recipe.id} accepts only {sorted(allowed)}"
            )
    elif modality not in allowed:
        raise CaseError(
            f"case {case.id!r} declares modality {modality!r}, but recipe {recipe.id} accepts only {sorted(allowed)}"
        )


def _check_template_shapes(recipe: Recipe, case: Case) -> None:
    """The template declares every shape the case's sides need (the runner fits those shapes)."""
    template = recipe.client.template
    if template is None:
        return
    declared = set(template.shapes())
    needed: set[str] = {"pair"} if recipe.role == "rerank" else {"query", "document"}
    missing = sorted(needed - declared)
    if missing:
        raise CaseError(
            f"case {case.id!r} needs the {', '.join(missing)} shape(s), which recipe {recipe.id}'s template "
            f"does not declare (it declares {', '.join(sorted(declared)) or 'none'})"
        )


def _check_lengths(recipe: Recipe, cases: list[Case]) -> None:
    """The long inputs of every long_under / long_over case, measured with the product tokenizer.

    A ``long_under`` case: no text input measures over ``client.max_tokens``, and at least one measures
    within 5% under it (``>= 0.95 * max_tokens``) -- the stratum is only real if it is near the budget.
    A ``long_over`` case: at least one text input measures over ``client.max_tokens``. Media carry no
    text and are not measured (their token counts are the engine's, not the tokenizer's).
    """
    if recipe.client.max_tokens is None:
        return  # recorded as skipped by the caller: a hosted profile declares no client-side budget
    max_tokens = recipe.client.max_tokens
    tokenizer = _tokenizer_of(recipe)
    for case in cases:
        if case.strata.batch == "mixed_length" and case.strata.length != "mixed":
            measured_batch = [
                tokenizer.count(text)
                for text in [query.text for query in case.inputs.queries]
                + [document.text or "" for document in case.inputs.documents]
            ]
            if len(set(measured_batch)) < 2:
                raise CaseError(
                    f"case {case.id!r}: batch mixed_length, but every text input measures "
                    f"{measured_batch[0]} tokens; the batch holds no mixed lengths"
                )
        if case.strata.length not in ("long_under", "long_over"):
            continue
        measured = [
            (label, item.id, tokenizer.count(text))
            for label, items in (("query", case.inputs.queries), ("document", case.inputs.documents))
            for item in items
            if (text := item.text) is not None
        ]
        counts = [count for _, _, count in measured]
        if case.strata.length == "long_under":
            over = next(((label, item_id, count) for label, item_id, count in measured if count > max_tokens), None)
            if over is not None:
                label, item_id, count = over
                raise CaseError(
                    f"case {case.id!r}: long_under, but {label} {item_id!r} measures {count} tokens over the "
                    f"budget ({max_tokens}); that is the long_over stratum"
                )
            if not any(count >= _LONG_UNDER_SHARE * max_tokens for count in counts):
                raise CaseError(
                    f"case {case.id!r}: long_under, but no input measures within 5% under the budget "
                    f"(>= {int(_LONG_UNDER_SHARE * max_tokens)} of {max_tokens} tokens); measured {counts}"
                )
        elif not any(count > max_tokens for count in counts):
            raise CaseError(
                f"case {case.id!r}: long_over, but no input measures over the budget ({max_tokens} tokens); "
                f"measured {counts}"
            )


def _tokenizer_of(recipe: Recipe) -> Any:
    """The product tokenizer the recipe's client block names (the harness's own loader)."""
    from rcp_ndcg_vllm.equivalence.fitting import tokenizer_of as harness_tokenizer_of

    try:
        return harness_tokenizer_of(recipe)
    except Exception as error:
        raise CaseError(
            f"recipe {recipe.id}: loading its tokenizer ({recipe.client.tokenizer!r}) failed: {error}; a recipe "
            "whose tokenizer lives on the Hub needs it cached or a network run (load_cases(..., "
            "check_lengths=False) records the check as skipped instead)"
        ) from error


def _check_coverage(recipe: Recipe, cases: list[Case]) -> None:
    """The strata grid of one recipe's case directory: every applicable cell holds at least one case.

    The applicable cells (case-format.md): the lengths ``short``, ``long_under`` and ``long_over``; the
    ``mixed_length`` batch; for a recipe that accepts images the ``image`` modality, for one that
    accepts video the ``video`` modality, and for a recipe with two or more input kinds the
    ``mixed_modality`` batch. A cell nobody can fill is not required; a missing one is a coverage gap
    and fails the load, so an incomplete grid cannot pass CI.
    """
    covered: set[str] = set()
    for case in cases:
        covered.add(f"length:{case.strata.length}")
        covered.add(f"batch:{case.strata.batch}")
        covered.add(f"modality:{case.strata.modality}")
    required = [f"length:{length}" for length in ("short", "long_under", "long_over")]
    required.append("batch:mixed_length")
    if "image" in recipe.input:
        required.append("modality:image")
    if "video" in recipe.input:
        required.append("modality:video")
    if len(recipe.input) >= 2:
        required.append("batch:mixed_modality")
    missing = [cell for cell in required if cell not in covered]
    if missing:
        raise CaseError(
            f"recipe {recipe.id}: the strata grid is incomplete; no case covers {', '.join(missing)} "
            f"(covered: {', '.join(sorted(covered)) or 'nothing'})"
        )
