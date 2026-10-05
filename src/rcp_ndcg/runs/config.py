"""The configuration of one run: which dataset, which candidates, which judge, which steps.

One YAML file validates into :class:`RunConfig`. Every section is typed and
unknown keys are refused (``extra="forbid"``), so a typo fails at load instead
of being ignored; the free-form exceptions are a dataset reader's ``options``
and a plugin runner's ``options``, which that reader or runner checks. A config
may start from another with ``extends: <path>`` (deep-merged, see
:mod:`rcp_ndcg.support.config`), and ``--set key=value`` overrides one field::

    extends: ../base.yaml
    dataset: hf://fabianschmidt-cohere/rcp-ndcg-nanobeir/NanoNFCorpusRetrieval
    candidates:
      from: retrieval
      retrieval: {kind: bm25}
      depth: 150
    judge: gpt_oss_120b
    steps: [retrieve, tournament, rubric, calibrate, evaluate]

The dataset is a URI of :func:`rcp_ndcg.data.load_dataset` (or a mapping with ``uri``, ``subset``, ``revision``
and reader ``options``). A ``judge:`` names a shipped judge config (:mod:`rcp_ndcg.llm.judges`) or gives a path
to one; the file is read when an override names one of its fields (``--set judge.base_url=...``), so the
override applies to the file's values.

The schedules default to the paper's (:class:`~rcp_ndcg.llm.TournamentSchedule`,
:class:`~rcp_ndcg.llm.RubricSchedule`); the number of rubric criteria is the
rubric prompt's, never a config field.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag, model_validator
from rcp_ndcg_core.irt import Priors

from rcp_ndcg.data.preprocess import Preprocessing
from rcp_ndcg.errors import ConfigError
from rcp_ndcg.llm.client import JudgeConfig
from rcp_ndcg.llm.schedule import RubricSchedule, TournamentSchedule
from rcp_ndcg.retrieval import RerankerConfig, RetrieverConfig
from rcp_ndcg.runners.kubernetes import KubernetesOptions
from rcp_ndcg.runners.local import LocalOptions
from rcp_ndcg.runners.slurm import SlurmOptions
from rcp_ndcg.runs.mirror import DEFAULT_INTERVAL_S
from rcp_ndcg.support.identity import FieldRole
from rcp_ndcg.support.serve import ServeConfig

#: The steps of a run, in the order they run.
StepName = Literal["retrieve", "rerank", "tournament", "rubric", "calibrate", "evaluate"]
STEPS: tuple[StepName, ...] = ("retrieve", "rerank", "tournament", "rubric", "calibrate", "evaluate")

#: Steps that call the judge.
JUDGE_STEPS: frozenset[str] = frozenset({"tournament", "rubric"})

_FORBID = ConfigDict(extra="forbid", populate_by_name=True)
_CONTENT = FieldRole.CONTENT


class DatasetSource(BaseModel):
    """The dataset of a run: a :func:`~rcp_ndcg.data.load_dataset` URI and its arguments.

    Written as the URI alone (``dataset: jsonl:rows.jsonl``) or as a mapping.

    Attributes:
        uri: The dataset URI (``hf://``, ``suite:``, ``beir:``, ``jsonl:``, ``images:``, ...).
        subset: The subset of a ``hf://`` repository or ``suite:``.
        revision: The Hub revision.
        options: Reader options of the reader schemes (e.g. ``qrels_uri`` for ``images:``).
    """

    model_config = _FORBID

    uri: str
    subset: str | None = None
    revision: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _a_uri_alone(cls, value: Any) -> Any:
        return {"uri": value} if isinstance(value, str) else value

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema: Any, handler: Any) -> dict[str, Any]:
        """The JSON Schema accepts both forms: the URI alone, or the mapping."""
        return {"anyOf": [{"type": "string", "description": "The dataset URI alone."}, handler(core_schema)]}

    def load(self):
        """The :class:`~rcp_ndcg.data.Dataset` (read lazily: queries and corpus on first access)."""
        from rcp_ndcg.data import load_dataset

        return load_dataset(self.uri, subset=self.subset, revision=self.revision, **self.options)

    def identity(self) -> dict[str, Any]:
        """What names the data: the source and, for a Hub dataset, the commit it resolves to now."""
        from rcp_ndcg.data.revisions import dataset_uri_revision

        payload = self.model_dump(mode="json", exclude_defaults=True)
        commit = dataset_uri_revision(self.uri, self.revision)
        return {**payload, "resolved": commit} if commit is not None else payload


class CandidatesConfig(BaseModel):
    """Where each query's candidate pool comes from.

    Every field changes the numbers (the retriever and the reranker declare their own roles, runtime fields and
    all, in :mod:`rcp_ndcg.retrieval.config`), so the whole section is content: a run's step identities hold it by
    :func:`rcp_ndcg.support.identity.identity_payload`.

    Attributes:
        source: ``"dataset"`` (the dataset's own pools, else its judged documents), ``"rankings"`` (a rankings
            file of :func:`~rcp_ndcg.data.load_rankings`) or ``"retrieval"`` (first-stage retrieval). Written
            ``from:`` in YAML.
        rankings: The rankings file, for ``from: rankings``.
        system: The system of a rankings file that holds several.
        retrieval: The retriever (:data:`~rcp_ndcg.retrieval.RetrieverConfig`), for ``from: retrieval``.
        rerank: A reranker applied to the pools (the ``rerank`` step).
        depth: Candidates per query kept for judging.
    """

    model_config = _FORBID

    IDENTITY_ROLES: ClassVar[dict[str, FieldRole]] = {
        "source": _CONTENT,
        "rankings": _CONTENT,
        "system": _CONTENT,
        "retrieval": _CONTENT,
        "rerank": _CONTENT,
        "depth": _CONTENT,
    }

    source: Literal["dataset", "rankings", "retrieval"] = Field(default="dataset", alias="from")
    rankings: str | None = None
    system: str | None = None
    retrieval: RetrieverConfig | None = None
    rerank: RerankerConfig | None = None
    depth: int = Field(default=150, ge=1)

    @model_validator(mode="after")
    def _source_has_its_input(self) -> Self:
        if self.source == "rankings" and not self.rankings:
            raise ValueError("candidates.from: rankings needs candidates.rankings (a rankings file)")
        if self.source == "retrieval" and self.retrieval is None:
            raise ValueError("candidates.from: retrieval needs candidates.retrieval (a retriever config)")
        return self


class CalibrationOptions(BaseModel):
    """How the calibrate step fits (see :func:`rcp_ndcg.calibration.calibrate`)."""

    model_config = _FORBID

    mode: Literal["auto", "tournament", "rubric_only"] = "auto"
    judges: Literal["single", "pooled"] = "single"
    priors: Priors = Priors()


class EvaluationOptions(BaseModel):
    """What the evaluate step scores.

    Attributes:
        k: The metric cutoff.
        systems: ``{name: rankings file}`` of further systems to score with the run's calibration, next to the
            run's reference systems (``candidates``, the pool order, and ``judge``, the judge's own abilities).
            A file of one system is scored under ``name``; ``<file>#<system>`` picks one system of a file that
            holds several; a file of several systems without ``#`` scores each under its own name.
    """

    model_config = _FORBID

    k: int = Field(default=10, ge=1)
    systems: dict[str, str] = Field(default_factory=dict)


class _Runner(BaseModel):
    model_config = _FORBID

    def option_values(self) -> dict[str, Any]:
        """The options the config sets (a runner's own defaults apply to the rest), as keyword arguments."""
        options: Any = self.options  # type: ignore[attr-defined]
        return dict(options) if isinstance(options, dict) else options.model_dump(exclude_unset=True)


class LocalRunnerConfig(_Runner):
    """Run on this host (in this process, unless ``run start --runner local`` or ``--detach``)."""

    name: Literal["local"] = "local"
    options: LocalOptions = LocalOptions()


class SlurmRunnerConfig(_Runner):
    """Run as one ``sbatch`` job."""

    name: Literal["slurm"]
    options: SlurmOptions = SlurmOptions()


class KubernetesRunnerConfig(_Runner):
    """Run as one Kubernetes ``batch/v1`` Job."""

    name: Literal["kubernetes"]
    options: KubernetesOptions = KubernetesOptions()


_PUBLIC_RUNNERS = ("local", "slurm", "kubernetes")


class PluginRunnerConfig(_Runner):
    """Run on a runner another installed package registers under the ``rcp_ndcg.runners`` entry-point group; its
    options are that runner's (``resources``, ``image`` and ``env`` describe the job). Its name is none of the
    public runners', whose options are typed, so the JSON Schema of a runner section is one branch exactly."""

    name: str = Field(json_schema_extra={"not": {"enum": list(_PUBLIC_RUNNERS)}})
    options: dict[str, Any] = Field(default_factory=dict)


def _runner_tag(value: Any) -> str:
    name = value.get("name", "local") if isinstance(value, dict) else getattr(value, "name", "local")
    return name if name in _PUBLIC_RUNNERS else "plugin"


#: Where the steps run: a public runner with its typed options, or a plugin's with free ones.
RunnerConfig = Annotated[
    Annotated[LocalRunnerConfig, Tag("local")]
    | Annotated[SlurmRunnerConfig, Tag("slurm")]
    | Annotated[KubernetesRunnerConfig, Tag("kubernetes")]
    | Annotated[PluginRunnerConfig, Tag("plugin")],
    Discriminator(_runner_tag),
]


class RunConfig(BaseModel):
    """The resolved configuration of one run.

    Attributes:
        label: A readable fragment of the run id.
        dataset: The dataset (its queries, corpus, labels and pools), by URI.
        candidates: Where the candidate pools come from, and their depth.
        judge: The judge: an inline :class:`JudgeConfig`, a shipped config's name, a path to one, or
            ``"fake"`` for the offline judge.
        steps: The steps to run (run in :data:`STEPS` order whatever order they are listed in).
        tournament: The tournament schedule; ``None`` is the paper's for the corpus's modality.
        rubric: The rubric schedule; ``None`` is the paper's for the corpus's modality.
        calibration: How the calibrate step fits.
        evaluation: What the evaluate step scores.
        preprocessing: The text policy, chunking and image policy the judging steps apply.
        mirror: Any fsspec URI (e.g. ``s3://bucket/runs``) the run directory is mirrored to while it runs, and
            restored from on resume (:mod:`rcp_ndcg.runs.mirror`).
        mirror_interval_s: Seconds between two uploads of the mirror.
        seed: The run's seed: of the judging schedules that set none (default: the schedules' own), of the
            offline judge (``judge: fake``) and of the evaluation's bootstrap intervals.
        limit: Judge only the first ``limit`` queries.
        runner: Where the steps run.
        serve: The judge's engine, started beside the run's job by the ``slurm`` or ``kubernetes`` runner
            (:class:`~rcp_ndcg.support.serve.ServeConfig`); the judge then uses its replicas' URLs. Omit it to
            bring your own endpoint (``judge.base_url``).
    """

    model_config = _FORBID

    label: str | None = None
    dataset: DatasetSource
    candidates: CandidatesConfig = CandidatesConfig()
    judge: JudgeConfig | str | None = None
    steps: list[StepName] = Field(default_factory=lambda: ["tournament", "rubric", "calibrate", "evaluate"])
    tournament: TournamentSchedule | None = None
    rubric: RubricSchedule | None = None
    calibration: CalibrationOptions = CalibrationOptions()
    evaluation: EvaluationOptions = EvaluationOptions()
    preprocessing: Preprocessing | None = None
    mirror: str | None = None
    mirror_interval_s: float = Field(default=DEFAULT_INTERVAL_S, gt=0)
    seed: int = TournamentSchedule.model_fields["seed"].default
    limit: int | None = Field(default=None, ge=1)
    runner: RunnerConfig = LocalRunnerConfig()
    serve: ServeConfig | None = None

    @model_validator(mode="after")
    def _steps_have_their_inputs(self) -> Self:
        if "retrieve" in self.steps and self.candidates.source == "dataset":
            raise ValueError("the retrieve step needs candidates.from: retrieval or rankings")
        if "rerank" in self.steps and self.candidates.rerank is None:
            raise ValueError("the rerank step needs candidates.rerank (the reranker's settings)")
        if JUDGE_STEPS & set(self.steps) and self.judge is None:
            raise ValueError("the tournament and rubric steps need a judge (judge: <config path> | fake | {...})")
        if self.serve is not None and (self.judge is None or self.judge == "fake"):
            raise ValueError("serve: starts the judge's engine, and this run has no served judge (judge: fake | none)")
        return self

    @model_validator(mode="after")
    def _schedules_take_the_runs_seed(self) -> Self:
        # Written into the config, so the recorded config (and a resume) carries the seed the schedule ran with.
        for stage in ("tournament", "rubric"):
            schedule = getattr(self, stage)
            if schedule is not None and "seed" not in schedule.model_fields_set:
                setattr(self, stage, schedule.model_copy(update={"seed": self.seed}))
        return self

    @property
    def ordered_steps(self) -> list[StepName]:
        """The configured steps in run order."""
        return [step for step in STEPS if step in self.steps]

    @classmethod
    def load(cls, path: str | Path, *, overrides: Sequence[str] = ()) -> RunConfig:
        """Read a run config YAML (``extends:`` resolved, ``key=value`` overrides applied).

        Relative paths in the file (the dataset, a rankings file, evaluation systems, a judge config, a prompt
        file) are relative to the file that declares them (a base config's to the base); the overrides' paths
        stay relative to the working directory.

        Args:
            path: A YAML file, or the name of a packaged run config (:mod:`rcp_ndcg.examples`: ``tiny``,
                ``rejudge_nfcorpus``, ...).
            overrides: ``dotted.key=value`` overrides.

        Raises:
            ConfigError: the file does not validate; the message names the fields.
            MissingInputError: ``path`` is neither a file nor a packaged config's name.
        """
        from pydantic import ValidationError

        from rcp_ndcg.support.config import apply_overrides, config_error, load_config

        path = _config_file(path)
        data = load_config(path, relative=_relative_to)
        data = inline_judge(data, overrides, base=path.parent)
        try:
            return cls.model_validate(apply_overrides(data, overrides))
        except ValidationError as exc:
            raise config_error(exc, model=cls, source=str(path), overrides=overrides) from exc

    @classmethod
    def from_data(cls, data: dict[str, Any], *, overrides: Sequence[str] = ()) -> RunConfig:
        """Validate a config mapping (e.g. a manifest's recorded config) with ``key=value`` overrides applied.

        Raises:
            ConfigError: the result does not validate.
        """
        from pydantic import ValidationError

        from rcp_ndcg.support.config import apply_overrides, config_error

        try:
            return cls.model_validate(apply_overrides(inline_judge(data, overrides), overrides))
        except ValidationError as exc:
            raise config_error(exc, model=cls, source="the run's config", overrides=overrides) from exc

    def resolved(self) -> dict[str, Any]:
        """The config as JSON-ready data (what ``run.yaml`` and the manifest record)."""
        return self.model_dump(mode="json", by_alias=True, exclude_none=True)

    def local_inputs(self) -> list[str]:
        """``"<field>: <path>"`` of every input the config reads from this host's filesystem.

        A job whose runner does not share this host's files (Kubernetes) cannot read them: the dataset (a local
        reader scheme, or a reader option ``*_uri``), a rankings file, evaluation systems, a judge config file and a
        prompt file. A remote URI (``hf://``, ``s3://``, ``https://``, ...) and a shipped name are not local.
        """
        from rcp_ndcg.llm.judges import judge_names
        from rcp_ndcg.llm.prompts import PROMPT_FILES
        from rcp_ndcg.storage import is_remote

        found: list[str] = []

        def local(field: str, location: str | None) -> None:
            if location and not is_remote(location):
                found.append(f"{field}: {location}")

        scheme, sep, rest = self.dataset.uri.partition(":")
        if sep and scheme in _LOCAL_SCHEMES:
            local("dataset", rest)
        for key, value in self.dataset.options.items():
            if key.endswith("_uri") and isinstance(value, str):
                local(f"dataset.options.{key}", value)
        local("candidates.rankings", self.candidates.rankings)
        for name, location in self.evaluation.systems.items():
            local(f"evaluation.systems.{name}", location.partition("#")[0])
        if isinstance(self.judge, str) and self.judge != "fake" and self.judge not in judge_names():
            local("judge", self.judge)
        for stage in ("tournament", "rubric"):
            schedule = getattr(self, stage)
            if schedule is not None and schedule.prompt is not None and schedule.prompt not in PROMPT_FILES:
                local(f"{stage}.prompt", schedule.prompt)
        return found

    def judge_config(self) -> JudgeConfig:
        """The judge as a :class:`JudgeConfig`.

        Raises:
            ConfigError: no judge is configured.
        """
        if self.judge is None:
            raise ConfigError("this run has no judge", hint="set judge: <config path> | fake")
        if isinstance(self.judge, JudgeConfig):
            return self.judge
        if self.judge == "fake":
            return JudgeConfig.fake(self.seed)
        return JudgeConfig.load(self.judge)


def _config_file(path: str | Path) -> Path:
    """The run config file: ``path`` itself, or the packaged config of that name when no such file exists."""
    candidate = Path(path)
    if candidate.is_file() or candidate.suffix in (".yaml", ".yml") or len(candidate.parts) > 1:
        return candidate
    from rcp_ndcg.examples import run_config_path

    return run_config_path(str(path))


#: Reader schemes whose location is a local path (``hf://`` and ``suite:`` name remote data).
_LOCAL_SCHEMES = ("beir", "jsonl", "images", "videos", "frames", "pdf")


def _relative_to(data: dict[str, Any], base: Path) -> dict[str, Any]:
    """``data`` with the relative paths a run config names made relative to ``base`` (the config's directory)."""
    data = dict(data)

    def local(location: Any) -> Any:
        if not isinstance(location, str) or not location or "://" in location or Path(location).is_absolute():
            return location
        return str((base / location).resolve())

    def uri(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        scheme, sep, rest = value.partition(":")
        return f"{scheme}:{local(rest)}" if sep and scheme in _LOCAL_SCHEMES else value

    dataset = data.get("dataset")
    if isinstance(dataset, str):
        data["dataset"] = uri(dataset)
    elif isinstance(dataset, dict):
        options = {k: local(v) if k.endswith("_uri") else v for k, v in (dataset.get("options") or {}).items()}
        data["dataset"] = {**dataset, "uri": uri(dataset.get("uri")), **({"options": options} if options else {})}
    candidates = data.get("candidates")
    if isinstance(candidates, dict) and candidates.get("rankings"):
        data["candidates"] = {**candidates, "rankings": local(candidates["rankings"])}
    evaluation = data.get("evaluation")
    if isinstance(evaluation, dict) and isinstance(evaluation.get("systems"), dict):
        data["evaluation"] = {**evaluation, "systems": {k: local(v) for k, v in evaluation["systems"].items()}}
    judge = data.get("judge")
    if isinstance(judge, str) and (base / judge).is_file():  # else a shipped judge's name (or fake)
        data["judge"] = local(judge)
    for stage in ("tournament", "rubric"):
        schedule = data.get(stage)
        if (
            isinstance(schedule, dict)
            and isinstance(schedule.get("prompt"), str)
            and (base / schedule["prompt"]).is_file()
        ):
            data[stage] = {**schedule, "prompt": local(schedule["prompt"])}
    return data


def inline_judge(data: dict[str, Any], overrides: Sequence[str], *, base: Path | None = None) -> dict[str, Any]:
    """``data`` with a ``judge:`` path replaced by the file's mapping when an override names a judge field.

    ``--set judge.base_url=...`` then applies to the file's values instead of failing on a string.

    Args:
        data: A run config mapping.
        overrides: The ``key=value`` overrides about to be applied.
        base: The directory a relative judge path is read from first (the run config's); else the working
            directory.
    """
    judge = data.get("judge")
    if not isinstance(judge, str) or judge == "fake" or not any(o.startswith("judge.") for o in overrides):
        return data
    from rcp_ndcg.llm.judges import judge_config_path
    from rcp_ndcg.support.config import load_config

    path = Path(judge)
    if base is not None and not path.is_absolute() and (base / path).is_file():
        path = base / path
    return {**data, "judge": load_config(judge_config_path(path))}


__all__ = [
    "JUDGE_STEPS",
    "STEPS",
    "CalibrationOptions",
    "CandidatesConfig",
    "DatasetSource",
    "EvaluationOptions",
    "KubernetesRunnerConfig",
    "LocalRunnerConfig",
    "PluginRunnerConfig",
    "RunConfig",
    "RunnerConfig",
    "SlurmRunnerConfig",
    "StepName",
    "inline_judge",
]
