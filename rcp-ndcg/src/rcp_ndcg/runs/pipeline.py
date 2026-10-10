"""The pipeline: run a config's steps against one run directory.

Each step calls the library function that owns its work -- retrieval, the
reranker, :func:`rcp_ndcg.judging.judge`, :func:`rcp_ndcg.calibration.calibrate`,
the evaluation -- and writes into the layout of :mod:`rcp_ndcg.runs.layout`.
The manifest records every step's identity and the content hashes of what it
read and wrote, so resuming a run re-does exactly the steps whose identity or
inputs changed. The judging steps resume at window granularity on top: their
store is append-only, so a stopped judging step asks the judge only for the
windows that are missing.

The dataset is read through :func:`rcp_ndcg.data.load_dataset`. The judging
steps read each query's pool from ``candidates.parquet`` (a
:class:`~rcp_ndcg.data.Rankings` table): the retrieved
(:func:`rcp_ndcg.retrieval.retrieve`) or supplied rankings, the dataset's own
pools (written when the run starts), or, with a ``rerank`` step, the first-stage
pools rescored by :func:`rcp_ndcg.retrieval.rerank`. The first stage is then kept
in ``work/first_stage.parquet``, so the reranker always reads the pools it was
configured on.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from rcp_ndcg_core.records import TEXT_FORMATTING_VERSION

from rcp_ndcg.data import Dataset, Rankings
from rcp_ndcg.errors import ConfigError, DataError, IdentityError, MissingInputError
from rcp_ndcg.judging.client import Usage
from rcp_ndcg.judging.prompts import load_prompt, shipped_prompts_digest
from rcp_ndcg.runs.config import JUDGE_STEPS, RunConfig
from rcp_ndcg.runs.layout import RunLayout, new_run_id
from rcp_ndcg.runs.manifest import DatasetRef, RunManifest, RunStatus, StepStatus
from rcp_ndcg.storage import local_dir
from rcp_ndcg.storage.artifacts import ArtifactRef, artifact_ref
from rcp_ndcg.support.identity import hash_payload, identity_payload
from rcp_ndcg.support.logging import get_logger
from rcp_ndcg.support.paths import runs_dir as default_runs_dir
from rcp_ndcg.support.serve import ENGINES_ENV, EngineRole, EngineURLs, parse_engines_env
from rcp_ndcg.support.step_budget import StepBudget, step_budgeted

if TYPE_CHECKING:
    from rcp_ndcg.judging.schedule import Modality as ScheduleModality

logger = get_logger(__name__)


class Pipeline:
    """The steps of one run against one run directory.

    Args:
        config: The run's config.
        runs_dir: Where a new run's directory is created: ``<runs_dir>/<run_id>``; default
            :func:`rcp_ndcg.support.paths.runs_dir` (``$RCP_NDCG_RUNS_DIR``, else ``runs``).
        layout: The run directory of an existing run (then ``runs_dir`` is not used).
        manifest: The existing run's manifest.
        only: Run only these of the config's steps in this invocation (``run resume --only``); the config's
            ``steps`` stay as recorded.
        engines: The engines overlay the job already carries for this invocation (a phase's
            ``RCP_NDCG_ENGINES``, or ``run resume --engine``); ``None`` reads the environment variable.

    Raises:
        ConfigError: ``only`` names a step the config does not have, or the engines overlay names a role this
            run cannot point at its engine.
    """

    def __init__(
        self,
        config: RunConfig,
        *,
        runs_dir: str | None = None,
        layout: RunLayout | None = None,
        manifest: RunManifest | None = None,
        only: Sequence[str] | None = None,
        engines: Mapping[EngineRole, EngineURLs] | None = None,
    ):
        self.config = config
        # Set here, not only in _run_step: a resume whose identity check raises before the step starts (a
        # judge config file gone) reaches the failure handler, which records a judging step's usage -- without
        # this, the handler itself crashes with an AttributeError that masks the typed error.
        self._judge_usage: Usage | None = None
        self._engines = dict(engines) if engines is not None else _engines_overlay()
        if self._engines:
            _check_engines(config, self._engines)
        self.only = list(only) if only else None
        if self.only is not None:
            missing = [step for step in self.only if step not in config.steps]
            if missing:
                raise ConfigError(
                    f"{', '.join(missing)} is not a step of this run (its steps: {', '.join(config.ordered_steps)})",
                    hint="name one of the run's steps, or start a new run with the steps you need",
                )
        # Named but not created: --estimate and --dry-run must not leave an empty run behind.
        root = local_dir(runs_dir or default_runs_dir(), "the runs directory")
        self.layout = layout or RunLayout.at(root / new_run_id(config.label))
        self.manifest = manifest or RunManifest.new(self.layout.run_id, config=config.resolved())
        self._dataset: Dataset | None = None

    @property
    def dataset(self) -> Dataset:
        """The run's dataset, loaded once (queries and corpus are read on first access)."""
        if self._dataset is None:
            self._dataset = self.config.dataset.load()
            if self._dataset.subsets:
                raise ConfigError(
                    f"{self._dataset.name!r} is a suite; a run evaluates one dataset",
                    hint="name one subset: dataset: {uri: ..., subset: <name>}",
                )
        return self._dataset

    # ------------------------------------------------------------------
    # Entry points
    # ------------------------------------------------------------------

    @classmethod
    def resume(
        cls,
        run_dir: str | Path,
        *,
        overrides: list[str] | None = None,
        only: Sequence[str] | None = None,
        engines: Mapping[EngineRole, EngineURLs] | None = None,
    ) -> Pipeline:
        """Reopen a run directory with its recorded config (``key=value`` overrides applied).

        ``only`` runs just those steps in this invocation; the recorded ``steps`` stay (see :class:`Pipeline`).
        ``engines`` overlays the role configs at runtime (the engines a job started); ``None`` reads
        ``RCP_NDCG_ENGINES``.
        """
        layout = RunLayout.at(run_dir)
        manifest = RunManifest.load(layout)
        config = RunConfig.from_data(manifest.config, overrides=overrides or [])
        return cls(config, layout=layout, manifest=manifest, only=only, engines=engines)

    @property
    def steps(self) -> list[str]:
        """The steps this invocation runs, in run order: the config's, narrowed by ``only``."""
        return [step for step in self.config.ordered_steps if self.only is None or step in self.only]

    def estimate(self, *, resume: bool = True):
        """The judging steps' calls, tokens and wall time (:func:`rcp_ndcg.judging.estimate`); the judge is not called.

        Before a retrieval-sourced run has retrieved, each query's pool is assumed to hold ``candidates.depth``
        documents of the corpus (:meth:`_assumed_pools`), and the estimate's ``assumptions`` say so. Before a
        rankings-sourced run has read its rankings file, the pools are read from that file.

        Raises:
            IdentityError, ConfigError: what :meth:`run` would refuse (:meth:`preflight`).
        """
        from rcp_ndcg.judging.cost import estimate

        self.preflight(resume=resume)
        stages = [step for step in self.steps if step in JUDGE_STEPS and not (resume and self._is_current(step))]
        schedules = {name: schedule for name in stages if (schedule := self.schedule(name)) is not None}
        pools, assumed = self._planned_pools()
        projected = estimate(
            self.dataset,
            pools,
            self.config.judge_config(),
            stages=stages,  # type: ignore[arg-type]
            schedules=schedules,
            preprocessing=self.config.preprocessing,
        )
        notes = []
        if resume and len(stages) < len([step for step in self.steps if step in JUDGE_STEPS]):
            notes.append("judging steps already done with the same identity and inputs are skipped, as the run would")
        for stage in stages:
            stored = _windows_stored(Path(self.layout.judgements) / f"{stage}.jsonl")
            if stored:
                notes.append(
                    f"the {stage} store already holds {stored} judged windows: they are counted here, and the run "
                    "asks only for the windows its store lacks"
                )
        if assumed:
            notes.append(
                f"the retrieval has not run: each query's pool is assumed to be {self.config.candidates.depth} "
                "documents (candidates.depth, at most the corpus), their lengths those of an evenly spaced sample of "
                "the corpus"
            )
        return projected.model_copy(update={"assumptions": [*projected.assumptions, *notes]}) if notes else projected

    def preflight(self, *, resume: bool = True) -> None:
        """Every refusal :meth:`run` with the same ``resume`` would give before it judges; nothing is written.

        A mirror is checked for an installed filesystem, and each judging step that would run is checked as
        :func:`rcp_ndcg.judging.judging.preflight` checks a pass: its settings, and a store holding judgements
        of another identity. ``--dry-run`` and ``--estimate`` call it, so they refuse what the real command refuses.

        Raises:
            DependencyError: no installed filesystem serves the mirror's URI.
            IdentityError: a judging step's store holds judgements of another identity.
            ConfigError: settings the judging pass refuses.
        """
        from rcp_ndcg.judging.judging import preflight
        from rcp_ndcg.runs.mirror import check_target

        if self.config.mirror is not None:
            check_target(self.config.mirror)
        stages = [s for s in self.steps if s in JUDGE_STEPS and not (resume and self._is_current(s))]
        if not stages:
            return
        pools, _ = self._planned_pools()
        for stage in stages:
            with _run_identity_hint(self.layout, stage):
                preflight(
                    self.dataset,
                    pools,
                    self.config.judge_config(),
                    stage=stage,  # type: ignore[arg-type]
                    out=self.layout.judgements,
                    schedule=self.schedule(stage),
                    preprocessing=self.config.preprocessing,
                )

    def _dataset_identity(self) -> dict[str, Any]:
        """The dataset part of every step's identity: the source and its resolved commit (so a moved upstream
        re-runs), plus the TEXT formatting rule's version (:data:`TEXT_FORMATTING_VERSION`) -- so a resume
        never reuses candidates, judgements or scores built from other strings.

        The formatting's own inputs are in the step payloads: the title mode and the instruction policy are
        CONTENT fields of the retriever, the reranker and the judge configs (``identity_payload``), and the
        instruction a Hub source carries is part of the repository at the resolved commit. A resume check
        stays metadata-only -- it never loads the dataset (a ``mteb:`` prompt comes from mteb's own metadata,
        which no URI or commit pins: that is the code-version gap the retrieval review records as A5, and it
        is not this lane's).
        """
        payload = dict(self.config.dataset.identity())
        payload["text_formatting"] = TEXT_FORMATTING_VERSION
        return payload

    def _planned_pools(self) -> tuple[dict[str, list[str]], bool]:
        """The pools the judging steps will read, and whether they are assumed (a run that has not retrieved)."""
        retrieved = Path(self.layout.candidates).exists()
        if self.config.candidates.source == "retrieval" and not retrieved:
            return self._assumed_pools(), True
        if self.config.candidates.source == "rankings" and not retrieved:
            depth = self.config.candidates.depth
            return {query: pool[:depth] for query, pool in self._limited(self._supplied_pools()).items()}, False
        return self._judging_input(), False

    def _assumed_pools(self) -> dict[str, list[str]]:
        """Each query's pool as :meth:`estimate` assumes it before retrieval: ``candidates.depth`` documents.

        The same evenly spaced sample of the corpus (in id order) stands in for every query's retrieved pool: the
        number of calls depends on the pool size only, and the tokens on the documents' lengths.
        """
        dataset = self.dataset
        corpus = sorted(dataset.corpus)
        if not corpus:
            raise DataError(f"{dataset.name!r} has no corpus to retrieve from")
        depth = min(self.config.candidates.depth, len(corpus))
        sample = [corpus[i * len(corpus) // depth] for i in range(depth)]
        return self._limited({query: sample for query in dataset.queries})

    def schedule(self, stage: str):
        """The schedule a judging stage runs with: the configured one (seeded by the config: :class:`RunConfig`).

        ``None`` is the paper's schedule for the corpus's modality, whose seed is the run's default seed; with
        another run seed that schedule is resolved here, from the modality of the corpus, and given the run's seed.
        """
        from rcp_ndcg.judging import RubricSchedule, TournamentSchedule

        kind = TournamentSchedule if stage == "tournament" else RubricSchedule
        schedule = getattr(self.config, stage)
        if schedule is not None or self.config.seed == kind.model_fields["seed"].default:
            return schedule
        return kind.for_modality(self._corpus_modality()).model_copy(update={"seed": self.config.seed})

    def _corpus_modality(self) -> ScheduleModality:
        """``"video"``, ``"image"`` or ``"text"``: what the documents carry (the choice of the paper's schedule)."""
        from rcp_ndcg_core.content import Modality

        modalities = {document.as_content.modality for document in self.dataset.corpus.values()}
        if Modality.VIDEO in modalities:
            return "video"
        return "image" if modalities - {Modality.TEXT} else "text"

    def plan(self, *, resume: bool = True) -> list[dict[str, Any]]:
        """What :meth:`run` with the same ``resume`` would do, without doing it (``--dry-run``)."""
        rows = []
        for step in self.steps:
            current = resume and self._is_current(step)
            rows.append({"step": step, "status": "would skip" if current else "would run"})
        return rows

    def run(self, *, resume: bool = True) -> RunManifest:
        """Run the configured steps in order (those of ``only``, when given) and return the final manifest.

        A resume that changes the config (overrides) keeps the change only when it succeeds. When it fails, or is
        refused, the recorded config, ``run.yaml`` and status are restored,
        so the run resumes as it was without the change. The steps it re-ran before failing keep their new records,
        so the next resume redoes them under the recorded config. One exception: once a judging step has claimed its
        store under the changed config (a stage judged for the first time), that store holds judgements of the new
        config, so the new config stays and the run is ``failed``.

        The run is ``completed`` when every step of its config is done; after ``only`` left some undone, it is
        ``partial`` until a resume runs them.

        Raises:
            IdentityError: a judging step's store holds judgements of another identity (a resume with a changed
                judge, schedule, dataset or preprocessing). The run is left as it was (see above).
        """
        recorded = self.manifest.model_copy(deep=True)
        claims = self._store_claims()
        self.layout.ensure()
        self._write_config()
        self.manifest.status = RunStatus.RUNNING
        self.manifest.save(self.layout)
        if self.config.candidates.source == "dataset" and set(self.config.steps) - {"retrieve"}:
            # A reranker rescores the whole first stage; the depth cut comes after it.
            pools = self._dataset_pools()
            if "rerank" not in self.config.steps:
                pools = {query: pool[: self.config.candidates.depth] for query, pool in pools.items()}
            _write_rankings(self._first_stage, Rankings.from_orders(pools, system=CANDIDATES))
        ran: set[str] = set()
        for step in self.steps:
            try:
                if self._run_step(step, resume=resume):
                    ran.add(step)
            except IdentityError as exc:
                self._refused(ran, recorded)
                _identity_hint(exc, self.layout, step)
                raise
            except (Exception, KeyboardInterrupt) as exc:
                # An interruption (SIGINT, or SIGTERM from a scheduler) is recorded like a failure.
                usage = self._judge_usage if step in JUDGE_STEPS else None
                self.manifest.finish_step(step, status=StepStatus.FAILED, usage=usage, error=_describe(exc))
                if _substance(self.config) != _substance(RunConfig.from_data(recorded.config)) and (
                    self._store_claims() == claims
                ):
                    logger.warning("[run] %s failed; the run keeps its recorded config: %s", step, exc)
                    self._refused(ran, recorded, failed=step)
                else:
                    self.manifest.status = RunStatus.FAILED
                    if _cancelled_meanwhile(self.layout):  # `run cancel` stopped this job: the run stays cancelled
                        self.manifest.status = RunStatus.CANCELLED
                        self.manifest.finish_step(step, status=StepStatus.CANCELLED, usage=None, error=_describe(exc))
                    self.manifest.save(self.layout)
                raise
            self.manifest.save(self.layout)
        done = all((record := self.manifest.step(step)) is not None and record.succeeded for step in self.config.steps)
        self.manifest.status = RunStatus.COMPLETED if done else RunStatus.PARTIAL
        self.manifest.save(self.layout)
        return self.manifest

    def _store_claims(self) -> dict[str, Any]:
        """The judgement store's recorded identities (``{}`` before a judging step first claimed it)."""
        from rcp_ndcg.judging.store import JudgementStore

        return JudgementStore(self.layout.judgements).identities()

    def _refused(self, ran: set[str], recorded: RunManifest, *, failed: str | None = None) -> None:
        """Undo what a refused or failed change wrote: the recorded config and status, and the records of
        every step but those that ran (``ran``: their outputs were rewritten, and their identity now differs from
        the recorded config's, so the next resume redoes them) and the ``failed`` step, when the run had no record
        of it (its error stays visible). A run recorded ``submitted`` or ``running`` (a job's first attempt, or one
        that was stopped) is left ``failed``: no job runs it any more."""
        keep = ran | ({failed} if failed is not None and recorded.step(failed) is None else set())
        steps = []
        for record in self.manifest.steps:
            kept = record if record.name in keep else recorded.step(record.name)
            if kept is not None:
                steps.append(kept)
        self.manifest.steps = steps
        self.manifest.config = recorded.config
        pending = recorded.status in (RunStatus.SUBMITTED, RunStatus.RUNNING)
        self.manifest.status = RunStatus.FAILED if pending else recorded.status
        self.config = RunConfig.from_data(recorded.config)
        self._write_config()
        self.manifest.save(self.layout)

    # ------------------------------------------------------------------
    # Dispatch and resume
    # ------------------------------------------------------------------

    def _run_step(self, step: str, *, resume: bool) -> bool:
        """Run ``step`` unless it is current; return whether it ran."""
        if resume and self._is_current(step):
            # The record stays as it was: completed, with the time it took when it ran.
            logger.info("[run] %s: done with the same identity and inputs, skipping", step)
            return False
        logger.info("[run] %s: starting", step)
        self._judge_usage = None  # a judging step sets it, also when it fails
        if step == "evaluate":
            self.manifest.metrics = {}  # a failed re-run keeps no metrics of the attempt it did not finish
        self.manifest.start_step(step, identity=self._identity(step))
        self.manifest.save(self.layout)
        inputs = self._inputs(step)
        outputs, usage = self._budgeted_step(step)
        self.manifest.finish_step(step, inputs=inputs, outputs=outputs, usage=usage)
        return True

    def _budgeted_step(self, step: str) -> tuple[list[ArtifactRef], Usage | None]:
        """One step's work, under the run's wall-clock budget when it sets one (checked at the request seams)."""
        seconds = self.config.step_budget_s
        if seconds is None:
            return getattr(self, f"_step_{step}")()
        with step_budgeted(StepBudget(step, seconds)):
            return getattr(self, f"_step_{step}")()

    def _is_current(self, step: str) -> bool:
        """A recorded step is current when it succeeded with the same identity and inputs and its outputs exist."""
        record = self.manifest.step(step)
        if record is None or not record.succeeded or record.identity_hash != hash_payload(self._identity(step)):
            return False
        if {(r.path, r.sha256) for r in record.inputs} != {(r.path, r.sha256) for r in self._inputs(step)}:
            return False
        return all(Path(self.layout.resolve(ref.path)).exists() for ref in record.outputs)

    @property
    def _first_stage(self) -> str:
        """Where the first-stage pools live: ``candidates.parquet``, or beside it when a reranker reorders them."""
        if "rerank" in self.config.steps:
            return str(Path(self.layout.work) / "first_stage.parquet")
        return self.layout.candidates

    def _identity(self, step: str) -> dict[str, Any]:
        """Everything that decides a step's output (runtime knobs excluded)."""
        config = self.config
        dataset = {"dataset": self._dataset_identity()}
        common = {**dataset, "limit": config.limit, "seed": config.seed}
        if step == "retrieve":
            # Retrieval covers every query of the dataset and draws nothing at random: no limit, no seed. The
            # content payload only: a served encoder's URL, key variable, concurrency, timeouts and retries move
            # the work, not the numbers (the reranker keys the rerank step, not the retrieval). The encoder's
            # tokenizer digest (``identity_extra()``) is spliced in at the encoder: what cuts the text is
            # content, and the tokenizer's *name* is not.
            candidates = identity_payload(config.candidates)
            candidates.pop("rerank", None)
            retrieval = candidates.get("retrieval")
            retriever = config.candidates.retrieval
            if isinstance(retrieval, dict) and "encoder" in retrieval and retriever is not None:
                encoder = getattr(retriever, "encoder", None)
                assert encoder is not None
                retrieval["encoder"] = {
                    **retrieval["encoder"],
                    **encoder.identity_extra(),
                }
            from rcp_ndcg.retrieval import RETRIEVE_BEHAVIOUR_VERSION

            # The step's behaviour version: a change to what the first stage computes that moves no config
            # field still re-runs it (the index's own version covers a cached index build).
            return {
                **dataset,
                "candidates": candidates,
                "output": self.layout.relative(self._first_stage),
                "behaviour_version": RETRIEVE_BEHAVIOUR_VERSION,
            }
        if step == "rerank":
            reranker = config.candidates.rerank
            if reranker is None:
                rerank = None
            else:
                # The reranker's content payload plus its tokenizer's SHA-256 (the name itself is runtime).
                rerank = {**identity_payload(reranker), **reranker.identity_extra()}
            from rcp_ndcg.retrieval import RERANK_BEHAVIOUR_VERSION

            return {
                **common,
                "rerank": rerank,
                "depth": config.candidates.depth,
                "behaviour_version": RERANK_BEHAVIOUR_VERSION,
            }
        if step in JUDGE_STEPS:
            schedule = self.schedule(step)
            judge = config.judge_config()
            # The prompt the step resolves to, by its content (the family's prompt_hash): its name or path is
            # runtime -- the same text under another name is the same instrument, edited text is not. A schedule
            # that leaves the prompt unset resolves the shipped one from the corpus's modality at judging time,
            # so the step pins the stage's whole shipped set by content instead (the dataset's own identity,
            # in `common`, carries its instruction and the formatting version -- the reader's instruction
            # lookup may read the card or the queries, never the corpus for a Hub source).
            if schedule is not None and schedule.prompt:
                prompt_sha256: str = load_prompt(schedule.prompt).sha256
            else:
                # step is one of JUDGE_STEPS here ('tournament' or 'rubric'); the set is typed str.
                stage = cast(Literal["tournament", "rubric"], step)
                prompt_sha256 = shipped_prompts_digest(stage)
            return {
                **common,
                "depth": config.candidates.depth,
                "judge": {**judge.identity(), **judge.identity_extra()},
                "schedule": schedule.model_dump(mode="json", exclude={"prompt"}) if schedule is not None else None,
                "prompt_sha256": prompt_sha256,
                "preprocessing": config.preprocessing.model_dump(mode="json") if config.preprocessing else None,
            }
        if step == "calibrate":
            return config.calibration.model_dump(mode="json")
        if step == "evaluate":
            # The seed draws the bootstrap intervals; the dataset's qrels give qrel-nDCG; limit and depth cut the pools.
            return {**common, "depth": config.candidates.depth, "evaluation": config.evaluation.model_dump(mode="json")}
        raise ConfigError(f"unknown step {step!r}")

    def _inputs(self, step: str) -> list[ArtifactRef]:
        layout = self.layout
        paths: list[str] = []
        if step == "retrieve" and self.config.candidates.source == "rankings":
            paths = [str(self.config.candidates.rankings)]
        elif step == "rerank":
            if self.config.candidates.source == "rankings" and "retrieve" not in self.config.steps:
                # The rerank step reads the rankings file itself on this shape (see _step_rerank).
                paths = [str(self.config.candidates.rankings)]
            else:
                paths = [self._first_stage]
        elif step in ("tournament", "rubric"):
            paths = [self._pools_source()]
        elif step == "calibrate":
            paths = [layout.path("judgements", f"{stage}.jsonl") for stage in ("tournament", "rubric")]
        elif step == "evaluate":
            paths = [
                self._pools_source(),
                layout.path("calibration", "items.json"),
                layout.path("calibration", "thetas.parquet"),
                *(location.partition("#")[0] for location in self.config.evaluation.systems.values()),
            ]
        return [artifact_ref(path, layout=layout) for path in paths if Path(path).exists()]

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def _step_retrieve(self) -> tuple[list[ArtifactRef], Usage | None]:
        candidates = self.config.candidates
        output = self._first_stage
        if candidates.source == "rankings":
            first = Rankings.from_orders(self._supplied_pools(), system=CANDIDATES)
        else:
            from rcp_ndcg.retrieval import retrieve

            assert candidates.retrieval is not None
            retrieved = retrieve(
                self.dataset,
                self._overlaid_retrieval(),
                depth=candidates.depth,
                out=str(Path(self.layout.work) / "index"),
            )
            first = Rankings.from_scores(retrieved.queries(), system=CANDIDATES)
        _write_rankings(output, first)
        return [artifact_ref(output, layout=self.layout)], None

    def _supplied_pools(self) -> dict[str, list[str]]:
        """Each query's pool, best first, from the rankings file of a ``from: rankings`` run (its one system)."""
        from rcp_ndcg.data import load_rankings

        path = str(self.config.candidates.rankings)
        supplied = load_rankings(path)
        system = self.config.candidates.system or _only_system(supplied, path)
        pools = supplied.queries(system=system, dataset=self.dataset.name)
        return {query: _order(scores) for query, scores in pools.items()}

    def _step_rerank(self) -> tuple[list[ArtifactRef], Usage | None]:
        from rcp_ndcg.retrieval import rerank

        reranker = self.config.candidates.rerank
        assert reranker is not None
        if self.config.candidates.source == "rankings" and "retrieve" not in self.config.steps:
            # A `from: rankings` run without a retrieve step: the rankings file IS the first stage -- always
            # read from it, never from a work/first_stage.parquet an earlier config in this run dir left.
            first = Rankings.from_orders(self._supplied_pools(), system=CANDIDATES)
        else:
            if not Path(self._first_stage).exists() and "retrieve" in self.config.steps:
                # `run resume --only rerank` after a restore that left no work/ (the mirror skips it): the
                # configured retrieve step regenerates the first stage, instead of dying with "rankings file
                # not found" and wedging the documented --only path.
                logger.info("[run] rerank: the first stage is missing; regenerating it with the retrieve step")
                self._step_retrieve()
            first = _read_rankings(self._first_stage)
        pools = self._limited(first.queries())
        depth = max((len(pool) for pool in pools.values()), default=1)
        rescored = rerank(
            self.dataset,
            Rankings.from_scores({query: first.for_query(query) for query in pools}, system=CANDIDATES),
            self._overlaid_reranker(),
            depth=depth,
            out=str(Path(self.layout.work) / "rerank"),
        )
        # The whole first stage in the reranker's order; the judging steps take its best ``depth``.
        _write_rankings(self.layout.candidates, Rankings.from_scores(rescored.queries(), system=CANDIDATES))
        return [artifact_ref(self.layout.candidates, layout=self.layout)], None

    def _step_tournament(self) -> tuple[list[ArtifactRef], Usage | None]:
        return self._judge("tournament")

    def _step_rubric(self) -> tuple[list[ArtifactRef], Usage | None]:
        return self._judge("rubric")

    def _overlaid(self, endpoint: Any, role: EngineRole) -> Any:
        """``endpoint`` with the role's engine URLs (and outage wait) applied, when the overlay names the role.

        The overlay is runtime only: it is never written into ``run.yaml`` and never reaches an identity, since
        ``base_url`` and ``wait_on_outage_s`` are RUNTIME fields. This is where the runners resolve an engine
        onto a config, so a config whose ``api`` selects an adapter of a different engine role is refused here
        (through the one written mapping in :func:`rcp_ndcg.inference.adapters.base.check_engine_api`). The config
        is then rebuilt through its model, so the overlaid config passes every validator a configured one does (a
        URL is normalised, a fake:// replica list is refused), with the typed error for what a configured config
        would refuse.
        """
        from pydantic import ValidationError

        from rcp_ndcg.inference.adapters.base import check_engine_api
        from rcp_ndcg.support.config import config_error

        engines = self._engines.get(role)
        if engines is None:
            return endpoint
        check_engine_api(getattr(endpoint, "api", None), engine_role=role, where=f"the {role} engine overlay")
        update: dict[str, Any] = {"base_url": engines.urls[0] if len(engines.urls) == 1 else list(engines.urls)}
        if engines.wait_on_outage_s is not None:
            update["wait_on_outage_s"] = engines.wait_on_outage_s
        try:
            return type(endpoint).model_validate({**endpoint.model_dump(), **update})
        except ValidationError as exc:
            raise config_error(exc, model=type(endpoint), source=f"the {ENGINES_ENV} overlay") from exc

    def _judge_client_config(self) -> Any:
        """The judge config the client is built from: the engines overlay applied (runtime only)."""
        return self._overlaid(self.config.judge_config(), "judge")

    def _overlaid_retrieval(self) -> Any:
        """The retriever the retrieve step calls: its served encoder with the engine's URL, when overlaid."""
        from rcp_ndcg.retrieval.config import DenseConfig, LateInteractionConfig

        retrieval = self.config.candidates.retrieval
        assert retrieval is not None
        engines = self._engines.get("encoder")
        if engines is None or not isinstance(retrieval, (DenseConfig, LateInteractionConfig)):
            return retrieval
        return retrieval.model_copy(update={"encoder": self._overlaid(retrieval.encoder, "encoder")})

    def _overlaid_reranker(self) -> Any:
        """The reranker the rerank step calls, with the engine's URL, when overlaid."""
        reranker = self.config.candidates.rerank
        assert reranker is not None
        if "reranker" not in self._engines:
            return reranker
        return self._overlaid(reranker, "reranker")

    def _judge(self, stage: str) -> tuple[list[ArtifactRef], Usage | None]:
        from rcp_ndcg.judging.client import JudgeClient
        from rcp_ndcg.judging.judging import judge
        from rcp_ndcg.judging.store import JudgementStore

        client = JudgeClient.from_config(self._judge_client_config())
        try:
            judge(
                self.dataset,
                self._judging_input(),
                client,
                stage=stage,  # type: ignore[arg-type]
                out=self.layout.judgements,
                schedule=self.schedule(stage),
                preprocessing=self.config.preprocessing,
            )
        finally:
            self._judge_usage = client.usage  # a failed step records what it asked too
            record = self.manifest.step(stage)
            if record is not None:
                record.engines = client.engines
        for entry in JudgementStore(self.layout.judgements).identities().values():
            from rcp_ndcg_core.schemas import Family

            family = Family.model_validate(entry["family"])
            self.manifest.families[family.key] = family
        store_file = self.layout.path("judgements", f"{stage}.jsonl")
        return [artifact_ref(store_file, layout=self.layout)], client.usage

    def _step_calibrate(self) -> tuple[list[ArtifactRef], Usage | None]:
        from rcp_ndcg.calibration import calibrate, judged_bt_l2, read_judgements

        options = self.config.calibration
        calibration = calibrate(
            read_judgements(self.layout.judgements),
            mode=options.mode,
            judges=options.judges,
            priors=options.priors,
            judged_bt_l2=judged_bt_l2(self.layout.judgements),
        )
        calibration.save(self.layout.calibration)
        return [
            artifact_ref(self.layout.path("calibration", name), layout=self.layout)
            for name in ("items.json", "queries.parquet", "thetas.parquet")
        ], None

    def _step_evaluate(self) -> tuple[list[ArtifactRef], Usage | None]:
        from rcp_ndcg.eval import compare

        report = self.evaluation_report()
        Path(self.layout.metrics).write_text(report.to_json(indent=2), encoding="utf-8")
        outputs = [artifact_ref(self.layout.metrics, layout=self.layout)]
        if len(report.systems) > 1:
            comparison = compare(report, baseline="candidates", seed=self.config.seed)
            Path(self.layout.comparison).write_text(comparison.to_json(indent=2), encoding="utf-8")
            outputs.append(artifact_ref(self.layout.comparison, layout=self.layout))
        # The judge's own order scores RCP-nDCG 1 by construction: it is in the report, not in the headline.
        self.manifest.metrics = {
            f"{row.system}/{row.metric}@{row.k}": row.value
            for row in report.summary
            if row.system != JUDGE and row.value is not None
        }
        return outputs, None

    def evaluation_report(self):
        """The run's :class:`~rcp_ndcg.eval.EvalReport`: every system scored with the run's calibration.

        The systems are the candidate order (``candidates``), the judge's calibrated abilities (``judge``) and
        the configured ``evaluation.systems``, on the queries the calibration covers. qrel-nDCG is added when
        the dataset has labels for them.
        """
        from rcp_ndcg.calibration import Calibration
        from rcp_ndcg.eval import evaluate

        calibration = Calibration.load(self.layout.calibration)
        name = self.dataset.name
        gains = calibration.gains(name)
        thetas = calibration.theta_map(dataset=name)
        pools = {query: pool for query, pool in self._judging_input().items() if query in gains}
        parts = [Rankings.from_orders(pools, system=CANDIDATES)]
        names = {CANDIDATES: "the run's candidate order", JUDGE: "the judge's abilities"}
        for key, location in self.config.evaluation.systems.items():
            for system, queries in _evaluation_systems(key, location, name).items():
                if system in names:
                    raise DataError(
                        f"evaluation.systems.{key}: system {system!r} is already scored ({names[system]})",
                        hint="rename the system, or pick one system of the file with <file>#<system>",
                    )
                names[system] = location
                kept = {q: docs for q, docs in queries.items() if q in gains}
                parts.append(Rankings.from_scores(kept, system=system))
        parts.append(Rankings.from_scores(thetas, system=JUDGE))
        qrels = {query: dict(self.dataset.qrels[query]) for query in pools if self.dataset.qrels.get(query)}
        dataset = Dataset(name=name, qrels=qrels, candidates=pools)
        return evaluate(
            Rankings.concat(parts),
            dataset=dataset,
            gains=gains,
            protocol="plain",
            k=self.config.evaluation.k,
            metrics=("rcp_ndcg", "qrel_ndcg") if qrels else ("rcp_ndcg",),
            seed=self.config.seed,
        )

    # ------------------------------------------------------------------
    # Inputs of the judging steps
    # ------------------------------------------------------------------

    def _dataset_pools(self) -> dict[str, list[str]]:
        """The dataset's own pools: its candidates, else each query's judged documents (``from: dataset``)."""
        dataset = self.dataset
        if dataset.candidates is not None:
            return self._limited({query: list(pool) for query, pool in dataset.candidates.items()})
        return self._limited({query: list(docs) for query, docs in dataset.qrels.items() if docs})

    def _limited(self, pools: dict[str, Any]) -> dict[str, Any]:
        """The first ``limit`` queries of ``pools`` (all without a limit)."""
        if self.config.limit is None:
            return pools
        return dict(list(pools.items())[: self.config.limit])

    def _pools_source(self) -> str:
        """The file the steps that read the first-stage pools consume: the candidates file a rerank (or
        retrieve) step writes -- or, when no configured step writes one, a `from: rankings` run's rankings
        file itself (the dataset source's preamble writes the first stage)."""
        if (
            self.config.candidates.source == "rankings"
            and "retrieve" not in self.config.steps
            and "rerank" not in self.config.steps
        ):
            return str(self.config.candidates.rankings)
        return self.layout.candidates

    def _judging_input(self) -> dict[str, list[str]]:
        """Each query's pool, best first, at the configured depth: what the judging steps judge.

        Raises:
            MissingInputError: the candidates come from retrieval or rankings and the step that writes them
                (the retrieve step, or the rerank step when one is configured) has not run.
        """
        path = self._pools_source()
        if self.config.candidates.source == "rankings" and path != self.layout.candidates:
            # The rankings file IS the first stage here: read straight from it, never a leftover file's.
            pools = self._supplied_pools()
        elif not Path(path).exists():
            if self.config.candidates.source != "dataset":
                writer = "rerank" if "rerank" in self.config.steps else "retrieve"
                raise MissingInputError(
                    f"{path} does not exist yet: the candidates come from {self.config.candidates.source}",
                    hint=f"run the {writer} step first",
                )
            pools = self._dataset_pools()
        else:
            pools = {query: _order(scores) for query, scores in _read_rankings(path).queries().items()}
        depth = self.config.candidates.depth
        return {query: pool[:depth] for query, pool in self._limited(pools).items()}

    def _write_config(self) -> None:
        import yaml

        resolved = self.config.resolved()
        self.manifest.config = resolved
        Path(self.layout.config).write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
        if self.manifest.dataset is None:
            resolved = self.config.dataset.identity().get("resolved")
            self.manifest.dataset = DatasetRef(
                name=self.dataset.name,
                revisions={self.config.dataset.uri: resolved} if resolved else None,
                subset=self.dataset.subset,
                split=self.dataset.split,
                task=self.dataset.task,
            )


def _substance(config: RunConfig) -> dict[str, Any]:
    """The config without its runtime-only fields, which never make a resume a config change: the mirror, the
    step budget, and the runtime fields the job's engines carry (the judge's URLs and outage wait, the served
    encoder's and reranker's URLs, keys, concurrency, timeouts and batch sizes)."""
    from rcp_ndcg.support.identity import FieldRole, declared_roles

    data = config.resolved()
    data.pop("mirror", None)
    data.pop("mirror_interval_s", None)
    data.pop("step_budget_s", None)
    data["candidates"] = identity_payload(config.candidates)
    if config.judge is not None:
        try:
            judge = config.judge_config()
        except MissingInputError:  # a judge file that is gone: compare what the config names
            return data
        runtime = {name for name, role in declared_roles(type(judge)).items() if role is FieldRole.RUNTIME}
        data["judge"] = judge.model_dump(mode="json", exclude=runtime)
    return data


def _engines_overlay() -> dict[EngineRole, EngineURLs]:
    """The engines ``RCP_NDCG_ENGINES`` carries for this invocation (unchecked; the caller checks them)."""
    text = os.environ.get(ENGINES_ENV)
    if not text:
        return {}
    return parse_engines_env(text)


def _check_engines(config: RunConfig, engines: Mapping[EngineRole, EngineURLs]) -> None:
    """Every role an engines overlay names must have a config the run can point at its engine.

    The overlay is applied to the role configs at runtime only (:meth:`Pipeline._overlaid`); it names a phase's
    engines, which the runner started for this run's ``serve:`` (validated there), or the roles ``run resume
    --engine`` spells.

    Raises:
        ConfigError: the overlay names a role this run has no served config for (the offline fake judge, a BM25
            or hosted encoder, no reranker), or gives a retrieval role more than one replica URL (this release's
            retrieval clients address one URL).
    """
    from rcp_ndcg.retrieval.config import ServedEmbedding, ServedPooling, ServedReranker

    for role, engine in engines.items():
        if role == "judge":
            if config.judge is None:
                raise ConfigError(
                    "the engines overlay names the judge, and this run has none",
                    hint="drop the judge role, or set judge in the run config",
                )
            if config.judge == "fake" or config.judge_config().is_fake:
                raise ConfigError(
                    "the engines overlay names the judge, and this run's judge is the offline fake",
                    hint="the fake judge is answered in process: drop the judge role, or set judge to a real "
                    "endpoint's config",
                )
        elif role == "encoder":
            encoder = getattr(config.candidates.retrieval, "encoder", None)
            if not isinstance(encoder, ServedEmbedding | ServedPooling):
                raise ConfigError(
                    "the engines overlay names the encoder, and this run has no served encoder to point at it",
                    hint="the encoder role serves a run's openai_embeddings or vllm_pooling encoder; drop the "
                    "role, or start its engine with serve.encoder",
                )
        else:
            if not isinstance(config.candidates.rerank, ServedReranker):
                raise ConfigError(
                    "the engines overlay names the reranker, and this run has no served reranker to point at it",
                    hint="the reranker role serves a run's api: rerank reranker; drop the role, or set "
                    "candidates.rerank to one",
                )
        if role != "judge" and len(engine.urls) != 1:
            raise ConfigError(
                f"the engines overlay gives the {role} {len(engine.urls)} replica URLs, and the retrieval client "
                "addresses one URL",
                hint="run one replica (replicas: 1) for a served encoder or reranker",
            )


def _describe(exc: BaseException) -> str:
    """A step's recorded error."""
    if isinstance(exc, KeyboardInterrupt):
        return "Interrupted: interrupted (SIGINT or SIGTERM)"
    return f"{type(exc).__name__}: {exc}"


def _cancelled_meanwhile(layout: RunLayout) -> bool:
    """Whether ``run cancel`` recorded the run as cancelled while this process ran it."""
    return Path(layout.manifest).exists() and RunManifest.load(layout).status is RunStatus.CANCELLED


def _identity_hint(exc: IdentityError, layout: RunLayout, step: str) -> None:
    """Give a run's refusal of a judging store of another identity the run's own way out."""
    store = layout.path("judgements", f"{step}.jsonl")
    exc.hint = exc.cli_hint = (
        f"start a new run with the changed config (`rcp-ndcg run start`), or move {store} aside to "
        f"judge the {step} step again in this run"
    )


@contextmanager
def _run_identity_hint(layout: RunLayout, step: str) -> Iterator[None]:
    try:
        yield
    except IdentityError as exc:
        _identity_hint(exc, layout, step)
        raise


def _windows_stored(store: Path) -> int:
    """The judged windows a judgement store file holds (its non-empty lines)."""
    from rcp_ndcg.judging.store import records_stored

    return records_stored(store)


#: The system name of a run's candidate pools in its rankings files.
CANDIDATES = "candidates"
#: The system name of the judge's calibrated abilities in a run's evaluation.
JUDGE = "judge"
#: The reference systems every run scores beside the user's own: the pool order and the judge's abilities.
REFERENCE_SYSTEMS: tuple[str, str] = (CANDIDATES, JUDGE)


def _order(scores: dict[str, float]) -> list[str]:
    """Document ids best first (score descending, then the lower document id: the retrieval stack's one tie
    rule, the same one ``Rankings.top`` and the first stage's cut apply)."""
    return sorted(scores, key=lambda doc: (-scores[doc], doc))


def _only_system(rankings: Rankings, where: str) -> str:
    if len(rankings.systems) != 1:
        raise DataError(
            f"{where} holds {len(rankings.systems)} systems {rankings.systems}; name the one to use",
            hint="set candidates.system",
        )
    return rankings.systems[0]


def _evaluation_systems(key: str, location: str, dataset: str) -> dict[str, dict[str, dict[str, float]]]:
    """``{system name: {query_id: {doc_id: score}}}`` of one ``evaluation.systems`` entry, on the run's ``dataset``.

    A file of one system is named ``key``; ``<file>#<system>`` picks that system, named ``key``; a file of several
    systems without ``#`` gives each under its own name.
    """
    from rcp_ndcg.data import load_rankings

    path, _, system = location.partition("#")
    supplied = load_rankings(path)
    if system:
        if system not in supplied.systems:
            raise DataError(
                f"evaluation.systems.{key}: {path} holds no system {system!r}; it holds {supplied.systems}",
                hint=f"write {path}#<one of {supplied.systems}>",
            )
        return {key: supplied.queries(system=system, dataset=dataset)}
    if len(supplied.systems) == 1:
        return {key: supplied.queries(dataset=dataset)}
    return {name: supplied.queries(system=name, dataset=dataset) for name in supplied.systems}


def _write_rankings(path: str, rankings: Rankings) -> None:
    rankings.save(path, format="parquet")


def _read_rankings(path: str) -> Rankings:
    from rcp_ndcg.data import load_rankings

    return load_rankings(path, format="parquet")


__all__ = ["Pipeline"]
