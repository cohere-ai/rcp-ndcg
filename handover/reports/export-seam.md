# Lane `export-seam`: the public results-export seam

## 1. Status

**DONE.** The public results-export seam ships in 0.0.1: the versioned `rcp-ndcg.result-record.v1` record
(one row per system x dataset x metric x cutoff), the `rcp_ndcg.results` entry-point group with the built-in
`jsonl`/`parquet`/`null` sinks and the shared conformance check, the `rcp-ndcg results` command group, the
exported JSON Schema, the how-to and the compatibility page, the contract snapshots and the CHANGELOG. The
private consumer's adapter is out of this lane, as the brief says.

## 2. Commits

| Commit | Subject |
|---|---|
| `19490a4a` | results: the rcp-ndcg.result-record.v1 record and the sink seam |
| `3c8d75c5` | results: the rcp-ndcg results command group and the declared sinks |
| `109c0922` | contract: the result-record schema, the CLI tree and the packaging snapshots |
| `d5cf5542` | docs: the results-export how-to, the compatibility contract and the CHANGELOG entry |
| `14c6eebb` | contract: rcp_ndcg.results is a pinned public module |
| `442a0786` | results: the verifier round-1 findings |
| `3890c9f8` | contract: the DatasetRef and ReportInputs provenance fields |
| `04b879da` | docs: the record's provenance sentence and the results layer in the charter |
| `fbbc6d1e` | results: the verifier round-2 findings |
| `9bae7a49` | Merge branch 'rfc-0001' into lane/export-seam (merged `c5ccc851`) |
| `6510affa` | Merge branch 'rfc-0001' into lane/export-seam (merged `75a341d9`, the gated tip) |

The lane base is `b67699c0` (the rfc-0001 tip at lane start).

## 3. What changed

* **The record** (`rcp-ndcg/src/rcp_ndcg/results.py`, public as `rcp_ndcg.results`, re-exported by the
  facade): frozen pydantic models `ResultSubject`, `ResultDataset`, `ResultMetric`, `ResultArtifact`,
  `ResultRecord` (`rcp-ndcg.result-record.v1`, `schema` alias). `record_identity(subject, dataset, metrics)`
  is the deterministic digest of the spec's formula. `ResultDataset.protocol` is the spec's preset name and
  the additive `protocol_spec` carries the full `rcp_ndcg_core.protocol.Protocol` (qrel gain, tie rule, pool
  restriction, rounding), so a run scored under another convention can be stated and two records differing
  only in protocol never compare equal. The JSON Schema is exported as `schemas/result-record.v1.json`.
* **The seam**: `ResultsSink` (name, `uri` first, abstract `emit`, `flush`), the built-ins `JsonlResultSink`
  (one record per line, replace by default, `options={"append": "1"}` appends), `ParquetResultSink` (one row
  per metric row, flat columns, nested fields as JSON) and `NullResultSink`; the `rcp_ndcg.results`
  entry-point group (declared in `rcp-ndcg/pyproject.toml`); `result_sink_class`/`registered_result_sinks`
  with the readers' `ConfigError` pattern; `rcp_ndcg.testing.results_conformance` (name, `uri` first and not
  positional-only, emit without mutation, idempotent flush, unknown schema refused).
* **The builders**: `records_from_report(report, ...)` and `records_from_run(run_dir, ...)`, one record per
  row. A run's records carry `subject.run_id`, a per-system `subject.identity` (the rankings digest, or the
  judge's family-key digest), run-relative `artifacts` with SHA-256 hashes, the dataset name and resolved Hub
  revision, and `provenance` with the code version and the resolved-config digest. The reference systems
  (`candidates`, `judge`) are excluded unless `include_reference` or named.
* **The CLI** (`rcp-ndcg/src/rcp_ndcg/cli/results.py`): `rcp-ndcg results sinks` lists the registry;
  `rcp-ndcg results export --run DIR [--report FILE] --sink NAME [--out URI] [--system NAME ...]
  [--include-reference]` builds records and writes them through the sink. The brief's `eval export --to` was
  superseded by the spec's `results export --sink/--out` (the brief says the spec wins).
* **Provenance the record states**: the manifest's `DatasetRef` gains `subset`, `split` and `task` (recorded
  by the pipeline from the loaded `Dataset`), and `ReportInputs` gains `split`/`task` (filled by
  `eval score --out`), so an exported record states the real provenance instead of the `test` default.
  `REFERENCE_SYSTEMS` has one home in `rcp_ndcg.runs.pipeline`, re-exported by the CLI and the results module.
* **Docs**: `docs/how-to/export-results.md` (Python and CLI, the record fields, the sink plugin seam) and
  `docs/reference/results-record.md` (the compatibility contract: versioning, fields, protocol/comparability,
  the sink contract), plus `docs/reference/cli.md`, `docs/api/evaluate.md`, the skill bullet and the
  `mkdocs.yml` navigation.
* **Contract**: `schemas/result-record.v1.json`, `schemas/results-export.v1.json`,
  `schemas/results-sinks.v1.json`, the regenerated `run-manifest.v1.json`/`eval-report.v1.json`/
  `run-summary.v1.json`, and the contract snapshots; `rcp_ndcg.results` in `PUBLIC_MODULES`; the CHANGELOG
  entry under `### Public surface`.

## 4. Verification

Two adversarial verifiers, both on DeepSeek-V4.1-flash `:xhigh`, fresh context, neither shown the other's
output (the harness allows one subagent call per turn, so they ran sequentially). The worktree's installed
metadata predates the new entry-point group, so the registry tests inject the declared entries
(`tests/results/_sinks.py`) and the packaging snapshot pins the `pyproject.toml` declaration; the gate
rebuilds the environment from the changed manifest and exercises the real registration.

* **Lens A (correctness against the brief and the spec): PASS**, 10 minor findings, all fixed:
  1. the empty-report guard fired even with `include_reference=True` -- now returns no records, with the
     "only reference systems" error only when the caller did not ask for them;
  2. `split`/`task` were the spec defaults -- the manifest now records them and the record states them;
  3. the `dataset=` docstring said "unchanged" while the report's protocol overwrites it -- reworded;
  4. the conformance accepted a positional-only `uri` the CLI cannot pass -- now refused, with a test;
  5. the parquet sink crashed on a datetime in `provenance` -- it now serialises through
     `model_dump(mode="json")`, as the JSONL sink does;
  6. the resolved-commit path had no test -- added (a copied manifest with a fabricated commit);
  7. a `--report` artifact stored an absolute path -- it now stores the path as given;
  8. `ResultMetric.value` accepted NaN (unhashable in `record_identity`, invalid JSON) -- `allow_inf_nan=False`
     on `value`/`ci_low`/`ci_high`, with a test;
  9. the layering charter did not name `results` -- `AGENTS.md` and the test's comments updated;
  10. JSONL appended while parquet overwrote -- JSONL now replaces by default and appends only with the
      explicit option.
* **Lens B (regressions and hygiene): PASS**, 4 minor findings, all fixed:
  1. the two provenance writers had no failing test -- assertions added to `tests/runs/test_pipeline.py` and
     `tests/cli/test_eval.py` (the verifier showed the old suite passed with the writes deleted);
  2. `results export --report` put a raw URI in `dataset.name` -- it now prefers the report's per-dataset row
     name and falls back to the suite/dataset URI for a multi-dataset summary, with a suite test;
  3. the test file's charter comments still said "just above runs" -- fixed;
  4. the CHANGELOG said "the recipe or model identity" unconditionally -- reworded to the judge's and the
     candidates' identity, which is what the builder labels.
* No blocker or major finding was reported in either lens, so no confirmation round was run (the lane rules
  require one only after a blocker or a major).
* Verifier mutations (in their scratch directories, no tracked file touched): a dataset-blind
  `record_identity` made the protocol-difference test red; a schema-swallowing sink made the conformance red;
  a defaulted `_run_dataset` made the provenance test red; a fake `Literal` schema tag with no `SchemaEntry`
  made the schema walker red.

Red-first evidence: before the implementation, `pytest tests/results tests/cli/test_results.py` failed at
collection with five errors, `ModuleNotFoundError: No module named 'rcp_ndcg.cli.results'` (and the same for
`rcp_ndcg.results`); every fix in the two verifier rounds was made on a failing test first.

## 5. Checks

Last run on the gated tip `6510affa` (the gate's own environment):

```
GATE: PASS
ruff-check exit=0 All checks passed!
ruff-format exit=0 571 files already formatted
basedpyright exit=0 0 errors, 0 warnings, 0 notes
pytest exit=0 3509 passed, 99 skipped
contract-docs exit=0 304 passed, 55 skipped
mkdocs exit=0 Documentation built
test-pkg exit=0 570 passed, 225 skipped
recipes exit=0 recipes: no failure outside the baseline (34 baseline failures remain, 0 fixed)
vllm-pkg exit=0 9 passed
vllm-models exit=0 71 passed, 7 skipped
run_all exit=0 leaderboards 1022 checks, 987 match, 35 known deviations, 0 failed; human study 67/67;
                 external LLM judges 82/82
public-names exit=0 public-names: clean (2 baselined hits remain)
clean exit=0 clean
```

The lane's own runs (before the gate) were `heavy uv run --no-sync pytest tests/ -q -n 4
-p no:cacheprovider -o faulthandler_timeout=120` (3509 passed, 99 skipped), `pytest tests/contract tests/docs`
(304 passed, 55 skipped) and `pytest rcp-ndcg-test/tests` (570 passed, 225 skipped).

## 6. Open questions

* The spec's `ResultDataset.protocol: str | None` is kept exactly; the full convention lives in the additive
  `protocol_spec`. A consumer that only wants the preset name can ignore the new field; a consumer that wants
  the ideal explicitly reads it from `metric` + `protocol_spec.restrict_to_candidates` (documented on the
  compatibility page).
* The brief's `rcp-ndcg eval export --to <name|path>` was not implemented; the spec's
  `rcp-ndcg results export --sink <name> --out <uri>` is (the brief says the spec wins). If the owner wants
  the `eval` spelling too, it should be an alias, not a second home.
* The JSONL sink replaces by default and appends only with `options={"append": "1"}`; the parquet sink
  overwrites and buffers until `flush`. If an export is expected to stream millions of rows, the parquet sink
  would need an incremental writer.
* A `--report`-only export has no run to be relative to, so its report artifact keeps the path as given
  (documented); the record's dataset provenance comes from the report's `inputs`.
* `results sinks` and `results export` are not MCP tools (the MCP tool list is static). `results sinks` is
  read-only and would be a natural addition in a later lane.
* The lane environment predates the entry-point group; the gate rebuilds it from the changed manifest. A bare
  `uv run --no-sync` against an un-rebuilt environment has an empty registry, which is an environment state,
  not a code path.

## CHANGELOG entry

```markdown
- **The results-export seam** (owner decision 40): a versioned `rcp-ndcg.result-record.v1` record
  (`rcp_ndcg.results`: `ResultRecord`, `ResultSubject`, `ResultDataset`, `ResultMetric`, `ResultArtifact`),
  one row per system x dataset x metric x cutoff, carrying the run identity, the dataset revision, the recipe
  or model identity the run names (the judge's and the candidates') and the scoring protocol --
  `dataset.protocol` is the preset name and
  `dataset.protocol_spec` the full `Protocol` (qrel gain, tie rule, pool restriction, rounding), so an
  importer can state another convention and two records differing only in protocol never compare equal
  (`record_id` digests the protocol). The record's JSON Schema is exported as
  `schemas/result-record.v1.json`. Sinks are the `rcp_ndcg.results` entry-point group (the same seam as
  `rcp_ndcg.readers`/`writers`/`runners`), with the built-ins `jsonl` (one record per line), `parquet` (one
  row per metric row) and `null`, and the shared contract check
  `rcp_ndcg.testing.results_conformance`. `records_from_report` and `records_from_run` build records from an
  `EvalReport` or a run directory; the new `rcp-ndcg results` group lists the sinks (`results sinks`) and
  exports (`results export --run DIR [--report FILE] --sink NAME --out URI`, `--system`, `--include-reference`).
  The run manifest's `DatasetRef` records the subset, split and task the data was read at, and a report's
  `inputs` carry them too, so an exported record states the real provenance rather than the `test` convention.
  The record schema is a compatibility contract: additive fields only within `v1`, a change to an existing
  field's meaning or type a new schema id ([the compatibility page](docs/reference/results-record.md)).
```

## Public surface changes

* New public module `rcp_ndcg.results` (in `PUBLIC_MODULES`): `ResultArtifact`, `ResultDataset`,
  `ResultMetric`, `ResultRecord`, `ResultSubject`, `ResultsSink`, `JsonlResultSink`, `ParquetResultSink`,
  `NullResultSink`, `RESULT_SCHEMA`, `RESULTS_GROUP`, `REFERENCE_SYSTEMS`, `record_identity`,
  `records_from_report`, `records_from_run`, `registered_result_sinks`, `result_sink_class`.
* Facade `rcp_ndcg` re-exports `ResultArtifact`, `ResultDataset`, `ResultMetric`, `ResultRecord`,
  `ResultSubject`, `ResultsSink`, `records_from_report`, `records_from_run`.
* `rcp_ndcg.testing` gains `results_conformance`.
* CLI: the new `results` group with `results sinks` and `results export` (`--run`, `--report`, `--sink`,
  `--out`, `--system`, `--include-reference`), and the output schemas `rcp-ndcg.results-sinks.v1` and
  `rcp-ndcg.results-export.v1`.
* Entry points: the new `rcp_ndcg.results` group (`jsonl`, `parquet`, `null`).
* Schema: `schemas/result-record.v1.json`; `run-manifest.v1` gains `DatasetRef.subset/split/task`;
  `eval-report.v1`'s `ReportInputs` gains `split`/`task`.
* No exit codes changed; no existing flag changed.

## Files outside scope

Each is minimal and carries a test:

* `rcp-ndcg/src/rcp_ndcg/runs/manifest.py`, `runs/pipeline.py` -- `DatasetRef` gains `subset`/`split`/`task`
  so the record states the real dataset provenance (the spec's `ResultDataset` fields).
* `rcp-ndcg/src/rcp_ndcg/eval/evaluate.py`, `cli/eval.py` -- `ReportInputs` gains `split`/`task` and
  `eval score --out` fills them, for the same reason on the `--report` path.
* `rcp-ndcg/src/rcp_ndcg/runs/pipeline.py` -- `REFERENCE_SYSTEMS`, so `candidates`/`judge` have one home
  instead of a second tuple in the results module.
* `tests/test_schemas.py` -- the schema walker now skips a `schema_name` field that is not a single Literal
  (`ResultArtifact.schema_name` is an artifact's own schema id, `str | None`, per the spec).
* `tests/test_layering.py`, `AGENTS.md` -- the `results` layer in the charter and its comments.
* `skills/rcp-ndcg/SKILL.md` -- one line pointing at the export command (the skill's 200-line budget was
  respected by shortening the bullet).

## For the next lanes

* The private companion repository's adapter consumes `rcp_ndcg.results`: register
  `rcp_ndcg.results` -> its sink class, implement `emit` (call `self._known(record)` first), and run
  `results_conformance` with `registered=False` until the package is installed. The record deliberately
  carries no consumer concept: no view, no pivot, no path layout.
* `records_from_run` reads `metrics/report.json` (or recomputes it from the run's artifacts); a run without a
  finished `evaluate` step is refused. `records_from_report` needs `report.inputs` (a report written by
  `eval score --out`); a hand-written report without inputs is refused, never named by a default.
* The record's protocol is the report's, always; a caller-provided `ResultDataset` supplies the name, subset,
  split, task and revision only. `record_identity` covers the subject, the dataset (protocol included) and
  the metrics; `created_at` is deliberately not in it.
* The manifest's `DatasetRef.subset/split/task` and `ReportInputs.split/task` are available to any later
  consumer that wants the dataset provenance without loading the data.
