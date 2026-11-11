# Extension points

A seam is a place where a downstream product adds behaviour without a fork. The product reads eight
entry-point groups and one protocol. A config, a command or a URI names an entry, and the package loads it only
then. The seam stays in this package and the plugin stays in yours.

| Seam | The group the product reads | What names an entry |
|---|---|---|
| Job runner | `rcp.runners` | `runner.name` in a run config, or `rcp run start --runner` |
| Wire adapter | `rcp.adapters` | a config's `api`, through an entry named `<role>.<name>` |
| Dataset reader | `rcp.readers` | the URI scheme `<name>:<path>`, or `rcp data convert --format` |
| Dataset writer | `rcp.writers` | `rcp data convert --to` |
| Result sink | `rcp.results` | `rcp results export --sink` |
| Verified fake transport | `rcp.fake_transports` | a `fake://<engine>-<version>/<recipe>` URL |
| Metric | `rcp.metrics` | `evaluate(metrics=...)` or `rcp eval score --metric` |
| Run-event subscriber | `rcp.events` | nothing: every run publishes to every installed subscriber |
| Artifact cache (`CacheStore`) | no group: a protocol | your own code passes the store |

## The lookup that five groups share

Five of the eight groups resolve a name through `rcp.support.entrypoints`: runners, readers, writers, results and
metrics. One name has one lookup, and the three rules below are the policy of these five name-selecting groups:

- An unknown name is a `ConfigError` (exit 3). Its hint names the group and the installed entries.
- A name that two installed distributions provide is a `ConfigError` that names both providers. A plugin never
  replaces a built-in by publishing the built-in's name.
- An entry point that does not import is a `ConfigError` that names the entry point and its own failure. A broken
  plugin never disappears from the list silently.

The other three groups do not resolve a name through that lookup. `rcp.adapters` selects by `(role, name)` in its
own registry. `rcp.fake_transports` asks every provider for a URL until one answers. `rcp.events` loads every
installed subscriber and selects nothing.

## Job runners (`rcp.runners`)

A runner decides *where* a job runs. The `local`, `slurm` and `kubernetes` runners are entries of this group; a
third-party runner is a distribution that publishes one:

```toml
[project.entry-points."rcp.runners"]
my-runner = "my_package.runner:MyRunner"
```

`rcp.runners.get_runner("my-runner", **options)` loads the entry point only when the name is selected and calls
`MyRunner(**options)` with the run config's `runner.options`. A name provided by more than one installed
distribution is refused, so a plugin cannot shadow a built-in; an unknown name lists the installed runners. The
runner's `name` attribute is the entry point's name.

### The contract

| Member | What it must do |
|---|---|
| `name: str` | The entry point's name: a non-empty string. |
| `synchronous: bool = False` | Whether `submit` runs the job in this process: then the handle is the job's name, known before it runs, and the job's failure is raised from `submit`. A scheduler runner leaves it false. |
| `__init__(**options)` | Receive `runner.options` as keyword arguments. An invalid option raises `TypeError` or `ValueError`; the registry reports it as `ConfigError` naming the runner. |
| `submit(jobs) -> list[str]` | Submit the `JobSpec`s in order and return one non-empty handle per job. A runner that runs jobs in this process (see `synchronous`) returns each job's name and raises the job's failure here. |
| `status(handle) -> JobStatus` | The job's current state, non-blocking. An unknown handle answers `JobStatus.UNKNOWN`, not an exception. |
| `logs(handle, *, tail=None) -> str` | The job's output so far; `tail` keeps the last lines. |
| `cancel(handle) -> None` | Stop a pending or running job; a finished job is left as it is. |

`JobSpec` is one command line (`argv`) with its image, resources and environment, or the phases to run in order
(`phases`), exactly one of the two. The run layer reads four optional members when a runner declares them:

- `render(jobs) -> dict[str, str]` returns the script or manifest each job would submit, without submitting
  anything; `run start --dry-run` prints it. It is called before the run directory is written, so it is where a
  runner refuses a job it cannot run.
- `renders_phases = True` declares that `render` and `submit` take a phased job (its phases' engines are the
  runner's to start). A runner without it is handed the whole-run command as the job's `argv`, and a job whose
  phases start engines is refused.
- `run_root` is the directory a run is restored into inside the job, from the run's mirror; a runner whose jobs
  do not see the submitting host's files sets it. Runners without it share the files.
- `note(handle)` explains why a pending job is still pending (a scheduler's unschedulable reason).

`local`, `slurm` and `kubernetes` are the reference implementations; `LocalRunner` is the smallest and the one
to read first. A runner that adds no scheduling can delegate to it.

### Errors a runner may raise

- `ConfigError` (exit 3) for an unknown or ambiguous runner name, or options that are not the runner's options.
  The registry raises it around the constructor; a plugin does not need to import it for its own options.
- `RunnerError` (exit 6) for a scheduler command that failed, or a job the runner cannot submit as specified.
- A job that exits non-zero is the job's failure, not the runner's. A `synchronous` runner raises the error class
  of the job's exit code (`rcp.errors.error_class`, for example `MissingInputError` for 4) from `submit`; a code
  no error class uses is a non-retryable `RunnerError`. A scheduler runner returns a handle and reports
  `JobStatus.FAILED`; its output is in `logs`.

The typed error is what a caller branches on: `MissingInputError` names an input to produce, `ProviderError` a
failure to retry, and so on. Flattening a job's failure into a generic `RunnerError` loses that decision.

### A minimal runner

This runner adds one option and delegates the four methods to the local supervisor. A scheduler runner replaces
the bodies of the four methods; the signatures and the answers are the contract.

```python
from rcp.runners import JobSpec, JobStatus, LocalRunner


class MyRunner:
    name = "my-runner"
    synchronous = True

    def __init__(self, *, log_dir: str) -> None:
        self._local = LocalRunner(log_dir=log_dir)

    def submit(self, jobs: list[JobSpec]) -> list[str]:
        return self._local.submit(jobs)

    def status(self, handle: str) -> JobStatus:
        return self._local.status(handle)

    def logs(self, handle: str, *, tail: int | None = None) -> str:
        return self._local.logs(handle, tail=tail)

    def cancel(self, handle: str) -> None:
        self._local.cancel(handle)
```

A plugin's own tests call `rcp.testing.runner_conformance` with a constructed runner and a job it can actually
run. The helper checks the four methods and the answers' shapes, plus `render`, `renders_phases` and `run_root`
when declared; a failure is one assertion listing every broken check.

<!-- snippet: skip (the plugin is installed by the package under test) -->
```python
from rcp.runners import JobSpec, get_runner
from rcp.testing import runner_conformance

runner = get_runner("my-runner", log_dir="logs")
runner_conformance(runner, job=JobSpec(name="smoke", argv=("true",)))
```

The plugin's test should also run one real step and one job that exits 4: the step proves the handle, `status`
and `logs` describe the same job, and the failing job proves the failure is the typed error the built-in
`LocalRunner` raises, not a generic runner error. The repository's `tests/contract/test_runner_plugin.py`
installs a minimal runner as a test distribution, resolves it through this group, and makes both checks.

## Wire adapters (`rcp.adapters`)

An adapter turns one role's request into HTTP calls and reads the replies back into the role's result. The
transport does the rest: routing over replicas, retries, parking on an outage, credentials and usage. The
`rcp.adapters` group holds third-party wires, and each entry is named `<role>.<name>`:

```toml
[project.entry-points."rcp.adapters"]
embed.bedrock = "my_package.adapters:BedrockEmbed"
```

The role is one of `judge`, `embed`, `rerank` and `multi_vector`. A config selects the adapter with
`api: bedrock`, inside its own role's registry. The registry is scoped by role, so the same name may name a
different adapter per role. An entry whose name and class disagree on the role or the name is refused. A plugin
that would take a shipped adapter's `(role, name)` is refused, so a built-in is never shadowed.

### The contract

| Member | What it must do |
|---|---|
| `name: str` | The adapter's name within its role: the value `api` holds. It is the entry name's part after the dot. |
| `role: str` | The role it serves: `judge`, `embed`, `rerank` or `multi_vector`. It fixes the request and result types. |
| `calls(request, *, model) -> Sequence[Call]` | The calls `request` becomes, in order. A `Call` is a `method`, a `path` and a JSON body. |
| `interpret(request, replies) -> result` | The role's result. The transport gives one reply per call, in the calls' order. |
| `usage(reply) -> TokenCount \| None` | The tokens one reply reports, or `None` when its API reports none. Tokens, never characters. |

The class declares five credential facts: `HOSTED`, `API_KEY_ENV`, `KEY_REQUIRED`, `AUTH_HEADER` and
`DEFAULT_BASE_URL`. Registration refuses a class that omits one. `HOSTED` is declared by the class itself, never
inherited. The transport resolves the key and sends it; an adapter never touches a key. Four optional facts say
what the wire accepts: `MAX_BATCH`, `SUPPORTS_DIMENSIONS`, `ENCODING_FORMAT` and `REQUEST_SHAPES`.

`AdapterBase` carries those facts with their declared defaults, and the constructor convention. The role client
builds the adapter with the role config (`adapter_cls(config)`). An adapter whose requests read no config field
accepts and ignores it.

### Errors an adapter may raise

- `CapabilityError` (exit 8) when the endpoint cannot take this request at all. The causes are a refused answer
  schema, a media count beyond its limit, or a 413 on a batch cap.
- `RequestRejectedError` (exit 6) when the endpoint refuses this one request. Other requests would go through.
- `BackendUnavailableError` (exit 6) when the endpoint is down. The transport parks the replica and retries.
- `ConfigError` (exit 3) at load time, when the plugin cannot register. The causes are:
    - the entry point does not import;
    - a member or a fact is missing;
    - the class's role or name disagrees with the entry name;
    - the name is a shipped adapter's.

### A minimal adapter

This rerank adapter speaks one endpoint and reports no tokens. The transport still routes, retries and counts
around it.

```python
from collections.abc import Sequence
from typing import ClassVar

from rcp.inference.adapters.base import AdapterBase
from rcp.inference.types import Call, RerankRequest, RerankResult, Reply, TokenCount


class MyRerank(AdapterBase):
    """One POST per request, scores read back in reply order."""

    name: ClassVar[str] = "mine"
    role: ClassVar[str] = "rerank"
    HOSTED: ClassVar[bool] = False  # declared here: the base's default is not a declaration

    def calls(self, request: RerankRequest, *, model: str) -> Sequence[Call]:
        body = {"model": model, "query": request.query.text, "documents": [doc.text for doc in request.documents]}
        return [Call(method="POST", path="/rerank", json=body)]

    def interpret(self, request: RerankRequest, replies: Sequence[Reply]) -> RerankResult:
        scores = [float(entry["score"]) for entry in replies[0].body["results"]]
        return RerankResult.aligned(request, scores)

    def usage(self, reply: Reply) -> TokenCount | None:
        return None  # this wire reports no tokens
```

`rcp.testing.adapter_contract` checks a class without a request: the name, the role, the three members, the five
facts and the constructor. Give it a recorded request and its replies to check `interpret` and `usage` too.

```python
from rcp.testing import adapter_contract

adapter_contract(MyRerank)
```

## Dataset readers (`rcp.readers`)

A reader reads one dataset format into the canonical records. Its name is the URI scheme of
`rcp.data.load_dataset` and the `--format` of `rcp data convert`. Two readers are not schemes: `hf` (the Hub
layout is `hf://`) and `pdf` (a PDF has no queries). The built-ins are `beir`, `jsonl`, `hf`, `mteb`, `images`,
`videos`, `frames` and `pdf`; a third-party format is one class:

```toml
[project.entry-points."rcp.readers"]
my-format = "my_package.io:MyReader"
```

`load_dataset("my-format:<locator>")` resolves the reader, and `get_reader("my-format", uri=..., **options)`
builds it with reader options. The first constructor parameter is named `uri`, because both callers pass it by
keyword.

### The contract

| Member | What it must do |
|---|---|
| `name: str` | The format name: the entry-point name and the URI scheme. |
| `shapes: frozenset[DataShape]` | The shapes it serves: `corpus` (queries, documents, qrels) or `ranking` (one record per query). |
| `dataset_name: str` | The name the loaded dataset carries: usually the file or directory name. |
| `__init__(*, uri, **options)` | Read from `uri`. An option the reader does not take is a `ConfigError`. |
| `examples()` or `documents()` | One of the two must be native; the other is derived. A class that overrides neither is refused. |

`examples()` yields one `RankingExample` per query with its candidate list. `documents()` yields the corpus. The
corpus methods `queries()`, `qrels()`, `candidates()`, `excluded()`, `gains()` and `thetas()` narrow what the
source holds. Each has a declared default. `provenance` says where the data came from and how it was read. Both
iterators are re-iterable: the same call yields the same records again. A reader streams, so a corpus of 10^7
rows never has to fit in memory.

### Errors a reader may raise

- `DataError` (exit 12) when the data is refused. The causes are:
    - two rows with one id and different content;
    - a row with no id;
    - a qrels row that names a document the corpus lacks;
    - a label that is not a finite number.
- `ConfigError` (exit 3) at load time, when the reader cannot load. The causes are:
    - an unknown format name (the hint lists the installed readers);
    - an entry point that does not import;
    - a class that is not a `SourceReader`;
    - a reader option the reader does not take.

A reader's own tests call `rcp.testing.io_conformance`. The suite is the one definition of what a reader must do.
It checks re-iterable records, unique ids, aligned document fields, and a shape that exists exactly when
something is judged. It also checks the pool, exclusion and gains members.

### A minimal reader

This reader serves the ranking shape only. The corpus methods derive from it: the judged documents become the
candidate list.

```python
import json
from collections.abc import Iterator

from rcp.data.io import DataShape, SourceReader
from rcp_core.records import RankingExample


class CandidateJsonlReader(SourceReader):
    """One JSON object per line: query_id, query, doc_ids, docs, qrels."""

    name = "candidates"
    shapes = frozenset({DataShape.RANKING})
    dataset_name = "candidates"

    def __init__(self, *, uri: str, name: str | None = None) -> None:
        self.uri = uri
        if name is not None:
            self.dataset_name = name

    def examples(self) -> Iterator[RankingExample]:
        with open(self.uri, encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                yield RankingExample(
                    query_id=row["query_id"],
                    query=row["query"],
                    doc_ids=row["doc_ids"],
                    docs=row["docs"],
                    qrels=row["qrels"],
                )
```

<!-- snippet: skip (the plugin is installed by the package under test, and reads its own fixture) -->
```python
from rcp.testing import io_conformance

io_conformance(CandidateJsonlReader(uri="candidates.jsonl"))
```

## Dataset writers (`rcp.writers`)

A writer writes the canonical records in one format. Its name is the `--to` of `rcp data convert`. The built-ins
are `beir`, `jsonl` and `mteb`:

```toml
[project.entry-points."rcp.writers"]
my-format = "my_package.io:MyWriter"
```

### The contract

| Member | What it must do |
|---|---|
| `name: str` | The format name: the entry-point name. |
| `shapes: frozenset[DataShape]` | What it can write; `ranking` by default. |
| `write_examples(examples, uri) -> int` | Write the ranking records and return how many it wrote. |
| `write_corpus(documents, queries, qrels, uri) -> int` | Write the corpus shape and return how many documents it wrote. |

A shape the writer does not serve keeps the base class's body, which raises `NotImplementedError`. That is how
the writer table knows the shape is absent.

### Errors a writer may raise

- `DataError` (exit 12) when the data cannot be written in that format. A field the format has no place for, or a
  record the format refuses.
- `ConfigError` (exit 3) when the writer cannot load or cannot write. The causes are:
    - an unknown writer name (the hint lists the installed writers);
    - an entry point that does not import;
    - a class that is not a `SinkWriter`;
    - an option the writer refuses.

### A minimal writer

This writer writes the ranking shape as one JSON object per line. It reports the count the conversion prints. A
record's query text is `example.text`: `query` is an alias, for construction only.

```python
import json
from collections.abc import Iterable

from rcp.data.io import DataShape, SinkWriter
from rcp_core.records import RankingExample


class CandidateJsonlWriter(SinkWriter):
    """The inverse of the reader above: one JSON object per line."""

    name = "candidates"
    shapes = frozenset({DataShape.RANKING})

    def write_examples(self, examples: Iterable[RankingExample], uri: str) -> int:
        written = 0
        with open(uri, "w", encoding="utf-8") as handle:
            for example in examples:
                row = {
                    "query_id": example.id,
                    "query": example.text,
                    "doc_ids": example.doc_ids,
                    "docs": example.docs,
                    "qrels": example.qrels,
                }
                handle.write(json.dumps(row) + "\n")
                written += 1
        return written
```

One record, written to a file, returns the count:

```python
example = RankingExample(
    query_id="q1", query="What is RCP?", doc_ids=["d1"], docs=["A document."], qrels={"d1": 1.0}
)
assert CandidateJsonlWriter().write_examples([example], "candidates.jsonl") == 1
```

## Result sinks (`rcp.results`)

A sink consumes `rcp.result-record.v1` records; it never reads the run. The built-in sinks are `jsonl`, `parquet`
and `null`:

```toml
[project.entry-points."rcp.results"]
mine = "my_package.results:MySink"
```

`rcp results export --sink mine --out <uri>` builds the records from a run directory or a report file. It hands
them to the sink, and `--out` is the sink's `uri`.

### The contract

| Member | What it must do |
|---|---|
| `name: str` | The sink name: the entry-point name. |
| `__init__(uri=None, options=None)` | Take `uri` as its first parameter after `self`; the registry passes it by keyword. |
| `emit(record)` | Consume one `ResultRecord`. Never mutate it. Refuse a schema it does not know. |
| `flush()` | Write what it buffers. Called once after the last record, and idempotent. |

### Errors a sink may raise

- `ConfigError` (exit 3) for a record whose `schema` is not one the sink knows. A v2 record written as v1 would
  corrupt the export. `ResultsSink._known(record)` is the shared check.
- `DataError` (exit 12) for a record the export cannot carry in that format.
- A failure of the destination the sink writes to is the backend's own error. The registry does not wrap it.
- `ConfigError` (exit 3) at load time, when the sink cannot load. The causes are:
    - an unknown sink name;
    - an entry point that does not import;
    - a class that is not a `ResultsSink`.

A sink's own tests call `rcp.testing.results_conformance`. With `registered=False` it skips the entry-point check
while the package is under development.

### A minimal sink

This sink counts the records and keeps none. It passes the conformance as written.

```python
from rcp.results import ResultRecord, ResultsSink
from rcp.testing import results_conformance


class CountingSink(ResultsSink):
    """Count the records: a dry run, or a bridge to a counter of your own."""

    name = "counting"

    def __init__(self, uri: str | None = None, options: dict[str, str] | None = None) -> None:
        super().__init__(uri, options)
        self.count = 0

    def emit(self, record: ResultRecord) -> None:
        self._known(record)  # refuses a record of a schema this sink does not know
        self.count += 1


results_conformance(CountingSink(), registered=False)
```

The record's fields are the compatibility contract, not this seam's. See [the results record](results-record.md)
for every field, and [export results](../how-to/export-results.md) for the export in full.

## Verified fake transports (`rcp.fake_transports`)

A `fake://` URL is answered in process, so an offline test runs the real transport. Two answers exist. A URL whose
host is an engine version (`fake://vllm-0.33.0/<recipe>`) names a **verified fake engine**. The product asks the
`rcp.fake_transports` group for that URL. Every other `fake://` URL is answered by the product's own hash-seeded
fake. The product ships no provider; the unpublished `rcp-test` package registers the verified fake engines.

```toml
[project.entry-points."rcp.fake_transports"]
my-engines = "my_package.engines:transport_for"
```

### The contract

An entry point resolves to one callable, `provider(url) -> httpx.BaseTransport | None`. The product asks each
provider in turn until one answers. `None` means "not mine", and the next provider is asked. The first
non-`None` transport answers the URL.

### Errors a provider may raise

- Any error a provider raises propagates to the caller that asked for the transport. The product wraps nothing.
- Return `None` for a URL you do not answer. Raise only for a URL you answer and cannot serve.
- The product raises `ConfigError` (exit 3) when the URL names a verified engine and no installed provider
  answers it. The message names this group and `rcp-test`.

### A minimal provider

This provider answers one engine version and declines every other URL.

```python
import httpx


def transport_for(url: str) -> httpx.BaseTransport | None:
    """Answer one engine version; return None for every other URL."""
    if not url.startswith("fake://my-engine-1.0.0/"):
        return None

    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.0, 1.0]}]})

    return httpx.MockTransport(handle)
```

`rcp.emulators` is the test package's own group, not a product seam: the product never reads it. See
[use the verified fake engines](../how-to/use-verified-fake-engines.md) for both groups.

## Metrics (`rcp.metrics`)

A metric scores one query. The shipped `rcp`, `qrel_ndcg` and `count_ndcg` are entries of this group too, added
by this package. So one lookup covers built-ins and plugins, and one policy refuses a duplicate:

```toml
[project.entry-points."rcp.metrics"]
mrr = "my_package.metrics:MeanReciprocalRank"
```

`evaluate(metrics=["mrr"])` and `rcp eval score --metric mrr` select it. The entry-point name is the metric name,
and the class's `name` must equal it.

### The contract

| Member | What it must do |
|---|---|
| `name: str` | The metric name: the entry-point name. The two must agree. |
| `k_required: bool = True` | Whether the metric takes the cutoff. `False`: it is scored once, at `k=0`. |
| `score(query, ranking, labels, *, protocol, k, candidates=None, excluded=()) -> float \| None` | One query's value. `None` when the metric is undefined for that query. |

`ranking` is the system's `{doc_id: score}`. `labels` is the query's `{doc_id: grade}` qrels. `protocol` is the
dataset's scoring protocol, so the tie rule and the pool restriction stay `rcp_core`'s.
`rcp_core.protocol.candidate_docs` is the one ranking rule. A `None` value drops out of the means.

### Errors a metric may raise

- `ConfigError` (exit 3) at load time, when the metric cannot load. The causes are:
    - an unknown metric name, or a name two distributions provide;
    - an entry point that does not import;
    - a class that is not a `Metric`;
    - no `k_required` bool;
    - a class `name` that disagrees with the entry-point name.
- `DataError` (exit 12) when `score` returns something that is not a number and is not `None`. The evaluator
  names the metric and the query.

### A minimal metric

This metric takes no cutoff, so the evaluator scores it once. It ranks through the protocol's rule. A pool
restriction or an exclusion applies to it, as it does to the shipped three.

```python
from collections.abc import Collection, Mapping, Sequence

from rcp_core.protocol import PROTOCOLS, Protocol, candidate_docs

from rcp.eval import Metric


class MeanReciprocalRank(Metric):
    """The reciprocal rank of the first document with a positive grade."""

    name = "mrr"
    k_required = False

    def score(
        self,
        query: str,
        ranking: Mapping[str, float],
        labels: Mapping[str, float],
        *,
        protocol: Protocol,
        k: int,
        candidates: Sequence[str] | None = None,
        excluded: Collection[str] = (),
    ) -> float | None:
        relevant = {doc_id for doc_id, grade in labels.items() if grade > 0}
        ranked = candidate_docs(protocol, ranking, candidates=candidates, excluded=excluded, query_id=query)
        ranks = [rank for rank, doc_id in enumerate(ranked, start=1) if doc_id in relevant]
        return 1.0 / min(ranks) if ranks else None


assert MeanReciprocalRank().score(
    "q1", {"d1": 2.0, "d2": 1.0}, {"d2": 1.0}, protocol=PROTOCOLS["plain"], k=0
) == 0.5  # the only relevant document is ranked second
```

## Run-event subscribers (`rcp.events`)

A run publishes what it does as data. A dashboard or a progress line can follow it without a patch to the
pipeline. A subscriber is a callable, installed in the `rcp.events` group. The group is optional, and the product
declares no entry:

```toml
[project.entry-points."rcp.events"]
my-dashboard = "my_package.events:subscribe"
```

Nothing selects a subscriber by name: every run publishes to every installed subscriber, in entry-point name
order.

### The contract

One callable, `subscribe(event: RunEvent) -> None`. It is called once per event, synchronously, in the order the
run produces the events. The run does not produce the next event until the subscriber returns, so a slow
subscriber slows the run. The event carries `run_id`, `step`, `phase`, `at` (UTC), `usage` and `detail`. The four
phases are `step_started`, `step_finished`, `step_failed` and `usage`.

### Errors a subscriber may raise

- None reaches the run. A subscriber that raises is logged once for the run and then ignored. The other
  subscribers still receive every event.
- A subscriber whose entry point does not load, or that is not callable, is logged once per run and skipped.

The manifest never records a subscriber's failure. A broken dashboard never changes what the run writes.

### A minimal subscriber

This subscriber keeps the finished steps. `Emitter([...])` is the same emitter a run builds over the group.

```python
from rcp.support.events import Emitter, RunEvent

seen: list[RunEvent] = []


def subscribe(event: RunEvent) -> None:
    """Keep the steps that finished: the shape a progress bar or a log sink reads."""
    if event.phase == "step_finished":
        seen.append(event)


emitter = Emitter([subscribe])
emitter.emit(RunEvent(run_id="run-1", step="evaluate", phase="step_finished", detail={"status": "completed"}))
assert [event.step for event in seen] == ["evaluate"]
```

See [runs, engines and runners](../concepts/runs.md) for the phase table, the ordering guarantees and the
manifest's record.

## Artifact caches (`CacheStore`)

`rcp.storage.cache.CacheStore` is a protocol, not an entry-point group. It has no group and no name, because
nothing selects a store. Your own code holds a store and passes it to `lookup(store, key)`. A store is a
structural contract, so `isinstance(store, CacheStore)` checks that the four methods exist, and no registration is
needed.

### The contract

| Method | What it must do |
|---|---|
| `get(key) -> bytes \| None` | The bytes stored under the key, or `None`. A miss is an answer, never an exception. |
| `put(key, value) -> None` | Store the value under the key, replacing what the key held. |
| `keys(prefix="") -> Iterable[str]` | Every key that starts with `prefix`; `""` lists all of them. The order is not part of the contract. |
| `delete(key) -> None` | Remove the key. A later `get` never returns it. Deleting a key that is not there is not an error. |

A key is an opaque string. The package's own keys are `/`-separated and prefix-scannable
(`rcp.storage.cache.cache_key`), and nothing in the contract requires that.

### Errors a store may raise

- Any exception. `lookup` turns a store that raises into a logged miss. A failed cache read never fails a run
  that could compute the value.
- `MissingInputError` (exit 4) from the shipped `UriCacheStore.put`. An entry records the source's identity to be
  fresh for. A key whose source does not exist would never be served.

Two implementations ship: `MemoryCacheStore` for tests, and `UriCacheStore` over the URI cache's own entries.
`tests/storage/test_cache_store.py` holds `assert_cache_store_contract`, one contract checked over both
implementations; a downstream runs it over its own keys.

### A minimal store

This store keeps its values in a dictionary. It is the smallest implementation of the four methods.

```python
from rcp.storage.cache import CacheStore, cache_key, lookup


class DictStore:
    """A store of your own: the four methods, over any backend."""

    def __init__(self) -> None:
        self._entries: dict[str, bytes] = {}

    def get(self, key: str) -> bytes | None:
        return self._entries.get(key)

    def put(self, key: str, value: bytes) -> None:
        self._entries[key] = value

    def keys(self, prefix: str = "") -> list[str]:
        return [key for key in self._entries if key.startswith(prefix)]

    def delete(self, key: str) -> None:
        self._entries.pop(key, None)


store: CacheStore = DictStore()
row = cache_key("run-1", "rows", "004001")  # "run-1/rows/004001"
store.put(row, b'{"query_id": "q1"}')
assert lookup(store, row) == b'{"query_id": "q1"}'
assert store.keys("run-1/rows/") == [row]
store.delete(row)
assert lookup(store, row) is None
```

See [judges, the judgement store and estimates](../concepts/judges.md) for the shipped stores and the resumption
they serve.
