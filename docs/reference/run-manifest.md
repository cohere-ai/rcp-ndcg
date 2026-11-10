# The run manifest

`rcp-ndcg.run-manifest.v1` is the record of one run: what it read, which steps ran, what each step wrote, what
the judge cost and which numbers came out. Every run directory has one `manifest.json`, updated atomically as
each step finishes. A run is resumable exactly when every completed step's identity and inputs are unchanged;
the [runs concept](../concepts/runs.md) describes the steps and the resume rule.

The manifest is a public artifact. Its JSON Schema is `schemas/run-manifest.v1.json`, exported from
`rcp.runs.RunManifest`; a consumer validates against the schema or reads the pydantic model, and both are
pinned by `tests/contract`.

## Versioning

The tag in the record's `schema` field is `rcp-ndcg.run-manifest.v1`, independent of the package version and
bumped only when the record changes incompatibly. Within `v1`:

* fields are only **added**, and an added field is optional or has a default;
* an existing field never changes meaning or type;
* an old `v1` manifest still validates, with the added fields taking their defaults;
* a reader of the first `v1` reads a later `v1` record: it reads the fields it knows and the additions are
  optional, so it does not need them.

The rule is the repository's artifact rule, spelled out for the exported results in
[the results record](results-record.md) and for every artifact in
[compatibility and versioning](versioning.md#artifact-tags). A change to an existing field's meaning or type is
a new tag (`rcp-ndcg.run-manifest.v2`), never a silent reinterpretation of `v1`.

## The fields

| Field | Type | Notes |
|---|---|---|
| `schema` | `"rcp-ndcg.run-manifest.v1"` | the artifact tag |
| `layout` | `"rcp-ndcg.run-layout.v1"` | the on-disk run layout |
| `run_id` | `str` | the run directory's name |
| `created_at`, `updated_at` | `datetime` | timezone-aware; `updated_at` moves on every write |
| `status` | enum | `submitted`, `running`, `completed`, `failed`, `partial`, `cancelled` |
| `code` | object | the package version, Python version, platform and, when known, the git commit and dirty flag |
| `config` | object | the resolved run config, with credentials and runtime-only fields removed |
| `dataset` | object or `null` | the run's primary dataset: name, revisions, subset, split, task |
| `path` | `str` or `null` | **added in `v1`**; the run directory the manifest was last written in. The artifact paths in `steps` are relative to it |
| `datasets` | list | **added in `v1`**; every dataset the run read, default `[]` (see below) |
| `suites` | list | **added in `v1`**; the suites the run read, default `[]` (see below) |
| `rows` | object or `null` | **added in `v1`**; the downstream row schema, default `null` (see below) |
| `families` | object | `{family_key: JudgementFamily}` of the judgements the run wrote |
| `steps` | list | one record per step: status, identity, inputs and outputs with hashes, usage, engines, timing |
| `metrics` | object | `{"<system>/<metric>@<k>": value}` from the evaluate step's summary |
| `usage` | object | requests and tokens summed over the run |

### The added `v1` fields

`path`, `datasets[]`, `suites[]` and `rows` were added to `v1` for consumers that need to record provenance the
first `v1` had no home for. Each is optional: a manifest written by the first `v1` loads unchanged, and the
added fields take their defaults.

* **`path`** — the run directory the manifest was last written in. `RunManifest.save` records it; a manifest
  written before the field existed reads `null`. The `steps[].inputs[].path` and `steps[].outputs[].path`
  values are run-relative, so this is what resolves them when a manifest is read away from its run directory.
* **`datasets[]`** — the datasets the run read. Each entry has the `dataset` provenance (`name`, `revisions`,
  `subset`, `split`, `task`) plus:
  * `internal` (`bool`, default `false`): whether the source is internal to the producer and not for
    publication;
  * `args` (`object`, default `{}`): the reader arguments that selected the slice (the options passed to
    `load_dataset`).
  `dataset` remains the run's primary dataset; `datasets` is the full list when a run read more than one.
* **`suites[]`** — the suites the run read. Each entry has the suite's `name` and `entries`, the suite entries
  the run evaluated (default `[]`, when the suite ran whole or the entries were not recorded).
* **`rows`** — `{schema: <name>}`: the downstream row schema the run's rows were written as. The package never
  interprets the name; it is a free string, and `rows` is `null` when the producer recorded none.

```python
from rcp.runs import RunManifest

manifest = RunManifest.model_validate(
    {
        "run_id": "20260102-030405-tiny-ab12cd",
        "code": {"package_version": "0.0.1", "python_version": "3.12", "platform": "linux"},
        "path": "runs/20260102-030405-tiny-ab12cd",
        "datasets": [
            {"name": "tiny", "subset": "test", "internal": False, "args": {"qrels_uri": "qrels.jsonl"}},
        ],
        "suites": [{"name": "nanobeir", "entries": ["NanoNFCorpusRetrieval"]}],
        "rows": {"schema": "example.rows.v1"},
    }
)

payload = manifest.model_dump(by_alias=True)
assert payload["datasets"][0]["args"] == {"qrels_uri": "qrels.jsonl"}
assert payload["suites"][0]["entries"] == ["NanoNFCorpusRetrieval"]
assert payload["rows"] == {"schema": "example.rows.v1"}

# A manifest written by the first v1 has none of the added fields; it still validates, and they default.
first_v1 = {name: value for name, value in payload.items() if name not in ("path", "datasets", "suites", "rows")}
assert RunManifest.model_validate(first_v1).path is None
```

## Reading a manifest

`RunManifest.load` accepts a run directory, a `RunLayout` or the manifest file itself. It refuses a record whose
`schema` tag it does not know, with a hint to restore the run from its mirror or re-run it; a damaged manifest is
never read as a valid one. The result sinks read the manifest for the run's identity and artifact hashes; the
export contract is [the results record](results-record.md).
