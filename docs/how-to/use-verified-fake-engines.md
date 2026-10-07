# Use the verified fake engines

One **verified fake engine** per (engine, version, recipe, behaviour fingerprint) answers
`fake://`-URLs with an engine-version host -- `fake://vllm-0.31.0/qwen3-embedding-0.6b` -- inside
every offline test and replay. The protocol is emulated (routes, request validation, error bodies,
result ordering and framing, usage counts and token counting with the recipe's **real tokenizer
files**), the model outputs are **replayed** for observed inputs and come from a declared
deterministic surrogate for unseen ones, clearly marked in the reply.

## Selecting an emulator

A verified engine URL is `fake://<engine>-<version>/<recipe>`, optionally with `?fingerprint=<sha>`
when several fingerprints of one recipe are registered (a migration). The routing seam is
`rcp_ndcg.inference.fake` (a `fake://` URL whose host names an engine and a version); the emulator
itself lives in `rcp_ndcg.testing.engines`:

<!-- snippet: skip (the corpus paths are the repository's; run it from a checkout) -->
```python
from rcp_ndcg.testing.engines import load_corpus, registry
from tests._engines import emulator_for  # the recipe wiring of this repository's tests

emulator_for("qwen3-reranker-0.6b")  # builds the emulator of its corpus and registers it
transport = registry.resolve("vllm", "0.31.0", "92c037f2639d6ec78f93e30ebc17041a819077f59fdc3e2e8bd9298c1c14020e", "qwen3-reranker-0.6b")
```

An emulator records the engine version and recipe revision it was verified against (`Verified`) and
**refuses another** (`require_verified_for` names the fields that moved). Out-of-tree models (a
private repository's plugins) register their emulators through the `rcp_ndcg.emulators` entry-point
group: each entry point resolves to a callable returning emulators, kept in the same registry. An
out-of-tree corpus lives in its own repository and answers the same URL shape against its own
registry entry.

## What a corpus is

An observation corpus is raw-first (OBSERVATIONS-SPEC): `manifest.json` (provenance and hashes),
`exchanges.jsonl.gz` (the wire, one record per exchange) and `verification.jsonl` (the append-only
verification record). Cabinets live at
`tests/contract/engines/<engine>-<version>/<recipe>/<behaviour-fingerprint>/`; the shared
`_tokenizers/` store vends the `tokenizer.json` files whose SHA-256 the fingerprints hash, so the
whole suite runs offline. Every derived view (parsed bodies, tolerances) is recomputed from the raw
records by versioned code (`NORMALISATION_VERSION` names the volatile fields a comparison strips --
request ids and `created` stamps); a corpus format registers with
`rcp_ndcg.testing.engines.register_corpus_format` and older document schemas migrate at load
(`register_line_migration`), so `rcp_ndcg_vllm.observe`'s writer plugs in when it lands.

## Conformance, staleness and goldens

The conformance suite (`tests/conformance/`) replays every recorded exchange against its emulator
and requires identical status and body within the recorded non-determinism (measured from the
corpus's own repeats, never assumed). The staleness check recomputes each recipe's **behaviour
fingerprint** (`rcp_ndcg_vllm.fingerprint.behaviour_fingerprint`: the checkpoint id and revision, the
serve block, the template file's bytes, the tokenizer's SHA-256 and the request-shaping client
fields) and fails **naming the changed inputs** when it moved; the only way past is a dated entry in
`tests/conformance/waivers.json`, which the release checklist requires to be empty.

The golden replays (`tests/e2e/test_golden_replay.py`) run one NanoBEIR-shaped and one ViDoRe-shaped
subset through rcp-ndcg's retrieval and rerank path against the emulators and reproduce the golden
metrics to 1e-9 -- every input observed (`x-rcp-ndcg-emulator-source: replayed` on every reply, and
the run fails otherwise; a surrogate is marked `surrogate` or `mixed`):

<!-- snippet: skip (a repository checkout with the corpora) -->
```bash
RCP_UPDATE_GOLDENS=1 uv run --no-sync pytest tests/e2e/test_golden_replay.py  # regenerate the goldens
```

## What moved between two corpora

`rcp-ndcg-vllm`'s `python -m rcp_ndcg_vllm.changes` reports change handling (OBSERVATIONS-SPEC
section 7): `changed` is the re-record-changed-only selection (per recipe: `unchanged`, `changed`,
`new` or `unloadable`, with the changed inputs named -- the wave runner's `--changed-since` mode
records the `changed` and `new` ones), and `diff` writes the behaviour diff of two corpora of one
recipe (per input, the score or vector deltas, changed statuses, changed refusals and changed
protocol behaviour, summarised by stratum) for review before a new corpus replaces the old.

<!-- snippet: skip (a repository checkout with the corpora) -->
```bash
python -m rcp_ndcg_vllm.changes changed --recipes-root packages/rcp-ndcg-vllm/recipes --corpora-root tests/contract/engines/vllm-0.31.0
python -m rcp_ndcg_vllm.changes diff --before <corpus-dir> --after <corpus-dir>
```