# Use the verified fake engines

One **verified fake engine** per (engine, version, recipe, behaviour fingerprint) answers
`fake://`-URLs with an engine-version host -- `fake://vllm-0.31.0/qwen3-embedding-0.6b` -- inside
every offline test and replay. The protocol is emulated (routes, request validation, error bodies,
result ordering and framing, usage counts and token counting with the recipe's **real tokenizer
files**); the model outputs are **replayed** for observed inputs and come from a declared
deterministic surrogate for unseen ones, marked in every reply.

## Selecting an emulator

A verified engine URL is `fake://<engine>-<version>/<recipe>`, with `?fingerprint=<sha>` when several
fingerprints of one recipe are registered (a migration). The routing seam is
`rcp_ndcg.inference.fake` (a `fake://` URL whose host names an engine and a version); the emulators
live in `rcp_ndcg.testing.engines`, and this repository's tests build them from the committed corpora:

<!-- snippet: skip (the corpus paths are the repository's; run it from a checkout) -->
```python
from rcp_ndcg.testing.engines import registry
from tests._engines import emulator_for  # the recipe wiring of this repository's tests

emulator = emulator_for("qwen3-reranker-0.6b")  # builds the emulator of its corpus and registers it
assert emulator.verified is not None
fingerprint = emulator.verified.behaviour_fingerprint
assert registry.resolve("vllm", "0.31.0", fingerprint, "qwen3-reranker-0.6b") is emulator
```

An emulator records the engine version and recipe revision it was verified against (`Verified`) and
**refuses another** (`require_verified_for` names the fields that moved). Out-of-tree models (a
private repository's plugins) register their emulators through the `rcp_ndcg.emulators` entry-point
group: each entry point resolves to a callable returning emulators, kept in the same registry.

## What a reply tells you

- `x-rcp-ndcg-emulator-source`: `replayed` (every input observed), `surrogate`, `mixed`, or
  `refused-unmodelled`. A test that asserts numbers can only use replayed inputs.
- `x-rcp-ndcg-emulator-route`: `observed` when the corpus recorded the route, `unobserved` when the
  reply follows the engine's source without a recording (`VllmEmulator.unobserved_routes`; the
  provisional corpus records no `/pooling` and no `/tokenize`).

An output is replayed only for the exact context it was observed under: the engine prompt plus every
request field that changes what the model returns (`use_activation`, `dimensions`,
`add_special_tokens`, `task`; `FIELD_CLASSES`). Another value answers the surrogate; a field the
emulator does not model (`instruction`, `truncate_prompt_tokens`, ...) is refused with a 400 marked
`refused-unmodelled`; a field the engine's request model does not declare is ignored, as the engine
ignores it.

## What a corpus is

An observation corpus is raw-first (OBSERVATIONS-SPEC) and has one format and one reader,
`rcp_ndcg.testing.corpus`: one record per exchange (the request and the response as they crossed the
wire, the headers that matter), `nondeterminism.json` (the measured differences between repeated
sendings of the same request and the tolerances derived from them) and `manifest.json` (provenance,
the behaviour fingerprint and its named inputs, integrity hashes). The repository keeps subsets
(`records.jsonl.gz` and an `index.json` naming the full corpus) at
`tests/contract/engines/<engine>-<version>/<recipe>/<behaviour-fingerprint>/`, found by scanning their
manifests (`find_corpora`), each with its append-only `verification.jsonl` beside the recorded files.
The shared `_tokenizers/` store vends the `tokenizer.json` files whose SHA-256 the fingerprints hash,
so the whole suite runs offline. The emulators read records through `exchanges_of`; every derived view
(parsed outputs, normalised bodies) is recomputed by versioned code, and the verification tolerance
is the corpus's own (`corpus_tolerance`).

The committed corpus is **provisional**: the GPU shakedown's recordings, valid to build and test the
emulators, not release evidence (every manifest says so in its `provisional` block). It sent every
request once, so its non-determinism is **unmeasured**: replays are compared exactly. It kept parsed
bodies only, so its records declare their raw bytes reconstructed and conformance compares no raw bytes
for them.

## Conformance, staleness and goldens

The conformance suite (`tests/conformance/`) replays every recorded exchange through the emulator's
HTTP surface and compares the status, the recorded headers, the body and, where recorded, the raw
bytes. The staleness check recomputes each recipe's **behaviour fingerprint**
(`rcp_ndcg_vllm.fingerprint.behaviour_fingerprint`: the checkpoint id and revision, the serve block,
the template file's bytes, the tokenizer's SHA-256 and the client fields that change the request
bytes) and fails **naming the changed inputs** when no committed corpus carries it; the only way past
is a dated, reasoned, unexpired entry in `tests/conformance/waivers.json`, which the release checklist
requires to be empty. A corpus left stale by a recipe change and awaiting its re-recording is declared in
`tests/conformance/stale.json` (the recipe, the recorded fingerprint, the inputs that moved, why and when):
the replays skip it, and the suite checks that it fails the staleness gate naming exactly those inputs;
the release checklist requires that list empty too. A corpus whose moved inputs shape none of its recorded
exchanges (metadata only, such as a pooler setting restating the engine's default) is re-keyed instead:
its manifest carries the current fingerprint and a `rekeyed` entry naming what moved and why. `RCP_APPEND_VERIFICATION=1` appends the suite's result to every corpus's
verification record.

The golden replays (`tests/e2e/test_golden_replay.py`) run a NanoBEIR-shaped and a ViDoRe-shaped mini
through rcp-ndcg's retrieval and rerank path against the emulators and pin nDCG@10 and RCP-nDCG@10 to
1e-9 -- every input observed, or the run fails. The pins are **regression pins** computed by this code
from the provisional corpus (`kind: regression-pin` in `tests/e2e/golden/goldens.json`), not
independent GPU-run numbers; the RC0 subset corpus replaces them. The ViDoRe retrieval view is waived
until a corpus observes a page image.

<!-- snippet: skip (a repository checkout with the corpora) -->
```bash
RCP_UPDATE_GOLDENS=1 uv run --no-sync pytest tests/e2e/test_golden_replay.py  # recompute the pins
RCP_APPEND_VERIFICATION=1 uv run --no-sync pytest tests/conformance -k verification_record
```

## What moved between two corpora

`rcp-ndcg-vllm`'s `python -m rcp_ndcg_vllm.changes` reports change handling (OBSERVATIONS-SPEC
section 7): `changed` is the re-record-changed-only selection (per recipe: `unchanged`, `changed`,
`new` or `unloadable`, with the changed inputs named -- the wave runner's `--changed-since` mode
records the `changed` and `new` ones), and `diff` writes the behaviour diff of two corpora of one
recipe (per input, the score or vector deltas, changed statuses and changed protocol behaviour --
headers, framing, error bodies -- summarised by route) for review before a new corpus replaces the old.

<!-- snippet: skip (a repository checkout with the corpora) -->
```bash
python -m rcp_ndcg_vllm.changes changed --recipes-root packages/rcp-ndcg-vllm/recipes --corpora-root tests/contract/engines/vllm-0.31.0
python -m rcp_ndcg_vllm.changes diff --before <corpus-dir> --after <corpus-dir>
```
