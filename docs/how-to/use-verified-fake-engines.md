# Use the verified fake engines

One **verified fake engine** answers `fake://`-URLs inside every offline test and replay. There is one such
engine per (engine, version, recipe, behaviour fingerprint). The URL host names an engine version, for example
`fake://vllm-0.33.0/qwen3-embedding-0.6b`. The engine emulates the protocol: routes, request validation, error bodies, result ordering and framing, and usage counts. It also emulates token counting with the recipe's **real tokenizer files**. It **replays** model outputs for observed inputs. For unseen inputs, it uses a declared deterministic
surrogate. Every reply marks which source it used.

**The committed corpora are still the `vllm-0.31.0` recording.** An emulator registers under the version that its corpus was recorded on. `rcp-ndcg-test/corpora/vllm-0.31.0/` was recorded on the v0.31.0
image. Re-recording on the v0.33.0 image is the GPU waves' work. `rcp-ndcg-test/corpora/vllm-0.33.0/` is the
destination; re-record renames the directory into it. The `fake://vllm-0.33.0/...` host that this page uses is
what the corpora register under once that recording is made. Until then, a `vllm-0.33.0` lookup is refused by
name. This is the version gate below, not a missing feature.

## Selecting an emulator

A verified engine URL is `fake://<engine>-<version>/<recipe>`. Add `?fingerprint=<sha>` when several
fingerprints of one recipe are registered (a migration). The routing extension point is
`rcp_ndcg.inference.fake`: a `fake://` URL whose host names an engine and a version. The emulators live in
`rcp_ndcg_test.engines`. This repository's tests build them from the committed corpora:

<!-- snippet: skip (the corpus paths are the repository's; run it from a checkout) -->
```python
from rcp_ndcg_test.engines import registry
from tests._engines import emulator_for  # the recipe wiring of this repository's tests

emulator = emulator_for("qwen3-reranker-0.6b")  # builds the emulator of its corpus and registers it
assert emulator.verified is not None
fingerprint = emulator.verified.behaviour_fingerprint
# the committed corpora are the v0.31.0 recording; the re-record registers them under 0.33.0
assert registry.resolve("vllm", "0.31.0", fingerprint, "qwen3-reranker-0.6b") is emulator
```

An emulator records the engine version and recipe revision that it was verified against (`Verified`). The
registry keys all four of them. `registry.resolve` refuses a lookup for an engine version that the corpus was
not recorded on. It names the version that the recipe is registered for. The routing extension point
(`transport_for`) refuses the same way. A corpus recorded on another engine version is refused, never
replayed. Out-of-tree models (a private repository's plugins) register their emulators through the
`rcp_ndcg.emulators` entry-point group. Each entry point resolves to a callable that returns emulators, kept
in the same registry.

### The two entry-point groups

Two groups carry the fake engines. Different packages read them:

- `rcp_ndcg.fake_transports` is the **product's** extension point (`rcp_ndcg.inference.fake`). Each entry
  point resolves to a callable `provider(url) -> httpx.MockTransport | None`. The product asks each one for a
  `fake://<engine>-<version>/<recipe>` URL until one answers. `None` means "not mine". This is how a served-role request reaches an emulator without the product importing `rcp_ndcg_test`. That package needs the observation corpora and the recipes. This repository's provider is `rcp_ndcg_test.engines`
  (`rcp-ndcg-test = "rcp_ndcg_test.engines:transport_for"` in its `pyproject.toml`). Thus installing
  `rcp-ndcg-test` beside `rcp-ndcg` is what makes a `fake://` URL resolve at all. Without it, the product
  raises a `ConfigError` that names that package and the group.
- `rcp_ndcg.emulators` is the **test package's** own extension point
  (`rcp_ndcg_test.engines.registry.load_entry_points`). Each entry point resolves to an emulator or to a
  callable that returns them. The emulators are registered in the same `registry` that `from_corpus` fills.
  It exists for emulators built outside this repository (a private model's), which have no product code to
  reach.

## What a reply tells you

- `x-rcp-ndcg-emulator-source`: `replayed` (every input observed), `surrogate`, `mixed`, or
  `refused-unmodelled`. A test that asserts numbers can only use replayed inputs.
- `x-rcp-ndcg-emulator-route`: `observed` when the corpus recorded the route. `unobserved` when the reply follows the engine's source without a recording (`VllmEmulator.unobserved_routes`). The committed corpora record no `/pooling`, and every one of them records `/tokenize`.

An output is replayed only for the exact context that it was observed under. That context is the engine prompt plus every request field that changes what the model returns. The fields are `use_activation`, `dimensions`, `add_special_tokens`, `task` and `FIELD_CLASSES`. Another value answers the surrogate. A field that the
emulator does not model (`instruction`, `truncate_prompt_tokens`, ...) is refused with a 400 marked
`refused-unmodelled`. A field that the engine's request model does not declare is ignored, as the engine
ignores it. A chat-shaped (`messages`) request is a prompt carrier. The emulator's strategy renders its
conversations (`ChatPrompts`). A media part is keyed by its content identity: the sent bytes' SHA-256 plus the
recipe's declared processing (`MediaIdentity`). Thus the same image or clip under the same policy replays, and
another one answers the surrogate. Building an emulator never fails on one unmodelled record. The record is
skipped and named in `VllmEmulator.unmodelled_records`; the verification record carries the list. The rest of
the corpus replays. A request for the skipped record answers the marked `refused-unmodelled` 400.

## What a corpus is

An observation corpus is raw-first (OBSERVATIONS-SPEC). It has one format and one reader,
`rcp_ndcg_test.corpus`. There is one record per exchange. It holds the request and the response as they crossed the wire, and the headers that matter. The corpus also has `nondeterminism.json` and `manifest.json`. That second file gives the measured differences between repeated sendings of the same request and the tolerances derived from them. The manifest records provenance: the engine's image, argv, `VLLM_*` environment and patch opt-in. It also records the model's facts, the behaviour fingerprint and its named inputs, and integrity hashes. The `VLLM_*` environment is the engine's own
knobs. The pod's Kubernetes Service links ride under the same prefix (the Services are named
`vllm-service-...`). `rcp_ndcg_test.observe.provenance` drops them, so no internal address or Service name
enters a manifest. The repository keeps subsets (`records.jsonl.gz` and an `index.json` that names the full
corpus) at `rcp-ndcg-test/corpora/<engine>-<version>/<recipe>/<behaviour-fingerprint>/`. `find_corpora` finds
them by scanning their manifests. Each subset has its append-only `verification.jsonl` beside the recorded
files. The shared `_tokenizers/` store vends the `tokenizer.json` files whose SHA-256 the fingerprints hash.
Thus the whole suite runs offline. The emulators read records through `exchanges_of`. Every derived view
(parsed outputs, normalised bodies) is recomputed by versioned code. The verification tolerance is the
corpus's own (`corpus_tolerance`).

The committed corpora are **release recordings**: the wave's own exchanges, cut to a subset. They are keyed by the behaviour fingerprint they were recorded at. Each has its append-only verification record; no manifest carries
a `provisional` block any more. `nondeterminism.json` measures the differences between repeated sendings of one
request: twice in one engine process and again after a restart. The corpus's replay tolerance is derived from
them (`corpus_tolerance`). A corpus whose requests were never repeated declares its non-determinism unmeasured
and is compared exactly. The records keep the raw bytes, so conformance compares them, not the parsed bodies
alone.

## Conformance, staleness and goldens

The conformance suite (`tests/conformance/`) replays every recorded exchange through the emulator's HTTP
surface. It compares the status, the recorded headers, the body and, where recorded, the raw bytes. The
staleness check recomputes each recipe's **behaviour fingerprint**
(`rcp_ndcg_test.fingerprint.behaviour_fingerprint`, rule `rcp-fp/4`). The fingerprint has these inputs: the checkpoint id and revision. It also has the engine image, its version floor and the serve block. It also has the source hash of every plugin module that the recipe's engine runs. That includes its declared architectures' modules and its opted-in patches'. The template file's bytes, the tokenizer's SHA-256 and the client fields that change the request bytes are inputs too. The
check fails and **names the changed inputs** when no committed corpus carries the fingerprint. The only way
past is a dated, reasoned, unexpired entry in `tests/conformance/waivers.json`. The release checklist requires
that file to be empty. A corpus left stale by a recipe change and awaiting its re-recording is declared in `tests/conformance/stale.json`. The entry gives the recipe, the recorded fingerprint, the inputs that moved, why and when. The
replays skip it. The suite checks that it fails the staleness gate naming exactly those inputs. The release
checklist requires that list empty too, and `RCP_NDCG_RELEASE=1` makes the suite enforce it. The way to empty
it is a new recording (the wave runner's `--changed-since`), never an edit. A corpus whose moved inputs shape none of its recorded exchanges is re-keyed instead. Such inputs are metadata only, such as a pooler setting that restates the engine's default. Its manifest carries the current fingerprint and a `rekeyed` entry that names what moved and
why. A re-key touches no record, by this procedure:

1. Recompute the recipe's inputs with `rcp_ndcg_test.fingerprint.fingerprint_inputs`. Compare them with the
   manifest's `recipe.fingerprint_inputs` (`fingerprint_changes`). Go on only when the changed names are
   exactly the ones you declare metadata-only.
2. In `manifest.json`, set `recipe.behaviour_fingerprint` and `recipe.fingerprint_inputs` to the current
   values. Append to `recipe.rekeyed` an entry with `from_behaviour_fingerprint`, `from_manifest_sha256` and `from_manifest_file_sha256`. Those are the old manifest's digest and file SHA-256. They keep the link to the full corpus that the subset was cut from. The entry also has `changed_inputs` (each name with its old and new value), `reason` and `date`.
3. Recompute `integrity.manifest_sha256` with `rcp_ndcg_test.corpus.manifest_digest`. Then recompute the
   corpus's `index.json` (`full_corpus.manifest_sha256` and `manifest_file_sha256`, the file's SHA-256).
4. `git mv` the directory to the new fingerprint. Move its entry in the engine's `index.json`
   (`<recipe>/<fingerprint>`, with the new `behaviour_fingerprint` and `manifest_sha256`).

The conformance suite then replays the corpus under its new key. A re-key whose changed inputs did shape a
recorded exchange fails that replay. `RCP_APPEND_VERIFICATION=1` appends the suite's result to every corpus's
verification record.

A committed manifest is a **redacted view** of the recording when the recording carried private infrastructure. The tracked tree may not name that infrastructure (AGENTS.md: public names only). Examples are the pod's Kubernetes Service
links in `engine.env`, or the pod's staging directory in a `serve_argv` path. The store's file stays the
original. The tracked file says what was removed, so nothing is silent. A redaction touches no record, by this
procedure:

1. Remove the field from `manifest.json`. It can be the environment keys that the Service links injected. It can also be the staging directory in a `--chat-template` value. Reduce it to its file name as the family goldens do it. Append to `engine.redacted` an entry with `field_path`, what was removed, `reason`, `date`, `from_manifest_sha256` and `from_manifest_file_sha256`. Those are the original manifest's digest and file SHA-256, from git. They keep the link to the store's file that the subset was cut from.
2. Recompute `integrity.manifest_sha256` with `rcp_ndcg_test.corpus.manifest_digest`. Then recompute the corpus's `index.json`. There, `full_corpus.manifest_file_sha256` is the tracked file's SHA-256 and `full_corpus.manifest_sha256` its digest, the fields that the conformance suite's integrity check reads.
   Then recompute the engine's `index.json` (`write_corpus_index`). The `gs://` pointer becomes the
   `gs://YOUR-BUCKET/...` placeholder (`docs/how-to/release-candidates.md`: a tracked file never names a real
   one).

A recording made after the fix needs no redaction. The engine environment's own filter
(`rcp_ndcg_test.observe.provenance`) keeps the Service links out.
`test_no_committed_corpus_names_private_infrastructure` holds the committed bytes to that.

### Behaviour fingerprint versions

The fingerprint rule's version is an input of its own (`fingerprint_schema`). Thus two corpora keyed by
different rules never collide. `rcp-fp/4` is current. It keys the engine image and version floor
(`engine.image`, `engine.min_version`) and the plugin code that the recipe's engine runs
(`plugin_sha256.<module>`). That plugin code is the shared entry modules and every declared `serve.plugin_architectures` module. It also includes every module that an opted-in `serve.patches` patch runs (its own module and the backend it registers). These inputs sit beside version 3's inputs. The rule bumps to `rcp-fp/5` when the input set or its canonicalisation changes. A change is adding or removing an input, or moving a field between the request-shaping and reply-processing classes. A bump is deliberate and recorded here, in the fingerprint
module and in the CHANGELOG. Every committed corpus and every per-variant golden is then either re-keyed or re-recorded on the next wave. A re-key is allowed only when the moved inputs shaped none of the recorded exchanges.
`tests/conformance/stale.json` declares the ones awaiting their recording.

The golden replays (`tests/e2e/test_golden_replay.py`) run a NanoBEIR-shaped and a ViDoRe-shaped mini through
rcp-ndcg's retrieval and rerank path against the emulators. They pin nDCG@10 and RCP-nDCG@10 to 1e-9. Every
input must be observed, or the run fails. The pins are **regression pins** computed by this code from the
committed corpora (`kind: regression-pin` in `tests/e2e/golden/goldens.json`). They are not independent GPU-run
numbers. The ViDoRe retrieval view is waived until a corpus observes a page image.

<!-- snippet: skip (a repository checkout with the corpora) -->
```bash
RCP_UPDATE_GOLDENS=1 uv run --no-sync pytest tests/e2e/test_golden_replay.py  # recompute the pins
RCP_APPEND_VERIFICATION=1 uv run --no-sync pytest tests/conformance -k verification_record
```

## What moved between two corpora

`rcp-ndcg-vllm`'s `python -m rcp_ndcg_test.changes` reports change handling (OBSERVATIONS-SPEC section 7).
`changed` is the re-record-changed-only selection: per recipe, `unchanged`, `changed`, `new` or `unloadable`,
with the changed inputs named. The wave runner's `--changed-since` mode records the `changed` and `new` ones.
`diff` writes the behaviour diff of two corpora of one recipe. Per input, it gives the score or vector deltas and changed statuses. It also gives changed protocol behaviour, such as headers, framing and error bodies. It summarises them by route. Use it for
review before a new corpus replaces the old.

<!-- snippet: skip (a repository checkout with the corpora; the path is the v0.31.0 recording's until the re-record) -->
```bash
python -m rcp_ndcg_test.changes changed --recipes-root rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes \
    --corpora-root rcp-ndcg-test/corpora/vllm-0.31.0  # renamed to vllm-0.33.0 after the re-record
python -m rcp_ndcg_test.changes diff --before <corpus-dir> --after <corpus-dir>
```
