# rcp-ndcg-test

The validation tooling for [rcp-ndcg](https://github.com/cohere-ai/rcp-ndcg) models served with
[vLLM](https://docs.vllm.ai): reference cases and the conformance suite (every case through the product's role
clients, never raw HTTP and never a copy of the client), the equivalence harness, the engine recorder, the
**verified emulators** replayed from the observation corpora, and the GPU wave runner.

**Unpublished on purpose.** This package is never uploaded to PyPI: the repository's CI, the product's
pytest suite and the GPU waves install it from the uv workspace (the root's `dev` dependency group) or
from the staged wheelhouse; anywhere else it installs **from a git subdirectory** (owner decision 22):

```bash
pip install "rcp-ndcg-test @ git+https://github.com/cohere-ai/rcp-ndcg.git@v0.0.1#subdirectory=rcp-ndcg-test"
```

No published package names it, and the release workflow builds the three
published distributions by name, so it is never built or released with them. It depends on `rcp-ndcg` and
`rcp-ndcg-vllm` at the release's version — never the other way round.

## What a case is

One case per file, `cases/<recipe-id>/<case-slug>.yaml` (the recipe id is the canonical one from the
serving-recipes package's `recipes/` root):

```yaml
id: qwen3-embedding-0.6b/card-asymmetry
recipe: qwen3-embedding-0.6b
role: embed
source:
  kind: model_card                      # model_card | generated
  url: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
  revision: "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"   # 40-hex commit of the README
  section: "Transformers Usage"
  quote: |                              # the exact lines of the card's example, verbatim
    query = "What is the capital of China?"
    ...
    embeddings[0][:5]
strata:
  modality: text                        # text | image | video | mixed
  length: short                         # short | long_under | long_over | mixed
  batch: single                         # single | uniform | mixed_length | mixed_modality
inputs:
  instruction: "Given a web search query, retrieve relevant passages"
  queries:   [{id: q1, text: "What is the capital of China?"}]
  documents: [{id: d1, text: "The capital of China is Beijing."}]
expected:
  kind: similarity_matrix               # similarity_matrix | scores | ranking | none
  values: [[0.76]]                      # null while the GPU wave has not filled it
  tolerance: {abs: 5e-4}
  origin: published                     # published | reference | engine
  status: reproduced                    # published_unverified | reproduced | pending_gpu
notes: "the card prints 4 decimals; a bf16 margin is declared"
```

A `model_card` case copies its inputs and printed outputs verbatim from the card at the pinned README
revision (the tolerance reflects the card's rounding plus a declared margin). If the card prints no
outputs, the case declares `expected.kind: none` (it exercises the path only) or `ranking` derived from
the card's text — say which in `notes`. A `generated` case fills a
stratum the cards do not cover: every recipe needs at least one case per applicable cell of
modality × length × batch — `short`, `long_under` (within 5% under `client.max_tokens`, never cut),
`long_over` (over it; the budget cuts it — and the anchors must survive),
`mixed_length` batches, and for
vision-language recipes `image` / `video` (if the model takes video) and `mixed_modality` batches. The
labels are cross-checked against the case's own inputs at load (a `modality: image` case with no image
document, or a `mixed_length` batch whose text inputs all measure the same, is refused), so a mislabel
cannot satisfy the grid.
Generated cases start `origin: reference, status: pending_gpu, values: null`; the GPU waves fill them from
the reference implementation and the engine. `origin: engine` means one recorded run of the recipe's
engine produced the values -- for the *shipped fixture cases* that engine is their recipe's named
deterministic test fake (each fixture's `notes` names it), never a real model. Long inputs are deterministic (their construction is
recorded in `notes`), measured with the recipe's tokenizer; media files live under the case's
`media/` directory (`cases/<recipe-id>/media/<slug>.<ext>`), and nothing is fetched at test time. An
image media case carries its file to the wire through the adaptation step's `prepare_image` and out
as `image_url` content parts (one media item per encode call on the pooling route) — it *runs* on
the pool and rerank routes (the wire shapes whose clients prepare media). Two declared *skips*
(recorded before every send, each with its reason): a `video` case on every route (the product's
media lowering sends images; a video container is refused until a frames reader lands), and any media
case on the embed route (its `EmbeddingClient` takes text only and refuses media before
preparation).

A generated text may be stored **by reference** instead of literally: `text_ref:
{generator: <name>@<version>, params: {...}, sha256: <hex of the UTF-8 text>}` — usable wherever a case
has `text:` (the two are exclusive). The loader materializes the text with the named generator
(`rcp_ndcg_test.generators` — the lanes' own constructions, stdlib only, no network) at load and refuses
a hash mismatch with a typed error, so a mutated param or a changed generator fails the load instead of
silently testing different bytes. The lanes' generators are ported verbatim; a construction change is a
new generator version. `model_card` texts stay literal (the card's bytes are the case). The length strata are measured as the engine
sees the inputs — the product's `fit` renders each case (the recipe's template and the tokenizer's
specials included) and the stratum is judged on the rendered input: `long_under` must render whole within
5% under `client.max_tokens`, `long_over` must be cut by it, `short` must render whole.

### How to add one

1. Write `cases/<recipe-id>/<case-slug>.yaml` — the slug is the file stem, the directory is the recipe id.
2. Run the validator (from the repository root; it needs no GPU):

   ```bash
   uv run --no-sync python -c "
   from rcp_ndcg_test.cases import load_cases
   from rcp_ndcg_vllm.recipe import load_recipe
   recipe = load_recipe('<variant-id>')  # the family layout resolves the id (decision 34)
   bundle = load_cases('rcp-ndcg-test/cases', recipe)
   print(bundle)
   "
   ```

3. The product suite enforces the same thing on every push: `tests/test_cases_package.py` loads every
   case (file level everywhere, recipe level where the recipe exists; the measured token lengths run in
   the network-marked test, because a recipe whose tokenizer lives on the Hub cannot load offline), so a
   malformed case fails CI.

Media: only files whose licence allows redistribution in an Apache-2.0 repository; otherwise generate a
synthetic image or video with a script kept in the case directory as `media/make_media.py` and say so in
`notes`. Never touch another recipe's directory.

## The conformance suite

One runner, two targets, one typed report:

```python
import rcp_ndcg_test.fakes  # noqa: F401  (registers the shipped fixture fake)

from rcp_ndcg_test.cases import load_cases
from rcp_ndcg_test.conformance import run_suite
from rcp_ndcg_test.fakes import fake_engine_for, fixture_path
from rcp_ndcg_vllm.recipe import load_recipe

# The packaged fixture recipe and its shipped fake: the runner's end-to-end exercise, CPU only.
recipe = load_recipe(fixture_path("recipes", "fake-embed"))
bundle = load_cases(fixture_path("cases"), recipe, recipes_root=fixture_path("recipes"))
report = run_suite(recipe, bundle.cases, target="fake")
print(report.summary())
assert report.ok

# A live engine serving the recipe (a vLLM on a GPU slot):
# run_suite(recipe, bundle.cases, target="engine", base_url="http://127.0.0.1:8100/v1")
```

The report is per case (`CaseResult`: `compared`, `passed`, `skipped` with a reason, `detail` with the
worst delta) and per run (`ConformanceReport`: `ok`, `failures`, `skipped`, `summary`). `values: null`
(pending the GPU wave) is a skip, never a pass; `kind: none` runs the path and compares nothing; a
tolerance breach is a failure with the worst cell in its detail; a product error (an engine refusal, an
unusable reply) fails the case carrying the product's typed message.

Both targets send through the **product's role clients** built from the recipe's `client` block —
`EmbeddingClient`, `PoolingClient` or `RerankClient` over the product's adapter and transport. The
engine is reached with a real `httpx` pool; a fake sits *below* the transport (an `httpx` transport the
product's `Transport` sends through), so routing, retries and the adapters are the product's in both.
Both targets send the case's **raw** contents: the product's role client runs the recipe's own
tokenizer, template and `max_tokens` in its adaptation (the snapshot & send) step and names the fit in
`notes`; a recipe with per-side prompts reaches the wire untouched. Media documents go out as
`image_url` content parts prepared by the same step (see What a case is).

## Fake engines

`rcp_ndcg_test.fakes` is the seam a recipe-level fake implements: the protocol
(`recipe_id`, `name`, `handle(method, path, body) -> FakeReply`) and the registry by recipe id. A fake
must answer the role's route(s) plus `GET /models` and `POST /tokenize` (the engine's tokenization ground
truth, so anything that cross-checks the client's `fit` can do it offline). No model-level fake ships
yet — the verified emulators are built from the GPU recordings later. What ships is one registered test
fake for the packaged fixture recipe (`fake-embed`: recipe, tokenizer and cases under
`rcp_ndcg_test/fixtures/`, all in the wheel), registered on import; the package's own tests rebuild
deterministic pool and rerank test fakes for its `fake-pool` and `fake-rerank` fixture recipes
(`tests/fakes_fixture.py`):

```python
from rcp_ndcg_test.fakes import FakeEmbedEngine, fake_engine_for

engine = fake_engine_for("fake-embed")  # the shipped one
assert isinstance(engine, FakeEmbedEngine)
```

A test fake is a deterministic surrogate, clearly marked — a test that asserts numbers may only do so
against inputs whose answer it recorded.

## The pytest plugin

Opt-in, one import — installing the package never injects tests into another suite:

```python
import pytest

from rcp_ndcg_test.plugin import conformance_params


@pytest.mark.parametrize("run", conformance_params("fake"))
def test_conformance(run):
    run.assert_passes()
```

`conformance_params(target, *, recipes_root=None, cases_root=None, base_url=None, check_lengths=False)`
builds one param per case (id `<recipe>/<case-slug>`). A recipe whose target is not wired is silently
absent from the list (no fake registered for the recipe, or no recipe at the recipes root yet) — a suite
that must not collect green against a half-merged tree asserts on `load_cases`'s bundle instead. A
skipped case (pending values) becomes `pytest.skip` with its reason; a failed case an `AssertionError`
with the detail. Against a live engine from the CI job's environment variable:

<!-- snippet: skip (needs a live engine's URL) -->
```python
import os

import pytest

from rcp_ndcg_test.plugin import conformance_params


@pytest.mark.parametrize("run", conformance_params("engine", base_url=os.environ["RCP_NDCG_CONFORMANCE_URL"]))
def test_served_conformance(run):
    run.assert_passes()
```

The package's own tests run from the root venv (the dev dependency group installs it):

```bash
uv run --no-sync pytest rcp-ndcg-test/tests -q
```

The real cases under `cases/` are repository data, staged to a GPU node by the wave runner; the fixture
recipe, its tokenizer and its fixture cases are package data (in the wheel), so the suite runs end to end
anywhere the package is installed. See `docs/how-to/add-a-model.md` for the recipe side and
`docs/how-to` for the wider guides.

## The validation tooling (moved with the layout move)

The equivalence harness, the engine recorder and the GPU wave jobs live under `rcp_ndcg_test`:

- `rcp_ndcg_test.equivalence` — stages 1-3, the reference subprocess, the gates and the report
  (`python -m rcp_ndcg_test.equivalence --recipe ... `); the harness drives the product's role clients (R30),
  never a re-derived fit.
- `rcp_ndcg_test.record` — the engine recorder for the observation corpora; `rcp_ndcg_test.observe` — the
  request generator (`python -m rcp_ndcg_test.observe.requests`, pairs under `pairs/`), the media set, the
  provenance probe and the negative controls.
- `rcp_ndcg_test.jobs` — the wave runner (`run_wave.py`), the node bootstrap, the RC build, `submit.sh`,
  `wave0` and the T4 end-to-end driver (`python -m rcp_ndcg_test.e2e`, scenarios under `scenarios/`).
- `rcp_ndcg_test.engines` and `rcp_ndcg_test.corpus` — the **verified emulators** (one per engine version,
  recipe and behaviour fingerprint, replayed from `corpora/`) and the corpora's one reader. The product's
  `rcp_ndcg.inference.fake` routes `fake://<engine>-<version>/<recipe>` URLs here through the
  `rcp_ndcg.fake_transports` entry point.
- `rcp_ndcg_test.changes` and `rcp_ndcg_test.fingerprint` — the corpora's behaviour fingerprints, the
  staleness gate and the re-record selector; the corpora to re-record are declared in
  `tests/conformance/stale.json`.
