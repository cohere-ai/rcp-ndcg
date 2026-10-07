<!-- Handover copy of the operator's working note `research/docs-release/work/blueprint.md` (sanitized: private names, the private repository, the job platform and storage locations removed; `<operator-notes>/...` paths point to files that are NOT available — the content you need is in handover/). Treat it as the specification it was for its lane. -->

# docs-release — `blueprint.md`: the 0.0.1 release and contributor surface

Ref under review: `18c86862276e` (`rfc-0001`). Final state = that ref + the layout move
(`drafts/layout-move.md`), 18 GPU-validated recipes (`drafts/recipes.tsv`), the owner decisions in
`STATE.md`, and the queued fixes in `research/sweep/TRIAGE.md` (assumed to land). Paths below are the **final**
paths, i.e. after the move `packages/rcp-ndcg-core` → `rcp-ndcg-core`, root package → `rcp-ndcg`,
`packages/rcp-ndcg-vllm` → `rcp-ndcg-vllm`, `packages/rcp-ndcg-test` → `rcp-ndcg-test`. Where an edit depends on an
unresolved owner question it names the question and is written for the recommendation. Severity codes (F1…F12) point
into `report.md`.

---

## 1. Target structure

### 1.1 The file set and its audiences

| File (final path) | Audience | Purpose — the one thing it owns |
|---|---|---|
| `CHANGELOG.md` (repo root) | A user reading the release; an upgrader | The public surface stated in prose + what changed in each released version. The contract narrative beside `tests/contract/snapshots/` and `schemas/`. |
| `AGENTS.md` (repo root) | Contributors and coding agents changing this repo | The repo contract: layout, layering, one home per concept, the rules, the release procedure. |
| `REPRODUCIBILITY.md` (repo root) | A reader of the paper verifying or re-deriving its numbers | The three reproduction levels and the complete ledger of deviations from the paper's code. |
| `rcp-ndcg/README.md` | A new user at GitHub or at the PyPI page of `rcp-ndcg` | Landing narrative + install + the three ways in + serving pointers. This becomes the package's PyPI long description. |
| `rcp-ndcg/README.root.md` → repo-root `README.md` | Visitor of github.com/cohere-ai/rcp-ndcg | A short landing card: pitch, install one-liner, links to the four directories, docs, citation, license. (See Q1; written for the recommendation.) |
| `rcp-ndcg-core/README.md` | A metric-only user at PyPI | What `rcp-ndcg-core` is, install, one worked example, pointer to `rcp-ndcg`. |
| `rcp-ndcg-vllm/README.md` | An engine operator at PyPI | The lean serving package: `serve <recipe-id>`, the 18-recipe catalogue, the model plugins, the lean-install contract. |
| `rcp-ndcg-test/README.md` | Contributors only (unpublished distribution) | What the test package holds and how to run its two suites; the "not published" statement. |
| `LICENSE` + `NOTICE` in all four distributions | Anyone receiving a wheel/sdist | Apache-2.0 text + third-party attribution (one shared pair, byte-identical, per Q4). |
| `MANIFEST.in` per published distribution | The release build | sdist contents, in-tree paths only (F4). |
| `CITATION.cff` (repo root) | Citing users; GitHub "Cite this repository" | The paper citation (already `version: 0.0.1`). |

Boundary rules (one home per claim — the "one home" rule applied to docs):

- **Install commands live only in the owning README.** `rcp-ndcg/README.md` may show the combined client install;
  `rcp-ndcg-core/README.md` shows only `pip install rcp-ndcg-core`; `rcp-ndcg-vllm/README.md` shows the three
  contexts (engine `--no-deps`, client, plain Python). `REPRODUCIBILITY.md` shows install lines only for its own
  reproduction steps. Nothing re-worded in four places.
- **Meanings of public names live in the CHANGELOG of their release** ("Public surface"); their machine-pinned
  spellings live in `tests/contract/snapshots/` and `schemas/`; how-to prose lives in `docs/` (owned by the docs
  lanes — this blueprint only coordinates).
- **Deviations from the paper live in `REPRODUCIBILITY.md`**; CHANGELOG "Changed" links there and never re-explains
  numbers (F7).
- **Contributor rules live in `AGENTS.md` only**; never duplicated into `docs/` or the READMEs.
- **The recipe catalogue is data**: `rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/<id>/recipe.yaml` is the source of
  truth; the `rcp-ndcg-vllm/README.md` table renders `drafts/recipes.tsv`'s 18 canonical ids (id / model / role /
  input / plugin / status) and is the only rendered copy; `rcp-ndcg/README.md` and `docs/` link to it.

### 1.2 `CHANGELOG.md` — target outline

```
# Changelog
## Versioning                         — edited (F1): four distributions; lockstep versions; the surface definition
                                      names the three published ones; module list sourced from the release's
                                      python_api snapshot
## 0.0.1 — <release date>             — the only version heading (F1; replaces `## Unreleased` + `## 0.1.0`)
  ### Overview                        — from the 0.1.0 draft, rewritten for the final state
  ### Highlights                      — the first eight bullets a user reads
  ### Public surface                  — merged 0.1.0 "Public surface" + Unreleased "Public surface", final state only
      **Distributions.**              — rcp-ndcg-core, rcp-ndcg, rcp-ndcg-vllm (published, lockstep);
                                        rcp-ndcg-test (unpublished, no surface promise)
      **Python modules.**             — regenerated from the release's python_api snapshot (19 product modules
                                        today + rcp_ndcg_vllm's public modules per Q3)
      **Command line.**               — `mcp` (serve) only (F12)
      **Exit codes.**                 — kept from the 0.1.0 draft (unchanged: 0-12, 7 retired)
      **MCP tools.**                  — the `mcp serve` manifest; the `mcp tools` shell call ERASED here and
                                        recorded under Removed (F12)
      **JSON Schemas.**               — kept, plus the recipe schema if it ships (Q3)
      **Serving recipes.**            — 18 rows (the recipes.tsv columns) + the sentence "every shipped recipe
                                        passed the release's GPU validation waves" (Q5)
      **rcp-ndcg-test.**              — one paragraph: unpublished distribution; cases, conformance, fakes,
                                        equivalence, record, GPU jobs
  ### Fixed                           — the 0.1.0 tournament-fix narrative as the lead (its own two paragraphs +
                                        link to REPRODUCIBILITY), then the Unreleased "Fixed" bullets, deduped
  ### Changed                         — Unreleased "Changed" + the 0.1.0 "Changed from the paper's code: text
                                        limits count the judge's tokens" section merged, each external-behaviour
                                        change one bullet; links to REPRODUCIBILITY for every paper deviation
  ### Removed                         — Unreleased "Removed" + `rcp-ndcg mcp tools` + the `local`/`vllm` extras
                                        consolidation (already present, keep) + the plugin distributions (folded
                                        into rcp-ndcg-vllm)
  ### Security                        — kept verbatim from `## Unreleased`
```

What keeps / merges / drops (the mechanical rules for the docs lane):

1. **KEEP** (self-contained, true of the final tree): every `Fixed` bullet's outcome sentence; the tournament-fix
   narrative (`CHANGELOG.md:1089-1106`-area, i.e. the current `### Fixed: tournament answers the paper's code could
   not parse` body) and the `### Changed from the paper's code: text limits count the judge's tokens` body; the
   `Security` section (`CHANGELOG.md:809-817`); the `Removed` bullets that name real removals (in-process model
   paths, `--judge-urls`, `[local]`/`[vllm]` extras, the OpenAI SDK, pinned action tags).
2. **MERGE**: the at-least-6 "tests/contract snapshots and the exported schemas regenerated for X" bullets
   (`CHANGELOG.md:718, 722, 725, 733, 738, 768`-area) into **one** closing bullet under "Public surface": "The
   contract snapshots and the exported schemas record every name above; regenerate with `uv run pytest
   tests/contract --update-snapshots`." The three `JobSpec.phases` entries (`CHANGELOG.md:32, 184, 243`) merge into
   one "Public surface" bullet (the `32` wording wins: "takes exactly one of `argv` and `phases`", plus
   `with_argv`). Every pair of "(change)" + "(the schemas follow)" bullets merges to one.
3. **DROP** (interim or internal — F2): every clause describing a state the code moved past — "until the text-budget
   mechanism is wired" (`CHANGELOG.md:273`), "the interim refusal of `max_tokens` is gone" (`CHANGELOG.md:70`), "no
   transport behaviour yet" (`CHANGELOG.md:246`), "the judge client adopts it later" (`CHANGELOG.md:54`); every
   internal review id — 11 lines cite ids like `(R5/R14/R15)`, `(R6)`, `(item 4)`, `(F7)` (command:
   `grep -nE '\b(R[0-9]{1,2}|F[0-9]{1,2}|item[s]? [0-9])\b' CHANGELOG.md` → 11): delete the parenthetical, keep the
   substance; lane bookkeeping ("which its own commit left out of the snapshot", `CHANGELOG.md:725`);
   `## 0.1.0` heading and everything in it that the new entry does not carry over (it was never released — see 1.3).
4. **CORRECT against the final tree**: the module list (source: the release's `tests/contract/snapshots/python_api.json`;
   the current draft says "Eighteen" and omits `rcp_ndcg.inference`, `CHANGELOG.md:830-834` — the count is 19 today
   per `tests/contract/surface.py:166-185`); retrieval configs spelled `api:` not `provider:` (the draft's
   `rcp_ndcg.retrieval` paragraph, `CHANGELOG.md:944`-area, still lists `provider:`, `Local`, `LocalEncoder`); no
   extras named `local` or `vllm` in the Distributions paragraph (`CHANGELOG.md:859`); `run resume --judge-urls`
   only under Removed, not in the CLI paragraph (`CHANGELOG.md:1052`).

How the earlier `## 0.1.0` section relates (the owner's question): it is the pre-unified-inference draft of the
first release's notes, written when the version was going to be 0.1.0 and the API described (`provider:`
discriminators, in-process `Local`, `mcp tools`, `--judge-urls`) was the shipping one. None of it shipped. It has no
version-number meaning — `0.1.0` never existed as code (`pyproject.toml` says `0.0.1` at every ref since; known item
14 shows the same drift in a snapshot). Therefore: **dissolve it into `## 0.0.1`** (its Overview and per-module
listing are the skeleton of the new entry), do not keep it as a historical entry, and let Git history preserve the
draft. Rationale for the docs lane to quote: a changelog version that was never released misleads every dependency
bot and upgrader; `CHANGELOG.md:3-22`'s own versioning rules are about released changes.

### 1.3 `AGENTS.md` — target outline

1. **Title + pitch** (kept as is, lines 1-4).
2. **Set up, check, test** — kept; edit line 11's comment to "all four packages' `src/` trees"; add one line:
   `uv run pytest tests/conformance` is not a root command — conformance lives in `rcp-ndcg-test` and runs per its
   README. (Nothing else moves: `uv sync --locked --extra dev` from the root still sets up the workspace.)
3. **Layout and layering** — rewrite the bullets to the four-distribution tree of `drafts/layout-move.md`:
   `rcp-ndcg-core/`, `rcp-ndcg/`, `rcp-ndcg-vllm/`, `rcp-ndcg-test/`, root = uv workspace + `docs/`, `experiments/`,
   `examples/`, `skills/`, `schemas/`, `.github/`. Layering: `rcp_ndcg_core → support → storage → data → inference →
   retrieval → llm → calibration → eval → runners → runs → schemas | mcp → cli` (kept) plus the two cross-package
   rules verbatim from layout-move items 4-5: "`rcp-ndcg` may import `rcp_ndcg_vllm`'s recipe data lazily
   (`recipe:<id>` resolution, missing package = typed error with the install hint); `rcp-ndcg-vllm` never imports
   `rcp-ndcg`, torch or the engine — its dependencies are pydantic and PyYAML"; "`rcp-ndcg-test` imports the
   product and the serving package; nothing imports it". Entry points: `rcp_ndcg.runners`, `rcp_ndcg.adapters`, and
   `vllm.general_plugins` (one lazy entry in `rcp-ndcg-vllm`).
4. **One home per concept** — keep the table, add four rows:
   | Serving recipes, the recipe schema, `serve` | `rcp_ndcg_vllm` |
   | Model plugins for served checkpoints | `rcp_ndcg_vllm.models/` |
   | Reference cases, conformance, model-level fakes, equivalence/recording/GPU job tooling | `rcp_ndcg_test` |
   | Judge/role text budgets, templates and their cut policy | `rcp_ndcg.data.preprocess`, `rcp_ndcg.data.templates` |
5. **Rules** — keep all nine bullets; add six (wording fixed here so the docs lane does not re-decide):
   - **Explicit budgets.** A self-hosted role config declares `tokenizer` + `max_tokens`; a hosted vendor profile
     without a tokenizer declares `budget_source: vendor` and sends content uncut. Over budget, the default is
     `cut`, recorded in the census; chunk aggregation is `max`; media are counted in tokens, never money.
   - **Anchors are reserved, never engine-side cut.** Client-side cuts are anchor-preserving; the paper code's
     anchor drops are declared `known_deviations` in a recipe and compared under the cap only.
   - **Recipe ids are the lowercased canonical Hub repo name** (never a redirecting short name), pinned by a test;
     a recipe's CHANGELOG bullets fold into the one release entry.
   - **Every recipe ships validated.** A recipe merges only with its GPU waves green (one 0.0.1 with everything);
     `status: unverified` does not ship in a tag.
   - **No GPU pytests.** GPU work produces observation corpora per `OBSERVATIONS-SPEC`; verified fake engines on
     CPU with conformance + golden replays stand in for tests. Model-level fakes live in `rcp-ndcg-test`; the
     generic `fake://` stays in the product.
   - **Versions move together.** All four distributions carry the tag version; `rcp-ndcg` pins
     `rcp-ndcg-core==<version>`; `rcp-ndcg-test` is never published.
6. **Releasing** — keep the trusted-publishing prose and the environment table unchanged; rewrite the paths:
   "the version is that of `rcp-ndcg/`, `rcp-ndcg-core/` and `rcp-ndcg-vllm/`'s `pyproject.toml` (and
   `rcp-ndcg-test/`'s, unpublished)"; publish order `core → rcp-ndcg → vllm` (from `RELEASE_PUBLISH_JOBS`' `needs`
   chain, `tests/docs/test_packaging.py:139-188`); delete "from its own directory: it is deliberately outside the
   uv workspace" (the move adds it to the workspace members); add the release gates from layout-move's gate line:
   fresh-venv wheel install of each dist, `pip install --no-deps` freeze-delta check for `rcp-ndcg-vllm` in an env
   that has only pydantic/PyYAML, `rcp-ndcg-vllm serve <id> --dry-run` for every recipe, `twine check`.
7. **Where to look** — add "the shipped recipes: `rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/`, catalogued in the
   `rcp-ndcg-vllm` README".

### 1.4 `REPRODUCIBILITY.md` — target outline

1. Intro paragraph — kept.
2. `## 1. Recompute the paper's tables from the released data (no LLM)` — kept verbatim except line 14:
   `pip install ./rcp-ndcg-core ./rcp-ndcg` (or `uv sync`) (F4). No extras here: this level "needs no GPU, no LLM
   and no credentials" (`REPRODUCIBILITY.md:10-11`) and refits nothing — verifier S9 rejected my earlier
   `[calibrate]` wording as importing torch into the cheapest path.
3. `## 2. Score your own system like the paper (no LLM)` — kept verbatim.
4. `## 3. Re-judge a pool with your own LLM endpoint` — **heading unchanged** (deep-linked from
   `CHANGELOG.md:1105`, F11). Body edits: line 55's path becomes `rcp-ndcg/src/rcp_ndcg/llm/judges/`; add one bullet:
   "The first-stage retrievers and rerankers of the paper's runs are served from the shipped recipes:
   `rcp-ndcg-vllm serve <recipe-id>` builds the engine command for the stock `vllm/vllm-openai:v0.31.0` image, and a
   run config's `recipe:<id>` reads the recipe's client block (18 recipes, each validated end to end on GPU in the
   release's waves; see the `rcp-ndcg-vllm` README)."
5. **NEW `## Deviations from the paper's code`** — the complete ledger (purpose: a reproducer must be able to name
   every cause of a moved number). Four subsections, fixed wording:
   - *Tournament answers* — "See [Tournament answers the paper's code could not parse](#tournament-answers-the-papers-code-could-not-parse)."
   - *Text limits* — "See section 3: this package counts the judge's tokens, the paper's code counted characters."
   - *Over-cap truncation preserves the anchor* — "The paper's code cut document text without regard to where the
     model reads its answer (dropping e.g. the trailing assistant header of a last-token reranker). This package
     reserves the template's anchors before cutting and re-attaches the frame after. A recipe declares
     `reference.known_deviations: [anchor_drop_over_cap]` for the models whose paper run used the anchor-dropping
     cut, and the equivalence gates compare under-cap pairs only. Only an over-cap document can end at a different
     place." (Source: `rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/zerank-1-reranker/reference.py` docstring + STATE
     owner decisions.)
   - *Judging defaults* — "The judge's documents text policy defaults to 32768 tokens (`on_overflow` default `cut`,
     every cut recorded; chunk aggregation `max`; media counted in tokens). Every shipped preset and paper config
     that relied on the previous 20000-token default pins it explicitly, so no judgement identity moves."
     (Source: STATE, Owner decisions 13:12.)
6. `## Tournament answers the paper's code could not parse` — **heading + body kept verbatim** (deep-linked from
   `CHANGELOG.md:1098`; it is the receipt for the forthcoming paper correction).
7. `## Tests` — kept; one added line: conformance and the fake-engine replays live in `rcp-ndcg-test` (unpublished)
   and run per its README.

### 1.5 The READMEs (PyPI pages)

**A. `rcp-ndcg/README.md`** (= today's root `README.md` moved; the PyPI long description of `rcp-ndcg`). Keep the
whole skeleton (logo, pitch, pipeline figure, Datasets table, Three ways in, MTEB, Documentation, Citation,
License). Edits only where the final state changed (all in the edit list):
- the "This repository holds…" sentence becomes the four-distribution sentence;
- Install: mention all three published packages and the two install contexts (client vs engine node), fix the git
  subdirectory paths (F4), keep the exit-10 sentence;
- "The runners' phased rendering is pending, and until it lands such runs are refused" (`README.md:120`) DELETED
  (queued fix: TRIAGE sweep-docs P-README; phased rendering exists since l4b), and "Until the release is up, or"
  deleted from `README.md:38-39` (keep "To work from the repository…": 0.0.1 IS the release, S10);
- new section `## Serving models` (before `## MTEB`): 18 recipes, `rcp-ndcg-vllm serve <recipe-id>` example on the
  stock image, `recipe:<id>` in a run config's retrieval fields, link to the `rcp-ndcg-vllm` PyPI page and its
  catalogue table (absolute URLs — the PyPI rule of `tests/docs/test_readme_pypi.py`);
- the skills bullet: "`mcp serve` exposes it as MCP tools" kept (correct), no `mcp tools` text (F12);
- License section: "...see [LICENSE]... Each distribution carries LICENSE and NOTICE" + one clause.

Repo root `README.md` (landing card, per Q1): ~25 lines — logo, the pitch paragraph and paper link (shared
sentences with A, allowed as the one duplication: a landing card), `pip install "rcp-ndcg[hf,calibrate]"` one-liner,
a four-row table of the directories (rcp-ndcg-core / rcp-ndcg / rcp-ndcg-vllm / rcp-ndcg-test with one line each and
links to their READMEs), links to docs, REPRODUCIBILITY, CHANGELOG, AGENTS, CITATION, license line.

**B. `rcp-ndcg-core/README.md`** — keep as is in structure and example; three edits: (1) the trailing sentence
"the `rcp-ndcg` package in the same repository" → "the `rcp-ndcg` package"; (2) one added sentence: "`rcp-ndcg`
pins `rcp-ndcg-core==<version>`; the two release in lockstep." (3) re-verify the example against the release's
python_api snapshot (it already matches names in `rcp_ndcg_core.__all__` — keep the code block, it is run by
`tests/docs`).

**C. `rcp-ndcg-vllm/README.md`** — full rewrite (the current draft on `lane/harness` describes the pre-move package
with the harness inside and a hard `rcp-ndcg` dependency; both are gone). Fixed outline:
1. `# rcp-ndcg-vllm` — "The serving half of RCP-nDCG: 18 vetted serving recipes for retrieval models, the
   `rcp-ndcg-vllm serve` command that turns one into a `vllm serve` command for the stock
   `vllm/vllm-openai:v0.31.0` image, and the model plugins that make two released checkpoints serveable on it."
2. `## Install` — three contexts (fixed wording):
   - engine environment: `pip install --no-deps rcp-ndcg-vllm` — "the wheel's only dependencies are pydantic and
     PyYAML, which the image ships; a `pip freeze` before and after differs by exactly this wheel";
   - client environment: `pip install rcp-ndcg rcp-ndcg-vllm` — the second package resolves `recipe:<id>` and runs
     `serve`;
   - recipe data only: `import rcp_ndcg_vllm` never imports vLLM, torch or `rcp-ndcg`.
3. `## Serve a recipe` — `rcp-ndcg-vllm serve <recipe-id> [--port ...]` builds the `vllm serve` argv (the chat
   template from the recipe's package data, the media flags, the pooler config) and execs it; `--dry-run` prints
   the argv and exits; a model needing a plugin is refused with the exact install line. One complete example with a
   public id from the catalogue.
4. `## The recipes` — the 18-row table (columns: recipe id, model, role, input, plugin, status) rendered from
   `drafts/recipes.tsv` canonical ids; the caption sentence: "Every recipe in this table was validated end to end on
   GPU against its reference implementation before v0.0.1 (equivalence, quality and end-to-end waves); each
   `recipe.yaml` records its model revision and `sources`." Each row links to its `recipe.yaml` in the repository.
5. `## Model plugins` — `topk-embed-v1-small` and `pplx-embed-v2-context-9b-preview` fold into
   `rcp_ndcg_vllm/models/` under one `vllm.general_plugins` entry point; registration is lazy
   (`"module:Class"` strings — importing this package never imports vLLM or torch); one version guard pins the
   tested vLLM line (`>=0.31,<0.32`) and refuses others loudly.
6. `## What this package is not` — judging, pipelines and the metric are `rcp-ndcg` / `rcp-ndcg-core`; the
   equivalence harness, recorder, cases and GPU job tooling are `rcp-ndcg-test` (unpublished). "The engine is
   reached over HTTP only; this package never imports `rcp-ndcg`. A recipe's `client` block is plain data,
   validated when `rcp-ndcg` reads it."
7. `## License` — Apache-2.0; `LICENSE` and `NOTICE` ship in every wheel and sdist.

**D. `rcp-ndcg-test/README.md`** — one screen (fixed outline): what it holds (`src/rcp_ndcg_test/`: `cases/` +
generated strata per `case-format.md`, `conformance/` (one suite, two targets: live engine | recipe-level fake
engine, through the product's role clients), `fakes/` (model-level fakes built from GPU recordings),
`equivalence/`, `record/`, `jobs/` (wave runner, bootstrap, submit); `cases/` data at the package top); how to run
(CPU conformance always; GPU waves on the release node layout — engine/client/reference environments never mixed);
"rcp-ndcg-test is intentionally not published to PyPI"; license line.

### 1.6 LICENSE / NOTICE / MANIFEST coverage

- One `LICENSE` (Apache-2.0) and one `NOTICE`, byte-identical in all four distributions (Q4 recommendation; keeps
  the current test design of `tests/docs/test_packaging.py:282-290`, extended to four directories — F9).
- Every `pyproject.toml` keeps `license = "Apache-2.0"` and `license-files = ["LICENSE", "NOTICE"]` (root and core
  already do, `pyproject.toml:13-14`; `rcp-ndcg-vllm` on `lane/harness` does; `rcp-ndcg-test` must).
- The shared `NOTICE` content after the move (F3, F10):
  - delete the "contains no ported third-party code" stance — it lives on the plugin lanes' and rec-zerank-1's
    copies and is false the moment the fold lands (verifier receipts in item 13);
  - add the `vLLM` attributions from `lane/plug-topk`'s plugin NOTICE re-pointed to
    `rcp-ndcg-vllm/src/rcp_ndcg_vllm/models/…`, plus `lane/plug-pplx`'s thin-subclass admission ("a thin subclass
    of vLLM's own `Qwen3_5ForCausalLMBase`");
  - extend the `ZeroEntropy zerank` and `Qwen3-Reranker` entries to name
    `rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/<id>/reference.py` beside `experiments/paper/...`; walk the other 16
    recipes' reference headers and add an entry for every upstream family a reference adapts (the recipe template
    requires `sources` provenance, so this is a mechanical pass);
  - re-point `src/transformers...`-derived paths: `src/rcp_ndcg/data/resolution.py` →
    `rcp-ndcg/src/rcp_ndcg/data/resolution.py` (`tests/data/_media_reference.py` and `experiments/...` stay).
- `MANIFEST.in` per published distribution, in-tree paths only (F4; note Q1 decisions): `rcp-ndcg/MANIFEST.in`
  includes `LICENSE NOTICE README.md` (+ `CHANGELOG.md` copy if Q1 says the sdist carries it) and
  `recursive-include src/rcp_ndcg …` + `recursive-include schemas *.json` **only if** Q1 moves `schemas/` in-tree;
  `rcp-ndcg-core/MANIFEST.in` (new; core currently relies on defaults — adds LICENSE/NOTICE/README to the sdist
  explicitly); `rcp-ndcg-vllm/MANIFEST.in` grafts `src/rcp_ndcg_vllm/recipes`, its schemas and the plugins' moved
  code. `prune` nothing outside the tree (impossible to reference anyway).
- Release artifact check in the gate (already in AGENTS after edit 2.x): unzip each wheel and list
  `*.dist-info/licenses/`; build each sdist and grep its file list for LICENSE/NOTICE (the "nothing cut silently"
  rule applied to packaging).

---

## 2. Numbered edit list

Each item: **file — what to change — why (source of truth)**. Items marked *(tests)* are the checks that keep the
docs honest; a docs lane executes the prose items and pairs with a test lane for the code items where noted.

1. **`CHANGELOG.md`** — replace `## Unreleased` + `## 0.1.0` with one `## 0.0.1 — <release date>` entry built per
   outline 1.2; apply the keep/merge/drop/correct rules (1.2.1-4). *Why:* F1/F2; no version was ever released as
   0.1.0 (`pyproject.toml` versions; `CHANGELOG.md:819` vs `CHANGELOG.md:2`). Source: STATE ("one 0.0.1 with
   everything"), `CHANGELOG.md:3-22`'s own versioning rules. Verifier additions (S3/S4/S11): also execute items
   23 (TRIAGE/queued-fix reconciliation incl. the owner-mandated 32768-token CHANGELOG bullet and the Security
   addition) and 24 (extended drop/correct lists); the Overview rewrite qualifies "ships as two distributions" →
   four (three published) and "starts, builds and configures no engine" → the judging package builds none, while
   `rcp-ndcg-vllm serve` builds the retrieval engine command (`CHANGELOG.md:826, 836`).
2. **`CHANGELOG.md` `## Versioning`** — "…the two are released together" → "the four distributions carry one
   version; `rcp-ndcg` pins `rcp-ndcg-core==<version>`; `rcp-ndcg-vllm` carries the same version; `rcp-ndcg-test`
   is never published". The public-surface definition names the three published distributions and adds
   `rcp_ndcg_vllm`'s modules to the `PUBLIC_MODULES` sentence (Q3). *Why:* F1/F6. Source: `layout-move.md` item 6,
   `tests/contract/surface.py:35` (as it must become).
3. **`CHANGELOG.md`, `Public surface` per-module listing** — regenerate from the release's
   `tests/contract/snapshots/python_api.json`; fix known wrongs first: count 19 not "Eighteen" (`CHANGELOG.md:830`),
   add `rcp_ndcg.inference` (absent at `CHANGELOG.md:831-834`, present at `tests/contract/surface.py:180`; the
   draft in fact lists 17 of the 19 — it also omits `rcp_ndcg_core` itself), replace
   every `provider:`/`Local`/`LocalEncoder` retrieval statement (`CHANGELOG.md:938`) with the `api:`-discriminated
   config sentence already written at `CHANGELOG.md:110-118`. *Why:* F1/F2. Source: the snapshot (the CHANGELOG's own
   definition of the surface, `CHANGELOG.md:9-13`).
4. **`CHANGELOG.md`, Distributions paragraph** (old `CHANGELOG.md:856-860`) — three published distributions
   (+ one unpublished), extras `hf`, `calibrate`, `mteb`, `data`, `s3`, `azure`, `http`, `dev`, `docs`
   (no `local`, `vllm`; removal recorded at `CHANGELOG.md:795`); license sentence unchanged. *Why:* F2 (`local`/`vllm`
   extras are under Removed). Source:
   `pyproject.toml` `[project.optional-dependencies]`, `CHANGELOG.md:795`.
5. **`CHANGELOG.md`, `Removed`** — add: "`rcp-ndcg mcp tools` is gone; `rcp-ndcg mcp serve` stays"; add: "the
   separate plugin distributions are folded into `rcp-ndcg-vllm`". *Why:* F12/F3. Source: owner decision (5) in
   STATE; `layout-move.md` item 3.
6. **`REPRODUCIBILITY.md`** — as outline 1.4: fix line 14's install, fix line 55's judges path, add the recipe
   bullet in section 3, insert the new `## Deviations from the paper's code` section (four subsections, wording
   fixed in 1.4.5), keep the two deep-linked headings verbatim. *Why:* F4/F7/F11. Source: STATE owner decisions
   13:12 and the dated ones; `rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/zerank-1-reranker/reference.py` docstring
   (`known_deviations`, under-cap gating).
7. **`rcp-ndcg/README.md`** (moved root README) — as outline 1.5.A: four-distribution sentence; install block with
   `./rcp-ndcg-core "./rcp-ndcg[hf,calibrate]"` and git `#subdirectory=rcp-ndcg-core` / `#subdirectory=rcp-ndcg`;
   delete the "phased rendering is pending… are refused" clause at line 119-121; new `## Serving models` section
   (18 recipes, `rcp-ndcg-vllm serve`, `recipe:<id>`, absolute links to the `rcp-ndcg-vllm` page); license line
   mentions NOTICE. *Why:* F4 + TRIAGE P-README (queued) + the recipes have no user-facing home today. Source:
   `layout-move.md` items 2/5, `drafts/recipes.tsv`.
8. **repo-root `README.md`** — the landing card per 1.5.A-final (Q1 assumption). *Why:* F4 — the root is no longer
   a package, so the old git-install promise at `README.md:41-48` cannot hold there. Source: `layout-move.md`
   root-tree line.
9. **`rcp-ndcg-core/README.md`** — the three edits of 1.5.B. *Why:* the lockstep sentence is the one install fact a
   core-only user needs. Source: `pyproject.toml` pin `rcp-ndcg-core==0.0.1`.
10. **`rcp-ndcg-vllm/README.md`** — full PyPI-page rewrite per outline 1.5.C (replace the `lane/harness` draft,
    which promises the harness and a hard `rcp-ndcg` dependency). Verifier additions (S5/S7): this edit runs AFTER
    (supersedes) any TRIAGE-F2 "amend the package README" change (see item 25); it also fixes the README-adjacent
    `rcp-ndcg-vllm/pyproject.toml` `description` (the PyPI summary still promises "the equivalence harness, the
    engine recorder and the GPU wave runner") — covered in item 27. *Why:* F3-adjacent correctness of the published
    page; the harness/dependency statements become false at the move. Source: `layout-move.md` items 2-4;
    `drafts/recipes.tsv`.
11. **`rcp-ndcg-test/README.md`** — new, per outline 1.5.D. *Why:* an unpublished distribution with runner tooling
    needs one contributor-facing entry point. Source: `layout-move.md` item 4; STATE owner decision (3).
12. **`AGENTS.md`** — per outline 1.3 (sections 2-7 rewritten/bordered). *Why:* F5. Source: `layout-move.md` items
    1-6; STATE owner decisions (budgets, anchors, 32768, no GPU pytests, recipes validated, `mcp tools` removal).
13. **`LICENSE`, `NOTICE` (shared copy, ×4 locations)** — content per 1.6. Verifier receipts (S7/F3 correction):
    merge order is (1) start from `lane/harness`'s `packages/rcp-ndcg-vllm/NOTICE` (already a byte-copy of the root
    NOTICE), (2) add `lane/plug-topk`'s topk vLLM entries (`model.py`, `weights.py`) and `lane/plug-pplx`'s
    thin-subclass admission re-pointed to `rcp_ndcg_vllm/models/<name>/`, (3) delete the "contains no ported
    third-party code" sentence — it survives only on the plugin lanes' and rec-zerank-1's copies — and (4) keep
    "smart_resize" (pinned by `tests/docs/test_packaging.py:293`). *Why:* F3/F10; Apache-2.0 §4(d). Source:
    the named branch files, recipe `reference.py` headers, root `NOTICE`.
14. **`MANIFEST.in` ×3 published dists** — in-tree only; per 1.6. *Why:* F4 (silent sdist truncation). Source:
    `layout-move.md` item 2 ("the wheel ships them" = recipes as package data) and Q1.
15. **`CITATION.cff`** — no change (`version: 0.0.1` already); extended by item 28 (its `version:` joins the
    release checklist, and the sdist content set after the move is declared policy). *Why:* F4 edge. Source:
    `CITATION.cff:5`.
16. *(tests)* **`tests/docs/test_readme_pypi.py`** — scan every distribution README (root landing card,
    `rcp-ndcg/README.md`, `rcp-ndcg-core/README.md`, `rcp-ndcg-vllm/README.md`) instead of `ROOT/"README.md"` only;
    refined by item 26 (assert shaping for imageless pages). *Why:* F8. Source: `tests/docs/test_readme_pypi.py:57,65`.
17. *(tests)* **`tests/docs/test_packaging.py:282`** — extend and rename
    `test_both_distributions_ship_the_license_and_the_notice` to all four distribution directories (add explicit
    `license-files` assertion for `rcp-ndcg-vllm/` and `rcp-ndcg-test/`). *Why:* F9 (addendum to known item 5,
    which anticipated three — the fourth is `rcp-ndcg-test`). Source: `layout-move.md` tree + `tests/docs/test_packaging.py:282-290`.
18. *(tests)* **`tests/contract/surface.py`** — `PACKAGES` gains `rcp_ndcg_vllm` (public modules per Q3) and the
    snapshot regenerates; then edit list item 3 lists the same names. *Why:* F6. Source: layout-move items 2-3 (a
    new public CLI), `CHANGELOG.md:9-13`'s premise.
19. *(tests)* keep `tests/docs/test_links.py` green: no heading it resolves may move (REPRODUCIBILITY's two
    anchors); run `uv run pytest tests/docs tests/contract` after each batch. *Why:* F11. Source:
    `tests/docs/test_links.py:30`.
20. *(coordination, not this surface's files)* — `skills/rcp-ndcg/SKILL.md:198` and `docs/reference/cli.md:135`
    also describe `mcp tools`; the lane that lands p1-tail item 2j (now in fix-cli) removes them in the same
    change. Flag in that lane's brief. *Why:* F12. Source: owner decision (5).

— items 21-28 added after the verifier round (S1-S12 of `verifier-sweep.md`; every one reproduced there with
  command output) —

21. *(tests)* **`tests/docs/test_readme_datasets.py`, `tests/docs/test_snippets.py`** — re-point the hard-coded
    page lists the README split re-binds: `test_readme_datasets.py:26` (`["README.md", "docs/data.md"]`) must name
    `rcp-ndcg/README.md` (the landing card of item 8 has no Datasets table — decide once, recommended: the table
    stays on `rcp-ndcg/README.md` and the test names it); `test_snippets.py:20` `PAGES` must name the moved
    `rcp-ndcg-core/README.md`; `:57` `OTHER_PAGES` must name `rcp-ndcg/README.md` (today `ROOT/"README.md"` still
    *exists* post-move, so a mechanical rename silently re-binds the snippet-parity rule to the landing card).
    Give `rcp-ndcg-vllm/README.md`'s and `rcp-ndcg-test/README.md`'s Python/CLI blocks the `skip` marker (marker
    vocabulary `{None, skip, network, example}` unchanged) or keep those pages prose/bash only. *Why:* S1 — three
    more `tests/docs` lists silently govern the wrong pages otherwise. Source: the cited test lines.
22. **`requirements-constraints.txt`** — re-derive the header export command for the workspace-only root: package
    -select `rcp-ndcg` (`uv export --frozen --no-hashes --no-emit-workspace --no-dev --package rcp-ndcg --extra
    calibrate --extra hf --extra s3 --extra azure -o requirements-constraints.txt`; confirm the exact uv form
    empirically at the move), regenerate the file there (`uv.lock` changes), keep the four extras ==
    `COORDINATOR_EXTRAS` (`tests/docs/test_packaging.py:121`, `src/rcp_ndcg/runners/script.py:50`); the test at
    `tests/docs/test_packaging.py:124-137` re-runs the header command and the release gate checks the file against
    the lock. *Why:* S2 — the root loses `[project]` at the move, so `--extra` stops selecting what the header
    claims. Source: `requirements-constraints.txt:1-2`, the cited tests.
23. **`CHANGELOG.md` (part of item 1, listed for the mandatory reconciliation)** — walk
    `research/sweep/TRIAGE.md` (and the other queued briefs) item by item before the merge; every landed fix with
    a user-visible effect gets one bullet. First of them, mandated by the owner's wording (`TRIAGE.md:58`,
    "documented transparently (docs, docstring, **CHANGELOG**)"): the judge's documents text-policy default of
    32768 tokens + the explicit pinning of every existing preset/paper config at 20000 (no identity moves).
    `### Security` = kept verbatim **plus** one bullet for the key-scoping fix (`TRIAGE.md:52-55`: a profile's
    default key variables no longer travel to a foreign `base_url`). Fold `CHANGELOG.md:1107-1115` ("Also in this
    release": `experiments/`, `examples/`, judge configs, `serve/`, `skills/`) into Overview/Highlights. Re-verify
    the `eval score --system` wording (`CHANGELOG.md:153-154`, "exit 3") against the queued sweep-cli exit-class
    fix. *Why:* S3. Source: `TRIAGE.md`, the owner decisions.
24. **`CHANGELOG.md` (part of item 1: extended drop/correct lists)** — add these keep-list escapees to DROP:
    `CHANGELOG.md:310-311`, `:433`, `:754-755` (all "until the text-budget/retrieval port"); to CORRECT:
    `:782-785` and `:806-807` ("the last from its own directory, outside the uv workspace" / the vllm pin
    sentence — restate as "three distributions built from their per-distribution directories, publish order core →
    rcp-ndcg → vllm; `rcp-ndcg-vllm` declares no `rcp-ndcg` dependency"), `:943` (checkpoint-key prose superseded
    by the kept `:568-572` fix), `:826`, `:836` (see item 1). Mechanical pass:
    `grep -nE 'until|adopt|interim|later|outside the uv workspace' CHANGELOG.md` → drop;
    `grep -n 'two distributions\|no engine' CHANGELOG.md` → correct. *Why:* S4/S11.
25. *(coordination)* — TRIAGE F2's doc half ("amend … the package README" for the hard `rcp-ndcg` dependency) is
    **superseded by item 10**: `layout-move.md` items 2/5 delete that dependency. Re-scope the RFC §6.4 amendment
    to: "`rcp-ndcg` reads `rcp-ndcg-vllm`'s recipe data through a lazy import when it is installed; judges and
    engines are reached over HTTP only". Landing order: item 10 runs over any F2 text. *Why:* S5 — two queued
    changes would otherwise rewrite each other's text.
26. *(tests)* **`tests/docs/test_readme_pypi.py` assert shaping** — assert non-emptiness (images/links) once, on
    `rcp-ndcg/README.md`; absoluteness per scanned page (`rcp-ndcg-core/README.md` embeds no image, so a blanket
    per-file "expected to embed images" assert — `test_readme_pypi.py:59,:67` — fails on it). Spell the new
    pages' links absolute from the start (S6): recipe rows in outline 1.5.C.4 →
    `https://github.com/cohere-ai/rcp-ndcg/blob/main/rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/<id>/recipe.yaml`, the
    landing card's directory links likewise; `tests/docs/test_links.py:25-30` then validates those targets (anchors
    included) against the post-move tree. *Why:* S6.
27. **`rcp-ndcg-vllm/pyproject.toml` + the artifact gate** (extends items 10/14 and §1.6) — fix `description` (drop
    "the equivalence harness, the engine recorder and the GPU wave runner": they move to `rcp-ndcg-test`); assert
    in the gate the lean `dependencies = [pydantic, PyYAML]`; add `[tool.setuptools.package-data]` for
    `rcp_ndcg_vllm/recipes/**` (`recipe.yaml`, `template.jinja`, `reference.py`) and `schema/recipe.schema.json` —
    `MANIFEST.in` `graft` is sdist-only (the harness build receipt: wheel of 20 files, 0 `recipes/`, 0 `schema/`,
    `TRIAGE.md` sweep-infra); the §1.6 artifact check lists `recipes/<id>/recipe.yaml` + `recipe.schema.json` inside
    the wheel; note in the gate that `rcp-ndcg-test/cases/` is source-run and deliberately not wheel-shipped (it
    sits outside `src/`). *Why:* S12 (+ S7's summary line).
28. **`CITATION.cff` + the sdist content policy** (extends item 15) — add `CITATION.cff`'s `version:` to the AGENTS
    "Releasing" rewrite and the release checklist (a fifth version carrier `release.yml` does not check); record in
    `### Changed` the declared sdist content set after the move: `CITATION.cff`, `requirements-constraints.txt`,
    `CHANGELOG.md`/`AGENTS.md`/`REPRODUCIBILITY.md`, `schemas/`, `skills/`, `examples/` leave the `rcp-ndcg` sdist
    (they stay in the repository) — per the rule "a truncation is declared policy and recorded". *Why:* S8.

---

## 3. Open questions for the owner

Q1. **Where do the README/CHANGELOG/CITATION/schemas/skills/examples live after the move, and what does each sdist
carry?** Tension: GitHub's landing page wants `README.md` at the root; `rcp-ndcg/pyproject.toml`'s PyPI long
description wants it beside the manifest; setuptools cannot reach `../`. Options: (a) full README in `rcp-ndcg/`,
thin landing card at the root (this blueprint's recommendation); (b) full README at root and a copied/templated
one in `rcp-ndcg/`; (c) `readme = "../README.md"` (works, but sdist contents become fragile). Same question for
`CHANGELOG.md`, `REPRODUCIBILITY.md`, `AGENTS.md`, `CITATION.cff` (recommend: repository-level, stay at root and
NOT inside sdists) and `schemas/`, `skills/`, `examples/` (layout-move says root keeps them — then the
`rcp-ndcg` sdist loses them; recommend accepting that loss and stating it in the sdist's contents test, unless
users need `skills/rcp-ndcg` from an sdist).

Q2. **`rcp-ndcg-test` versioning and surface promises.** Recommend: same version number as the published three
(aids traceability), never published, no `tests/contract` surface pin (internal API), a CHANGELOG entry only where
it affects the public packages, own LICENSE/NOTICE, its own README per 1.5.D.

Q3. **Which `rcp_ndcg_vllm` names are public?** Recommend: `rcp_ndcg_vllm.recipe` (`Recipe`, `load_recipe`,
`iter_recipes`, the serve-argv builder) and the `rcp-ndcg-vllm` console script tree (`serve`, with `--dry-run`);
everything else internal. Needed to write the 0.0.1 surface listing truthfully (edit item 3) and to pin it (item
18). If the recipe schema ships as `schema/recipe.schema.json`, that file is public too.

Q4. **One shared NOTICE or per-distribution NOTICE?** Recommend: one byte-identical merged NOTICE in all four (the
current test's design), accepting that `rcp-ndcg-core`/`rcp-ndcg-test` mention code they do not contain. The
alternative (attributions filtered per distribution) is more precise but needs a per-distribution review at every
move. F3's merge happens either way.

Q5. **Do we publish per-recipe validation status?** Recommend: yes — a `status` column in the catalogue table
("validated: T0-T4 + E2E wave" or the record's own words) sourced from each recipe's `status:` field, with
`sources` in `recipe.yaml` as the receipts. If the owner prefers one blanket sentence ("every shipped recipe
validated before the tag"), the table drops the column and edit 10 shrinks.

Q6. **Add `SECURITY.md`?** The CHANGELOG has a Security section and the sweep found a shipped key-handling issue
now fixed (TRIAGE sweep-docs SECURITY item). A standard `SECURITY.md` (reporting policy + supported versions) to
link from the three READMEs would complete a professional release surface. Recommend yes; content is 20 lines.

Q7. **Date and heading style of `## 0.0.1 — …`?** Recommend heading with the ISO release date filled at tag time
(edit item 1 leaves `— <release date>` and the release checklist sets it), given "0.1.0-style plain heading" is
also fine if the owner prefers no dates.

---

## 4. Execution order for the docs lane (dependency order)

1: tests 16-19 and 21-22, 26 sketched first (red where the surface is wrong — they are the failing tests the fixes answer).
2: edits 13-15 (LICENSE/NOTICE/MANIFEST + 27's package-data half) at the same time as the layout move touches paths.
3: edits 8-11 (the four READMEs) — item 10 consumes Q3-Q5 answers or the recommendations and runs over TRIAGE F2 (item 25).
4: edit 6 (REPRODUCIBILITY), edits 7 (rcp-ndcg README), 1-5 (CHANGELOG last: it quotes every finished page), with
   items 23-24 (the TRIAGE reconciliation and the drop/correct passes) folded into the CHANGELOG step.
5: edit 22 (constraints, at the move) and 28 (CITATION + sdist policy) before the CHANGELOG pass, since CHANGELOG
   quotes the sdist policy.
6: edit 12 (AGENTS) last of all (it references the finished layout and gates).
7: `uv run pytest tests/docs tests/contract` + `ruff format --check` over Markdown-embedded code; register the
   regenerated snapshot/schema diff and fold its bullets per rule 1.2.2.
