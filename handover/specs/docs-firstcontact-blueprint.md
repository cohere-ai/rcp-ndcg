<!-- Handover copy of the operator's working note `research/docs-firstcontact/work/blueprint.md` (sanitized: private names, the private repository, the job platform and storage locations removed; `<operator-notes>/...` paths point to files that are NOT available — the content you need is in handover/). Treat it as the specification it was for its lane. -->

# blueprint.md — the first-contact surface of rcp-ndcg 0.0.1 (final state)

For lane `l5-docs` (per `drafts/l5-docs-notes.md`, executed after the layout move against the merged tree).
Companions: `report.md` (findings + evidence + the five journey walkthroughs), `digest.md` (verdict).
Line anchors in the edit list are from ref `18c86862276e`; where the queued TRIAGE fixes already shift a line, the
anchor is the section heading. Every item is final-state (post-layout-move) unless marked otherwise.

---

## Part 1 — Target structure

Audiences, in reading order. **(a) New user, 60 seconds** (root README on GitHub/PyPI): decides whether this
release solves their problem. **(b) New user, 30 minutes** (quickstart, machine in front): completes one journey.
**(c) Agent user** (SKILL.md): completes a task for their human via `--json`/MCP. **(d) Contributor/QA**
(rcp-ndcg-test README, AGENTS-linked).

### 1. `README.md` (repository root) — audience (a)
Purpose: what RCP-nDCG is, how to install it, and which journey to start; deliberately brief, linking depths.
Outline:
1. Logo, title, one-paragraph definition + paper link + pipeline figure (unchanged, README.md:1-23).
2. **Install** — PyPI one-liner `pip install "rcp-ndcg[hf,calibrate]" --extra-index-url https://download.pytorch.org/whl/cpu`
   + `uvx` alternative + **one line** "from a checkout: `uv sync --extra hf --extra calibrate`" linking quickstart for
   git installs and the extras table (canonical install documentation lives in quickstart — one home).
3. **Distributions** (new, 4 lines): `rcp-ndcg` (pipeline + CLI), `rcp-ndcg-core` (metric + IRT, numpy/pydantic),
   `rcp-ndcg-vllm` (18 serving recipes + `rcp-ndcg-vllm serve` for stock vLLM images). One sentence each.
4. **Four paths** (was "Three ways in"), same order everywhere:
   1. Score a system on the released data (existing §1 + one clause: "no rankings file yet? path 2 produces one").
   2. **Serve an open model and score it** (NEW — journey 3): engine side `pip install --no-deps rcp-ndcg-vllm` on
      stock `vllm/vllm-openai`, `rcp-ndcg-vllm serve <recipe-id> --port 8000`; client side `rcp-ndcg retrieval …`
      with `recipe:<id>`, then `eval score`; link the recipe catalog and quickstart path 2.
   3. Re-judge a pool with your own endpoint (existing §2, stale clause gone per TRIAGE, one clause on the
      32 768-token default text policy and recorded cuts).
   4. Reproduce a table of the paper (existing §3, unchanged).
5. **Datasets** table (unchanged).
6. **Run the whole pipeline as a job** (2-3 lines replacing README.md:118-121's tail): `serve:` one engine per role on
   SLURM/Kubernetes, recipes compose (`rcp-ndcg-vllm serve <id>` as the engine command), link quickstart's job
   section and `docs/concepts/serving.md`.
7. **Documentation** map (updated bullets: quickstart, concepts, the recipe catalog, SKILL, AGENTS, CHANGELOG).
8. **Citation**, **License** (unchanged).

### 2. `rcp-ndcg/README.md` (NEW — the PyPI long description of the flagship distribution) — audience (a)
Purpose: the PyPI page; self-contained pitch + install + three hero commands + links. `rcp-ndcg/pyproject.toml`
keys `readme = "README.md"` here (today the root doubles as it: `pyproject.toml:6,9` at the ref).
Outline: 3-line pitch (verbatim from root lede) · install one-liner + extras table pointer · hero commands
(`eval score`, `run start rejudge_nfcorpus`, `retrieval search` + `recipe:<id>`) · links (docs/index.md on the repo,
root README for the full guide, `rcp-ndcg-core`, `rcp-ndcg-vllm`, the paper, Apache-2.0). No datasets table, no job
material — those live in the root README and docs.

### 3. `rcp-ndcg-core/README.md` (moved from `rcp-ndcg-core/README.md`) — audiences (a)+(d)
Purpose: the dependency-light core on its own PyPI page. Content is sound today (checked, full read): pitch,
two install lines, one runnable snippet, the "RCP-, qrel- and Count-nDCG differ only in their gains" paragraph.
Changes at the move: verify no `packages/`-relative wording (none found), the docs link keeps working (docs/ stays
at the root), and the metric.md link target stays.

### 4. `rcp-ndcg-vllm/README.md` (REPLACED wholesale) — audiences (a)+(operators)
Purpose: the PyPI page of the **lean serving package** and the recipe catalog home. The current lane-harness README
must not survive (findings B1). Outline:
1. Pitch: 18 public serving recipes as package data + `rcp-ndcg-vllm serve <recipe-id>`; runs on the unmodified
   `vllm/vllm-openai` image; dependencies pydantic + pyyaml only; never imports torch/vllm at import time.
2. **Install into the engine image**: `python3 -m pip install --no-deps rcp-ndcg-vllm==<version>` — the one permitted
   change to the engine environment (freeze differs by exactly this wheel); the topk/pplx model plugins ship inside
   this wheel (`vllm.general_plugins`, lazy registration).
3. **Serve**: `rcp-ndcg-vllm serve <recipe-id> [--port] [--dry-run]` — builds the `vllm serve` argv from the recipe's
   package data and execs it; a missing prerequisite prints the exact install line.
4. **Recipe catalog**: table from `drafts/recipes.tsv` (18 rows): `id | model | role (embed/multi_vector/rerank) |
   input (text/image/video) | notes (plugin, context)`, each GPU-validated on the pinned engine before v0.0.1.
5. **Use a recipe from rcp-ndcg**: `recipe:<id>` as a retrieval/rerank config value (lazy resolution via
   `rcp_ndcg_vllm`, install hint otherwise); budgets are explicit per recipe, over-budget content is cut client-side
   with anchors preserved, every cut recorded.
6. **Recipe authors / validation**: link `docs/how-to/add-a-model.md`; equivalence harness, recorder, conformance,
   wave tooling live in `rcp-ndcg-test` (unpublished). CHANGELOG + Apache-2.0.

### 5. `rcp-ndcg-test/README.md` (NEW, unpublished) — audience (d)
Purpose: one page telling contributors where QA assets went and how to run them. Outline: what it holds (model-card
reference cases + generated strata, one conformance suite with two targets through the product's role clients, fakes,
the equivalence harness, the recorder, the RC build/node bootstrap/wave tooling) · not published to PyPI · the three
conformance entry commands · link AGENTS.md. Public names only.

### 6. `docs/quickstart.md` — audience (b)
Purpose: the 30-minute, copy-runnable tour. Canonical home for install-from-source and the extras table.
Outline (section → purpose):
1. **Install** (rewrite): PyPI one-liner, `uvx`, checkout (`uv sync`, `pip install "./rcp-ndcg[hf,calibrate]"
   ./rcp-ndcg-core`), git-tag installs with `#subdirectory=rcp-ndcg-core` (and `rcp-ndcg`), full extras table
   (unchanged content), exit code 10 hint.
2. **Four paths** (was "Three paths"; same order as the root README):
   1. *Score a system on the released data* — current content + `--subset`/`dataset` column rules (keep).
   2. *Serve an open model and score it* (NEW): engine on the stock vLLM image (`pip install --no-deps`,
      `rcp-ndcg-vllm serve qwen3-embedding-0.6b --port 8000`, `--dry-run` to inspect the argv); a `recipe:<id>`
      retriever/reranker YAML; `retrieval index`/`search` and `retrieval rerank`; `eval score --suite nanobeir`;
      the budget/cut clause; the catalog link.
   3. *Re-judge a pool with your own endpoint* — current content + one sentence on the 32 768-token default text
      policy (explicit budgets; cuts recorded; `--set judge.tokenizer=…` counts exactly).
   4. *Reproduce a table of the paper* — current content (keep).
3. **Run the whole pipeline as a job** (NEW section, journey 4): one complete run config (candidates from a served
   encoder through `recipe:<id>`, a judge, `serve:` with `command: ["rcp-ndcg-vllm", "serve", "<id>"]`,
   `runner: name: kubernetes` + `mirror:`), then `--runner kubernetes --dry-run`, submit, `run status`/`run logs`,
   `run resume --runner`, failure semantics in one paragraph; link `docs/concepts/serving.md` for depth.
4. **The metric alone** (keep; the core snippet — acceptable overlap with rcp-ndcg-core/README, different audience).

### 7. `skills/rcp-ndcg/SKILL.md` — audience (c)
Purpose: an agent's complete operator manual for the release. Existing skeleton (paths, invariants, reading
results, exit codes, primitives, offline practice, MCP) is strong; keep its discipline (envelopes, exit-code
contract). Changes: add the model-evaluation story (retrieval + recipes) and the budget invariant; remove the
`mcp tools` sentence. Outline deltas in Part 2 items 17-21.

### 8. `examples/` — audiences (b)+(c)
Purpose: runnable, tested companions (tests/docs runs every `NN_*.py`; only public imports allowed —
`tests/docs/test_examples.py:19,21`). Keep 01-07; add 08 (offline retrieve → score) and 09 (serve recipe → score,
guarded); fix two install docstrings.

### What moves where (mapping)
| Today | In 0.0.1 |
|---|---|
| root `README.md` serves as PyPI long description (`pyproject.toml:9`) | split: root README = repo front door; new `rcp-ndcg/README.md` = PyPI page |
| root README's source/git install block + quickstart's copy (25 identical lines) | quickstart owns the full install story; root keeps one line + link |
| `rcp-ndcg-core/README.md` | `rcp-ndcg-core/README.md` (git mv, content unchanged) |
| `rcp-ndcg-vllm/README.md` (recipes + harness + record + jobs, "pulls in the pinned rcp-ndcg") | replaced: lean `rcp-ndcg-vllm/README.md` (serve + catalog); harness/record/jobs prose → `rcp-ndcg-test/README.md` |
| `rcp-ndcg-vllm/recipes/README.md` (recipe-author file list) | folded into `docs/how-to/add-a-model.md` territory and rcp-ndcg-test README (also covered by harness FOLLOWUP-2 item 8 — do not re-decide its content) |
| README.md:118-121 "serve: … phased rendering is pending … refused" | deletion already queued (TRIAGE sweep-docs); replaced by item 6's job blurb + quickstart job section |

---

## Part 2 — Numbered edit list

Format: **N. `file` — what to change — why — source of truth.** Executable in order; nothing here re-decides
anything (open questions Q1-Q9 have recommended defaults; write to the default if the owner does not answer).

1. **`README.md` Install block (README.md:27-51) AND `docs/quickstart.md`'s install lines (quickstart.md:35,40-41)** — replace the checkout/git variants in BOTH files with the new-layout commands:
   `uv sync --extra hf --extra calibrate` from the root; `pip install "./rcp-ndcg[hf,calibrate]" ./rcp-ndcg-core`;
   git tag variant with `#subdirectory=rcp-ndcg-core` and `#subdirectory=rcp-ndcg`. Keep the "Until the release is
   up" hedge only until the tag (sweep-docs row 24). — why: B2 (commands fail after the layout move; verifier A confirmed the gap: quickstart.md:35,40-41 must be rewritten in this item, not left to the dedup). — source:
   `drafts/layout-move.md` target tree (root = workspace only; dists at top level); `report.md` B2 evidence;
   `work/verifier-A.out.md` B2.
2. **`README.md` extras paragraph (README.md:36-38)** — list the extras that exist (`hf, calibrate, mteb, s3, azure,
   http, data, dev, docs`) or say "the extras table in the quickstart". — why: sweep-docs row 27 (names 3 of 9);
   kept here because it is first contact. — source: root `pyproject.toml` `[project.optional-dependencies]` at the
   ref.
3. **`README.md` + `docs/quickstart.md` duplication** — README keeps 3-line install + link; the git-install block and
   extras table exist only in quickstart (the surviving quickstart block is the item-1-corrected one — the broken
   `packages/` and root-`.[…]` paths must not survive the dedup); delete README's copies (m1). — source:
   `report.md` m1 evidence (25 identical lines) + AGENTS "One home per concept" spirit.
4. **`README.md` "Three ways in" → "Four paths"** (README.md:62+) — insert path 2 "**Serve an open model and score
   it**": engine side `python3 -m pip install --no-deps rcp-ndcg-vllm` on `vllm/vllm-openai`, `rcp-ndcg-vllm serve
   qwen3-embedding-0.6b --port 8000`; client side `retrieval search --retriever <YAML with recipe:qwen3-embedding-0.6b>`
   (per Q1 default: `encoder: {recipe: qwen3-embedding-0.6b, base_url: http://127.0.0.1:8000/v1}`) then
   `eval score --suite nanobeir`; link the catalog and quickstart path 2. Demote path 1's hero command mention of
   "my_system.parquet" with a bridge clause "no rankings yet? path 2 produces one" (M2). — why: M1, M2. — source:
   `drafts/layout-move.md` items 2 and 5; STATE owner decisions (6),(7); `drafts/recipes.tsv`; `report.md` evidence.
5. **`README.md` path 1** — add the bridge sentence from item 4 (keep the rest). — why: M2. — source: journey-1
   walkthrough in `report.md`.
6. **`README.md` §2 tail (README.md:118-121)** — after the queued TRIAGE deletion of "the runners' phased rendering
   is pending … refused", replace the remaining clause with: "a run config's `serve:` section starts one engine per
   role (the judge, the retrieval encoder, the reranker) in a SLURM or Kubernetes job — with the recipes, the engine
   command is `rcp-ndcg-vllm serve <id>` (Quickstart's job section; Serving for depth)." — why: M3. — source:
   `report.md` M3; layout-move item 2.
7. **`README.md` Documentation map** — add bullets: the four paths' order names (align "Three paths"/"Three ways
   in" everywhere to **"Four paths"** — m2), `rcp-ndcg-vllm` + recipe catalog, `rcp-ndcg-test` for contributors;
   keep "Every command except `mcp serve` takes `--json`" (true after `mcp tools` removal). — why: M4, M5, m2. —
   source: STATE decision (5); `recipes.tsv`.
8. **`README.md` guard** — if the sweep-docs fixes for rows 23-24 ("is on PyPI" hedge) are not yet in the tree, apply
   their exact wording; do not re-litigate. — source: `research/sweep-docs/work/report.md` rows 23-24.
9. **`rcp-ndcg/README.md` (NEW)** — write per Part 1 §2; key `rcp-ndcg/pyproject.toml` `readme = "README.md"` at it.
   — why: M6 (flagship PyPI page undefined at the split). — source: `pyproject.toml:6,9` at the ref
   (`name = "rcp-ndcg"`, `readme = "README.md"` proves the root currently is the long description); layout-move
   target tree.
10. **`rcp-ndcg-core/README.md`** — mechanical `git mv` (layout-move item 1 owns the move); apply only: re-check the
    docs link and drop any `packages/`-relative phrasing (none found at the ref; re-verify). — source: Part 1 §3.
11. **`rcp-ndcg-vllm/README.md`** — replace wholesale per Part 1 §4 (lean pitch; `--no-deps` engine install;
    `rcp-ndcg-vllm serve <recipe-id> [--port] [--dry-run]`; the 18-row catalog from `drafts/recipes.tsv` with
    columns `id | model | role | input | notes`; `recipe:<id>` consumption with the budgets/anchor-cut clause;
    validation status per Q4 default "GPU-validated on `vllm/vllm-openai:v0.31.0` before v0.0.1"; pointers to
    `docs/how-to/add-a-model.md` and `rcp-ndcg-test`). — why: B1 (current README documents tooling that moves and a
    dependency story that dies; commands it documents will not exist in the package). — source: `drafts/layout-move.md`
    items 2-4; STATE owner decision (7); `report.md` B1 evidence.
12. **`docs/quickstart.md` — new path 2** ("Serve an open model and score it", full commands per Part 1 §6.2.2):
    engine host (stock `vllm/vllm-openai` image): `python3 -m pip install --no-deps rcp-ndcg-vllm==<version>` →
    `rcp-ndcg-vllm serve qwen3-embedding-0.6b --port 8000` (`--dry-run` prints the argv); client host: a retriever
    YAML with `recipe:qwen3-embedding-0.6b` (+ `base_url` per Q1 default), `rcp-ndcg retrieval index --dataset
    jsonl:tiny/rows.jsonl --retriever recipe.yaml --out index/` + `retrieval search … --out rankings.parquet` (or on
    a suite subset), then `rcp-ndcg eval score --rankings rankings.parquet --suite …`; one paragraph: budgets are
    explicit in each recipe, over-budget content is cut client-side at token boundaries with the template anchors
    preserved and every cut recorded (m3). — why: M1, M2, m3. — source: layout-move items 2,5; `recipe-template.md`
    "Explicit budgets"/"Anchors"; STATE budget decisions; TRIAGE owner 13:12 (F7 for the judge side goes in item 13).
13. **`docs/quickstart.md` — path 3** ("Re-judge…" keeps its content) + one sentence: "documents run whole up to the
    judge's text policy — 32 768 tokens by default (documented policy, every cut recorded); `--set judge.tokenizer=…`
    makes the counts exact." — why: m3. — source: TRIAGE "Owner decisions (13:12)" F7 + STATE budget line ("documents
    default to 32768 tokens").
14. **`docs/quickstart.md` — path/order + heading naming** — "Three paths" → "Four paths", order = README's order
    (score · serve+score · re-judge · reproduce). — why: m2. — source: `report.md` m2.
15. **`docs/quickstart.md` — NEW "Run the whole pipeline as a job"** per Part 1 §6.3: one complete run config
    (dataset `suite:nanobeir` or a single subset; `candidates: {from: retrieval, retrieval: <recipe:qwen3-embedding-0.6b
    config>, rerank: <recipe:qwen3-reranker-0.6b config>, depth: 50}`; judge by shipped name; `serve:` blocks whose
    `command` is `["rcp-ndcg-vllm", "serve", "<id>"]` per Q2 default with the engine-side `--no-deps` install as
    image preparation; `mirror:` for Kubernetes; `runner: {name: kubernetes, options: …}`), then the command walk:
    `rcp-ndcg run start my_run.yaml --runner kubernetes --dry-run` → submit → `run status` / `run logs` →
    `run resume --runner kubernetes`; one sentence each on phased GPU partitioning and engine-failure semantics; link
    `docs/concepts/serving.md` for depth. — why: M3 (walkable end to end today only in the reference page). — source:
    `docs/concepts/serving.md` at the ref (serve-by-role semantics, verbatim command rule); layout-move item 2;
    journey-4 walkthrough in `report.md`.
16. **`skills/rcp-ndcg/SKILL.md`** — add the retrieval/model story to path 1: "to produce a rankings file from a
    model, `retrieval index|search|rerank|fuse` (`schema show commands --json` lists flags; `--retriever`/`--reranker`
    take a YAML or `recipe:<id>` per Q1 default), then `eval score`"; keep the existing envelope discipline. — why:
    M2, M5. — source: `tests/contract/snapshots/cli.json` (the `retrieval` family exists at the ref and is
    undocumented for agents); `report.md` M2.
17. **`skills/rcp-ndcg/SKILL.md`** — add a "**Serve an open model (recipes)**" path: engine `pip install --no-deps
    rcp-ndcg-vllm`, `rcp-ndcg-vllm serve <id>`, `recipe:<id>` configs, the catalog link, and the invariant: "budgets
    are declared per recipe; over-budget content is cut client-side, anchors preserved, cuts recorded — never
    engine-side"; exit 10's install hint covers a missing `rcp_ndcg_vllm` (`pip install rcp-ndcg-vllm`). — why: M1,
    M5, m3. — source: layout-move items 2,5 ("typed error with the install hint otherwise"); STATE budget decisions.
18. **`skills/rcp-ndcg/SKILL.md:198`** — delete the `rcp-ndcg mcp tools --call …` sentence; replace with: "Without an
    MCP client, call the commands with `--json`; `rcp-ndcg schema show commands --json` describes the same surface."
    Keep the 13-tool list at :194-197 verbatim (verified against `tests/contract/snapshots/mcp_tools.json`, 13/13
    match). — why: M5 (a removed command instructed to agents). — source: STATE owner decision (5) ("remove
    `rcp-ndcg mcp tools` (keep `mcp serve`)").
19. **`skills/rcp-ndcg/SKILL.md` serve-by-role paragraph (:155-175)** — add the recipe form beside the verbatim
    command: `serve: {encoder: {command: ["rcp-ndcg-vllm", "serve", "<id>"], resources: {gpus: 1}, image:
    "vllm/vllm-openai:<tag>"}}`, and amend "a served `api: rerank` or `api: openai_embeddings` model without
    `base_url`" to "or a `recipe:<id>` value". — why: M3. — source: layout-move item 2; `docs/concepts/serving.md`
    "Starting the engines with the run".
20. **`skills/rcp-ndcg/SKILL.md` path 2 invariant** — add the judge text-policy default sentence (32 768 tokens,
    declared and recorded) after "Ask the user for the model's tokenizer". — why: m3. — source: TRIAGE owner
    decisions 13:12 (F7).
21. **`examples/01_score_released_suite.py:8`, `examples/07_mteb.py:7` docstrings** — `pip install ".[hf]"` →
    `pip install "rcp-ndcg[hf]"` (from a checkout: `pip install "./rcp-ndcg[hf]"`), same for `[mteb]`. — why: B2. —
    source: layout-move target tree.
22. **`examples/08_retrieve_and_score.py` (NEW, offline)** — BM25 retrieval on the tiny dataset (`rcp_ndcg.examples.
    tiny()`), write rankings JSONL, `rcp.evaluate` against the released gains, compare two systems. Public imports
    only; runs in tests/docs like 02/03. — why: M2 (the model→rankings→score chain has no runnable example). —
    source: `tests/docs/test_examples.py:19-21` contract; example-02's released-gains idiom.
23. **`examples/09_serve_recipe_score.py` (NEW, guarded)** — the J3 walkthrough as code: `rcp-ndcg-vllm serve
    qwen3-embedding-0.6b --dry-run` printing the engine argv, then a `recipe:qwen3-embedding-0.6b` retriever against
    `RCP_NDCG_ENGINE_URL` and `eval score`; skip with a clear reason when no engine/network (same mechanism the
    network examples use). Extend `tests/docs/test_examples.py`'s `PUBLIC` allow-list with `rcp_ndcg_vllm` if the
    example imports it (Q7 default: yes). — why: M1, m4. — source: layout-move item 5; `tests/docs/test_examples.py`.
24. **`examples/` guard** — rerun `tests/docs`; every existing snippet marker (`snippet: example`, `snippet: skip`)
    stays semantically true after items 1-23 (a `snippet: example` block in README copied into the README must match
    example 01 exactly — the harness enforces it). — source: `tests/docs/_markdown.py:15` marker contract.
25. **`skills/rcp-ndcg/SKILL.md:55-56` (`--only` semantics)** — replace "A run started with `--only` for some of its
    steps ends `partial` with exit code 0; `run resume --run <dir>` runs the rest" with: "`run start --only` records
    just those steps as the run's plan (the run then ends `completed`); `run resume --run <dir> --only <step>` runs
    just those now and leaves the run `partial` (exit 0), and a later `run resume --run <dir>` runs the rest." —
    why: N1 (verifier B reproduced on `tiny`: `run start --only tournament` records `steps: [tournament]`, ends
    `completed`, and `run resume` runs nothing more; `cli/run.py:315,386`, `runs/pipeline.py:277-278`). — source:
    `work/verifier-B.out.md` N1.
26. **`skills/rcp-ndcg/SKILL.md:103` (CREDENTIALS row)** — qualify: "missing or rejected credentials; the message or
    hint names the variable, never its value — a backend that rejects access below the transport layer may still
    surface as `INTERNAL`; treat a `Forbidden`/`denied` message as credentials." Queue the code fix separately
    (map storage auth failures to `CredentialsError`/`ProviderError`; scrub account identities from surfaced backend
    errors). — why: N2 (verifier B reproduced: denied `gs://` write surfaces `INTERNAL (1)` "this is a bug" and the
    raw message carries the caller's cloud account identity). — source: `work/verifier-B.out.md` N2.
27. **`skills/rcp-ndcg/SKILL.md:14-15` (--json envelope)** — show the real envelope `{"schema", "command", "ok",
    "data"|"error", "warnings", "meta"}` and say: read `data` **and check `warnings`** (each `{code, message}`:
    recorded cuts, invalid windows, uncalibrated documents…). — why: N5 (6-key envelope verified at the ref;
    `warnings` is where recorded-cut notices live and "Read `data`" hides them). — source:
    `work/verifier-B.out.md` N5; `rcp-ndcg/src/rcp_ndcg/errors.py:28,264-275`.
28. **`skills/rcp-ndcg/SKILL.md` budget (`tests/docs/test_skill.py:27` ≤ 200 lines; 198 at the ref)** — items 16-20
    and 25-27 must fit: move the exit-code table to `docs/reference/cli.md` as the single home (the test compares
    the two copies anyway) and compress "Reading results"; or deliberately amend the 200-line contract with a
    CHANGELOG entry (AGENTS rule). Never drop the M1/M5 content to fit. — why: N3 (verifier B). — source:
    `work/verifier-B.out.md` N3.
29. **`README.md:115` + `skills/rcp-ndcg/SKILL.md:149` (GCS mirror wording)** — "(S3 and Azure via the `s3`/`azure`
    extras, GCS after `pip install gcsfs`, or any fsspec filesystem you register)". — why: N4 (verifier B counted
    the 9 extras at the ref; none provides `gcsfs`). — source: `work/verifier-B.out.md` N4.
30. **`skills/rcp-ndcg/SKILL.md:102` and `:17-18` (two concrete rows)** — MISSING_INPUT row: "the message or hint
    names it"; the exit-3 sentence: "`error.details.errors` per **field** problem (`field`, `input`, `expected` **or**
    `did_you_mean`, `source`); a config error outside the field path has empty `details` — read `error.message`
    too." — why: N6, N7 (verifier B reproduced both counterexamples at the ref). — source:
    `work/verifier-B.out.md` N6-N7.
31. **`README.md:70-74`, `docs/quickstart.md:44-45`, `skills/rcp-ndcg/SKILL.md:31-32` (rankings-file contract)** — state
    the column contract once, completely: `query_id`, `doc_id`, `score` and every accepted alias — read
    `rcp-ndcg/src/rcp_ndcg/data/rankings.py:33` (`COLUMNS` aliases) at execution time and list exactly what it accepts
    (`qid`, `query`, `docid`, `doc`, `corpus_id`, `sim`, `subset`, …). — why: verifier B's precision nit on J1 (no
    surface states the full contract; the loader accepts aliases). — source: `work/verifier-B.out.md`, section
    "Anything wrong in report.md?".

Guard for all items: run `uv run pytest tests/docs tests/contract` at the end (AGENTS: docs describe current
behaviour; every public change updates snapshots + CHANGELOG — the CHANGELOG entry for the `mcp tools` removal and
`recipe:<id>`/`rcp-ndcg-vllm serve` additions belongs to those code lanes' entries; the docs lane adds its own entry
for documentation changes per AGENTS).

---

## Part 3 — Open questions for the owner

Each has a **recommended default** (Q#-D): the docs lane writes to the default unless the owner answers otherwise.

- **Q1 — the exact `recipe:<id>` spelling and the URL plumbing.** layout-move item 5 says "`recipe:<id>` as a
  retrieval/rerank config value"; it does not say whether this is (a) a string value of `encoder:`/`rerank:`
  (`encoder: "recipe:qwen3-embedding-0.6b"`), (b) a mapping (`encoder: {recipe: qwen3-embedding-0.6b, base_url:
  …}`), or (c) only a CLI shorthand (`--retriever recipe:qwen3-embedding-0.6b`), nor where a locally running
  engine's `base_url` is set (`--set`? a sibling field?). **Q1-D**: (b) mapping form with `base_url` a sibling field,
  and (c) accepted as shorthand with `--set retriever.encoder.base_url=…`; docs show (b), mention (c).
- **Q2 — the serve-by-role engine command and the wheel install.** May a run config's `serve:` block use
  `command: ["rcp-ndcg-vllm", "serve", "<id>"]`, and who runs `pip install --no-deps rcp-ndcg-vllm` in the engine
  environment (image build, the supervision script, or the command)? **Q2-D**: yes the command; the README documents
  the `--no-deps` install as **image preparation** (the engine image carries the wheel), and quickstart's YAML shows
  a `bash -lc "python3 -m pip install --no-deps rcp-ndcg-vllm==<version> && exec rcp-ndcg-vllm serve <id> --port
  8000"` fallback for images without it.
- **Q3 — README vs quickstart install split.** Should the root README keep the full git/checkout install block or
  link quickstart (item 3's premise)? **Q3-D**: link; README keeps the PyPI one-liner plus one `uv sync` line.
- **Q4 — the catalog's validation wording.** Owner decision (6) says none ships unverified; what goes in the status
  column at the tag — "validated (waves T0-T4 on `vllm/vllm-openai:v0.31.0`)" per row, or one table caption for all?
  **Q4-D**: caption "all recipes GPU-validated on `vllm/vllm-openai:v0.31.0` before v0.0.1 (T0-T4 + E2E)" + per-row
  known deviations where the recipe declares `known_deviations` (e.g. anchor drops compared under the cap only).
- **Q5 — does `rcp-ndcg-test` get a README on the front door?** **Q5-D**: a minimal README exists (Part 1 §5) but is
  linked only from AGENTS.md and `docs/how-to/add-a-model.md`, not from the user-facing front door.
- **Q6 — path order.** "Serve + score your own model" as path 2 (after "score released data") or as path 1 (the most
  common "how good is my retriever" question)? **Q6-D**: path 2; the released-data path is the only one needing no
  GPU and should stay first.
- **Q7 — examples 08/09.** Accept two new example files (and the `PUBLIC` allow-list widening in
  `tests/docs/test_examples.py`), or keep example 09 as a `snippet: skip` block in quickstart only? **Q7-D**: both
  files; 09 guarded like the network examples.
- **Q8 — judge model names at first contact.** quickstart's re-judge path uses `http://localhost:8000/v1` /
  `my-model` today; keep generic placeholders there (shipped judge names live in `docs/concepts/serving.md`)? This
  is the public-names scrub for the first-contact surface. **Q8-D**: generic placeholders in README/quickstart/SKILL;
  shipped judge config names (`gpt_oss_120b`, `qwen35_397b_*`, `qwen36_27b_fp8`) appear only in serving.md and the
  packaged configs' comments.
- **Q9 — one home for the recipe catalog.** If the docs-site lane (`research/docs-site`) adds a recipe page, the
  catalog table lives there and the `rcp-ndcg-vllm` README carries only the top 5 + link — or the reverse. **Q9-D**:
  table lives once, in `rcp-ndcg-vllm/README.md` (it is the distribution's PyPI page); a docs page links it.
