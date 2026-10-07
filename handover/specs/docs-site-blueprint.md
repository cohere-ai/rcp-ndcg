<!-- Handover copy of the operator's working note `research/docs-site/work/blueprint.md` (sanitized: private names, the private repository, the job platform and storage locations removed; `<operator-notes>/...` paths point to files that are NOT available — the content you need is in handover/). Treat it as the specification it was for its lane. -->

# docs-site — blueprint for the 0.0.1 documentation site

Tag `docs-site`. Reviewer: MiMo v2.6 Pro (xhigh). Named ref `<repo>` `18c86862276e`, worktree
`/tmp/docs-site/rcp-ndcg`. Findings and evidence: `report.md` (D1–D16). This file is the execution spec for the
final docs lane (`drafts/l5-docs-notes.md`: "execute the three final-output blueprints … edit lists + the owner's
answers to their open questions"). Assume, per the brief: the layout move (`drafts/layout-move.md`), the 18
recipes (`drafts/recipes.tsv`) GPU-validated before the tag, the owner decisions in `STATE.md`, and the queued
TRIAGE fixes.

**Ground rules for executing this list** (they are the repository's own):
- Keep `tests/docs` green at every step: `test_links.py` (every link resolves **and nav == pages exactly**),
  `test_commands.py` (every `rcp-ndcg …` quoted in any Markdown parses against the real click tree),
  `test_snippets.py` (every Python block of every `docs/` page runs, unless marked `skip`/`network`),
  `test_wording.py` (no retired names, and never an RFC/issue id — its `FORBIDDEN` bans `RFC 0001`-style ids).
- Public names only (AGENTS.md): no internal hosts, buckets, people, tools or unreleased models. Placeholders:
  `gs://YOUR-BUCKET/...`, `registry.example.com`, "your cluster's job CLI".
- Never retype chat-template special-token literals in shell, code or commit text; quote them only where the page
  already does.
- Numbers need a source. Where this blueprint says "copy from <file>", that file is the single source of truth.
- Every edit below names the file at `18c86862276e` (line numbers are at that ref), what to change, why (a D-finding
  or an owner decision), and the backing source.

---

## 1. Target structure

Three tiers — **Concepts** (what the machinery is and why), **How-to** (run one task end to end), **Reference**
(look up an exact name, flag, schema, or id). Top-level `Home / Quickstart / Data` are entry pages.

```text
nav:
  - Home: index.md                       # audience: everyone; the map and the pitch
  - Quickstart: quickstart.md            # audience: first 30 minutes; install + three paths
  - Data: data.md                        # audience: benchmark users; released datasets and loaders
  - Concepts:                            # audience: understand before running
      concepts/metric.md            "The metric: RCP-nDCG"
      concepts/protocols.md         "Scoring protocols and tie rules"
      concepts/tournament.md        "Stage A: the tournament"
      concepts/rubric.md            "Stage B: the rubric"
      concepts/calibration.md       "The calibration"
      concepts/primitives.md        "Primitives: re-annotation, insertion and several judges"
      concepts/preprocessing.md     "Preprocessing and chunking"          (judge policy only; REWRITTEN)
      concepts/text-budgets.md      "Text budgets for served roles"       (NEW by split)
      concepts/inference.md         "The inference layer: one transport for every role"
      concepts/retrieval.md         "Retrieval and reranking"             (NEW)
      concepts/embeddings.md        "Embedding endpoints"
      concepts/late-interaction.md  "Late interaction: pooling and MaxSim"
      concepts/judges.md            "Judges, the judgement store and estimates"   (NEW by split)
      concepts/runs.md              "Runs, engines and runners"                   (NEW by split)
  - How-to:                              # audience: one task, start to finish (was "Tutorials")
      how-to/calibrate-your-benchmark.md "Calibrate your benchmark"
      how-to/mteb-integration.md         "MTEB integration"
      how-to/reproduce-the-paper.md      "Reproduce the paper"
      how-to/serve-a-model.md            "Serve a retrieval model"          (NEW)
      how-to/add-a-model.md              "Add a serving recipe"             (NEW; harness lane co-owns)
      how-to/validate-a-recipe.md        "Validate a recipe on GPUs"        (NEW; = release-candidate workflow)
  - Reference:                           # audience: exact names
      reference/cli.md             "Command line"
      reference/recipes.md         "Recipes and serving models"           (NEW)
      reference/rcp-ndcg-test.md   "The rcp-ndcg-test package"            (NEW)
      api/metric.md                "rcp_ndcg_core"
      api/evaluate.md              "rcp_ndcg.eval"
      api/inference.md             "rcp_ndcg.inference (the rerank wire and client)"
```

Section purposes and the split boundaries:

- **Concepts — preprocessing.md vs text-budgets.md.** `preprocessing.md` answers "what does the judge read"
  (load-time text policy, window budget, page images and video, the judging identity). `text-budgets.md` answers
  "what does a served model read" (the anchor rule, `TemplateSpec`, `TextBudget`/`fit`, chunk pooling, vendor
  budgets, media in served requests). The seam is the page's existing `## Text budgets for served roles` heading
  (line 256 at the ref).
- **Concepts — judges.md vs runs.md.** `judges.md` = the judge contract (config, serving a judge, runtime checks,
  replicas, client behaviour, the judgement store, estimates): `serving.md` §§ lines 8–253. `runs.md` = orchestration
  (run configs and steps, runners, the self-installing coordinator, `serve:` engines, job phases, GPU partitioning,
  failure policy, deployment modes, the job interface, durability/mirrors, one engine many jobs): lines 254–662.
  This cut keeps every `#anchor` that other pages link to (see E4's link map).
- **How-to.** Task-shaped pages with numbered steps and copy-paste commands; no new concepts introduced (link
  back). `validate-a-recipe.md` is where the release-candidate workflow lives, because it is a procedure a recipe
  author runs.
- **Reference.** Exact spells only. `reference/recipes.md` owns the 18 ids and the `rcp-ndcg-vllm` surface;
  `reference/rcp-ndcg-test.md` owns the unpublished contributor package; `api/*.md` stay where they are (three
  pages is the 0.0.1 API reference; no new api pages — see OQ-8).

## 2. Page-by-page verdicts (all 21 pages at the ref)

| Page (at `18c86862276e`) | Verdict | Reason |
|---|---|---|
| `index.md` | **rewrite** | The map lists neither the recipes/serving surface nor the new tiers; the @ref concepts list must match the split nav. |
| `quickstart.md` | **rewrite** (parts) | D1 (git paths break at the tag), D7 ("without touching it"), missing third distribution; the rest is sound. |
| `data.md` | **keep** | Claims verified against `rcp_ndcg.data`; unaffected by the final-state changes. (Link update when `tutorials/` moves: E2.) |
| `concepts/metric.md` | **keep** | Matches `rcp_ndcg_core.metric/gain`; no final-state change touches it. |
| `concepts/protocols.md` | **keep** | Matches `rcp_ndcg_core.protocol` incl. the fixed tie rule at the cut (post-`89896b8` code still credits the class mean — `metric.py:118-127` — so the page's "expected nDCG over all orders of the tie" stays true). Link update (E2). |
| `concepts/tournament.md` | **keep** + 1 edit | Schedule and fit match `rcp_ndcg.llm.schedule`/fit; the parsing paragraph (D9) must follow the M5 fix; identity wording (D8) in the "record_id" sentence. |
| `concepts/rubric.md` | **keep** + 1 edit | Criteria C1–C5 correct (rule: exactly five, never reword); family sentence (D8) gains the sampling fields (B2). |
| `concepts/calibration.md` | **keep** + 1 edit | Model, fit and artifacts match `rcp_ndcg.calibration`; "different families" list (line 89) stale after B2 (D8). |
| `concepts/primitives.md` | **keep** + 2 edits | Behaviour matches `rcp_ndcg.calibration.extend`; terminology only: "the second recipe" (K), "anchor report" disambiguation (K). |
| `concepts/preprocessing.md` | **rewrite + split** | 488 lines, two audiences (D13); number wrong after owner decision (D2). Keeps the judge policy; the served-role budget mechanism moves to `text-budgets.md`. |
| `concepts/inference.md` | **keep** + 3 edits | The transport contract matches `rcp_ndcg.inference`; credential row (D4), judge-on-`RoleClient` (D15), tokenizer-digest dedup (the canonical sentence lives in `text-budgets.md`, per E16). |
| `concepts/embeddings.md` | **keep** + 4 edits | Wire and client correct; credential rows (D4), `recipe:<id>` resolution missing (layout-move item 5), "config without `max_tokens`" scoping (D10), the hosted `dimensions` refusal timing (verifier B V3 → E26). |
| `concepts/late-interaction.md` | **keep** + 2 edits | Wire and MaxSim match the post-`89896b8` code; `dimensions` refusal (D10) and the same `max_tokens` scoping (D10). |
| `concepts/serving.md` | **split + rewrite** | 662 lines, two audiences (D13); three claims false at the tag (D7, D8, D16). Becomes `judges.md` + `runs.md`. |
| `reference/cli.md` | **rewrite** (parts) | D3 (`mcp tools` gone), D6 (`--plan`), D4 (credentials), D11 (anchor wording in the tree line). |
| `api/metric.md` | **keep** + 1 edit | Content correct; three blob links rot after the move (D14/D1). |
| `api/evaluate.md` | **keep** + 2 edits | The duplicated gains-keying clause (D5) and the unreachable spellings `explain.score_delta` / `bootstrap_interval` (verifier B V1 → E24). Link update (E2). |
| `api/inference.md` | **keep** + 3 edits | Correct; `max_tokens` scoping (D10); the missing `instruction: system` mode (verifier B V2 → E25); its `Identity` section duplicates `embeddings.md` verbatim — keep one canonical sentence and link (D11 cleanup). |
| `tutorials/calibrate-your-benchmark.md` | **keep** (move to `how-to/`) | Solid walk-through; unaffected except its `serving.md` links after the split. |
| `tutorials/mteb-integration.md` | **keep** (move to `how-to/`) | Matches `rcp_ndcg.eval.mteb`; tie-rule warning is right. |
| `tutorials/reproduce-the-paper.md` | **keep** (move to `how-to/`) | Matches `experiments/`; public judge names only. |

Nothing is deleted outright; no two pages merge (the fixes are splits and targeted rewrites — merges would lose
working links and snippets for no gain).

## 3. What moves where

| From | To | Content mapping |
|---|---|---|
| `docs/tutorials/*` | `docs/how-to/*` | Whole directory move (`git mv`); contents unchanged. Link sites to fix: `docs/api/evaluate.md:38`, `docs/concepts/protocols.md:103`, `docs/data.md:74`, `docs/index.md:27-29`, `docs/quickstart.md:22,82,94` = **9 docs sites**, plus **root `README.md:127,149`** (GitHub-URL links; `test_links.py` checks them via `REPO_URL`) and **`REPRODUCIBILITY.md:43,57`** (relative links; checked because `markdown_files()` covers every tracked `.md` — inventory completed by verifier B) = **13 link sites total**. |
| `docs/concepts/preprocessing.md` §`Text budgets for served roles` (line 256: "The template as data", "The budget and the fit", "Chunked documents pool by maximum", "Explicit budgets, and hosted vendor profiles") + §`Page images and video` → subsection `Retrieval roles: one preparation path…` (from line 443) | `docs/concepts/text-budgets.md` | Move verbatim, then apply edits E6/E15 there. First paragraph of the new page = `preprocessing.md`'s anchor-rule paragraph (lines 258-266), retitled "The anchor rule". |
| `docs/concepts/serving.md` §§ lines 8–253 ("The judge config", "Serving a judge", "What the client checks at run time", "Several replicas", "How the client behaves", "The judgement store", "Estimating a pass") | `docs/concepts/judges.md` | Keep headings verbatim so anchors survive (`#the-judgement-store` is linked from `tournament.md:47,109`). |
| `docs/concepts/serving.md` §§ lines 254–662 ("Runs and job runners" … "One engine, many judge jobs") | `docs/concepts/runs.md` | Keep headings verbatim (`#starting-the-engines-with-the-run` and `#durability-local-runs-and-a-mirror` are linked from `reference/cli.md:74,88`; `#runs-and-job-runners` from `tournament.md:123`; the root `README.md:117-118` links both). |
| — | everything else | stays put. |

## 4. Terminology standard (apply with edits E16)

| Term | One meaning | Notes and renames |
|---|---|---|
| **budget** | "text budget" = a `TextBudget` (tokenizer + `max_tokens` + `on_overflow` + chunk geometry); "window budget" = the judge's per-window share; "pair budget" = the rerank request shape's budget; "pixel budget" / "video budget" = an `ImagePolicy`/`VideoPolicy` range for images and frames (verifier B V6). | Never "token budget" alone; text budgets are counted in the named tokenizer's tokens, pixel and video budgets in pixels. |
| **anchor** | reserved for the **template anchor** (the fixed position a served model reads its output from; `anchor: last\|first\|mean\|marker`; "anchor-preserving cut"). | The insertion check is the **anchor report** (`Extension.anchor_report`, public name — do not rename in docs; write "the insertion anchor report" on first use per page). "Template anchor" on first use per page. |
| **recipe** | reserved for a **serving recipe** (`rcp-ndcg-vllm` package data, `recipes/<id>/recipe.yaml`). | `primitives.md:41` "In the second recipe" → "In the second procedure". |
| **role** | the pipeline role (`judge`, `embed`, `multi_vector`, `rerank`; the role clients). | Chat-template roles always spelled "chat-template role markers" (`preprocessing.md` window-budget item 2). |
| **phase** | always qualified: "schedule phase" (`random\|stratified\|adaptive`) or "job phase" (`runs.md`). | |
| **engine** | the served process answering HTTP (vLLM, SGLang). Hosted APIs are "hosted APIs"/"vendors", never engines. | "engine role" (`ENGINE_ADAPTER_ROLES`) and "engine environment" (the node runtime) defined once each — `inference.md` / `how-to/validate-a-recipe.md`. |

## 5. Edit list

### Cross-cutting moves and navigation

- **E1 — `mkdocs.yml`.** Replace the `nav:` block with the tree of section 1 (titles exactly as there). Update the
  top comment to mention the three tiers. *Why:* D13; `test_links.py` requires nav == `docs/**.md` exactly. *Source:*
  section 1 of this file.
- **E2 — move `docs/tutorials/` → `docs/how-to/`** (`git mv`), then update the **13** link sites listed in
  section 3 (9 in `docs/`, 2 in root `README.md`, 2 in `REPRODUCIBILITY.md:43,57` — the last two found by
  verifier B), plus `tutorials/` → `how-to/` inside `index.md`'s "Data and tutorials" section heading → "Data and
  how-to guides"). *Why:* D13. *Source:* repo rules (docs describe current behaviour;
  `test_links.py`).
- **E3 — split `docs/concepts/preprocessing.md` → keep + `docs/concepts/text-budgets.md`** per section 3. Keep the
  judge page's section list (`## Text`, `## Page images and video`, `## Identity`) intact and end with a link to
  `text-budgets.md`; the new page links back once. *Why:* D13 (488 lines, two audiences). *Source:* the page's own
  heading structure (line 256 is the seam).
- **E4 — split `docs/concepts/serving.md` → `docs/concepts/judges.md` + `docs/concepts/runs.md`** per section 3; add
  one intro sentence to each naming the sibling page. Update in-bound links to keep the anchors:
  `tournament.md:47,109` (`serving.md#the-judgement-store` → `judges.md#the-judgement-store`),
  `tournament.md:123` (`serving.md#runs-and-job-runners` → `runs.md#runs-and-job-runners`),
  `reference/cli.md:74` (`../concepts/serving.md#starting-the-engines-with-the-run` → `../concepts/runs.md#...`),
  `reference/cli.md:88` (`#durability-local-runs-and-a-mirror` likewise),
  `preprocessing.md:384` and `inference.md:13` (`serving.md` → `judges.md`), root `README.md:117-118` (full URLs →
  `docs/concepts/runs.md#durability-local-runs-and-a-mirror` and `docs/concepts/judges.md`). *Why:* D13; D7/D8/D16
  edits land in the new files. *Source:* link inventory in report.md; `tests/docs/test_links.py` checks
  `REPO_URL` links too.

### Blocker edits

- **E5 — `docs/quickstart.md` (D1, D7).** Line 40: `#subdirectory=packages/rcp-ndcg-core` →
  `#subdirectory=rcp-ndcg-core`; alongside it mention the third distribution in the "Install" section:
  `rcp-ndcg-vllm` ("the lean serving package: 18 GPU-validated serving recipes and
  `rcp-ndcg-vllm serve <recipe-id>`", link `reference/recipes.md`) and note `rcp-ndcg-test` is an unpublished
  contributor package (link `reference/rcp-ndcg-test.md`). Line 31 paragraph ("Every retrieval model is served
  now … installs cleanly next to an engine image without touching it"): restate as — the client talk HTTP to the
  engines and installs beside an engine untouched **in its own venv**; serving a shipped recipe via
  `rcp-ndcg-vllm` adds exactly one pure-Python wheel to the engine's environment (`pip install --no-deps`), and
  nothing else changes; the `experiments/` sentence and the "old `[local]`/`[vllm]` extras are gone" sentence
  stay. *Why:* the command cannot succeed at the tag and the deployment claim is false. *Source:*
  `drafts/layout-move.md` tree; STATE runtime principle 21:11 item 1 (`pip freeze` differs by exactly that wheel).
- **E5b — `docs/api/metric.md:48-50` (D14).** Drop `packages/` from the three blob URLs
  (`…/blob/main/rcp-ndcg-core/src/rcp_ndcg_core/{metric,gain,protocol}.py`) and retitle the nav entry
  "`rcp_ndcg_core`". *Source:* same.
- **E6 — `docs/concepts/preprocessing.md` (D2).** Line 55: "`max_tokens` defaults to 20,000" → "defaults to 32,768
  tokens for `truncate` and `fail`; shipped presets and the paper configs pin their cap explicitly, so their
  identities do not move". Line 20's YAML comment: change `max_tokens: 20000` to `max_tokens: 32768` (or show it
  omitted with the default named). Apply the same number wherever the move puts the sentence (E3). *Why:* wrong
  number changes how much of a document the judge reads. *Source:* TRIAGE `sweep/TRIAGE.md:58-60`
  "Owner decisions (13:12)" (F7, read as the
  2^15 typo); the landing commit text `a3a07b3` "…defaults to 32768 tokens for truncate/fail without a declared cap
  (owner decision, was 20000)"; after it lands the code's `DEFAULT_MAX_TOKENS` is the source of truth — re-read
  `src/rcp_ndcg/data/preprocess.py` (or `rcp-ndcg/…` post-move) and quote its value.
- **E7 — `docs/reference/cli.md` (D3).** Line 35 command tree: `rcp-ndcg mcp serve | tools [--call TOOL --args JSON]`
  → `rcp-ndcg mcp serve`. Lines 135-136: delete the "mcp tools --call" paragraph; if a shell alternative is worth
  one sentence, name `rcp-ndcg schema show commands --json`. Check that no other page quotes `mcp tools`
  (`grep -rn "mcp tools" docs/` at the ref shows only these). *Why:* owner decision (5): `rcp-ndcg mcp tools`
  removed (p1-tail 2j). *Source:* STATE "Owner decisions (cumulative)"; `tests/docs/test_commands.py` parses the
  docs' commands against the tree.

### Major edits

- **E8 — credential rule (D4).** Three pages, one rule, spelled once in `reference/cli.md` and linked:
  *"A profile's default key variables apply only when the request goes to the profile's own default host. Any other
  `base_url` receives a key only from an explicit `api_key_env`."*
  - `docs/concepts/embeddings.md:21` (table row) and `:32` (the "A served engine takes no key…" sentence): replace
    "the profile's own … or the one `api_key_env` names" with the rule (a served engine takes a key only from
    `api_key_env`).
  - `docs/concepts/inference.md:23` (table row): same rule.
  - `docs/reference/cli.md:103-105` ("Credentials are read only under the name a config declares…"): keep, add the
    same-host qualifier and the full variable list as the canonical spelling.
  *Why:* the documented behaviour sends a user's OpenAI key to arbitrary hosts; the fix changes it. *Source:* TRIAGE
  `sweep-docs` SECURITY finding + its accepted fix wording ("default key variables apply only when the request goes
  to the profile's own default host"); after the fix lane lands, `src/rcp_ndcg/inference/adapters/` is the truth.
- **E9 — `docs/api/evaluate.md:7-12` (D5).** Delete the duplicated clause "… follows the same keying rule. For a
  suite, `gains` and `count_gains` keys may be `\"<subset>/<query_id>\"` -- and" (one of the two copies) so the
  paragraph states: gains come from `gains` else the released `gain` column; `count_gains` follows the same keying;
  suite keys may be `<subset>/<query_id>` and must be when subsets share ids; one style per subset; an unknown
  subset prefix is refused. *Source:* `src/rcp_ndcg/eval/evaluate.py` (keying + refusals as merged in `89896b8`
  and `7cf2167`).
- **E10 — `--plan` (D6).** `docs/reference/cli.md`:
  - line 62: after "The same flag means the same thing on every command that has it:", add "(one exception until it
    is unified: `--plan` — see the table)" or replace with the exact wording the fix-cli lane lands; re-check the
    flag against the click tree (`tests/docs/test_commands.py` will catch a wrong name — **the executed spelling must
    be whatever `src/rcp_ndcg/cli/calibration.py` and `cli/llm.py` define after fix-cli**);
  - line 76 table row: split into two rows, per current shape: `--plan FILE` (`judge tournament`: ask exactly the
    windows of an insertion plan file) and `--plan` (`calibration insert`: pick `--n` opponents and write the plan
    with `--out`, judging nothing);
  - line 21 tree row for `insert`: keep "--plan picks opponents" wording aligned with the flag's final name.
  - `docs/concepts/primitives.md:36-42` (the three CLI recipes): verify each command still parses; rename only if
    fix-cli renamed the flag. *Why:* the flag-invariant claim is false today. *Source:* TRIAGE `sweep-x-api` F6;
  `src/rcp_ndcg/cli/calibration.py:254` (verifier A: the field; :276 is its use site); fix-cli's landing tree.
- **E11 — engine claims (D7).**
  - `docs/concepts/serving.md` opener (→ `judges.md`/`runs.md`, E4): "The package never builds an image, never pins
    an engine and never translates engine flags" → scope to the judge/run side ("for your own engines and judge"),
    and add one sentence: "For the 18 shipped retrieval models `rcp-ndcg-vllm serve <recipe-id>` composes the engine
    command from the GPU-validated recipe (link `how-to/serve-a-model.md`); anything else is your command, verbatim."
  - `docs/quickstart.md:31` — covered by E5.
  - `reference/recipes.md` and `how-to/serve-a-model.md` (new) state the one-wheel rule and the `pip freeze`
    check. *Source:* layout-move items 2–3; STATE runtime principle 21:11 item 1; `drafts/recipe-template.md`
    "Out-of-the-box image (owner)".
- **E12 — identity and family claims (D8).**
  - `serving.md` §The judgement store (→ `judges.md`): the identity bullet (lines 163-167) becomes: the judgement
    family = judge model + revision, prompt (by content hash), criteria, parse version, decoding, preprocessing and
    tokenizer SHA **plus, when they differ from the defaults, `temperature`, `max_output_tokens`,
    `context_tokens`, `extra_body` and `api`** (B2); every record id and the store identity **name the dataset and
    its revision for every input form, including in-memory records** (B1); replace line 166's "the same file under
    another name or path is the same identity" with the code's rule: "a local dataset enters with its path absolute
    and normalised, so the same file named from another directory (`./rows.jsonl`) is one identity — the same bytes
    copied under another directory are not" (`src/rcp_ndcg/llm/judging.py:270-289`);
  - `serving.md` job-runs paragraph (→ `runs.md`): "Resuming … reuses every step whose identity is unchanged" gains
    "(a judging step's identity includes the prompt's text hash and the tokenizer's SHA-256)" (N1);
  - `docs/concepts/calibration.md:89`: "(another prompt, parse version, decoding or preprocessing)" gains "or
    sampling fields"; same list check in `rubric.md:24` and `tournament.md:45` (they describe `decoding`/`prompt`
    specifics — add the sampling fields to the one place that enumerates the family, `judges.md`, and link).
  *Why:* three fixes change the rules the docs state as settled. *Source:* TRIAGE `sweep-llm`/`sweep-x-arch` B1, B2,
  N1 (all CONFIRMED, fixes queued); `rcp_ndcg_core.schemas` `Family.key` and `src/rcp_ndcg/runs/pipeline.py`
  `JUDGE_STEP` after they land.
- **E13 — the wheelhouse comment (D16).** `serving.md` (→ `runs.md`) lines 314-316: the comment "rcp-ndcg and the
  rcp-ndcg-core it pins exactly; the pyproject.toml versions must match" → "every workspace member:
  `rcp-ndcg-core`, `rcp-ndcg`, `rcp-ndcg-vllm` and the unpublished `rcp-ndcg-test`; the published three share one
  version, which the tag must match". Keep the commands' flags identical to `.github/workflows/release.yml`'s at
  the tag (open it at the moved paths and diff the two command lines word for word). *Why:* release instructions
  must match the release workflow exactly. *Source:* layout-move items 1 and 6 (three published dists, publish
  order core → rcp-ndcg → vllm); `release.yml`.

### Minor edits

- **E14 — `docs/concepts/tournament.md:28-31` (D9).** Restate the parse rule in §"Parsing an answer" from the fixed
  parser's docstring (fix-llm-runs, TRIAGE M5: the orphan-marker handling "wipes a complete answer" today; the fix
  restricts what is deleted). One sentence, exactly matching `src/rcp_ndcg/llm/_parsing` afterwards. Quote the page's
  special-token literals as they already stand; never retype them elsewhere. *Source:* TRIAGE `sweep-llm` M5.
- **E15 — scoped "no `max_tokens`" sentences and the `dimensions` refusal (D10).**
  - `late-interaction.md:114`, `api/inference.md:77`, `embeddings.md` (client bullet, "a config without
    `max_tokens` sends every item whole"): scope to "a hosted profile that declares no limit sends … whole" (an
    explicit budget is mandatory for a self-hosted config).
  - `late-interaction.md:38`: "`dimensions` is never sent: `/pooling` refuses it" → "a `PoolingEndpoint` refuses
    `dimensions` at construction (`/pooling` has no such field)" per the B12/S1 fix.
  *Source:* STATE explicit-budget decision; TRIAGE `sweep-x-api` B12/S1.
- **E16 — terminology sweep (D11).** Apply section 4:
  - `primitives.md:41` "In the second recipe" → "In the second procedure above";
  - `serving.md:476` (→ `runs.md`) "So the recipe depends on the platform" → "So the deployment shape depends on
    the platform" (verifier B V5 — "recipe" is reserved for the serving recipe);
  - `primitives.md:170` and `reference/cli.md:21`: first use → "the insertion **anchor report** (the refit check…)" ;
    the budget pages say "template anchor" on first use;
  - `tournament.md`/`calibration.md` schedule phases vs `runs.md` job phases: qualify ("schedule phase" / "job
    phase") on first use per page;
  - `preprocessing.md` window-budget item 2: "chat-template role markers";
  - `judges.md` (from `serving.md`) `#phases` section title may stay "Phases", but its first sentence says "job
    phases"; `api/inference.md` §Identity duplicates `embeddings.md`'s tokenizer-digest paragraph nearly verbatim —
    keep the canonical one in `text-budgets.md`, one-sentence restates plus links elsewhere.
- **E17 — drop the transitional claim (D15).** `docs/concepts/inference.md:142`: "…the multi-vector pooler; the judge
  ports onto it later)" → "one client per role (the judge, the embedder, the multi-vector pooler, the reranker), all
  derived from `rcp_ndcg.inference.clients.RoleClient`" — if the judge is not yet on `RoleClient` at execution time
  (p1-tail item 4), write the shipped fact and nothing more. *Source:* STATE plan-to-release P1 ("judge onto
  RoleClient"); `src/rcp_ndcg/inference/clients` at execution time.

### New pages (write as specified)

- **E18 — `docs/concepts/retrieval.md` "Retrieval and reranking"** (D12). Audience: someone building candidate pools.
  Sections, each with its source:
  1. *What retrieval contributes* — the `retrieve`/`rerank` run steps and `candidates: from: retrieval` (source:
     `src/rcp_ndcg/runs` step docs and the run-config schema `schemas/run-config.v1.json`).
  2. *Retriever kinds* — one table over `kind: bm25 | dense | late_interaction`, each row's fields copied from
     `src/rcp_ndcg/retrieval/config.py` (`BM25Config`, `DenseConfig`, `LateInteractionConfig`; `stemmer` is a
     Snowball language and is part of the index identity — cf. `reference/cli.md`'s BM25 note), plus rerank as a
     second stage (`RerankerConfig`).
  3. *Indexes, checkpoints and identity* — content vs runtime fields (`IDENTITY_ROLES` in `config.py`); the rerank
     checkpoint keys on every content field of the reranker and the candidate texts' digest, so changed texts or
     settings re-score (source: the landed fix `7c5b4e0` "The rerank checkpoint keys on the reranker's content and
     the exact texts"; CHANGELOG), and depth/fuse/resume refusals.
  4. *On the wire* — link `inference.md`, `embeddings.md`, `late-interaction.md`; budgets and anchors link
     `text-budgets.md`.
  No commands beyond what parses (`retrieval index|search|rerank|fuse`).
- **E19 — `docs/how-to/serve-a-model.md` "Serve a retrieval model"** (D12, D7). Sections: (1) the 18 recipes and
  `rcp-ndcg-vllm` (link `reference/recipes.md`); (2) `rcp-ndcg-vllm serve <recipe-id> [--port …] [--dry-run]` —
  builds the `vllm serve` argv from package data: template file path, media flags, pooler config; refuses a missing
  plugin with the exact install line (layout-move item 2 — copy the real flags from `rcp-ndcg-vllm`'s CLI at
  execution); (3) the engine environment on a node: stock vLLM image, the one pure-Python wheel with `--no-deps`,
  the `pip freeze`-differs-by-exactly-one-wheel check (STATE 21:11 item 1); (4) the client venv and
  `recipe:<id>` in the retriever/run config (layout-move item 5: resolved through `rcp_ndcg_vllm`, lazy, typed
  error with the install hint when it is not installed); (5) hosted APIs need nothing of this.
- **E20 — `docs/how-to/add-a-model.md` "Add a serving recipe"** (D12). Co-owned with the harness lane
  (SWEEP-KNOWN harness FOLLOWUP items 8 and 12 already queue its wording — do not re-decide those, verify them).
  Sections follow `drafts/recipe-template.md` verbatim in content: the recipe directory (`recipe.yaml`, template
  file, `reference.py`); the `client` block IS the endpoint config + `TemplateSpec` + budget fields, validated by
  `load_recipe`; explicit budgets (`tokenizer` `repo@commit` + `max_tokens`, `query_max_tokens` where the reference
  caps the query, `on_overflow: cut`); specials by name in templates and the declared `anchor`;
  `serve.chat_template` ships with the recipe (R10); media: `serve.mm_processor_kwargs` pins pixels = the client
  media policy (R20); multi-vector: `embed_dtype: float16`, `dim`; `use_activation` explicit for rerankers;
  reference interface in its own environment; `status`/`sources`; recipe id = the lowercased canonical Hub repo
  name; then the CPU checks (stage 1 token-id equality + anchor check on ≥20 sampled pairs incl. 5 over-long) and
  the deletion-style mutation that must turn the check red (AGENTS "every fix gets a failing test first" flavour).
  Apply the recipe-sweep conventions from `drafts/recipe-sweep-after-p1-tail.md` (normalize strips, per-shape query
  caps, one `requirements-reference.txt` convention) as the list of things a new recipe must not get wrong.
- **E21 — `docs/how-to/validate-a-recipe.md` "Validate a recipe on GPUs"** (= the release-candidate workflow)
  (D12). Sections: (1) *Why before the tag* — owner decision 6: no recipe ships unverified; every recipe runs its
  waves first, `status:` records the outcome (quote the schema's vocabulary from `rcp_ndcg_vllm/recipe.py` — do not
  invent status names); (2) *Three environments on the node* — engine (the image's Python; untouched except the one
  `--no-deps` wheel; `pip freeze` check), client (`uv venv …`, staged wheels + the release's constraints file, runs
  the harness/recorder/CLI/wave runner over HTTP only, no torch), reference (`--system-site-packages` venvs or a
  pinned `requirements.txt` venv per reference; each run records versions) (STATE runtime principle 21:11; `uv` via
  `pip install --target`, `UV_CACHE_DIR` on local disk); (3) *The waves* — T0 smoke (engine up from the recipe's
  serve argv; `GET /v1/models`; one request per role route; record engine version/fingerprint), T1 observations
  (the recorded corpus), T2 equivalence (served vs reference, with the engine's token count checked), T3 quality
  (MTEB suites), T4 end to end with a served judge (link `judges.md`; no judge model names beyond the shipped
  configs); (4) *Fakes and conformance on CPU afterwards* — one conformance suite, two targets (live vLLM | the
  recipe-level fake engine through the product's role clients), golden replays and forward compatibility; GPU is
  never exercised by pytest (GPU produces observation corpora; CPU verifies the fakes) — link
  `reference/rcp-ndcg-test.md`; (5) *Release candidate loop* — wheelhouse + constraints (`runs.md` §wheelhouse),
  waves A–D, RC fix rounds, GPU rerun of what changed, then the tag. Public placeholders only.
  *Source:* STATE owner decisions and runtime principle, `GPU-VALIDATION.md` tiers T0–T4 (public subset),
  `OBSERVATIONS-SPEC.md` (describe the corpus idea only), `drafts/layout-move.md` item 4.
- **E22 — `docs/reference/recipes.md` "Recipes and serving models"** (D12). Sections: (1) the `rcp-ndcg-vllm`
  distribution: `recipes/` package data read via `importlib.resources`, `rcp-ndcg-vllm serve <id>`, `models/<topk|pplx>/`
  plugins under one `vllm.general_plugins` entry point, lazily registered (importing the package never imports
  torch or vllm), one version guard (layout-move item 3); (2) the table of the 18 recipes — columns
  `id | model | role | modality | plugin`, **id and model copied from `drafts/recipes.tsv`, role/modality/plugin
  copied field-by-field from each merged `rcp-ndcg-vllm/recipes/<id>/recipe.yaml`** (research drafts say
  `role: multi_vector` for `topk-embed-v1-small` and `pplx-embed-v2-context-9b-preview`, rerank for the rerankers,
  embed for the encoders — verify against the merged files, never from this note); (3) budgets: "every recipe
  declares `client.tokenizer`, `client.max_tokens` and (where the reference caps queries) `query_max_tokens`;
  over-budget content is cut client-side, anchor-preserving, recorded; no number is repeated here — the recipe file
  is the source"; (4) equivalence policy: served recipes use anchor-preserving cuts; the paper code's anchor drops
  are `known_deviations` and compared under the cap only (STATE); (5) `recipe:<id>` resolution in `rcp-ndcg`
  (optional dependency; typed error with the install line when `rcp-ndcg-vllm` is not installed).
- **E23 — `docs/reference/rcp-ndcg-test.md` "The `rcp-ndcg-test` package"** (D12). Audience: contributors. State
  first: unpublished (no PyPI); install from a checkout of the workspace. Then one section per module area with the
  source: `cases/` (model-card reference cases + generated strata: multimodal, short, long-under/over the budget,
  mixed batches — STATE owner decision 3, `drafts/case-format.md`), `conformance/` (one suite, two targets: live
  vLLM | the recipe-level fake engine, through the product's role clients), `fakes/` (model-level fakes rebuilt
  from the GPU observation recordings; the generic `fake://` stays in the product — `inference.md` §The offline
  fakes), `equivalence/` + `record/` (the equivalence harness and the recorder; R30: they send through the product's
  role clients), `jobs/` (the RC build, node bootstrap, wave runner — link `how-to/validate-a-recipe.md`).
  Every command quoted here must actually run from a checkout (verify before merge; `test_commands.py` only parses
  `rcp-ndcg …` quotations, so `python -m …`/`pytest …` lines are unchecked by CI — check them by hand).
  *Source:* `drafts/layout-move.md` item 4 and the tree in its item 1; STATE owner decision (3).

- **E24 — `docs/api/evaluate.md` (verifier B V1).** Spell the two callables importable: `explain.score_delta` →
  `from rcp_ndcg.eval.explain import score_delta`, and the bare `bootstrap_interval` →
  `from rcp_ndcg.eval.evaluate import bootstrap_interval` (the modules `rcp_ndcg.eval.evaluate`/`.explain` are
  shadowed by the same-named functions the package re-exports, so the documented spellings raise
  `ImportError`/`AttributeError` — reproduced in `work/verifier-b-round1.md`; alternatively re-export both from
  `rcp_ndcg.eval`: public-surface change = failing test first, `tests/contract/snapshots` regen, CHANGELOG
  (see OQ-10)).
- **E25 — `docs/api/inference.md` (verifier B V2).** The query-text rule enumerates `fold`/`field`/`none` and
  claims completeness; `src/rcp_ndcg/inference/config.py:389` declares four values including `system`. Add
  `system` ("the instruction as a system message", one-line restatement of `preprocessing.md:298`), described as
  the fixed rerank client sends it (after sweep-budget B1 lands), and drop "for every path" if modes keep
  per-wire restrictions.
- **E26 — `docs/concepts/embeddings.md:27` (verifier B V3).** "a config that sets `dimensions` on one is refused"
  → "… is refused when the request is built (the API has no such parameter)" — the config constructs fine today;
  the refusal is the request-builder's `CapabilityError` (`adapters/embeddings.py:199-201`). If the B12/S1 fix
  family moves these refusals to construction (as it does for `PoolingEndpoint`), say "at construction".
- **E27 — `docs/reference/cli.md` table gaps (verifier B V7).** Add a `--baseline` Flags row ("the system a
  comparison compares against"; `cli/eval.py:228`) and an `RCP_NDCG_ENGINES` Environment-variables row (the
  engine overlay the job runners set; `runs/execution.py:480`; link `runs.md` §Starting the engines with the
  run) — or explicitly scope each table ("shared flags" / "the CLI's own environment variables") and link out.

## 6. Cross-checks before merging any edit (the docs lane's self-gate)

1. `uv run pytest tests/docs tests/contract` (nav == pages; every link incl. root `README.md`'s GitHub URLs; every
   `rcp-ndcg` quotation parses; every snippet runs or is marked).
2. `uv run mkdocs build --strict` (the CI check).
3. `grep -rn "mcp tools\|packages/rcp-ndcg\|20,000\|tutorials/" docs/ README.md` is empty.
4. Every quoted status name, flag and config field matches the schema it names (`rcp-ndcg schema show commands
   --json`, `schemas/`).

## 7. Open questions for the owner

- **OQ-1 — `rcp_ndcg.testing`'s home.** Does `FakeJudge`/`build_tiny_world` (`src/rcp_ndcg/testing.py`) stay in
  `rcp-ndcg`, or move to `rcp-ndcg-test` with the other fakes? Owner decision (3) moves "model-level fakes built
  from the GPU recordings" and keeps the generic `fake://` in the product, but does not settle this module —
  `concepts/calibration.md`, `concepts/primitives.md` and `examples/03_*` depend on it. The edit list assumes
  **stays**; if it moves, two snippets and the examples need re-pointing (and `test_examples`).
- **OQ-2 — rename "anchor report"?** The insertion check (`Extension.anchor_report`, `src/rcp_ndcg/calibration/
  extend.py`) collides with the shipped "template anchor" vocabulary of the budget mechanism. The edit list keeps
  the public name and disambiguates in prose (no code change). If you prefer a rename (`refit report`), it is a
  public-surface change: failing test first, `tests/contract/snapshots/python_api.json` regenerated, CHANGELOG
  entry (AGENTS rules).
- **OQ-3 — the `--plan` fix shape.** Does fix-cli unify `calibration insert --plan` (boolean) and
  `judge tournament --plan FILE`, or keep two meanings? E10 documents the two meanings and carves out the
  invariant sentence; if it is unified, drop the carve-out and use the unified name in `primitives.md`'s procedure.
- **OQ-4 — how public is the GPU validation detail?** E21 publishes wave names (T0–T4), the three-environment node
  runtime and the conformance targets, with placeholder infrastructure. Confirm that is the intended public level
  (the alternative is to keep wave specifics in the repository's contributor docs only and leave the site with a
  short "validation" paragraph in `reference/recipes.md`).
- **OQ-5 — the recipe table's `kind` column** (`paper` / `public` in `drafts/recipes.tsv`): publish it (it says
  which recipes are compared against a paper reference) or leave it as lane metadata? The page as specified omits
  it.
- **OQ-6 — `rcp_ndcg`'s optional `recipe:<id>` dependency story.** The site should say how a user installs it
  (`pip install "rcp-ndcg[recipes]"`? or always via `rcp-ndcg-vllm` installed separately). E5/E19 say "install
  `rcp-ndcg-vllm` alongside"; confirm there is no extra alias at the tag and name it if there is.
- **OQ-7 — p1-tail in or out of 0.0.1** (see report.md S2). If in, the budget pages need one more pass for per-shape
  query budgets, LI skip ids and the empty-query policy; if out, `preprocessing.md`'s `request_shape: token_ids`
  sentence (already implemented at the ref) stays but nothing new is documented.
- **OQ-8 — API reference depth.** 0.0.1 ships three `api/` pages (`rcp_ndcg_core`, `rcp_ndcg.eval`,
  `rcp_ndcg.inference`); `rcp_ndcg.calibration`'s public functions are documented only inside
  `concepts/primitives.md`. Is that acceptable for the tag, or does the owner want a fourth api page
  (`api/calibration.md`, ~1 page, from `rcp_ndcg.calibration`'s `__all__`)? The edit list assumes **acceptable**.
- **OQ-9 — one 0.0.1, one version string.** The site says `0.0.1` in the git-install example. Confirm the tag is
  `v0.0.1` across all three published distributions (layout-move item 6) before the quickstart number is frozen.
- **OQ-10 — export `score_delta` and `bootstrap_interval` from `rcp_ndcg.eval`?** (verifier B's D17/E24). The
  docs currently promise `explain.score_delta` and bare `bootstrap_interval`, which cannot be imported as
  spelled. Either fix the spellings in the docs (E24 as written, no code change) or re-export the two callables
  (a public-surface change: failing test first, `tests/contract/snapshots/python_api.json` regen, CHANGELOG).
  The edit list assumes **docs spellings only**.
