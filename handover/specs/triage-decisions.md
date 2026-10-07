<!-- Handover copy of the operator's working note `research/sweep/TRIAGE.md` (sanitized: private names, the private repository, the job platform and storage locations removed; `<operator-notes>/...` paths point to files that are NOT available — the content you need is in handover/). Treat it as the specification it was for its lane. -->

# Sweep triage (operator; each finding checked against the code before it becomes lane work)
Legend: CONFIRMED (I read the code), ACCEPT (evidence in the report is sufficient; reproduction on disk), OWNER (a
policy question), KNOWN (already queued), REJECT (with reason).

## sweep-x-arch
- N1 judge-step identity lacks prompt content hash and tokenizer SHA: CONFIRMED, narrowed — store records are keyed by
  family (prompt_hash), so judgements never mix; but the STEP identity (runs/pipeline.py, JUDGE_STEPS) holds only
  JudgeConfig.identity() + schedule + preprocessing, so `run resume` can treat a judge step as done after a custom
  prompt or the tokenizer changed. Fix: splice the judge's identity_extra() (tokenizer SHA) and the resolved prompt's
  content hash into the judge-step identities. MAJOR.
- F2 rcp-ndcg-vllm depends on rcp-ndcg (RFC §6.4 says they meet over HTTP): deliberate (R30, consume the product) ->
  amend RFC §6.4 and the package README; no code change.
## sweep-x-api
- F7 judge TextPolicy fills max_tokens=20000 for truncate/fail: CONFIRMED (data/preprocess.py DEFAULT_MAX_TOKENS) —
  a documented judge default, inconsistent with the explicit-budget rule for served roles: OWNER.
## sweep-tests
- "C deletes TestOutageBoundaries": REJECT as worded — C branched before test-gaps-1 merged into rfc-0001; the merge
  restores them (verify at the stage-budget -> rfc-0001 merge gate).
## sweep-retrieval-eval
- blocker maxsim `_grouped_max`/`_grouped_sum` clamp: CONFIRMED by reading P maxsim.py:85-98 (an empty item's start
  == n_cols is clamped to n_cols-1, so the preceding group ends one column early; the empty group is zeroed after) —
  wrong late-interaction scores whenever an empty item follows another in a block. Fix: append a sentinel column/row
  (or compute ends explicitly) instead of clamping; tests with empty items at every position.
- blocker rerank checkpoint key omits instruction/use_activation/recipe/api/tokenizer/text: ACCEPT (matches l3e's own
  report: key kept byte-identical to the pre-0.0.1 writer so old checkpoints resume) — correctness wins: key on every
  CONTENT field of the reranker + the candidate texts' digest; old checkpoints re-score (CHANGELOG).
- pickle BM25 index: ACCEPT (load of attacker-writable pickle) -> bm25s save/load without pickle, or refuse.
## sweep-llm
- B2 Family.key omits temperature/max_output_tokens/context_tokens/extra_body/api: CONFIRMED (core schemas.py:186-241).
  OWNER: recommend adding each to the family digest only when it differs from the default (the existing `tokenizer`
  pattern), so every existing family key (the paper's) is unchanged.
- B1 judgement_record_id omits the dataset; row-sequence inputs get a constant store identity: CONFIRMED in signature
  (schemas.py:244-266) — fix: the store identity names the dataset (and its revision) for every input form.
- M1/M2 store writer races (claim/note_engines RMW; append torn-tail truncation of a peer's line): ACCEPT (repro on
  disk, 28/30 via the real API) -> a single-writer lock (fcntl) on the store dir for every writer.
- M3 "C carries the pre-fix identity.json write": merge hazard — at the stage-budget -> rfc-0001 merge keep P's hunk
  and P's two regression tests (check explicitly).
## sweep-runs
- judge-step prompt identity: same as x-arch N1 (CONFIRMED). squeue non-zero for finished jobs breaks `run status`:
  ACCEPT (real Slurm behaviour) -> fall through to sacct. `pip install uv` bootstrap ignores the wheelhouse: ACCEPT
  (air-gapped path) -> install uv from the wheelhouse too (stage the uv wheel in it). AttributeError masking on resume:
  ACCEPT.
## sweep-docs
- C/H doc hunks written against an older tree (in-process extras, provider: retrieval, transitional phase notes,
  deleted --system/SNAPSHOT_LISTING docs, usage.calls): merge hazard -> at each merge keep P's pages where P owns the
  code; tests/docs must stay green after each merge (H's three snippet pages fail at H: take C's versions).
- P README.md:120 "phased rendering pending / refused": stale since l4b -> delete (fix lane).
- SECURITY (from embeddings.md:31 finding): CONFIRMED at C adapters/embeddings.py:330-335 — `openai_embeddings`
  (hosted OpenAI AND every self-hosted engine) declares `API_KEY_ENV = ("OPENAI_API_KEY",)` optional, so with the
  variable set the transport sends the user's OpenAI key to ANY `base_url` (a self-hosted engine, a third party).
  MAJOR. Fix: a profile's default key variables apply only when the request goes to the profile's own default host;
  any other base_url gets a key only from an explicit `api_key_env`. Audit every adapter with API_KEY_ENV the same way.
## Owner decisions (13:12)
- F7: documents default to 32768 tokens (owner wrote 32678; read as the 2^15 typo), documented transparently (docs,
  docstring, CHANGELOG); every shipped preset/paper config that relies on today's 20000 default pins it explicitly
  so no existing identity moves (run_all unchanged).
- Everything else is the operator's design call; B2 Family.key: add each content field only when non-default.
- The operator verifies every blocker and major himself before it becomes lane work.
## Operator verification pass (13:15-13:45), blockers and majors
### sweep-infra (5/5 CONFIRMED)
- release.yml builds packages/rcp-ndcg-vllm, absent at P (ls-tree) -> tag only after the harness merges (checklist).
- vllm wheel: built at H -> 20 files, 0 recipes/, 0 schema/; default_recipes_root() = parents[2]/recipes (source-tree
  relative) -> recipes become package data under the import package (with the layout move).
- publish-vllm `needs: build` only (P release.yml:134-136) -> needs publish-rcp-ndcg (and core); fix the test.
- plugins in no CI or release job (rg empty) -> build + test + attach as release assets.
- C carries mutable action tags (C ci.yml:23) -> merge keeps P's pinned workflows.
### sweep-x-api (CONFIRMED: F1, F2, F5, F6, B1, B12; ACCEPT: F3 census, B2, B3; F7 owner-decided)
- F1 reproduced at C: same config family raises ConfigError+hint or bare ValidationError (no hint).
- F2 reproduced at C: probe() AssertionError for image_processor+image_policy without tokenizer.
- F5 reproduced: hint names `preprocessing.image` on a role config (field is `image_policy`).
- F6 confirmed: --plan boolean (calibration insert) vs plan file (judge tournament) vs cli.md "same meaning".
- B1/N2 confirmed: H recipe.py:524 client_config overwrites client.recipe with recipe.id (content field).
- B12/S1 confirmed: PoolingEndpoint inherits `dimensions`, pooling adapter never sends it, pool client never refuses.
### sweep-x-arch (CONFIRMED: N1, S1, F1 (JudgeConfig base_url='' -> urls ('',)), N2, N3 (3 VL recipes declare image
  input with no max_images/image_processor/image_policy); ACCEPT: F3 adapter-contract kit, F4 media inlining x3, F5
  atomic-write copies; F2 = deliberate R30 -> amend RFC §6.4)
### sweep-retrieval-eval (CONFIRMED by running the reviewer's scripts at P: maxsim empty-item truncation + -inf leak,
  checkpoint key same under changed query/doc texts, count_gains bare-key cross-attribution, k-cut tie rule broken;
  ACCEPT: mixed gains keys drop, pickle BM25)
### sweep-llm (CONFIRMED: B1, B2, M5 (orphan think-end regex wipes a complete answer), ACCEPT: M4 empty planned
  windows, M6 torn census line; M3 = merge hazard; M1/M2 NOT REPRODUCED: the reviewer's own repro_store_races.py
  at rfc-0001 gives 0/300 lost claims, 0/300 corrupted append rounds, 0/300 torn-tail rounds -> SUSPECTED: the fix lane
  writes a multi-process stress test first and fixes only if it is red)
### sweep-runs (5/5 CONFIRMED: prompt identity (reproduced), _judge_usage AttributeError masking MissingInputError
  (reproduced), rerank-without-retrieve crashes on work/first_stage.parquet (reproduced), squeue before sacct
  (read slurm.py:432), uv bootstrap ignores the wheelhouse (read script.py:149))
### sweep-tests (CONFIRMED: C probe-leak guard compares the CHARACTERS of the temp dir path; tournament "rank
  preservation" re-derives a monotone map of a sorted list; ACCEPT: 45 tests no CI job runs, identity OR guard,
  C media check measuring itself; "C deletes boundary tests" REJECTED (branch base))
### sweep-docs (CONFIRMED: README.md:118-121 stale "phased rendering pending/refused"; the OPENAI_API_KEY-to-any-host
  SECURITY finding; C/H stale doc hunks = merge hazard)
### sweep-vllm (CONFIRMED by its repro at H: #1 BLOCKER `_compare_shape` compares only served_vectors[0] (read
  stages.py:879) -> multi-vector stage 2 wrong; #2 BLOCKER `_fit_pair` returns the uncut query (142 vs fit's 48;
  173 > 160 sent); over-cap flag counts the raw query; EQUIVALENCE.md hides the stored failure; fixture references
  call the product's fit (tautological render check). ACCEPT: failed --record never affects the verdict, recorder
  fixtures drifted from the adapters, topk/pplx reference `embed` nesting incompatible, `shape`-field rows dropped
  from the render comparison) -> all into harness FOLLOWUP-2 (items 2-4/7/12 already cover part).
### sweep-budget (CONFIRMED: B1 BLOCKER `instruction: system` is handled nowhere in the rerank client (only `fold`
  changes the query, `field` goes to the adapter) -> silently drops the instruction; majors ACCEPT: every one
  reproduced by its verifier A) -> inference/budget fix lane.
### sweep-data (CONFIRMED by its repros at P: file:// exists() False for a live file + junk `file:` cwd tree; BEIR
  reader drops id-less rows and last-wins duplicate labels; image query silently text-only in BeirWriter; remote cache
  serves stale bytes; ACCEPT: storage.relative `..` containment escape (reviewer + verifier A), qrels unknown grade
  column -> all 1.0, cache payload/.meta two-rename race)
### sweep-core (CONFIRMED by its repros at P: inf/NaN gains -> NaN nDCG silently; BT drops unknown-doc comparisons
  silently; record_id collides across datasets; NaN score disables the ranking sort; ACCEPT: the two unpinned SE ridge
  terms (mutation survivors), non-finite public records)
### sweep-cli (CONFIRMED by its repro: --args non-object -> INTERNAL; eval_score readOnlyHint true with an `out`
  input; ACCEPT: malformed tools/call kills the stdio server (reviewer + verifier); unknown --system/--baseline exit
  class CONFIG vs USAGE)
### sweep-recipes (verified by the operator on int-recipes = rfc-0001 9ff594e + 18 recipes + 2 plugins, 20:2x)
- CONFIRMED #1 ctxl instruction split: paper configs ctxl_rerank_{1b,2b,6b}.yaml all `instruction: none`; recipes
  1b/2b `fold`, 6b `none` -> decision: `none` for all three (the paper is what is reproduced), references never fold.
- CONFIRMED #2 (follows from #1): 1b/2b references fold inside themselves; removed with #1.
- CONFIRMED #3 qwen3-reranker over-cap: 4b declares `anchor_drop_over_cap`, 0.6b/8b `[]` -> one family policy; a
  deviation is declared only when the reference really drops the anchor.
- CONFIRMED #4 4b/8b references `tok.decode(...)` the render (NFC round trip); 0.6b cuts by offsets -> 0.6b's way
  family-wide; a non-NFC render test.
- CONFIRMED #5 "binds on overflow only" wording in ctxl-2b/6b, zerank-1-small, qwen3-reranker-4b (+ others) -> state
  the merged rerank client's actual settle rule, read from the client.
- CONFIRMED #6 R20 pixel pin: nested `images_kwargs` (qwen3-vl-embedding) vs flat (qwen3-vl-reranker, topk) ->
  decide from the vLLM v0.31.0 tag source which shape reaches the HF processor; one shape.
- CONFIRMED #7 no client media policy declared in the media recipes; also qwen3-vl-embedding's "rcp-ndcg's lowering
  never emits video_url" is stale since p1-tail 2e.
- CONFIRMED #8 pplx recipe.yaml:152 "role client also refuses a max_tokens config outright" is false at rfc-0001.
- ACCEPT #9 contract tests weak (3/5 mutants green) on the reviewer's mutation evidence.
- Minors: cheap ones in the recipe sweep. Assigned: lane `recipe-sweep` (MiMo exec, GLM+MiMo verifiers).
- OPERATOR DECISION (09:2x, supersedes #3's "declare only when the anchor drops"): references stay faithful to the paper /
  model card and NEVER port the client's cut (recipe-common's f1d1e0a did that for qwen3-reranker-8b and
  zerank-1-small: reverted by the owning families). A reference whose over-cap cut differs from the client's while
  keeping the anchors declares the new `over_cap_cut_differs` (harness: lane/recipe-sweep 9b6bfb5); over-cap rows are
  reported, not gated. 4b's `anchor_drop_over_cap` becomes `over_cap_cut_differs` if its reference keeps the anchors.
