# Lane `harness-media`: the harness models and probes media and frames (pre/post-processing review A1, A3, A4)

**Status:** DONE (one item reported with its limit: the flake, item 6 — see its section).

Branch `lane/harness-media`, base `rfc-0001` after harness-fix and media-rules (`afecce00`).

## Commits

| Commit | Subject |
|---|---|
| `a2399d8c` | The generator's identity splits its semantic version from its sampling seed: `GENERATOR_VERSION` 2 names the artifact, `GENERATOR_SEED` freezes the stream version 1 drew from; all 34 pairs files regenerated with the fixed generator (the 27 older byte-identical, the 7 added since harness-fix gain its fixes) and the manifest records both |
| `165b7db9` | The verified fake engines model media and chat-shaped records: `messages` is a prompt carrier, `ChatPrompts`/`RequestPrompts`/`PairPrompts` derive the prompts, `MediaIdentity` keys a media part by content + declared processing, an unmodelled record is skipped and named (`unmodelled_records`, the verification record) with `EmulatorUnmodelledError` as the typed refusal |
| `bb98a3ba` | The stage-1 frame probes and the harness gaps: media row per shape in the template check, the engine `usage.prompt_tokens` probe, the `anchor: mean` audit, the every-text render comparison, the width gate, the fps-arm test, the video protocol edges, the stub's pair-prompt count |
| `f8e44dc6` | The media gate is labelled an INPUT gate (`MEDIA_GATE_SCOPE`/`scope_note`, the report and the docs); `fetch_tokenizer` refuses a stale warm cache naming pin, hash and fix |
| `f7f5058e` | The CHANGELOG and the docs cover the lane; the stub sets `RLIMIT_CORE` to 0 |
| `154d6e4c` | The wave runner's process and cross-wave state hardened (the test-pkg SIGSEGV flake): the stop signals the engine's own session by `popen.pid` (never a second `getpgid`) and is serialized per engine; per-wave slot-TMPDIR token; the closing state is per wave |
| `52ce3471` | `ruff format`: the wire module's `prompt_tokens` helper keeps its blank line |
| `ec1165df` | The lane report (this file) |
| `6e8048ad` | The round-1 verifier findings: the every-text render comparison gets its failing test, one shared `sent_media_content`, the per-shape media row pick, the placement in the media key, the stub's pair-prompt pin, `CORPUS_PLAN_VERSION` 2, `__all__`/docstrings/dead code |
| `b4340f3e` | Merge `rfc-0001` (40 commits: core-records, content-wire A5/A7/A8, fp-v4, the judge recipes) with the conflicts resolved |
| `283ddb6a` | The five re-keyed corpora's verification records re-appended under the merged code |
| `7ef002a4` | The report's checks, the round-1 fixes and the merge note |
| `5427e08e` | The round-2 confirmation findings (the pairs wording, the dead wrapper, the fixture reference's query render, the video-only edges test, the report's provider id) |
| `0d31abdd` | The qwen3-vl-embedding stage-1 recipe test serves the stub the checkpoint's own chat template (the prompt-token probe's frame) |

## What changed

**1. A1, the media emulator (the release blocker).** `VllmEmulator.from_corpus` raised `DataError` for the
whole corpus on one `messages`/media record and `_text` raised a bare `ValueError` on a media part of a
rerank body. Now:

- `messages` is classified `prompt` (it carries the engine prompt), not `unmodelled`;
  `request_context` applies the chat routes' own `add_special_tokens: false` default, so an omitted field
  and an explicit `false` key one entry;
- `ChatPrompts` renders one engine prompt per conversation through a wiring-supplied render callable (the
  served chat template over the conversation's text parts) and models media through a wiring-supplied
  `media` callable; `RequestPrompts` dispatches an `input` body to `StringsPrompts` and a `messages` body
  to the chat strategy; `PairPrompts` takes the same `media` callable and models a `{"content": [...]}`
  side (its text joined as `Content.text` joins it);
- a media part is keyed by `MediaIdentity` (kind, the sent bytes' SHA-256, the recipe's declared processing
  -- the canonical JSON of its image/video policy and processor family) and its engine tokens come from the
  product's own `content_media_tokens` under the recipe's effective policies with the client's tokenizer;
- a record the model layer cannot model (an unmodelled field, a chat body without a chat strategy, a media
  part without a media model) is **skipped and named** in `VllmEmulator.unmodelled_records`; the rest of
  the corpus builds and replays; `verification_record` carries the list; a request for the skipped record
  answers the marked `refused-unmodelled` 400 (`EmulatorUnmodelledError` is the typed error behind both);
- `rcp_ndcg_test.errors.EmulatorUnmodelledError` is new; `rcp_ndcg_test.equivalence.wire` gains
  `recipe_config` (the validated endpoint config, one home with `role_client`) and `prompt_tokens` (the
  captured `usage.prompt_tokens`, one reader for the media stage and the new probe);
- tests: `tests/conformance/test_media_records.py` (14 tests) with a small in-test media corpus: a chat
  record with an image replays for its content identity and processing, another image/processing answers
  the surrogate, a batch keys every conversation, a rerank media side replays, an unmodelled field / a chat
  body without a chat strategy / a media part without a media model are skipped and named (the corpus still
  builds), and the wiring builds the media model for the vision fixture.

**2. A3, the frame probes.** Stage 1 gains `engine_prompt_tokens_check`: with an engine URL, the engine's
own `usage.prompt_tokens` of one captured request per shape must equal the count of the render the client
budgeted against (per captured exchange, so a batch's sum is compared with the batch's texts; a pointwise
reranker against the sum of its pair renders; a listwise recipe reports `not_run` with the reason). The
template check now renders **one media row per shape**: `stage1_prompts` probes one marked media row per
declared shape for a chat-shaped/media embed recipe (kept out of the anchor/render/tokenize checks), the
`messages` check renders those conversations and requires the render WITH the media to open and close with
the declared frame's fixed edges, and the file check renders a media row's text where the pairs file has
one.

**3. A4, the harness gaps.**

- `anchor: mean` is no longer vacuous: `_audit_mean` asserts the declared fixed edges survive (head
  measured in the assembled render, tail ids plus the post-processor's tokens) and at least one content
  token sits between them and the post-processor's tokens;
- the render comparison covers **every text of every row**: the reference is asked to render each document
  (one written row per document, the query beside it) and every captured text is compared; a per-text
  mismatch names its document and the over-cap carve-out attributes the change to that document's record;
- a width mismatch is a named gate failure with both widths (never a `ValueError` out of the cosine);
- the media gate's fps arm is pinned by a test: the client counts the declared fps rule's realised frames
  (16 of a 64-frame/8 fps clip at fps 2, not 64 and not the loader's 32) with its loaded tokenizer, the
  reference computes the same realised count from the clip's facts (the fixture reference gains the rule),
  the engine served with the declared pin counts the client's tokens, and an engine pinned to the wrong
  rate fails the engine check;
- `media_set.media_edges` gains `edge:too_many_videos` and `edge:corrupt_video` (bare, with strata present
  or absent and the reason), the video half of the image edges.

**4. The media gate is an INPUT gate.** `equivalence/media.py` exports `MEDIA_GATE_SCOPE` (`"input"`) and
`MEDIA_GATE_SCOPE_NOTE`; every stage document (run, `not_run`, `no_media_rows`) carries `scope` and
`scope_note`; `EQUIVALENCE.md` prints the scope; `docs/how-to/add-a-model.md` and `rcp-ndcg-test/README.md`
state that no media vector or score is compared (the media output half is a separate stage, ref-envs').

**5. Wave uploads** — skipped per the brief (moved to lane wave-integrity).

**6. The flake** — see "The flake, in full" below.

**7. The request generator's identity.** `GENERATOR_SEED = "1/rcp-observe-v1"` (the exact sampling stream
version 1 drew from) keys `_rng`; `GENERATOR_VERSION` (2) is the semantic version bumped for every change
to the generator's output, so a bump never re-draws a row; `write_manifest` records
`GENERATOR_VERSION`/`GENERATOR_SEED`/`SEED`; `observe/__init__` re-exports the constant; the corpus
collector block records `generator_seed`. All 34 pairs files were regenerated with the fixed generator
(`python -m rcp_ndcg_test.observe.requests --out rcp-ndcg-test/pairs --reference-python <venv python>`,
4m16s): the 27 files harness-fix regenerated are **byte-identical**, the 7 added since
(pplx-embed-v1-0.6b/-4b, pplx-embed-v2-late-9b, qwen3-embedding-4b/-8b, qwen3-vl-embedding-8b,
qwen3-vl-reranker-8b) gain the fixed generator's over-cap row; the four whose role sends the empty string
(pplx-embed-v2-late-9b, qwen3-embedding-4b/-8b, qwen3-vl-embedding-8b) also carry the corrected
empty-content row (its query side is the empty string now, `content:empty@query`), while the two
pplx-embed-v1 files keep that row absent by their declared `empty_doc: omit_zero` and qwen3-vl-reranker-8b
by its `empty_query: refuse` (it gains only the over-cap row). Tests:
`test_a_version_bump_does_not_redraw_a_row` (monkeypatches the version and requires identical rows) and
`test_the_committed_pairs_files_match_the_recorded_generator_identity` (the manifest's version, seed and
every file's sha256/bytes/rows).

**8. The fps arm, concretely.** Verified: `equivalence/media.py` passes the client's loaded tokenizer and
the clip's own `num_frames`/`fps` into the product's `content_media_tokens` (media-rules' port), and
`stub_engine.py` counts through the same product call with its own tokenizer; the new test in item 3 pins
the whole arm (client == reference == engine at the rule's realised 16 frames; a wrongly pinned engine
fails). What was left was coverage, not code — the fps path had no test.

**9. The stale tokenizer cache.** `tests/recipes/_served.py::fetch_tokenizer` returned a cached file whose
hash did not match its pin when the download failed ("better than an error"). It now raises a typed
`HarnessError` naming the pin, the cached hash, the cache entry and the fix (`rm <entry>` or restore the
network); a cold cache without network still skips with its reason. Tests: a corrupt warm cache with a
failing download raises and names all four; a cold cache still skips.

### The flake, in full

**What the operator reported.** `rcp-ndcg-test/tests/test_record_and_wave.py::test_wave_gives_each_slot_its_own_short_tmpdir`
killed the test-pkg run with SIGSEGV (exit 139, no faulthandler trace) under the gate's load, twice; once
more at `test_wave_records_disk_and_evicts_after_the_last_recipe`, and once at
`test_wave_corpus.py::test_the_corpus_is_keyed_by_the_engine_version_the_pod_reports` (a test that no
longer exists). 4 crashes in 155 gate runs (2.6%).

**What I established.** (a) `uv run` **execs** (`uv run python -c 'print(PPid)'` shows the shell as the
parent), so the 139 is the Python process's own status. (b) A SIGSEGV inside a pytest test under
`-X faulthandler` prints a full traceback (reproduced with a scratch test that segfaults) and, once
`/coredumps` exists, writes a core; the gate's four crashing logs contain no `Fatal Python error`/
`Current thread` line at all, and the last line is the *test name* (pytest flushes the nodeid before the
body), so the process died inside that test **without faulthandler reporting**. (c) A segfaulting *child*
(the engine) does not kill pytest (reproduced with a wave whose engine is a segfaulting process: the test
passed). (d) The cores `/coredumps` collected under load are all `SIGABRT` — the `--fault abort` stub
engines, not the flake.

**What I could not do.** Reproduce the crash: **~50 wave-file runs and 3 full test-pkg suites under load
(six parallel instances, the machine's load average 40-60, ~200 threads per pytest process) all passed**,
with `/coredumps` armed for a SIGSEGV core. So I cannot name the crashing instruction; the three hazards
below are what the code review found, each fixed and pinned by a test.

**The fixes (all in `jobs/run_wave.py`, plus the stub).**

1. **The signal path.** `_EngineRun.stop` sent `os.killpg(os.getpgid(self.popen.pid), ...)`: the
   `getpgid` lookup is a second syscall whose answer can name another process group if the child's number
   was reused between the two calls, and `stop()` is called from the worker's `_stop_after_failure` and
   the wave's end concurrently. The engine runs in its own session (`start_new_session`), so its process
   group **is** its pid: the stop now signals `popen.pid` directly and serializes the whole teardown on
   the engine's lock. Test: `test_stopping_an_engine_signals_its_own_session_by_pid` (monkeypatches
   `os.killpg`/`os.getpgid`; asserts one `SIGTERM` to the pid and that `getpgid` is never called, and that
   a second stop signals nothing).
2. **The wave-shared slot TMPDIR.** `_slot_tmp_dir(slot)` was `<tempdir>/rcp-s<pid>-<slot>` — the same
   path for **every wave in one process** (a test session runs dozens). A previous wave's leftover engine
   or abandoned step body could remove (`shutil.rmtree(run.tmpdir)`) or reuse the path the next wave's
   engine is running with. Each wave now carries a token (`_Wave.token`, `os.urandom(3).hex()`) into
   `_slot_tmp_dir(slot, wave=...)`, so no two waves share a path (still ~30 characters, far inside the
   107-character AF_UNIX budget). Test: `test_every_wave_gets_its_own_slot_tmpdirs`.
3. **The module-wide closing Event.** `_CLOSING` was one global `threading.Event`, cleared by every wave:
   a step body the executor abandoned (GPU-E1's deliberate abandon) could call `_start` **after** a later
   wave opened it, starting an engine under a recipe no wave tracks, on GPUs no wave booked, with a TMPDIR
   its own wave's cleanup removes. The closing state is now per wave (`_Wave.closed` +
   `_CURRENT_WAVE`): a start from a closed wave, or from a wave that is not the open one, is refused.
   Test: `test_no_engine_starts_once_the_wave_closes` (both refusals).
4. **The stub's core dumps.** The `--fault abort` stub engines dumped a 566 MB core per run once the
   machine's `core_pattern` directory existed. The stub sets `RLIMIT_CORE` to 0 (a deliberate abort is the
   test's point, not a bug to preserve).

**Stress loop.** `scratch/stress-fixed.sh`: six parallel instances of `test_record_and_wave.py`
(36 tests each) with `-X faulthandler`, four iterations each, under the machine's normal multi-lane load.
All 24 runs passed every test body (24 × 36 = 864 test executions green); six of the 24 runs ended with the
suite's teardown checkout-guard *error* because this report file was created while they were running (the
error names the file, not a test). The same loop on the pre-fix tree ran ~50 iterations green too, which is
why the report claims a hardening, not a proven root cause.

## Verification

**Round 1** (two fresh verifiers, DeepSeek-V4.1-flash at `:xhigh`, lens A correctness / lens B
regressions+hygiene; they ran the suites and their own reproductions and mutations).

**Lens A: PASS** (9 minors, no blocker/major). It reproduced every brief item with its own scripts
(`scratch/verify-a/`): the media emulator (replay by content identity, skip-and-name, the marked 400, the
typed `_text`), the prompt-token probe (a stub serving a different chat template fails it while
`template_render_check` passes — exactly the gap the probe closes), the media row per shape, the anchor-mean
audit, the every-text comparison (a second-document divergence is caught and named `document: 1`), the width
gate, the fps arm (16 frames on all three sides), the input-gate label, the generator identity (27 files
byte-identical, 7 changed), the fps wiring, `fetch_tokenizer`, and the flake hardening (accepted as honest,
with its own 20-run stress loop green). Its findings and what I did:
- A-F1/F2/F3/F4/F5/F6/F7/F8/F9 (minor): fixed — see the round-1 fix commit below. F1 (a non-typed media
  error could still fail the whole corpus), F2 (the query shape could get a document-only media row), F3
  (part order not in the replay key), F4 (the batch wording; the probe now records `None` for a multi-input
  capture), F5 (`__all__`), F6 (the pairs-diff wording), F7 (the missing CHANGELOG entry), F8 (a video-only
  recipe's edges), F9 (the stress-loop wording).

**Lens B: FAIL** (1 major, 9 minors) — the major is real and fixed:
- **B-1 (major): the every-text render comparison had no failing test.**  Its mutation (reverting
  `_reference_rows`/`_served_texts_by_row`) left the whole relevant suite green.  Fixed with
  `test_stage1_render_check_covers_every_document_of_a_row`: a scratch reference that diverges on the second
  document only; it is **red under the pre-fix behaviour** (verified in place: `assert 1 == 2`, "the
  reference must be asked to render every document") and green on the fix.
- B-2/B-3 (minor): the wiring's media model raised bare errors (a PIL `UnidentifiedImageError` still failed
  the whole corpus) and re-derived the image size / data-URI decode.  Fixed: one shared public
  `equivalence.media.sent_media_content` reads a sent part through the product's own
  `image_dimensions`/`MediaResolver.bytes_of`/`probe_video_header`, the wiring raises
  `EmulatorUnmodelledError` for a part it cannot read, and
  `test_a_media_part_the_wiring_cannot_read_is_skipped_and_named` pins it.
- B-4 (minor): the fixture reference's fps rule — decided and stated (independent restatement by design; see
  Open questions).
- B-5 (minor): the stub's pair-prompt count was unpinned.  Fixed with
  `test_the_stub_counts_a_rerank_pairs_rendered_prompt`: the probe passes against the stub for a rerank
  recipe with a served template, and a mutant stub that counts the spans fails it.
- B-6 (minor): `text_only_conversation` is now in `stages.__all__`; the unused `probe["_capture"]` is gone
  and the dead `_engines.text_only` wrapper is deleted.
- B-7 (minor): `CORPUS_PLAN_VERSION` 1 -> 2 (the new video edges change the plan; the pairs sampling is
  untouched).
- B-8 (minor): the stale docstrings fixed (`requests.py`, `media_set.py`, `media.py`).
- B-9 (nit): the vacuous assert replaced with `isinstance(..., MediaPrompt)`.

Both lenses' suites: root 3627 passed/102 skipped; test package 927 passed/222 skipped (their first run's
single teardown error was this report file appearing mid-run; their re-runs are clean); contract+docs 304
passed/55 skipped; recipe tests 250 passed/221 skipped; mkdocs strict clean; ruff/format/basedpyright clean;
the pairs manifest and the append-only verification records self-consistent; `git status` clean.

**Round 2** (one fresh confirmation verifier, both lenses, on the merged tree): **PASS**.  It confirmed
every round-1 fix with its own reproductions and mutations (the every-text test red under the reverted
comparison, the typed-media skip red under a bare error, the one sent-media reader, the stub mutant, the
per-shape row pick red under the pre-fix pick, the placement keys, `__all__`, `CORPUS_PLAN_VERSION`, the
docstrings, the video-only edges) and ran the suites on the merged tree (root 3674 passed/102 skipped, test
package 960 passed/222 skipped, contract+docs 301 passed/55 skipped, mkdocs clean, ruff/basedpyright clean).
Its five minors, all fixed in the round-2 commit: the pairs-diff wording was off by one
(qwen3-vl-reranker-8b gains only the over-cap row -- `empty_query: refuse`), the `_engines.text_only` wrapper
is now really deleted, `fixture-vl-video`'s reference render mode now emits its declared query shape too (a
full stage 1 with a reference failed `render_check` before: the reference rendered no query row), the
merge-note's corpora sentence is scoped to the five re-keyed corpora, and the video-only `media_edges` branch
has a test of its own.

## Checks

On the merged tree (see the merge note below), all run from the lane worktree:

- `uv run --no-sync ruff check .` — All checks passed; `ruff format --check .` — 587 files already
  formatted; `uv run --no-sync basedpyright` — 0 errors, 0 warnings, 0 notes.
- `heavy uv run --no-sync pytest tests/ -q -n 4 -p no:cacheprovider` — **3674 passed, 102 skipped**.
- `heavy uv run --no-sync pytest rcp-ndcg-test/tests -q -p no:cacheprovider -o faulthandler_timeout=300`
  — **960 passed, 222 skipped**.
- `uv run --no-sync pytest tests/contract tests/docs -q -p no:cacheprovider` — **301 passed, 55 skipped**.
- `uv run --no-sync mkdocs build --strict -d <scratch>/site` — Documentation built.
- **`bin/gate lane/harness-media` on `0d31abdd` (the first merged tree plus the round-2 fixes) — GATE:
  PASS**, every step green: ruff-check / ruff-format (587 files) / basedpyright (0 errors); `pytest` 3674
  passed, 102 skipped; `contract-docs` 301 passed, 55 skipped; `mkdocs` built; `test-pkg` 960 passed, 222
  skipped; `recipes` exit 0 (no failure outside the baseline, 0 baseline failures remain); `vllm-pkg` 49
  passed; `vllm-models` 72 passed, 7 skipped; `run_all` 1022 checks/987 match/35 known deviations/0 failed,
  human study 67/67, external LLM judges 82/82; `public-names` clean; `clean` (the checkout is clean).
- **`bin/gate lane/harness-media` on `af5976a3` (the second merged tree: the recipe line, wave-integrity,
  mrl-recipes, runner-security, retrieval-fixes) -- GATE: PASS**, every step green: ruff-check /
  ruff-format (592 files) / basedpyright (0 errors); `pytest` 3830 passed, 102 skipped; `contract-docs` 301
  passed, 55 skipped; `mkdocs` built; `test-pkg` 998 passed, 223 skipped; `recipes` exit 0 (no failure
  outside the baseline, 0 baseline failures remain); `vllm-pkg` 49 passed; `vllm-models` 72 passed, 7
  skipped; `run_all` 1022 checks/987 match/35 known deviations/0 failed, human study 67/67, external LLM
  judges 82/82; `public-names` clean; `clean`.
- **`bin/gate lane/harness-media` on `0cb88687` (the third merged tree: run-integrity, mrl-harness) -- GATE:
  PASS**, every step green: ruff-check / ruff-format (598 files) / basedpyright (0 errors); `pytest` 3869
  passed, 102 skipped; `contract-docs` 301 passed, 55 skipped; `mkdocs` built; `test-pkg` 1024 passed, 223
  skipped; `recipes` exit 0 (no failure outside the baseline, 0 baseline failures remain); `vllm-pkg` 49
  passed; `vllm-models` 72 passed, 7 skipped; `run_all` 1022 checks/987 match/35 known deviations/0 failed,
  human study 67/67, external LLM judges 82/82; `public-names` clean; `clean`.
- The stress loop (`scratch/stress-fixed.sh`, six parallel instances of `test_record_and_wave.py` × 4,
  `-X faulthandler`): **all 24 runs × 36 test bodies green** (six runs ended with the checkout-guard
  teardown error naming this report file, which was created while they ran; the guard is the suite's own
  invariant, not a test failure).  A second loop on the pre-merge tree repeated it; the runs that caught the
  tree mid-merge (the fp-v4 patches test's `_CLOSING` use, fixed in the merge) are excluded from the claim.

### The third merge with `rfc-0001`

`git merge rfc-0001` (29 further commits: run-integrity, mrl-harness) at `67e6ef25`; merged as `0cb88687`.
Conflicts and drift resolved keeping **both** lanes' behaviour:

- `equivalence/wire.py`: the MRL lane added a `full_width` strip of the Matryoshka selection for stage 2's
  ex-post gate and built the endpoint config inline; this lane's `recipe_config` is the one-home
  construction, so it gains `full_width` (popping `dimensions`/`mrl_dim`) and `role_client` passes it
  through -- one construction, both behaviours.
- `equivalence/stages.py`: the MRL lane's per-`k` comparison loop and this lane's width-mismatch gate are
  merged (the width check runs on the cut vectors inside the loop, each row carrying its `mrl_dim`).
- `observe/requests.py`: two lanes bumped `CORPUS_PLAN_VERSION` to 2 for **different** plans (the MRL
  stratum; this lane's media video edges).  The merged plan is **3**, with both additions named in the
  docstring -- two different plans must not share a version -- and the MRL lane's test pins 3.
- `tests/stub_engine.py`: this lane's `RLIMIT_CORE` guard and the MRL lane's `--hf-overrides` matryoshka
  parsing both kept.
- The CHANGELOG auto-merged; the version entry now says 3.  Every lane's report matches `rfc-0001`'s
  version except this lane's own; no generated file was hand-merged.

### The second merge with `rfc-0001`

`git merge rfc-0001` (66 further commits: the recipe line's family layout, wave-integrity, mrl-recipes,
runner-security, retrieval-fixes) at `b8832a2e`; merged as `af5976a3`.  Conflicts and drift resolved as: the
CHANGELOG's both-sides-added block kept; the qwen3-embedding-0.6b corpus (re-keyed once more) resolved by
keeping both appended verification lines and re-appending the record under the merged code
(`RCP_APPEND_VERIFICATION=1`); `run_wave.py`'s conflict (the wave-integrity lane's closing flag and upload
verdicts against this lane's per-wave state) resolved by keeping **both** behaviours -- the per-wave
`_Wave`/`_CURRENT_WAVE` guard and the verified per-recipe/wave uploads with `write_summary()` -- and
`test_record_and_wave.py`'s patches test keeps the per-wave `wave=` argument (the wave-integrity lane's
`_CLOSING.clear()` line is gone with the flag).  Per-recipe test modules rfam deleted stay deleted; no
generated file was hand-merged (the pairs manifest, goldens, `DELTAS.json`, schemas and snapshots come from
`rfc-0001`; the corpora's verification records were re-appended the documented way).  Every lane's report
matches `rfc-0001`'s version except this lane's own.

### The first merge with `rfc-0001`

`git merge rfc-0001` (40 commits: core-records, content-wire A5/A7/A8, fp-v4, the judge recipes) at
`26da5852`; merged as `b4340f3e`, plus the re-appended verification records (`283ddb6a`), the round-2 fixes
(`5427e08e`, `0d31abdd`) and the final gate on `0d31abdd`.  Conflicts and
drift resolved as: the CHANGELOG's both-sides-added blocks kept; the five re-keyed corpora's
`verification.jsonl` conflicts resolved by keeping both appended lines (mine from the old fingerprint,
upstream's from the rcp-fp/4 one) and then re-appending the five re-keyed corpora's records under the merged
code (`RCP_APPEND_VERIFICATION=1`, the documented append-only writer; the other seven corpora keep their
2026-10-07 records -- the key is additive and their next re-verification adds it); the two import-line unions in
`equivalence/media.py` (`reference_of` + `prompt_tokens`) and `stages.py`; and the fp-v4 test
`test_the_wave_start_renders_the_recipes_patches_into_the_engine_environment`, which monkeypatched the
removed `_CLOSING`, adapted to the per-wave `_Wave`/`_CURRENT_WAVE` state.  No lane behaviour changed in the
merge; the whole test package is green on the merged tree.

## Open questions

- **The flake's root cause is still unnamed.** The three fixed hazards are real and each has a test, but
  the SIGSEGV itself was never reproduced (~50 wave runs + 3 suites clean; `/coredumps` armed). If it
  recurs, the machine's `core_pattern` (`/coredumps/core.%t.%e.%p`) plus the stub's new `RLIMIT_CORE` will
  capture the crashing process's core (a SIGSEGV core, not the abort tests'): `rocgdb --batch -ex bt
  .venv/bin/python /coredumps/core.*` then names the frame. The one unexplained signature is that
  faulthandler reported nothing for a signal that should be catchable — a core would settle whether the
  fault is inside the handler, in a thread it cannot walk, or not a fault at all.
- **The every-text render comparison relies on the reference contract** "renders the row's first document"
  (the harness writes one row per document, so `row["documents"][0]` IS that document).  Every shipped
  reference follows it (checked for the fixture and read for the shipped families); a future reference that
  renders all documents itself would need that mapping revisited.
- **The emulator's rerank reply for a media document** composes `document.text` as the content's text
  parts joined (media dropped). No media rerank corpus exists yet; the first recording (E2) decides whether
  the engine echoes something else, and the conformance replay would fail loudly on the difference.
- **The emulator's usage for a media request** assumes the engine's prompt is the render of the text parts
  plus the product's media count — the same identity the media stage's engine check verifies on the node
  (engine with media minus engine without). If that check passes on the GPU, the emulator's count matches;
  if the engine's placeholder arithmetic differs from the product's count, both the media stage and the
  emulator's `usage` fail together, which is the right place to see it.
- **`GENERATOR_SEED` is frozen at the version-1 sampling string** (`"1/rcp-observe-v1"`), which is
  deliberately not the human-facing `SEED`: a future deliberate re-sample changes `GENERATOR_SEED` and
  bumps `GENERATOR_VERSION` together (the docstring says so).
- **The fps-arm fixture** adds the realised-frame rule to `fixture-vl-video`'s reference (a synthetic card
  restating the rule in its own constants, like its `card_resize`): the fixture reference stays independent of
  the product by design, so the rule is a deliberate second statement, not an import; the shipped qwen3-vl
  recipes' own references compute it (media-rules).  The fixture's declared *query* frame was also aligned
  with its served chat template (both are the document frame; the served template frames every conversation
  the same way), which the new per-shape media-probe test exposed: stage 1 on that fixture previously failed
  its template check for the query shape.
- **`all-retrieval.txt` and the recipe catalog** were untouched by this lane (the 34 pairs files are the
  recipe catalog's, not the wave list's).

## CHANGELOG entry

See `CHANGELOG.md` under `## Unreleased` (the entries added by this lane): "The request generator's
identity separates its semantic version from its sampling seed", "The verified fake engines model media
and chat-shaped records", "`rcp_ndcg_test.equivalence.wire` exposes `recipe_config`", "Stage 1 gains the
engine prompt-token probe", "The media stage declares its scope", "The media request set gains the video
protocol edges" (Public surface); "The whole corpus no longer fails on one unmodelled media or chat
record", "The anchor audit is no longer vacuous for `anchor: mean`", "The render comparison covers every
text of every row", "A width mismatch gates stage 2 with both widths", "The media gate's fps arm is pinned
end to end", "`fetch_tokenizer` refuses a stale warm cache", "The CPU stub engine counts a pair's rendered
prompt" (Fixed); "Every pairs file was regenerated with the fixed generator under one recorded version",
"Stage 1's template check renders a media row per shape", "The verification records of the current corpora
were re-appended" (Changed).

## Public surface changes

`rcp-ndcg-test` is unpublished; its names are not in the contract snapshots. New/changed public names:

- `rcp_ndcg_test.observe.requests`: `GENERATOR_SEED` (new), `GENERATOR_VERSION` (1 -> 2);
  `rcp_ndcg_test.observe` re-exports it.
- `rcp_ndcg_test.engines`: `ChatPrompts`, `MediaIdentity`, `MediaPrompt`, `RequestPrompts` (new),
  `Prompt` widened, `VllmEmulator.unmodelled_records` (new), `FIELD_CLASSES["messages"]` is `prompt`,
  `request_context`'s chat default.
- `rcp_ndcg_test.errors`: `EmulatorUnmodelledError` (new).
- `rcp_ndcg_test.equivalence.wire`: `recipe_config`, `prompt_tokens` (new).
- `rcp_ndcg_test.equivalence.media`: `MEDIA_GATE_SCOPE`, `MEDIA_GATE_SCOPE_NOTE` (new); the stage
  document carries `scope`/`scope_note`.
- `rcp_ndcg_test.equivalence.stages`: `text_only_conversation` (new); `stage1_prompts`' document gains
  `engine_prompt_tokens_check`; `_render_check`'s per-text shape.
- `rcp_ndcg_test.observe.media_set`: `media_edges` gains `edge:too_many_videos`, `edge:corrupt_video`.
- Corpus artifacts: the 5 current corpora's `verification.jsonl` gain one appended record (the new
  `unmodelled_records` key); all 34 pairs files and the pairs manifest regenerated.

No CLI, exit-code, product-schema or published-package change.

## Files outside scope

None. (The lane touched the files its brief assigns plus tests, the corpus verification records, the pairs
files, CHANGELOG and docs.)

## Docs updated

- `docs/how-to/add-a-model.md` — the media stage's INPUT scope; the `anchor: mean` audit; the
  every-text render comparison; the `engine_prompt_tokens_check`; the template check's media row.
- `docs/how-to/use-verified-fake-engines.md` — the chat/media prompt model, `MediaIdentity`, and the
  skip-and-name behaviour.
- `rcp-ndcg-test/README.md` — the media stage as an input gate (`scope: input`).
- `rcp-ndcg-test/schema/observation-corpus.md` — `GENERATOR_SEED`, the corpus collector's
  `generator_seed`, and the video protocol edges.
- `CHANGELOG.md` — the entries above.
- Greps run (over `docs/ README.md REPRODUCIBILITY.md skills/ examples/ experiments/` and
  `rcp-ndcg-test/schema`): `GENERATOR_VERSION`, `GENERATOR_SEED`, `MEDIA_GATE_SCOPE`,
  `engine_prompt_tokens_check`, `unmodelled_records`, `EmulatorUnmodelledError`, `MediaIdentity`,
  `ChatPrompts`, `RequestPrompts`, `edge:too_many_videos`, `max_videos`, `anchor: mean`,
  `template_render_check`, `render_check`, `anchor_check`.

## For the next lanes

- **ref-envs (the media output half)**: the input gate is labelled; when the media stage 2 lands, replace
  the label with the real scope in `equivalence/media.py`, `EQUIVALENCE.md` and
  `docs/how-to/add-a-model.md`.
- **E2 (the recording wave)**: the emulator now models media and chat records, so
  `qwen3-vl-embedding-2b` can be re-recorded; the first media corpus exercises `unmodelled_records` (it
  should be empty) and the rerank reply's `document.text` composition (Open questions).
- **The flake**: if it recurs, the core is in `/coredumps` (the stub no longer writes cores); the three
  fixed hazards are the first places to re-read.
