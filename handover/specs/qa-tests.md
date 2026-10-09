<!-- Handover copy of the operator's working note `drafts/qa-tests.md` (sanitized: private names, the private repository, the job platform and storage locations removed; `<operator-notes>/...` paths point to files that are NOT available — the content you need is in handover/). Treat it as the specification it was for its lane. -->

# QA review `qa-tests`: is the test suite evidence, or a sense of security?

**Crux.** The suite has about 2,400 tests in 123 files (24k lines) and 481 uses of mocks or monkeypatching, and it is
green. Establish, with measurements, how much of that is evidence that the package works on real engines and data,
and how much only re-asserts what its authors wrote. Then propose the test architecture the release should have, and
the concrete restructuring (what to delete, merge, rewrite, add) to get there.

Read-only for the repository. Work in your own detached worktree `/tmp/qa-tests/wt` (`git -C <repo>
worktree add --detach /tmp/qa-tests/wt <commit>`; remove it at the end with `git worktree remove`) and your own venv
(`.github/scripts/cpu-env.sh dev docs` inside that worktree, then `uv pip install coverage pytest-cov mutmut` into
that venv only). Never touch `<repo>/.venv` or another lane's worktree. Also read `drafts/qa-COMMON.md`
rules (`<operator-notes>/drafts/qa-COMMON.md`): evidence, severities, scope boundaries.

## Measure (each with the command, the number, and its referent)
1. **Coverage**, line and branch, per module of `rcp-ndcg/src/rcp_ndcg/` and `rcp-ndcg-core/src/` and
   `rcp-ndcg-vllm/src/`: which code paths no test executes; which modules are covered only through mocks
   (run coverage with the mocking tests excluded once, to see what real-path coverage remains).
2. **Mutation score** on the modules that decide numbers or correctness: `rcp_ndcg_core` (metric, gain, protocol, IRT
   fit entry points), `data/preprocess.py` (`fit`, `token_prefix`, chunking), `data/resolution.py`, `inference/transport.py`,
   `inference/adapters/*.py`, `eval/evaluate.py`, `runs/pipeline.py` (`_identity`), `support/serve.py` (`plan_phases`).
   Use mutmut (or a small AST mutator of your own if mutmut cannot run here) with a time budget; report killed /
   survived per module and list the surviving mutants that matter (each is a missing test).
3. **Test inventory**: per test file, its subject, its kind (pure unit, mocked I/O, recorded wire, fake-server
   integration, end-to-end pipeline, contract/snapshot, docs snippet), its runtime (`--durations=0`), and its mock
   count. Flag: tests that assert a mock's own canned value back (tautological), tests that only check types or shapes,
   near-duplicate tests (same behaviour asserted in several files), tests of removed or soon-removed code (the
   in-process paths lane l3e deletes), skipped tests that can never run, tests that write outside `tmp_path`, flaky
   tests (run the full suite 3 times with `-n 4` and different `-p randomly`-style orders if a plugin is available, or
   `--count`; report every non-deterministic test — `tests/test_end_to_end.py::test_a_job_runner_runs_the_same_pipeline_started_over_mcp`
   is a known one).
4. **Integration gaps**: list the behaviours that only a real engine or real data can show and that no offline test
   approximates — the wire shapes of vLLM v0.31.0 (`/v1/embeddings`, `/pooling` base64 and bytes, `/rerank`,
   chat completions, error bodies), real tokenizers' anchors, real media processors' token counts, the phased job
   script with real processes, the transport against a slow or failing real server. For each, say which offline test
   could replace a mock with a recording (RFC-0001 lane L7: recorded engine responses under
   `tests/contract/engines/vllm-0.31.0/`, produced by the GPU waves) or a real small model on CPU.

## Propose (deliverables, in `<operator-notes>/research/qa-tests/work/`)
- `report.md`, `digest.md` (your preamble's format), the measurement scripts and raw outputs.
- **Target architecture**: the test layers and their markers (e.g. `unit`, `contract` (snapshots + recorded engine
  replies), `integration` (fake servers, real tokenizers on CPU, real phased scripts with stub engines), `e2e`
  (offline full pipeline), `gpu` (only on the GPU node, from the wave runner), `network` (opt-in hosted APIs)), the
  directory layout, the shared fixtures, and what CI runs on every push versus nightly.
- **Restructuring plan**: a table of every file — keep, merge into X, rewrite as recorded-wire, delete (with the reason) —
  and the new tests to add, ranked by the defects they would catch (cite the surviving mutants and the gaps). The
  target is fewer, stronger tests: state the expected count after restructuring and why it is enough.
