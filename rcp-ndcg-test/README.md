# rcp-ndcg-test (unpublished)

The validation tooling for RCP-nDCG's served models — **never published to PyPI**. It exercises the product
(`rcp-ndcg`) and the serving recipes (`rcp-ndcg-vllm`) and reimplements neither (R30: the harness drives the
product's role clients).

What it holds:

- `rcp_ndcg_test.equivalence` — the three-stage equivalence harness (render audit with token-id equality and
  the anchor check; score gates against the reference subprocess; tie-robust metrics): `python -m
  rcp_ndcg_test.equivalence --recipe <dir> --base-url <url> --pairs <file> --out <dir>`.
- `rcp_ndcg_test.record` — the engine recorder: the product's role clients observed through one `httpx`
  transport hook, writing the observation corpus.
- `rcp_ndcg_test.jobs` — the RC build (`rc_build.sh`), the node bootstrap (`bootstrap.sh`), the wave runner
  (`run_wave.py`), wave 0 (`wave0.sh` + probes + report) and `submit.sh`; all node-side shellcheck-clean.
- `src/rcp_ndcg_test/cases/`, `conformance/`, `fakes/` — the case reader, the conformance targets and the
  verified model fakes, as the cases lanes land them.

Run its suites from a workspace checkout (`uv sync --locked --extra dev`, then `uv run --no-sync pytest
rcp-ndcg-test/tests -q`). Nothing here reaches PyPI, and `tests/contract` pins none of it (docs-release Q2).
