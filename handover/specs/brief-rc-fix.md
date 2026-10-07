<!-- Handover copy of the operator's working note `rc-fix/BRIEF.md` (sanitized: private names, the private repository, the job platform and storage locations removed; `<operator-notes>/...` paths point to files that are NOT available — the content you need is in handover/). Treat it as the specification it was for its lane. -->

# Lane `rc-fix`: the node-runtime bugs the GPU shakedown found (rc-build follow-up 3)

Read first: `<operator-notes>/COMMON.md` (binding; at most two verifier rounds), `AGENTS.md`,
`<operator-notes>/GPU-VALIDATION.md` section "Node runtime: isolation and resources" (binding),
`<operator-notes>/shake/FINDINGS.md` (the rows marked HARNESS are yours), and the operator's shakedown patches
as evidence (throwaway, not to be merged): `git -C <repo> log -p rfc-0001..shake1 -- packages/rcp-ndcg-vllm/jobs packages/rcp-ndcg-vllm/src/rcp_ndcg_vllm/jobs`.
Base: `rfc-0001`. Scope: `packages/rcp-ndcg-vllm/jobs/`, `packages/rcp-ndcg-vllm/src/rcp_ndcg_vllm/jobs/`, their tests and
`docs/how-to/release-candidates.md`. Never read or print the auth script or the token file.

## Items (failing test first for each; CPU tests with the stub engine and fake CLIs)
1. `jobs.plugins collect`: an invalid recipe in the wave list is reported (stderr, and the wave report marks that recipe
   failed with the validation message) and skipped; it never fails the job. The wave runner keeps "one failing recipe
   never stops the wave" end to end.
2. `submit.sh` creates `RCP_SUBMIT_DIR` when it does not exist.
3. `bootstrap.sh`: a plugin named by a recipe and not staged as a file installs from the staged wheelhouse only
   (`--no-index --find-links <stage>/wheelhouse`); a plugin found nowhere fails that recipe with the exact name.
4. `bootstrap.sh`: the reference venv (`--system-site-packages`) must keep the image's torch stack: install under a
   constraint of the image's own torch/torchvision/torchaudio/triton versions (from the engine python's freeze); a
   requirement that would replace them fails loudly. The report's `reference` block records torch and whether it is
   the image's build (CUDA) — a CPU torch on a GPU node is a failed bootstrap.
5. `weights.disk_free_bytes` (and every disk check) works before the cache directory exists (nearest existing parent).
6. Re-read FINDINGS.md before each verifier round: the operator appends rows during the night; every HARNESS row present
   when you start round 2 is in scope.

## Done when
`bin/gate <your branch>` passes every step (incl. `vllm-pkg`), shellcheck clean, and REPORT.md has the finding -> test ->
red/green table.
