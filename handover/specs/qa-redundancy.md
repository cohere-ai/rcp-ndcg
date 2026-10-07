<!-- Handover copy of the operator's working note `drafts/qa-redundancy.md` (sanitized: private names, the private repository, the job platform and storage locations removed; `<operator-notes>/...` paths point to files that are NOT available — the content you need is in handover/). Treat it as the specification it was for its lane. -->

# QA review `qa-redundancy`: duplication, dead code, and documentation or tests that no longer describe the code

**Crux.** Many lanes wrote code, tests and docs in parallel. Find what exists twice, what nothing uses, and what
describes behaviour the code no longer has.

## Watch for (each item: check it, report what you found, with evidence)
1. **Duplicate implementations**: near-duplicate functions and classes across modules (e.g. response parsing in the
   embedding and rerank adapters, base64 decoding, index realignment, retry or backoff code, token cutting helpers,
   fake servers in tests and in `inference/fake.py`, recipe validation in the recipe package and in rcp-ndcg). Use a
   similarity scan (e.g. a small script comparing normalised ASTs or token shingles of every function over 8 lines)
   and report every pair above a stated threshold with a judgement: merge, or keep with a reason.
2. **Dead code**: functions, classes, constants, parameters, config fields and CLI flags with no caller outside their
   own tests (`git grep` each public and private name you suspect; vulture-style scan in a venv under your scratch
   dir). Include leftovers of the removed in-process paths and the old serve shape.
3. **Unused or redundant dependencies**: every dependency in both `pyproject.toml` files and the recipe package is
   imported somewhere; nothing is imported that is not declared; extras match their use.
4. **Docs versus code**: every docs page under `docs/`, `README.md`, `REPRODUCIBILITY.md`, `skills/rcp-ndcg/`,
   `examples/` and `experiments/**/README.md` — find statements that are now false (removed names, old config shapes,
   old flags, old env variables, numbers that changed, "on PyPI" claims), and docstrings whose described behaviour
   differs from the code. Produce a table: file:line | statement | why false | fix.
5. **CHANGELOG**: the `## Unreleased` section read as one release note: duplicates, contradictions, entries describing
   intermediate states of a lane, entries for changes later reverted, missing entries for public changes (compare with
   the contract snapshot diff since `main`).
6. **Tests**: duplicated tests (same assertion on the same path in two files), tests of removed code, skipped tests that
   can never run, fixtures defined twice, tests that write outside `tmp_path`.
7. **Naming consistency**: the same concept named differently across modules (e.g. `max_tokens` vs `budget` vs
   `max_length`, `api` vs `provider`, `role` vs `side`); propose one name per concept.
Deliver also: the similarity-scan script and its raw output in your scratch dir.
