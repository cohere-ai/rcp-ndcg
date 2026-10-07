# Workstream 06: the final docs (after the layout move)

Read `handover/00-MASTER.md` first. Specs (binding): the three blueprints and their owner answers in
`handover/specs/`: `docs-firstcontact-blueprint.md` + `-answers.md`, `docs-release-blueprint.md` + `-answers.md`,
`docs-site-blueprint.md` + `-answers.md`. They were written against older refs: re-check every claim on the tip.

## Already done on `rfc-0001` (the path-independent half)
The site is split into Concepts / How-to / Reference (`docs/tutorials/` -> `docs/how-to/`; served-role budgets in
`docs/concepts/text-budgets.md`; `serving.md` -> `judges.md` + `runs.md`); `SECURITY.md`; four first-contact paths
(score, serve-and-score, re-judge, reproduce); the distributions section; the new `how-to/serve-a-model.md` and
`how-to/validate-a-recipe.md`; the p1-tail features documented; examples 08 and 09; the agent skill and the contributor
contract updated. Deliberately NOT yet written (they were future-true before the move): the `recipe: <id>` mapping and
`rcp-ndcg-vllm serve <id>` usage, the lean dependency set, the release gates added by the move (fresh-venv wheel
installs, the `--no-deps` freeze check, `serve --dry-run`), the 18-recipe catalog as servable.

## Your scope
1. Every docs edit that names moved paths: the README split (Q1(a): the full README into `rcp-ndcg/` as the PyPI long
   description, a short landing card at the root naming the four distributions and linking each README and the
   docs); install snippets (`#subdirectory=` paths), `AGENTS.md`'s layout and layering sections and its one-home table
   paths, `REPRODUCIBILITY.md`, `docs/how-to/release-candidates.md`, `docs/how-to/add-a-model.md`,
   `skills/rcp-ndcg/`, mkdocs nav.
2. The now-true sentences listed above, written from the code (run each snippet; `tests/docs` runs them).
3. The recipe catalog page: one row per public recipe with a status column read from each recipe's `status:`
   (`unverified` until GPU waves verify it — never claim otherwise), the reference column (the paper's code or the model
   card), no internal "kind" column (docs-site OQ-5).
4. `CHANGELOG.md`: fold `## Unreleased` into one coherent `## 0.0.1 — <date set at tag time>` section (the
   Versioning preamble stays; there is a stale "0.1.0 first public release" section from before the version reset —
   reconcile it into 0.0.1, nothing is lost); every recipe family's bullets folded into one recipes entry;
   `rcp-ndcg-test` mentioned only where it affects published packages (docs-release Q2).
5. Every number in the docs has a reproducible source (AGENTS.md); public names only.
Gate: the MASTER quality bar (`tests/docs` and `mkdocs build --strict` especially). Report in
`handover/reports/06-docs.md`.
