<!-- Handover copy of the operator's working note `research/docs-release/work/OPERATOR-ANSWERS.md` (sanitized: private names, the private repository, the job platform and storage locations removed; `<operator-notes>/...` paths point to files that are NOT available — the content you need is in handover/). Treat it as the specification it was for its lane. -->

# Operator answers to docs-release's open questions (18:03; binding for the docs lane and the layout move)
- Q1 (a): the full README lives in `rcp-ndcg/` (the PyPI long description); the root README is a short landing card
  (what the project is, the four distributions, links to each README and the docs). CHANGELOG.md, REPRODUCIBILITY.md,
  AGENTS.md, CITATION.cff, schemas/, skills/, examples/ stay at the repository root and are not in sdists; the sdists'
  contents test states it.
- Q2: rcp-ndcg-test carries the same version number, is never published, has no tests/contract surface pin, gets
  CHANGELOG lines only where it affects the published packages, and its own LICENSE/NOTICE and README.
- Q3: public `rcp_ndcg_vllm` names = `rcp_ndcg_vllm.recipe` (Recipe, load_recipe, iter_recipes, the serve-argv
  builder), the `rcp-ndcg-vllm` console tree (`serve`, `--dry-run`), and the exported recipe schema file; everything
  else internal. Pin them in tests/contract.
- Q4: one merged NOTICE, byte-identical in all four distributions (including the folded-in plugins' attributions).
- Q5: yes, a per-recipe status column sourced from each recipe's `status:` (its sources as the receipts).
- Q6: yes, add SECURITY.md (GitHub private vulnerability reporting; supported versions: the latest release), linked
  from the three published READMEs.
- Q7: `## 0.0.1 — <ISO date>`, the date set by the release checklist at tag time.
