# Report 02: recipe-common, the recipe line, and the harness gaps (integrated)

**Status:** DONE. Part A+B landed through the recipe line (merged into `rfc-0001` at M2); part C merged into
`rfc-0001` directly.

## A. recipe-common
- The WIP snapshot was dropped (its CHANGELOG wording contradicted decision 9; its other edits belonged to families).
- The over-length sampler refuses a tokenizer count that never reaches its target; a test bounds its runtime (the
  pre-fix sampler hung).
- One merged NOTICE attributing every ported or copied third-party file of the recipes and plugins; every attributed
  upstream file re-fetched at its pinned revision (SHA-256s match; licences match the Hub metadata).
- The case loader accepts a run-level instruction only as a whole delimited unit of the recipe template's QUERY frame.

## B. The recipe line brought up to date
`rfc-0001` merged into the recipe line; generated files regenerated with no drift; all 18 recipes load under the
stricter validators (two load-blocker fixes taken verbatim from their family branches).

## C. Harness gaps (all test-first)
- G1: token_ids bodies — the anchor audit reads the sent ids; the render check compares the sent ids with the
  reference text's ids under the shape's `add_special_tokens`.
- G2: messages bodies extract their text parts (joined with `TEXT_JOIN`) and media placeholders; an audit that checked
  nothing fails.
- G3: the `anchor: first` head edge is taken from the assembled render's offsets (no false failures on BPE joins).
  token_ids heads are a conservative lower bound (no decode; see the open items).
- The marker audit counts markers in the sent content, not the post-processor's tokens.
- The offline fake draws one seeded SHAKE-256 stream per vector with an exactly rounded norm: deterministic and
  bit-identical across BLAS kernels; 16k tokens x 2048 dims in about 2 s.
- The node bootstrap installs a named plugin from every staged wheelhouse (`EXTRA_DIRS` included).
- The Cloud SDK search reads `RCP_GCLOUD_SDK_DIRS` (one home in `jobs/gcs.sh`); the node-script tests are hermetic (a
  machine's real `gcloud` never stands in for an absent one).
