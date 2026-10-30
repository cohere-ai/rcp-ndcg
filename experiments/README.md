# Reproducing the paper's tables from public data

This folder recomputes the main tables and headline numbers of
[Rubric-Calibrated Preferences (arXiv:2609.35739)](https://arxiv.org/abs/2609.35739), as printed in its version 2,
from the public datasets on the Hugging Face Hub. No LLM is called and no credentials are needed. The folder is
not part of the installed `rcp_ndcg` package: the scripts import the library, and the library never imports them.

The existing table-reproduction scripts print the paper's value next to the reproduced one and exit with status 1
if any value falls outside its stated tolerance. The calibration pilot below reports measured diagnostics instead.

## Run

From the repository root:

```bash
pip install ./rcp-ndcg-core ./rcp-ndcg     # or: uv sync
pip install -r experiments/requirements.txt      # huggingface-hub, pyarrow
python experiments/fetch_data.py                 # about 150 MB into experiments/data/
python experiments/run_all.py                    # all checks, about one minute on a laptop
```

Each script also runs on its own (`python experiments/leaderboards.py --suite bright`, and so on) and fetches
the files it needs if they are missing. `RCP_EXPERIMENTS_DATA` moves the data directory for the checks; `fetch_data.py --data-dir` moves only
its own download (the checks resolve the environment variable, else `experiments/data`). Files land under `<data-dir>/<repo>/<revision>/`, so data fetched for arXiv v1 (release 1, or an
earlier checkout's unkeyed `experiments/data/<repo>/`) is never read for these checks; it can be deleted. The
dataset revisions are pinned in `fetch_data.py`: each dataset's release 2, with the corrected
Stage A parse and refit behind arXiv v2 (see
[REPRODUCIBILITY.md](../REPRODUCIBILITY.md#tournament-answers-the-papers-code-could-not-parse)).
`paper_values.json` holds the values arXiv v2 prints; release 1 and the arXiv v1 values are in the repository's
history. arXiv v2 credits tied reranker scores with their group's mean gain on every suite and for both metrics, as
the library's presets do, and leaves out the judged cases that no longer have, under the v2 fit, the property they
were selected for.

## Calibration refit pilot

`refit_calibration.py` exercises the released 2PL calibration API on the pinned public NanoBEIR Stage A tournament scores and Stage B rubric verdicts. The default example is one 50-query subset, not the paper's shared fit across all NanoBEIR subsets:

```bash
python -m experiments.refit_calibration --subset NanoSciFactRetrieval --holdout
```

The script fetches only that subset's two provenance files, chooses the `Qwen3.5-397B-A17B-NVFP4 (tournament, judge)` Stage A scores, and joins every rubric verdict by `(subset, query-id, corpus-id)`. It rejects missing or non-finite scores and any rubric document without a score, rather than silently accepting a partial fit. The data revision is the NanoBEIR commit pinned in `fetch_data.py` (`cb6c981c80a70b6a46f8b7ba3a60206a9fab5d8e`). No LLM, Cohere key, or GPU is needed. Install the `calibrate` extra (CPU PyTorch is enough) for the 2PL fit.

`--holdout` chooses 20% of distinct documents *within each query* by a stable SHA-256 ordering and removes all of those documents' repeated Stage B placements from the fit. Held-out Brier score and 10-bin expected calibration error use the existing library diagnostics. Two references use the same training placements: per-criterion training pass rates (deliberately weak, no tournament score) and one pooled logistic curve per criterion on raw tournament scores (no per-query affine map). This checks predictions for new judged documents within queries already fitted, given their public tournament scores. Without `--holdout`, the diagnostics are explicitly in-sample. `--all-nanobeir` fits one model across all 13 subsets; it is more costly and should be run before comparing item parameters with any paper-wide result. A fresh fit is not expected to reproduce the released gains byte for byte.

On a Linux CPU run of the command above, SciFact had 50 queries, 7,500 scored documents and 49,860 rubric placements. The document split left 6,000 documents and 39,873 placements for fitting, then 1,500 documents and 9,987 placements for evaluation; the fit skipped no observations and converged. All five criteria give 49,935 held-out binary verdicts. Lower Brier/ECE is better:

| Fit or reference | Pooled held-out Brier | Pooled held-out ECE (10 bins) |
|---|---:|---:|
| 2PL with per-query affine calibration | 0.016325 | 0.002706 |
| Per-criterion logistic on raw tournament score | 0.019570 | 0.004913 |
| Per-criterion training pass rate | 0.029294 | 0.004126 |

This is one fixed split of one subset, not a paper-wide claim. The 2PL Brier is lower than the raw-score logistic reference on C1-C3 but **higher on C4 and C5** (0.005353 vs 0.003840 and 0.005432 vs 0.004889). The command prints every criterion's counts and scores so those differences remain visible.

## What is reproduced

| Script | Paper artefact | Public input | Result |
|---|---|---|---|
| `leaderboards.py` | NanoBEIR table: qrel-nDCG@10 and RCP-nDCG@10 of 14 rerankers on 13 datasets and their mean (Qwen3.5-397B) | [`rcp-ndcg-nanobeir`](https://huggingface.co/datasets/fabianschmidt-cohere/rcp-ndcg-nanobeir) `qrels`, `top_ranked`, `provenance/runs` | all 392 cells |
| `leaderboards.py` | BRIGHT table (12 tasks, BRIGHT's `excluded_ids` removed) | [`rcp-ndcg-bright`](https://huggingface.co/datasets/fabianschmidt-cohere/rcp-ndcg-bright) `qrels`, `excluded`, `provenance/runs` | all 182 RCP cells; 157 of 182 qrel cells, the other 25 on TheoremQA Theorems and the means (see below) |
| `leaderboards.py` | ViDoRe v3 table (8 corpora, native-language questions) | [`rcp-ndcg-vidore-v3`](https://huggingface.co/datasets/fabianschmidt-cohere/rcp-ndcg-vidore-v3) native-language `qrels`, `top_ranked`, `provenance/runs` | all 252 cells |
| `leaderboards.py` | TREC-DL 2019/2020 reranker pairs (pool panel, Qwen3.6-27B, calibration CAL-A): pairs separated by a paired t-test, agreement with NIST on the NIST-decisive pairs, Kendall tau of the two leaderboards | [`rcp-ndcg-trecdl`](https://huggingface.co/datasets/fabianschmidt-cohere/rcp-ndcg-trecdl) `qrels`, `top_ranked`, `provenance/runs` | all 14 values |
| `human_study.py` | Blind human study: 311 contests, 46 annotators, 933 reviews, 7,080 counted grades on 2,268 documents; the verdict table (main measure 72.9% of 181, disagreements 72.0% of 150, both margins at least 0.02 77.0% of 126, and so on, with query-clustered CIs); hit rates and margin bands; margin AUCs | [`rcp-ndcg-external-validation`](https://huggingface.co/datasets/fabianschmidt-cohere/rcp-ndcg-external-validation) `human_study/` | all 67 values |
| `external_judges.py` | Tournament-order test with GLM-5.3-flash, DeepSeek-4.1-flash and Kimi-K3 (sets S1, controls, S2, S3; per judge, by majority, S2 per suite) and Study 1 with the same three judges (NanoBEIR and BRIGHT, both reference judges, the rebuilt BRIGHT cases), without the cases flagged `excluded_under_fit` | `rcp-ndcg-external-validation` `blind_comparisons/` | all 82 values |

The per-suite scoring rules (candidate set, excluded documents, tie rule, ideal ranking, rounding) and the nDCG
come from the library: `rcp_ndcg_core.protocol` and `rcp_ndcg_core.metric`. Every preset credits tied scores with
their group's mean gain, for qrel-nDCG@10 and RCP-nDCG@10.

The external-judge cases follow arXiv v2's exclusion rule: a case enters only if, under the v2 fit, it keeps the
property it was selected for. Release 2 records this per case in `excluded_under_fit`: in `external_judges_cases`
(21 of 480 S1 disagreements, 1 of 120 controls, 9 of 447 S2 and 1 of 100 S3 cases) and in `study1_cases` (35 of
2,969 cases whose qrel set has at least as much total RCP gain as the reference judge's set). `external_judges.py`
leaves the flagged cases out. A release without the column (release 1) keeps every case.

### Known deviations

The scripts accept one documented difference, with its own bound and its exact population of cells:

- **BRIGHT, qrel-nDCG@10 on TheoremQA Theorems.** The reranker runs on the two TheoremQA tasks were made on a later
  upstream BRIGHT revision. The paper's qrel column for TheoremQA Theorems used that revision's gold ids, which
  differ on 27 of 58 queries. The hub keeps the `xlangai/BRIGHT@a75a0eb4` qrels that mteb's `BrightRetrieval`
  uses. Under arXiv v2's group-mean ties, all 14 cells differ by up to 2.52 points, and 11 of the 14 qrel means by
  up to 0.22 points. RCP-nDCG is unaffected: all its BRIGHT cells, TheoremQA included, reproduce per query.
Its two populations (the 14 cells and the 11 means) are enforced, not just described: the checks fail a run in
which more cells deviate than documented -- or fewer, which would mean this page is stale. No other value of any
table or study may deviate at all.

## What is not reproduced here

These results need LLM calls or inputs that are not public, or are not covered by these scripts:

| Result | Status |
|---|---|
| Re-judging a pool (Stage A tournament, Stage B criteria) | Needs an LLM endpoint; use the `rcp-ndcg` pipeline (see the main README). |
| Full-suite refitting of the 2PL calibration, ICCs, item-parameter comparison and calibration ablations | Not covered by `run_all.py`. `refit_calibration.py` provides a bounded subset refit and a document-held-out Brier/ECE diagnostic; its `--all-nanobeir` mode can fit the shared NanoBEIR model, but it does not compare its parameters to the paper. The inputs are public (the tournament judge strategy in `provenance-runs` and the Stage B verdicts in `provenance-judge-criteria`). |
| Leaderboards with the second judge (gpt-oss-120b) and the TREC-DL calibrations CAL-B and CAL-C | Those gains are not in the public datasets. |
| TREC-DL analyses over 18 systems (the 14 rerankers plus 4 first-stage runs) | The first-stage runs are not released. |
| Count-nDCG leaderboards and sensitivity analyses | Not covered. |
| Re-running the external LLM judges | Needs the three judges' endpoints. Every prompt and answer is in `meta_judges/` of the external-validation dataset, and the verdicts checked above are parsed from them. |
| Study 1 judgments by the authors | Not released. |
