# Reproducing the paper's tables from public data

This folder recomputes the main tables and headline numbers of
[Rubric-Calibrated Preferences (arXiv:2609.35739)](https://arxiv.org/abs/2609.35739) from the public datasets on
the Hugging Face Hub. No LLM is called and no credentials are needed. The folder is not part of the installed
`rcp_ndcg` package: the scripts import the library, and the library never imports them.

Each script prints the paper's value next to the reproduced one and exits with status 1 if any value falls
outside its stated tolerance.

## Run

From the repository root:

```bash
pip install ./rcp-ndcg-core .           # or: uv sync
pip install -r experiments/requirements.txt      # huggingface-hub, pyarrow
python experiments/fetch_data.py                 # about 150 MB into experiments/data/
python experiments/run_all.py                    # all checks, about one minute on a laptop
```

Each script also runs on its own (`python experiments/leaderboards.py --suite bright`, and so on) and fetches
the files it needs if they are missing. `RCP_EXPERIMENTS_DATA` or `fetch_data.py --data-dir` moves the data
directory. The dataset revisions are pinned in `fetch_data.py`.

## What is reproduced

| Script | Paper artefact | Public input | Result |
|---|---|---|---|
| `leaderboards.py` | NanoBEIR table: qrel-nDCG@10 and RCP-nDCG@10 of 14 rerankers on 13 datasets and their mean (Qwen3.5-397B) | [`rcp-ndcg-nanobeir`](https://huggingface.co/datasets/fabianschmidt-cohere/rcp-ndcg-nanobeir) `qrels`, `top_ranked`, `provenance/runs` | all 196 qrel cells; 186 of 196 RCP cells within the printed rounding, 10 within 0.07 points (see below) |
| `leaderboards.py` | BRIGHT table (12 tasks, BRIGHT's `excluded_ids` removed) | [`rcp-ndcg-bright`](https://huggingface.co/datasets/fabianschmidt-cohere/rcp-ndcg-bright) `qrels`, `excluded`, `provenance/runs` | all 182 RCP cells; 157 of 182 qrel cells, the other 25 on TheoremQA Theorems and the means (see below) |
| `leaderboards.py` | ViDoRe v3 table (8 corpora, native-language questions, tied scores get their group's mean gain) | [`rcp-ndcg-vidore-v3`](https://huggingface.co/datasets/fabianschmidt-cohere/rcp-ndcg-vidore-v3) native-language `qrels`, `top_ranked`, `provenance/runs` | all 252 cells |
| `leaderboards.py` | TREC-DL 2019/2020 reranker pairs (pool panel, Qwen3.6-27B, calibration CAL-A): pairs separated by a paired t-test, agreement with NIST on the NIST-decisive pairs, Kendall tau of the two leaderboards | [`rcp-ndcg-trecdl`](https://huggingface.co/datasets/fabianschmidt-cohere/rcp-ndcg-trecdl) `qrels`, `top_ranked`, `provenance/runs` | all 14 values |
| `human_study.py` | Blind human study: 311 contests, 46 annotators, 933 reviews, 7,080 counted grades on 2,268 documents; the verdict table (main measure 72.4% of 185, disagreements 69.9% of 156, both margins at least 0.02 74.8% of 131, and so on, with query-clustered CIs); hit rates and margin bands; margin AUCs | [`rcp-ndcg-external-validation`](https://huggingface.co/datasets/fabianschmidt-cohere/rcp-ndcg-external-validation) `human_study/` | all 67 values |
| `external_judges.py` | Tournament-order test with GLM-5.3-flash, DeepSeek-4.1-flash and Kimi-K3 (sets S1, controls, S2, S3; per judge, by majority, S2 per suite) and Study 1 with the same three judges (NanoBEIR and BRIGHT, both reference judges, the rebuilt BRIGHT cases) | `rcp-ndcg-external-validation` `blind_comparisons/` | all 82 values |

The per-suite scoring rules (candidate set, excluded documents, tie rule, ideal ranking, rounding) and the nDCG
come from the library: `rcp_ndcg_core.protocol` and `rcp_ndcg_core.metric`.

### Known deviations

The scripts accept two documented differences, each with its own bound and its exact population of cells:

- **NanoBEIR, RCP-nDCG@10: 10 of the 182 dataset cells.** The hub's NanoBEIR gains come from a calibration fit that differs
  slightly from the one behind the paper's table (gains differ by at most 0.0065). Ten cells (FEVER 5, Quora 2,
  NFCorpus 2, HotpotQA 1) move by at most 0.07 points and print 0.1 away from the paper. All 14 means print
  identically.
- **BRIGHT, qrel-nDCG@10 on TheoremQA Theorems.** The reranker runs on the two TheoremQA tasks were made on a later
  upstream BRIGHT revision. The paper's qrel column for TheoremQA Theorems used that revision's gold ids, which
  differ on 27 of 58 queries. The hub keeps the `xlangai/BRIGHT@a75a0eb4` qrels that mteb's `BrightRetrieval`
  uses. Of the 14 cells, 13 differ by up to 2.5 points, and 12 of the 14 qrel means by up to 0.21 points. RCP-nDCG
  is unaffected: all its BRIGHT cells, TheoremQA included, reproduce per query.

Both populations are enforced, not just described: the checks fail a run in which more cells deviate than
documented -- or fewer, which would mean this page is stale. No other cell of any table may deviate at all.

## What is not reproduced here

These results need LLM calls or inputs that are not public, or are not covered by these scripts:

| Result | Status |
|---|---|
| Re-judging a pool (Stage A tournament, Stage B criteria) | Needs an LLM endpoint; use the `rcp-ndcg` pipeline (see the main README). |
| Refitting the 2PL calibration and its diagnostics (ICCs, item parameters, calibration ablations, Brier and ECE) | Not covered. The inputs are public (the judge's tournament scores are the `is_judge` runs in `provenance-runs`, the Stage B verdicts are `provenance-judge-criteria`), and the fit is the library's calibration API. |
| Leaderboards with the second judge (gpt-oss-120b) and the TREC-DL calibrations CAL-B and CAL-C | Those gains are not in the public datasets. |
| TREC-DL analyses over 18 systems (the 14 rerankers plus 4 first-stage runs) | The first-stage runs are not released. |
| Count-nDCG leaderboards and sensitivity analyses | Not covered. |
| Re-running the external LLM judges | Needs the three judges' endpoints. Every prompt and answer is in `meta_judges/` of the external-validation dataset, and the verdicts checked above are parsed from them. |
| Study 1 judgments by the authors | Not released. |
