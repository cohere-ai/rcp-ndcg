# Dependabot and pip-audit analysis (lane relmech, read-only)

**Status: analysis only, 2026-10-12. Nothing was dismissed, closed, upgraded or re-locked.** The owner decides
which alerts to dismiss and whether to move the lock.

## Method

- `gh api repos/cohere-ai/rcp-ndcg/dependabot/alerts --paginate` (authenticated as the owner's `gh`): **51
  alerts, 40 open and 11 fixed**. The three Dependabot pull requests (#1 oauthlib, #2 vllm, #3 transformers) are
  already closed (2026-09-30).
- Dependabot evaluates the repository's **default branch**, and `main` is still the initial commit
  (`de900f35`, "initial commit"): its `uv.lock` carries `vllm 0.16.0`, `sglang`, `oauthlib 3.3.1`,
  `transformers 4.57.6`, `setuptools 80.10.2`, `torch 2.9.1`, `xgrammar 0.1.29` and `diskcache 5.6.3`. Every
  alert below is therefore classified against the **release tree's** lock (`rfc-0001`, merged here), not
  `main`'s.
- Release tree versions, from `uv.lock` / `requirements-constraints.txt` in this worktree:
  `oauthlib 4.0.0`, `transformers 5.17.0`, `setuptools 84.0.0`, `torch 2.14.0`, `pyjwt 2.14.0`; **no** `vllm`,
  `sglang`, `xgrammar` or `diskcache` at all. `rcp-ndcg-vllm`'s dependencies are pydantic and PyYAML; the
  engine is the stock image `vllm/vllm-openai:v0.31.0` (its vLLM 0.31.0 is above every vLLM fix in the list,
  the highest being 0.28.0; its torch is 2.13.0, above the torch fix).
- `pip-audit` against the release tree's constraints file:
  `uvx pip-audit -r requirements-constraints.txt --no-deps --disable-pip` (the default mode creates a venv and
  fails here: this Python build has no `ensurepip`; `--disable-pip --no-deps` audits the pins directly against
  the advisory database). Result: **2 known vulnerabilities in 1 package: `pyjwt 2.14.0`** (PYSEC-2026-4141 /
  CVE-2026-101918 and PYSEC-2026-4183 / CVE-2026-102275; fixed in 2.15.0).

## Headline

**Nothing in the release tree's own lock needs an upgrade for 0.0.1.** Every open alert is either already fixed
in the release lock (`oauthlib`, `transformers`, `setuptools`, `torch`) or no longer a dependency at all
(`vllm` 27, `xgrammar`, `diskcache`); the alerts are stale against `main`'s initial lock and Dependabot will
re-scan when `rfc-0001` is merged. The one finding that is new and actionable is `pyjwt 2.14.0`, reachable only
through the optional `[azure]` extra (`adlfs -> azure-identity -> msal -> pyjwt`), so it is not in the default
install -- a next-lock-move item, not a release blocker.

## Open alerts (40)

One row per alert. "Release pin" is the version in this tree's `uv.lock` (`absent` = not a dependency).

| # | package | manifest | kind | severity | advisory | affected range | fixed | release pin | recommendation |
|---|---|---|---|---|---|---|---|---|---|
| 4 | `oauthlib` | requirements-constraints.txt | unknown | medium | `GHSA-hj66-6f7g-4r5v` | >= 0.6.1, <= 3.3.1 | 4.0.0 | 4.0.0 | supersede at the next lock move (the release tree pins 4.0.0, the fix) |
| 5 | `oauthlib` | requirements-constraints.txt | unknown | medium | `GHSA-xpv3-w29h-x7cv` | >= 3.0.0, < 4.0.0 | 4.0.0 | 4.0.0 | supersede at the next lock move (the release tree pins 4.0.0, the fix) |
| 7 | `diskcache` | uv.lock | transitive | medium | `GHSA-w8v5-vhqr-4h9v` | <= 5.6.3 | none | absent | not shipped (a vllm transitive that left with vllm; no upstream fix exists) |
| 8 | `xgrammar` | uv.lock | transitive | high | `GHSA-7rgv-gqhr-fxg3` | <= 0.1.31 | 0.1.32 | absent | not shipped (a vllm transitive that left with vllm) |
| 9 | `vllm` | uv.lock | direct | medium | `GHSA-v359-jj2v-j536` | >= 0.15.1, < 0.17.0 | 0.17.0 | absent | not shipped (no vllm dependency in the release tree; the engine image is v0.31.0) |
| 13 | `vllm` | uv.lock | direct | high | `GHSA-7972-pg2x-xr59` | >= 0.10.1, < 0.18.0 | 0.18.0 | absent | not shipped (no vllm dependency in the release tree; the engine image is v0.31.0) |
| 14 | `vllm` | uv.lock | direct | medium | `GHSA-3mwp-wvh9-7528` | >= 0.1.0, < 0.19.0 | 0.19.0 | absent | not shipped (no vllm dependency in the release tree; the engine image is v0.31.0) |
| 15 | `vllm` | uv.lock | direct | medium | `GHSA-pf3h-qjgv-vcpr` | >= 0.16.0, < 0.19.0 | 0.19.0 | absent | not shipped (no vllm dependency in the release tree; the engine image is v0.31.0) |
| 16 | `vllm` | uv.lock | direct | medium | `GHSA-pq5c-rjhq-qp7p` | >= 0.7.0, < 0.19.0 | 0.19.0 | absent | not shipped (no vllm dependency in the release tree; the engine image is v0.31.0) |
| 17 | `transformers` | uv.lock | direct | medium | `GHSA-69w3-r845-3855` | < 5.0.0rc3 | 5.0.0rc3 | 5.17.0 | not shipped by default (optional `[mteb]` extra only; 5.17.0 is above the fix) |
| 18 | `vllm` | uv.lock | direct | medium | `GHSA-hpv8-x276-m59f` | >= 0.6.1, < 0.20.0 | 0.20.0 | absent | not shipped (no vllm dependency in the release tree; the engine image is v0.31.0) |
| 19 | `vllm` | uv.lock | direct | low | `GHSA-x368-4g9h-fvv4` | < 0.19.1 | 0.19.1 | absent | not shipped (no vllm dependency in the release tree; the engine image is v0.31.0) |
| 22 | `vllm` | uv.lock | direct | medium | `GHSA-3ww4-5jv9-j5gm` | < 0.22.0 | 0.22.0 | absent | not shipped (no vllm dependency in the release tree; the engine image is v0.31.0) |
| 23 | `torch` | uv.lock | direct | low | `GHSA-qfhq-4f3w-5fph` | < 2.10.0 | 2.10.0 | 2.14.0 | not shipped by default (optional `[calibrate]`/`[irt]` extras only; 2.14.0 is above the fix) |
| 24 | `vllm` | uv.lock | direct | high | `GHSA-q8gq-377p-jq3r` | < 0.22.0 | 0.22.0 | absent | not shipped (no vllm dependency in the release tree; the engine image is v0.31.0) |
| 25 | `vllm` | uv.lock | direct | critical | `GHSA-94f4-hr76-p5j6` | >= 0.3.0, < 0.22.0 | 0.22.0 | absent | not shipped (no vllm dependency in the release tree; the engine image is v0.31.0) |
| 26 | `vllm` | uv.lock | direct | medium | `GHSA-hgg8-fqqc-vfmw` | <= 0.23.0 | 0.24.0 | absent | not shipped (no vllm dependency in the release tree; the engine image is v0.31.0) |
| 27 | `vllm` | uv.lock | direct | medium | `GHSA-6pr9-rp53-2pmc` | <= 0.23.0 | 0.24.0 | absent | not shipped (no vllm dependency in the release tree; the engine image is v0.31.0) |
| 28 | `vllm` | uv.lock | direct | medium | `GHSA-98f3-hwg4-4rf7` | <= 0.19.0 | none | absent | not shipped (no vllm dependency in the release tree; the engine image is v0.31.0) |
| 29 | `transformers` | uv.lock | direct | high | `GHSA-29pf-2h5f-8g72` | < 5.3.0 | 5.3.0 | 5.17.0 | not shipped by default (optional `[mteb]` extra only; 5.17.0 is above the fix) |
| 31 | `transformers` | uv.lock | direct | high | `GHSA-fgcw-684q-jj6r` | < 5.5.0 | 5.5.0 | 5.17.0 | not shipped by default (optional `[mteb]` extra only; 5.17.0 is above the fix) |
| 32 | `vllm` | uv.lock | direct | medium | `GHSA-5jv2-g5wq-cmr4` | >= 0.5.5, < 0.24.0 | 0.24.0 | absent | not shipped (no vllm dependency in the release tree; the engine image is v0.31.0) |
| 33 | `vllm` | uv.lock | direct | medium | `GHSA-8jr5-v98p-w75m` | >= 0.11.0, < 0.24.0 | 0.24.0 | absent | not shipped (no vllm dependency in the release tree; the engine image is v0.31.0) |
| 34 | `vllm` | uv.lock | direct | medium | `GHSA-7h4p-rffg-7823` | >= 0.8.5, <= 0.23.0 | 0.24.0 | absent | not shipped (no vllm dependency in the release tree; the engine image is v0.31.0) |
| 35 | `vllm` | uv.lock | direct | medium | `GHSA-6c4r-fmh3-7rh8` | >= 0.5.5, < 0.18.0 | 0.18.0 | absent | not shipped (no vllm dependency in the release tree; the engine image is v0.31.0) |
| 36 | `torch` | uv.lock | direct | low | `GHSA-rrmf-rvhw-rf47` | <= 2.12.1 | 2.13.0 | 2.14.0 | not shipped by default (optional `[calibrate]`/`[irt]` extras only; 2.14.0 is above the fix) |
| 37 | `vllm` | uv.lock | direct | high | `GHSA-rwxx-mrjm-wc2m` | < 0.24.0 | 0.24.0 | absent | not shipped (no vllm dependency in the release tree; the engine image is v0.31.0) |
| 38 | `vllm` | uv.lock | direct | high | `GHSA-33cg-gxv8-3p8g` | >= 0.12.0, < 0.24.0 | 0.24.0 | absent | not shipped (no vllm dependency in the release tree; the engine image is v0.31.0) |
| 39 | `setuptools` | uv.lock | transitive | medium | `GHSA-h35f-9h28-mq5c` | < 83.0.0 | 83.0.0 | 84.0.0 | not shipped by default (a transitive of torch; 84.0.0 is above the fix) |
| 41 | `vllm` | uv.lock | direct | high | `GHSA-wcwg-c5fc-9vrc` | >= 0.8.0, < 0.19.0 | 0.19.0 | absent | not shipped (no vllm dependency in the release tree; the engine image is v0.31.0) |
| 42 | `transformers` | uv.lock | direct | high | `GHSA-xrqw-3rrv-vx5w` | < 5.10.0 | 5.10.0 | 5.17.0 | not shipped by default (optional `[mteb]` extra only; 5.17.0 is above the fix) |
| 43 | `vllm` | uv.lock | direct | medium | `GHSA-8737-qx52-hjff` | < 0.26.0 | 0.26.0 | absent | not shipped (no vllm dependency in the release tree; the engine image is v0.31.0) |
| 44 | `vllm` | uv.lock | direct | medium | `GHSA-hwrm-c4cx-rf4j` | < 0.26.0 | 0.26.0 | absent | not shipped (no vllm dependency in the release tree; the engine image is v0.31.0) |
| 45 | `vllm` | uv.lock | direct | medium | `GHSA-48jh-3gj7-fg8v` | < 0.26.0 | 0.26.0 | absent | not shipped (no vllm dependency in the release tree; the engine image is v0.31.0) |
| 46 | `vllm` | uv.lock | direct | medium | `GHSA-7m6h-x95x-82q5` | < 0.27.0 | 0.27.0 | absent | not shipped (no vllm dependency in the release tree; the engine image is v0.31.0) |
| 47 | `vllm` | uv.lock | direct | medium | `GHSA-4hhp-h66f-j5j7` | < 0.26.0 | 0.26.0 | absent | not shipped (no vllm dependency in the release tree; the engine image is v0.31.0) |
| 48 | `vllm` | uv.lock | direct | medium | `GHSA-hcwq-8wjf-3gcr` | <= 0.23.0 | 0.24.0 | absent | not shipped (no vllm dependency in the release tree; the engine image is v0.31.0) |
| 49 | `vllm` | uv.lock | direct | medium | `GHSA-8pw2-6jv3-mj5j` | < 0.28.0 | 0.28.0 | absent | not shipped (no vllm dependency in the release tree; the engine image is v0.31.0) |
| 50 | `oauthlib` | uv.lock | transitive | medium | `GHSA-hj66-6f7g-4r5v` | >= 0.6.1, <= 3.3.1 | 4.0.0 | 4.0.0 | supersede at the next lock move (the release tree pins 4.0.0, the fix) |
| 51 | `oauthlib` | uv.lock | transitive | medium | `GHSA-xpv3-w29h-x7cv` | >= 3.0.0, < 4.0.0 | 4.0.0 | 4.0.0 | supersede at the next lock move (the release tree pins 4.0.0, the fix) |
### Why each group is what it is

- **`oauthlib` (4, medium).** `requirements-constraints.txt` and `uv.lock`, transitive through
  `rcp-ndcg -> gcsfs -> google-auth-oauthlib -> requests-oauthlib`. The release tree pins **4.0.0**, both
  advisories' fixed version; the alerts are against `main`'s 3.3.1. Dismiss after the merge with "the release
  lock pins oauthlib 4.0.0". No upgrade needed.
- **`vllm` (27, one critical / six high / twenty medium).** `main`'s lock only (0.16.0, direct there). The
  release tree declares no `vllm` dependency anywhere; the engine is the stock image at **0.31.0**, above every
  fixed version in this list. Dismiss after the merge with "vllm is not a dependency of the release tree (vLLM
  only lives in the engine image, at v0.31.0)".
- **`transformers` (4, three high / one medium).** Reachable only through the optional `[mteb]` extra and the
  dev tree (`mteb -> sentence-transformers`), never the default install. The release tree pins **5.17.0**,
  above every fixed version (the highest is 5.10.0). Dismiss after the merge; no upgrade needed.
- **`torch` (2 open, low; 2 fixed).** The optional `[calibrate]`/`[irt]` extras and the dev tree. The release
  tree pins **2.14.0**, above the 2.13.0 fix; the engine image's own torch is 2.13.0, also above. Dismiss after
  the merge; no upgrade needed.
- **`setuptools` (1 open, medium; 1 fixed).** A transitive of torch (dev/optional). The release tree pins
  **84.0.0**, above the 83.0.0 fix. Dismiss after the merge; no upgrade needed.
- **`xgrammar` (1, high).** A transitive of `main`'s vllm (0.1.29, affected; fixed 0.1.32); it left the tree
  with vllm. The image's own xgrammar is not readable offline, but it belongs to the engine image, not this
  repository. Dismiss after the merge; no upgrade needed.
- **`diskcache` (1, medium).** A transitive of `main`'s vllm (5.6.3, affected; **no fixed version exists
  upstream**); it left the tree with vllm. Dismiss after the merge; no upgrade needed.

## Fixed alerts (11)

Already resolved by the dependency leaving the lock (sglang, with the vLLM-only move) or by the constraints
pin moving (torch 2.14.0, setuptools 84.0.0). No action.

| # | package | manifest | severity | advisory | affected range | fixed |
|---|---|---|---|---|---|---|
| 1 | `torch` | requirements-constraints.txt | low | `GHSA-qfhq-4f3w-5fph` | < 2.10.0 | 2.10.0 | the constraints file now pins torch 2.14.0 |
| 2 | `torch` | requirements-constraints.txt | low | `GHSA-rrmf-rvhw-rf47` | <= 2.12.1 | 2.13.0 | the constraints file now pins torch 2.14.0 |
| 3 | `setuptools` | requirements-constraints.txt | medium | `GHSA-h35f-9h28-mq5c` | < 83.0.0 | 83.0.0 | the constraints file now pins setuptools 84.0.0 |
| 6 | `sglang` | uv.lock | medium | `GHSA-9w53-xr52-mwgj` | <= 0.5.3 | 0.5.4 | the dependency left the lock with the vLLM-only move |
| 10 | `sglang` | uv.lock | critical | `GHSA-jx93-g359-86wm` | <= 0.5.9 | 0.5.10 | the dependency left the lock with the vLLM-only move |
| 11 | `sglang` | uv.lock | critical | `GHSA-rgq9-fqf5-fv58` | <= 0.5.9 | 0.5.10 | the dependency left the lock with the vLLM-only move |
| 12 | `sglang` | uv.lock | high | `GHSA-hvwj-8w5g-28rg` | <= 0.5.9 | 0.5.10 | the dependency left the lock with the vLLM-only move |
| 20 | `sglang` | uv.lock | medium | `GHSA-6m5f-673f-5vh7` | <= 0.5.9 | none | the dependency left the lock with the vLLM-only move |
| 21 | `sglang` | uv.lock | critical | `GHSA-36m8-w8qf-g76p` | >= 0.4.1.post7, <= 0.5.12 | none | the dependency left the lock with the vLLM-only move |
| 30 | `sglang` | uv.lock | low | `GHSA-m2jr-x4gq-5rmj` | <= 0.5.10.post1 | none | the dependency left the lock with the vLLM-only move |
| 40 | `sglang` | uv.lock | low | `GHSA-jrcc-j37m-v8fg` | <= 0.5.11 | none | the dependency left the lock with the vLLM-only move |
## pip-audit findings

`uvx pip-audit -r requirements-constraints.txt --no-deps --disable-pip`:

```
Found 2 known vulnerabilities in 1 package
Name  Version ID              Fix Versions
----- ------- --------------- ------------
pyjwt 2.14.0  PYSEC-2026-4141 2.15.0
pyjwt 2.14.0  PYSEC-2026-4183 2.15.0
```

`pyjwt 2.14.0` is reached only through the optional `[azure]` extra
(`rcp-ndcg[azure] -> adlfs -> azure-identity -> msal -> pyjwt`); the default install never pulls it, and no
shipped distribution declares it. The two advisories: an unauthenticated recursion DoS in the pre-verification
payload parse (GHSA-42vr-xj54-vc7v / CVE-2026-101918) and inconsistent OKP `x`/`d` JWK handling
(GHSA-x33g-cr3x-6449 / CVE-2026-102275); both are fixed in 2.15.0. GitHub's advisory database has not flagged
`pyjwt` for this repository, so it does not appear among the Dependabot alerts above.

**Worth doing, at the next lock move (the owner's call):**

```bash
uv lock --upgrade-package pyjwt
python .github/scripts/check_constraints.py --write
git diff uv.lock requirements-constraints.txt   # review, then commit
```

## Recommendations (summary)

1. **Do nothing before the tag.** The release tree's lock and constraints are at or above every open alert's
   fixed version, or carry no such dependency; no shipped default install is affected by any open alert.
2. **After `rfc-0001` is merged to `main`,** let Dependabot re-scan, then dismiss the stale alerts with the
   reasons above (they disappear on their own for `vllm`/`xgrammar`/`diskcache`, which are no longer
   dependencies).
3. **One next-lock-move upgrade: `pyjwt` 2.14.0 -> 2.15.0** (the commands above). It moves an optional-extra
   transitive only; the owner decides whether it rides the next lock move or the release.
4. **Do not** bump `oauthlib`, `transformers`, `setuptools` or `torch`: the release tree already carries the
   fixed versions, and a bump would move the lock for no advisory reason.
