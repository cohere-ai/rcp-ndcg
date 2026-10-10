# Lane runner-security: injection, secrets and pod hardening (runner review B10, B11, D security)

## Status

DONE. Base `rfc-0001` tip `d630e4a6`; `rfc-0001` (`afecce00`, harness-fix `f409e1bc` + ci-recipes + the
integration commit) merged as `c5a997d5`. Every claim in the review's B10/B11/D-security items was re-verified
on the base first (all reproduced; none dropped), fixed test-first, and confirmed by two independent
adversarial verifier rounds. The final gate is on the head below.

## Commits

| Commit | Subject |
|---|---|
| `01d99ad5` | security: refuse injection strings and secret values, harden pods, keep run records owner-only |
| `c98cb80f` | harness: mount the HF token file instead of expanding it into the job CLI's argv |
| `c5a997d5` | Merge rfc-0001 into lane/runner-security (`rfc-0001` = `afecce00`) |
| `8b27c1bd` | security: round-1 review fixes: the live mirror keeps its credentials, run_as_non_root is opt-in |
| `a5a6db25` | security: round-2 minors: the restore temp forces 0600, a legacy state error is redacted on read |

## What changed

### B10 injection (security F1, F3, F12; slurm F4)

* **Boundary.** `rcp_ndcg.support.resources.no_control_characters` refuses every C0 control and DEL;
  `no_nul_byte` refuses NUL in a free-form value. Applied to the names, ids, queues and namespaces a renderer
  turns into lines or words: `SlurmOptions.log_dir`/`partition`/`account`/`qos` (which also refuse whitespace,
  because sbatch parses a `#SBATCH` line whitespace-separated) and `image`/`workdir`/`container_mounts`/
  `sbatch_args`; `KubernetesOptions.namespace` (now a DNS-1123 label, ≤ 63), `image`, `context`,
  `service_account`, `secrets`; `JobSpec.image`; `ServeConfig.image`; `RunConfig.mirror`.
* **Render site.** `runners/script.heredoc` raises a typed `ConfigError` when a body holds its terminator line
  (`RCP_NDCG_WORKER`, `RCP_NDCG_ENGINE_<ROLE>`, …), so a value that forges it cannot run the rest of itself as
  top-level script; the Kubernetes engine hosts are `shlex.quote`d where they become one shell word each in
  `wait_for_replicas`; `sbatch_args` may not take over the runner's `--output`/`--error`/`--job-name`, including
  attached short values (`-o/path`, `-Jname`) and unambiguous long abbreviations (`--out=`, `--job-n=`) that
  Slurm's getopt resolves.
* **Round-trip inert.** `$(...)`, backticks and newlines in a free-form argv/env/engine-command value stay
  inside the single quotes `shlex` gives them; tests execute the rendered sbatch script with bash and compare
  the values byte for byte. `bootstrap.sh` (security F2, the unquoted manifest-field expansion) is the
  wave-integrity lane's file and is untouched; see *For the next lanes*.

### B11 secrets (env F3, F4/F4b, F9; security F4, F10)

* **Refusal.** `runner.options.env` and `serve.<role>.env` refuse a literal value under a name that looks like a
  credential (`*_TOKEN`, `*_KEY`, `*SECRET*`, `*PASSWORD*`, `*_AUTH`, `*CREDENTIAL*`, word-bounded so `MONKEY`
  passes), with a hint at the submitting environment (local, SLURM) and the Kubernetes `secrets` reference.
* **Recording.** `RunConfig.recorded()` is the written form: a secret-looking `env` value becomes
  `<redacted>` and a mirror URI passes through `safe_url`. `run.yaml`, `manifest.json` and the manifest created
  at construction use it; `resolved()` stays the live form, so the job's argv and the submitting host's mirror
  flush keep the credentials the store needs (round-1 blocker fix). A config whose mirror carries userinfo,
  a query or a fragment warns at `job_for`/`check_target` and the CLI dry run.
* **Redaction.** `MirrorState.remote` and `MirrorState.last_error` are validators; every mirror log line
  (`restore`, flush failures, the `hf://` warning, `check_target`, a torn state file) goes through
  `safe_url`/`redact_urls`; `logs/jobs.json`'s submission error and `run status`'s mirror notes are redacted.
* **HF token.** `submit.sh` no longer expands the token into the job CLI's argv: it mounts
  `$RCP_HF_TOKEN_FILE` at `$RCP_HF_TOKEN_MOUNT` (default `/etc/rcp/hf_token`) and mounts a generated wrapper
  that reads the file inside the job, exports `HF_TOKEN` and execs the worker (engines inherit it). The
  wrapper is executed by the test suite, not just inspected.
* **Permissions.** The run directory and its subdirectories are `0700` (an existing directory is tightened when
  the run is resumed); `run.yaml`, `manifest.json`, `logs/jobs.json` and `logs/mirror.json` are `0600` through
  `storage.publish(..., mode=0o600)`; a restored file is created `0600` (`os.open` + `fchmod`, so a stale
  predictable temp cannot hand it an old mode).

### Pod/job hardening (security F6, F9, F13)

* Every rendered pod (Job and engine StatefulSet) carries `securityContext.seccompProfile: {type:
  RuntimeDefault}`, `automountServiceAccountToken: false` unless `automount_service_account_token: true`, and
  `allowPrivilegeEscalation: false` on every container. `run_as_non_root` is an opt-in (default false) because
  the default coordinator image and the stock `vllm/vllm-openai` image both run as root, and the kubelet
  refuses a root image with `runAsNonRoot`; an image with a non-root `USER` sets it true.
* `ServeConfig.image` must name an exact tag or a digest: `:latest` and an untagged reference are refused with
  a hint. All shipped recipe engine images pass (exact tags or `@sha256:`).
* `engine_objects(job, job_uid=None)` renders without an owner reference (the render happens before the Job
  exists; `submit` adds the applied Job's uid), so `rcp-ndcg run start --dry-run --runner kubernetes` emits a
  stream `kubectl apply` accepts, and `check_objects` validates it.

## Verification

**Round 1 (two independent verifiers, both on the merged head `c5a997d5`).**

* Verifier A (correctness against the brief): **VERDICT: FAIL** — 2 blockers + 3 minors:
  * F1 (blocker) `RunConfig.resolved()` redacted the mirror, and `prepare()` builds the pipeline from it, so the
    live job argv and the submitter's flush lost the credentials; the docstring/CHANGELOG claimed the opposite.
    *Fixed*: `resolved()` faithful; new `recorded()` for every written copy (`8b27c1bd`), with the end-to-end
    test `tests/runs/test_secrets.py::test_the_job_argv_keeps_the_full_mirror_uri`.
  * F2 (blocker) default `run_as_non_root: true` could not start the root coordinator image.
    *Fixed*: default false, opt-in; tests, schema, docs and CHANGELOG updated.
  * F3 (minor) a plugin runner's secret env reached the job as `<redacted>`. *Fixed* by the F1 change (the live
    config keeps the value; a secret-looking name is refused at the `JobSpec` boundary).
  * F4 (minor) a torn mirror-state warning printed pydantic's `input_value` (a legacy full URI). *Fixed* with
    `redact_urls`.
  * F5 (minor) the reserved `sbatch_args` guard was bypassed by attached/abbreviated flags. *Fixed* by prefix
    matching.
* Verifier B (regressions and hygiene): **VERDICT: FAIL** — 1 blocker + 6 minors, largely the same mirror and
  `run_as_non_root` blockers (fixed as above), plus: stale `sbatch_args` guard (F3, same fix); several
  validators and the host-quoting test were untested or vacuous (F4 — tests added, and the host-quoting test now
  monkeypatches a hostile host and asserts the rendered `wait_for_replicas` word); dead
  `JobOptions.resolved()` env redaction (F5 — removed); stale harness comments and an unexecuted wrapper
  (F6 — `RCP_HF_TOKEN_MOUNT` added, the wrapper is executed by the test, `e2e.sh`/`wave0.sh` comments fixed);
  the restore temp's 0644 window (F7 — `os.open`/`fchmod`). Its mutation run showed four fixes red (heredoc
  terminator, secret refusal, automount default, image pin) and flagged the untested ones.

**Round 2 (one fresh confirmation verifier, both lenses, head `8b27c1bd`).**

**VERDICT: PASS** — all nine round-1 fixes confirmed with real-path reproductions and 11 mutation tests (each
new test goes red when its fix is mutated; no survivors). Two new narrow minors were found and fixed in
`a5a6db25`, failing test first:

* F1 (minor) the predictable restore temp kept a stale file's mode (`O_CREAT`'s mode applies only when it
  creates the file) → `os.fchmod(fd, 0o600)`.
* F2 (minor) a *valid* legacy state file's `last_error` was printed unredacted by `run status` →
  `MirrorState.last_error` validator with `redact_urls`.

The round-2 verifier's independent quality bar on `8b27c1bd`: `ruff format --check`/`ruff check` clean,
`basedpyright` 0 errors, root suite `3697 passed, 102 skipped`, `rcp-ndcg-test` `903 passed, 222 skipped`,
`tests/contract tests/docs` `304 passed, 55 skipped`, `public-names-step` clean, `git status` empty. Its
background gate read: every finished step green (only `run_all`/`public-names`/`clean` pending at read time).
No third round was run: COMMON.md's verifier rule limits round 2 to one confirmation verifier, and the two
round-2 findings are narrow, each fixed with a red-then-green test.

## Checks

Last commands on the final head (`a5a6db25`), and their result lines:

```
uv run --no-sync ruff format --check .   -> 584 files already formatted
uv run --no-sync ruff check .            -> All checks passed!
uv run --no-sync basedpyright            -> 0 errors, 0 warnings, 0 notes
flock /tmp/rcp_heavy.lock uv run --no-sync pytest tests/ -q -n 4 -p no:cacheprovider
                                         -> 3698 passed, 102 skipped
uv run --no-sync pytest rcp-ndcg-test/tests -q -p no:cacheprovider
                                         -> 903 passed, 222 skipped
uv run --no-sync pytest tests/contract tests/docs -q -p no:cacheprovider
                                         -> 304 passed, 55 skipped
uv run --no-sync mkdocs build --strict   -> Documentation built
bin/gate lane/runner-security            -> GATE: PASS (see below)
```

The final gate on `a5a6db25` (slot 5):

```
rev lane/runner-security = a5a6db25 (slot 5)
ruff-check exit=0 All checks passed!
ruff-format exit=0 584 files already formatted
basedpyright exit=0 0 errors, 0 warnings, 0 notes
pytest exit=0 3699 passed, 102 skipped in 87.61s
contract-docs exit=0 304 passed, 55 skipped in 50.75s
mkdocs exit=0 Documentation built in 1.71 seconds
test-pkg exit=0 903 passed, 222 skipped in 624.64s
recipes exit=0 recipes: no failure outside the baseline (0 baseline failures remain, 0 fixed; pytest exit 0)
vllm-pkg exit=0 40 passed in 5.65s
vllm-models exit=0 72 passed, 7 skipped in 120.87s
run_all exit=0 leaderboards: 1022 checks, 987 match, 35 known deviations, 0 failed;
              human study: 67 checks, 67 match; external LLM judges: 82 checks, 82 match
public-names exit=0 public-names: clean (0 baselined hits remain)
clean exit=0 clean
GATE: PASS
```

The gate on the round-1 head `8b27c1bd` (read by the round-2 verifier) was green on every step it had finished;
the two round-2 minors landed after it and the gate above is the final one.

The failing-test-first evidence: `tests/runners/test_security.py tests/runs/test_secrets.py` was `40 failed,
8 passed` before the product fixes (the failing lines are the refusals and redactions above); the two round-2
tests were `2 failed` before their fixes.

## Open questions

* **`run_as_non_root` is opt-in.** The brief's "where the image allows" is a declaration the runner cannot make
  for an image: the stock coordinator and `vllm/vllm-openai` images run as root, so the secure default would
  have made the default Kubernetes run unstartable. The flag is documented as the opt-in for a non-root image.
* **A resume from the redacted `run.yaml` needs environment credentials** (or `--mirror` again) for a mirror
  whose URI carried them; `run status` reads the recorded (redacted) URI and cannot reach a userinfo-only
  store. This is the cost of never writing credentials down; the config warns at load.
* **`RunLayout.ensure` tightens an existing run directory to `0700`** on resume. That is deliberate for a
  shared cluster, but an operator who intentionally shared a run directory with a group loses that.
* A credentialed `wheelhouse`, dataset or rankings URL is still recorded in clear; the brief's redaction scope
  is `env` and the mirror. Worth a follow-up if any of those can carry credentials.
* The reserved `sbatch_args` guard is conservative (any `--o*`/`--e*`/`--j*` prefix that could resolve to
  `--output`/`--error`/`--job-name` is refused, as are `-o*`/`-e*`/`-J*`).

## CHANGELOG entry

The two entries added under `## Unreleased` (exact text; `### Public surface` and `### Security`):

> - **The Kubernetes runner's pod hardening** is configurable: `runner.options.run_as_non_root` (default false:
>   the stock coordinator and `vllm/vllm-openai` images run as root; set it true for an image with a non-root
>   `USER`, e.g. the `vllm-openai-nonroot` variant) and `runner.options.automount_service_account_token` (default
>   false). Every rendered pod carries a `RuntimeDefault` seccomp profile and no privilege escalation either way.
> - **`rcp_ndcg.support.resources`** exports the string rules the config boundary applies: `no_control_characters`,
>   `no_nul_byte`, `looks_like_secret`, `refuse_secret_value` and the `REDACTED` marker; `rcp_ndcg.storage.publish`
>   and `publish_bytes` take an optional `mode` (a run's records pass `0o600`). `RunConfig.recorded()` is the
>   written form of a config (secret `env` values and a mirror URI's credentials redacted) while `resolved()` stays
>   the live form. `rcp_ndcg.runners.kubernetes` exports `CONTAINER_SECURITY_CONTEXT`; its
>   `engine_objects(job, job_uid=None)` leaves the owner reference to `submit` (a render has no Job uid yet), and
>   the placeholder constant `JOB_UID` is gone.

> - **A config- or recipe-derived string cannot inject into a rendered job**: a newline in a value used to end
>   the job script's heredoc (the rest of the value ran as top-level script), a `#SBATCH` directive line, or a
>   Kubernetes shell word built from config. The heredoc renderer now refuses a body holding its terminator
>   (typed `ConfigError`), `#SBATCH` values (`partition`, `account`, `qos`, `log_dir`, `sbatch_args`) refuse
>   whitespace and control characters, an engine image refuses control characters, a namespace must be a
>   DNS-1123 label, and engine host names are quoted where they become shell words. `$(...)`, backticks and
>   newlines in a free-form argv/env value stay inert inside the single quotes the renderer already gave them.
> - **A credential cannot be recorded, and a secret-looking `env` name is refused**: `runner.options.env` and
>   `serve.<role>.env` refuse a literal value under a name that looks like a credential (`*_TOKEN`, `*_KEY`,
>   `*SECRET*`, `*PASSWORD*`, `*_AUTH`, `*CREDENTIAL*`), naming the environment and the Kubernetes `secrets`
>   routes; a value that reaches a recorded config by another route is written as `<redacted>`, never in clear
>   (`RunConfig.recorded()`; `resolved()` is the live form the job runs with). A mirror URI's credentials
>   (userinfo, query, fragment) never reach `run.yaml`, the manifest, the state file (its `remote` and its recorded
>   error), `run status`, `logs/jobs.json`'s recorded error or a log line -- the live config and the job's command
>   line keep the full URI, which is where it must reach the store, and a config that carries credentials warns
>   (a resume from the redacted `run.yaml` takes them from the environment or a `--mirror` override). The
>   harness's `submit.sh` no longer expands the HF token into the job CLI's argv: the token file is mounted and a
>   wrapper reads it inside the job (argv is world-readable on the submit host through `/proc/<pid>/cmdline`).
> - **Run artifacts are owner-only on a shared filesystem**: the run directory and its subdirectories are `0700`
>   (existing ones are tightened when the run is resumed), and `run.yaml`, `manifest.json`, `logs/jobs.json` and
>   `logs/mirror.json` are `0600` (restored files too), so a cluster where every user sees the shared filesystem no
>   longer exposes the config, the records or any env value.
> - **Kubernetes pods are hardened**: a `RuntimeDefault` seccomp profile, no privilege escalation and no
>   service-account token unless declared on every pod, and `run_as_non_root: true` for an image whose `USER` is
>   non-root (the stock images run as root, so the default is off and the kubelet does not refuse them); the engine
>   image the recipe declares must name an exact tag or a digest (`:latest` and an untagged reference are refused),
>   and `run start --dry-run --runner kubernetes` emits objects `kubectl apply` accepts (the engine objects no
>   longer carry an owner reference with a placeholder uid).

## Public surface changes

* Added to `rcp_ndcg.support.resources`: `REDACTED`, `looks_like_secret`, `no_control_characters`, `no_nul_byte`,
  `refuse_secret_value`.
* Added to `rcp_ndcg.runners.kubernetes`: `CONTAINER_SECURITY_CONTEXT`; `engine_objects(job, job_uid=None)` now
  takes an optional uid (previously defaulted to a placeholder); `JOB_UID` **removed**.
* Added `RunConfig.recorded()`; `rcp_ndcg.storage.publish`/`publish_bytes` take an optional `mode`.
* New runner options: `KubernetesOptions.run_as_non_root` (default false),
  `KubernetesOptions.automount_service_account_token` (default false); `KubernetesOptions.namespace` is now a
  DNS-1123 label; `ServeConfig.image` must be pinned; `RunConfig.mirror` refuses control characters.
* Schemas regenerated: `schemas/run-config.v1.json`, `schemas/run-start.v1.json`,
  `schemas/run-status.v1.json`; snapshot `tests/contract/snapshots/python_api.json`.
* CLI: no new flag or exit code. `submit.sh` gains the `RCP_HF_TOKEN_MOUNT` environment knob (default
  `/etc/rcp/hf_token`).

## Files outside scope

* `rcp-ndcg-test/src/rcp_ndcg_test/jobs/{submit.sh,e2e.sh,wave0.sh}` and
  `rcp-ndcg-test/tests/test_jobs_scripts.py` (the harness; the brief's HF-token item assigns it to this lane).
* `rcp-ndcg/src/rcp_ndcg/runs/{config,execution,layout,manifest,mirror,pipeline}.py` and
  `rcp-ndcg/src/rcp_ndcg/storage/core.py` are the recording/redaction paths the brief puts in this lane; the
  run-integrity lane's status/resume code is untouched.
* `rcp-ndcg-test/src/rcp_ndcg_test/jobs/bootstrap.sh` is untouched (wave-integrity's file).

## For the next lanes

* **wave-integrity**: security F2 stands — `bootstrap.sh:476-484` expands `CLIENT_ARGS` unquoted into the node's
  client wrapper and `manifest.json`'s `version` is not hashed, so a tampered stage can run `$(...)` on the node.
  Its comment at `:31` ("HF_TOKEN reaches the engines from the job's secret") is now stale: the token arrives as
  the mounted file at `$RCP_HF_TOKEN_MOUNT` (default `/etc/rcp/hf_token`), which `submit.sh`'s wrapper reads
  inside the job and exports as `HF_TOKEN`.
* **run-integrity**: this lane merged `rfc-0001` at `afecce00`, before your `a98742b9`/`e38f7dbe`. When your
  branch lands, expect conflicts in `runs/mirror.py` (the `MirrorState` validators and the restore temp) and
  `runs/execution.py` (`_write_record` now `publish_bytes(..., mode=0o600)` and `redact_urls` on the submission
  error). Keep both sides.
* **Release**: nothing in this lane changes the paper's numbers or anchors; the metric/gain/protocol code is
  untouched.
