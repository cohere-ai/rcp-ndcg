# Report 01: the inference-layer fixes (integrated)

**Status:** DONE, merged into `rfc-0001` in two merges (the lane, then a security follow-up). Gate: every step PASS;
shuffled single-process run of the root suite green.

## What landed
- `wip/fix-inference` finished and integrated with `rfc-0001` (the WIP snapshot's two "truth" fixes kept where true: the
  dead hosted-OpenAI batch constant removed; a docs claim decided by a test against the client).
- **One home per concept** (seven concepts had two implementations after the merge): usage folding
  (`RoleClient._record_usage`, the judge included), URL redaction (`rcp_ndcg.support.urls.safe_url` and `redact_urls`),
  the reply-index rule (`adapters.base._aligned_by_index`), the adapter shape check (registration and the contract kit),
  per-content slicing (`PreparedRequest.per_content`, replacing `select`, which made a media corpus encode quadratic),
  and one batch-cap rule gated on `HOSTED`.
- **Security (decision 24)**, each fix with a test red before it:
  - The key-host rule lives in the transport, per replica (`AuthProfile.homes`, `applies_to`): a vendor profile's
    default key reaches only its home URL (trailing slash ignored; any other difference, including case, port, userinfo,
    lookalike hosts and path tricks, gets no key); a named `api_key_env` reaches only the config's own URLs; an
    injected transport aimed elsewhere gets no key. Verified against 22 URL tricks, mixed replica lists under 400
    concurrent sends, threads, endpoint mutation and injection.
  - URL credentials (userinfo, query) never reach an `EngineInfo.error`, a log line, an exception message, `details`, a
    chained cause or a rerank error; URL-list config refusals are typed `ConfigError`s naming redacted URLs.
  - `Transport.close()` defers the pool's close to in-flight background runs (no deadlock when called from inside one).
- Every claimed fix of the original branch has a test that fails on the pre-integration code.

## Decisions
- A named key sent to a URL outside its homes goes out without a key (fail-closed) rather than raising.
- URL-list refusals moved from `ValueError` to `ConfigError` (pydantic would render the raw input of a ValueError).

## Open (see 00-MASTER section 9)
The security follow-up commits merged on red-then-green tests without a separate verifier round: re-check in QA.
