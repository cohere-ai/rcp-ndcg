<!-- Handover copy of the operator's working note `drafts/qa-correct.md` (sanitized: private names, the private repository, the job platform and storage locations removed; `<operator-notes>/...` paths point to files that are NOT available — the content you need is in handover/). Treat it as the specification it was for its lane. -->

# QA review `qa-correct`: correctness hazards a unit test suite misses

**Crux.** The suite is green. Find what is still wrong: behaviour that is incorrect on inputs the tests do not cover,
errors that surface as the wrong type, concurrency that only works on the happy path, numbers that drift, identities
that re-key or fail to, and anything unsafe.

## Watch for (each item: check it, reproduce what you can on CPU, report with evidence)
1. **Errors**: every failure path raises a typed error from `rcp_ndcg.errors` with a hint; find bare `ValueError`,
   `KeyError`, `assert` used for control flow, `except Exception` that swallows, and silent defaults that change
   numbers (AGENTS.md: "Nothing is cut or defaulted silently").
2. **Async and the sync bridge**: `Transport.run` inside and outside a running loop, repeated calls, cancellation
   (Ctrl-C) mid-request, closing; semaphores bound to the right loop; no request left in flight on error; the outage
   parking and rejection rule under concurrency (write a stress reproduction with `httpx.MockTransport` and many
   concurrent requests with random failures).
3. **Text budgets and anchors**: for each request shape and each recipe template, an over-length input keeps every
   anchor, an input under the budget is byte-identical to the uncut render, chunking gives every chunk the full
   template, and the census records each cut. Try adversarial text (CJK, emoji, combining marks, very long words,
   empty strings, whitespace only).
4. **Media**: token counts for images and videos equal what the processor computes, for the judge and the retrieval
   roles; no double resizing; the unpinned `video_url` refusal; an image that alone exceeds the budget.
5. **Numerics**: float16 multi-vector transfer decoded correctly (endianness, shape, ragged offsets); MaxSim
   accumulates in float32; normalisation idempotent; reranker scores aligned by `index` and never by position; score
   scales (probability, logit, cosine) preserved end to end.
6. **Identities**: no runtime field (URLs, keys, concurrency, timeouts, batch sizes) reaches any step identity; every
   content field (model, revision, tokenizer SHA-256, template, budgets, prompts, normalisation, dimensions,
   embed_dtype, use_activation, media policy) does. Test by perturbation on the real `_identity` functions.
7. **Security**: credentials only from the environment, never logged or written to manifests or run dirs (grep logs a
   test run produces for key-shaped strings); no `shell=True`, `eval`, `pickle`, `yaml.load`; subprocess argv lists;
   paths from configs resolved and contained; TLS never disabled; the GPU job scripts never print the auth script or
   the environment.
8. **Tests that pass vacuously**: sample 30 new tests; for each, does it fail when the behaviour it names breaks?
   Mutate the code under test for at least 10 of them and report which stayed green.
Deliver also: the reproductions as scripts under your scratch dir, and a list of the 10 mutations with their outcome.
