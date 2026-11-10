# Writing style

This page is the rule for the documentation in this repository. It states the style, the exceptions, and the
check.

## The style

The technical documentation uses ASD-STE100, Simplified Technical English. The rules are:

- Write sentences of 20 words or fewer.
- Count an inline code span as one word.
- Use the active voice.
- Put one idea in one sentence.
- Define a technical term the first time you use it.
- Do not use idioms, metaphors or jokes.

## What the style does not change

The style changes the prose only. Keep these items exactly as they are:

- code blocks, commands and paths;
- tables and their rows;
- link targets;
- numbers and their sources;
- the words that a test pins.

## The check

`tests/docs/test_style.py` reads every hand-written page in `docs/` and fails a sentence of more than 20 words.
The generated pages under `docs/reference/api/` are skipped. The check skips the code blocks, the tables and
the headings. It reads each bullet item as its own sentence. A sentence that a test pins is in the check's
allowlist, with the reason.

## How to edit a page

1. Rewrite the prose.
2. Keep the code, the tables and the numbers.
3. Run `uv run --no-sync pytest tests/docs -q`.
