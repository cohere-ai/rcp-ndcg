# Security

## Supported versions

Security fixes land on the latest release only. The published distributions (`rcp-ndcg`, `rcp-ndcg-core`,
`rcp-ndcg-vllm`) carry one version and are released together; upgrade to the newest one before reporting
anything that may already be fixed.

## Reporting a vulnerability

Report a vulnerability through GitHub's **private vulnerability reporting** on this repository (the Security
tab, "Report a vulnerability") so it reaches the maintainers without becoming public. Do not open a public
issue or a discussion for it.

A report should name the affected version and distribution, the steps to reproduce, the impact you see, and
any fix you have in mind. Never include credentials or other secrets in a report -- name the variable or
mechanism involved, not its value.

You get an acknowledgement and a decision (fix, or not a vulnerability) through the report thread, and a
release note in the [CHANGELOG](https://github.com/cohere-ai/rcp-ndcg/blob/main/CHANGELOG.md)'s Security
section for anything that ships a fix. A vulnerability in an unpublished checkout is the same report.

## What counts as a vulnerability

Anything that lets a caller read or write data they should not (credentials sent to the wrong host, a stored
answer rewritten silently), makes numbers wrong without the input being wrong (silent truncation, a gained
identity), or executes code from data. Deliberate refusals with typed errors and exit codes are documented
behaviour, not vulnerabilities -- the [command-line reference](https://github.com/cohere-ai/rcp-ndcg/blob/main/docs/reference/cli.md)
lists them.