"""The offline fakes: one wire adapter per role behind ``fake://`` URLs, so offline tests run the real transport.

They sit *below* the transport: a ``fake://`` base URL resolves to a fake adapter, and the transport above it
is the real one, so routing, retries, parking and usage run in every offline test. The fakes are:

* **the judge** -- :class:`rcp_ndcg.testing.FakeJudge`'s answer logic moves here unchanged, answering through the
  same messages and parsers as a real judge;
* **the encoder** -- deterministic hash-seeded unit vectors per content part, with an optional ragged
  multi-vector mode;
* **the reranker** -- each document scored by the same hidden ability the fake judge uses, so a tiny run's
  rerank, judge and calibration agree.

The transport work implements this module; nothing below is wired yet.
"""

from __future__ import annotations

#: The URL scheme that selects a fake adapter: ``fake://seed/<n>`` for the judge (``JudgeConfig.fake``), and the
#: per-role equivalents for the encoder and the reranker.
FAKE_SCHEME = "fake://"

__all__ = ["FAKE_SCHEME"]
