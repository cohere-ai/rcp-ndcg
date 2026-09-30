"""Endpoint: the served-model fields the judge and the hosted providers share."""

from __future__ import annotations

from rcp_ndcg import Endpoint
from rcp_ndcg.support.identity import check_declarations, identity_payload


def test_an_endpoint_keys_on_its_model_and_checkpoint_only() -> None:
    check_declarations(Endpoint)
    one = Endpoint(base_url="http://a:8000/v1/", model="m", revision="abc", concurrency=4)
    other = Endpoint(base_url="http://b:9000/v1", model="m", revision="abc", timeout_s=30)
    assert one.base_url == "http://a:8000/v1"
    assert identity_payload(one) == identity_payload(other) == {"model": "m", "revision": "abc"}
