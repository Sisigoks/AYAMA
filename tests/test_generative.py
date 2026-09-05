"""The generative-provider registry: what can run, what cannot, and why.

The tests that matter here are the ones about the models that *cannot* run.
"Sat2City is not wired in" and "Sat2City has published no code" are different
statements, and only the second is true; a registry that blurs them turns an
upstream fact into an apparent gap in this repository. So the reasons are
asserted, not just the booleans.

The other invariant under test is that no provider is metric. Every one of them
normalises its output into a unit cube, which is why they enter as scored
side-artifacts and never as the delivered geometry. If a future provider is
added with `metric=True`, that is a decision someone should have to make
deliberately, and this suite makes it visible.
"""
from __future__ import annotations

import pytest

from traksha.mesh import generative as G


# ---------------------------------------------------------------- the registry
def test_every_registered_provider_is_self_consistent():
    for key, p in G.PROVIDERS.items():
        assert p.name == key
        assert p.title and p.venue
        assert p.unavailable_reason, f"{key} is unreleased and gives no reason"


def test_no_provider_claims_to_be_metric():
    """Each one normalises into a unit cube. That is why none is the deliverable."""
    assert not any(p.metric for p in G.PROVIDERS.values())


def test_nothing_in_this_record_is_runnable_today():
    """Every system here has published a paper and no code. That is the record."""
    assert G.released() == []
    assert "mesh.trellis" in G.summary()


def test_an_unknown_provider_lists_the_alternatives():
    with pytest.raises(ValueError, match="sat2city"):
        G.get("nope")


def test_lookup_is_case_and_whitespace_tolerant():
    assert G.get("  SAT2CITY ").name == "sat2city"


def test_the_table_names_every_provider_and_its_state():
    text = G.table()
    for key in G.PROVIDERS:
        assert key in text
    assert "released" in text and "unreleased" in text


# ------------------------------------------------------- the unreleased ones
def test_sat2city_is_recorded_as_having_published_no_code():
    """Checked by reading the repository: it holds one file, README.md."""
    p = G.get("sat2city")
    assert not p.released
    assert "README.md" in p.unavailable_reason
    assert "Coming soon" in p.unavailable_reason


def test_sat2city_v2_is_recorded_as_having_no_repository():
    p = G.get("sat2city-v2")
    assert not p.released
    assert "Code Coming" in p.unavailable_reason
    assert p.repo is None


def test_sat2city_v2_records_that_it_is_not_metric_even_though_it_is_strongest():
    """Its own paper scales every asset into [-0.5, 0.5]^3."""
    p = G.get("sat2city-v2")
    assert not p.metric
    assert "[-0.5, 0.5]" in p.notes


