"""Tests for the offline entailment scorer. No network, no models."""

import pytest

from citecheck.entailment import (
    CONTRADICTED,
    SUPPORTED,
    UNSUPPORTED,
    WEAK,
    EntailmentResult,
    extract_numbers,
    score_entailment,
)


def test_entailed_claim_supported():
    r = score_entailment(
        "Metformin lowers HbA1c by 1.0 to 1.5 percentage points in most patients.",
        "In adults with type 2 diabetes, metformin therapy typically reduces "
        "HbA1c by 1.0\u20131.5 percentage points. Most patients tolerate it well.",
    )
    assert r.verdict == SUPPORTED
    assert r.score >= 0.6


def test_unrelated_passage_unsupported():
    r = score_entailment(
        "Metformin lowers HbA1c in diabetic patients.",
        "Completely unrelated text about cooking pasta with garlic and olive oil.",
    )
    assert r.verdict == UNSUPPORTED
    assert r.score < 0.3
    assert r.flags == []  # no noisy flags on unrelated text


def test_negated_claim_contradicted():
    r = score_entailment(
        "Statins do not increase the risk of diabetes.",
        "Statin therapy is associated with a small increased risk of new-onset diabetes.",
    )
    assert r.verdict == CONTRADICTED
    assert "negation_flip" in r.flags
    assert r.score <= 0.25


def test_negated_passage_contradicted():
    r = score_entailment(
        "The drug increases blood pressure.",
        "The drug does not increase blood pressure in any trial arm.",
    )
    assert r.verdict == CONTRADICTED
    assert "negation_flip" in r.flags


def test_antonym_pair_contradicted():
    r = score_entailment(
        "The drug is safe and effective.",
        "The drug is unsafe and ineffective.",
    )
    assert r.verdict == CONTRADICTED
    assert "antonym_pair" in r.flags


def test_numeric_mismatch_flagged_and_unsupported():
    r = score_entailment(
        "The trial enrolled 200 patients.",
        "The trial enrolled 150 patients across three sites.",
    )
    assert "numeric_mismatch" in r.flags
    assert r.verdict == UNSUPPORTED
    assert r.score <= 0.25


def test_numeric_match_percent_variants():
    r = score_entailment(
        "Response rate was 40 percent in the treatment arm.",
        "The treatment arm showed a 40% response rate.",
    )
    assert r.verdict == SUPPORTED
    assert "numeric_mismatch" not in r.flags


def test_ungrounded_number_penalized():
    supported = score_entailment(
        "HbA1c fell by 1.5 percentage points.",
        "HbA1c fell by 1.5 percentage points over 24 weeks.",
    )
    with_number = score_entailment(
        "HbA1c fell by 1.5 percentage points in 300 patients.",
        "HbA1c fell by 1.5 percentage points over 24 weeks.",
    )
    assert "ungrounded_numbers" in with_number.flags
    assert with_number.score < supported.score


def test_partial_support_is_weak():
    r = score_entailment(
        "The treatment may also support modest weight loss and improve "
        "long-term cardiovascular outcomes.",
        "Patients on metformin experienced modest weight reduction averaging "
        "2 kg over one year.",
    )
    assert r.verdict == WEAK
    assert 0.3 <= r.score < 0.6


def test_score_bounds_and_breakdown():
    r = score_entailment("Some claim here.", "Some passage here.")
    assert isinstance(r, EntailmentResult)
    assert 0.0 <= r.score <= 1.0
    for key in ("coverage", "overlap", "negation_flip", "numeric_mismatch",
                "antonym_hits"):
        assert key in r.signals
    assert isinstance(r.evidence, list) and r.evidence


def test_empty_inputs_unsupported():
    assert score_entailment("", "passage").verdict == UNSUPPORTED
    assert score_entailment("claim", "").verdict == UNSUPPORTED


def test_extract_numbers_units_and_ranges():
    nums = extract_numbers("reduced by 1.0 to 1.5 percentage points")
    assert (1.0, "pct") == (nums[0][0], nums[0][1])
    assert (1.5, "pct") == (nums[1][0], nums[1][1])
    assert extract_numbers("a 40% response")[0][1] == "pct"
    assert extract_numbers("lost 2 kg")[0] == (2.0, "kg", "2")
    # the "1" in HbA1c is not a number
    assert extract_numbers("HbA1c levels") == []
    # word numbers
    assert extract_numbers("enrolled twenty patients")[0][0] == 20.0
