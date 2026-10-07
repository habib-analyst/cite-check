"""Tests for the audit pipeline (answer-doc parsing + scoring)."""

import json

import pytest

from citecheck.audit import (
    audit_items,
    items_from_answer_doc,
    items_from_list,
    load_items,
)
from citecheck.fixtures import DEMO_DOC


def test_answer_doc_parsing():
    items = items_from_answer_doc(DEMO_DOC)
    assert len(items) == 4
    assert [i["id"] for i in items] == ["cite-1", "cite-2", "cite-3", "cite-4"]
    # markers stripped, sentences grouped per citation
    assert "[1]" not in items[0]["claim"]
    assert "Metformin lowers HbA1c" in items[0]["claim"]
    assert "Statins do not increase the risk of diabetes." == items[3]["claim"]


def test_audit_demo_end_to_end():
    result = audit_items(items_from_answer_doc(DEMO_DOC), threshold=0.6)
    by_id = {r["id"]: r for r in result["results"]}
    assert by_id["cite-1"]["verdict"] == "SUPPORTED"
    assert by_id["cite-2"]["verdict"] == "WEAK"
    assert by_id["cite-3"]["verdict"] == "UNSUPPORTED"
    assert by_id["cite-4"]["verdict"] == "CONTRADICTED"

    s = result["summary"]
    assert s["total"] == 4
    assert s["supported"] == 1
    assert s["pct_supported"] == 25.0
    assert s["verdict_counts"] == {
        "SUPPORTED": 1, "WEAK": 1, "UNSUPPORTED": 1, "CONTRADICTED": 1,
    }
    assert s["worst_offenders"][0]["id"] == "cite-3"


def test_threshold_controls_supported_count():
    items = items_from_answer_doc(DEMO_DOC)
    strict = audit_items(items, threshold=0.99)["summary"]["supported"]
    lax = audit_items(items, threshold=0.2)["summary"]["supported"]
    assert lax >= strict


def test_item_list_format(tmp_path):
    doc = {"items": [
        {"id": "a", "claim": "Water boils at 100 degrees Celsius.",
         "citation_text": "Water boils at 100 degrees Celsius at sea level.", "source_id": "src"},
        {"id": "b", "claim": "Cats can fly.",
         "citation_text": "Photosynthesis converts sunlight into chemical energy.",
         "source_id": "src"},
    ]}
    f = tmp_path / "items.json"
    f.write_text(json.dumps(doc))
    items = load_items(f)
    assert len(items) == 2
    result = audit_items(items)
    by_id = {r["id"]: r for r in result["results"]}
    assert by_id["a"]["verdict"] == "SUPPORTED"
    assert by_id["b"]["verdict"] == "UNSUPPORTED"


def test_bare_list_format(tmp_path):
    f = tmp_path / "list.json"
    f.write_text(json.dumps([
        {"claim": "Water boils at 100 degrees Celsius.",
         "citation_text": "Water boils at 100\u00b0C at sea level."}
    ]))
    assert len(load_items(f)) == 1


def test_unreferenced_citation_flagged():
    doc = {"answer": "No markers here at all.",
           "citations": [{"id": "1", "text": "Some passage.", "source": "s"}]}
    result = audit_items(items_from_answer_doc(doc))
    r = result["results"][0]
    assert r["verdict"] == "UNSUPPORTED"
    assert "citation_not_referenced" in r["flags"]
