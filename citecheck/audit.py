"""Audit pipeline: score every (claim, cited passage) pair in a document.

Input formats (auto-detected):

1. Item list -- a JSON array, or {"items": [...]}:
     [{"id": "c1", "claim": "...", "citation_text": "...", "source_id": "s1"}, ...]

2. RAG answer doc:
     {"question": "...",
      "answer": "Metformin lowers HbA1c [1]. Statins ... [2].",
      "citations": [{"id": "1", "text": "...", "source": "..."}, ...]}
   Sentences are split on [. ! ?], inline [n] markers are parsed, and each
   citation's claim becomes the concatenation of the sentences that cite it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .entailment import score_entailment

_MARKER_RE = re.compile(r"\[(\d+)\]")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _strip_markers(text: str) -> str:
    text = _MARKER_RE.sub("", text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)  # "diabetes ." -> "diabetes."
    return re.sub(r"\s+", " ", text).strip()


def items_from_answer_doc(doc: dict) -> list[dict]:
    answer = doc.get("answer", "")
    citations = {str(c.get("id")): c for c in doc.get("citations", [])}
    by_cite: dict[str, list[str]] = {cid: [] for cid in citations}
    for sent in _SENT_SPLIT_RE.split(answer.strip()):
        markers = _MARKER_RE.findall(sent)
        clean = _strip_markers(sent)
        if not clean:
            continue
        if markers:
            for m in markers:
                if m in by_cite:
                    by_cite[m].append(clean)
        # sentences without markers are context, not claims -> skipped
    items = []
    for cid, cite in citations.items():
        claim = " ".join(by_cite[cid])
        items.append({
            "id": f"cite-{cid}",
            "claim": claim,
            "citation_text": cite.get("text", ""),
            "source_id": cite.get("source", cid),
            "citation_id": cid,
            "referenced": bool(claim),
        })
    return items


def items_from_list(data: list | dict) -> list[dict]:
    raw = data["items"] if isinstance(data, dict) else data
    items = []
    for i, entry in enumerate(raw):
        items.append({
            "id": str(entry.get("id", f"item-{i + 1}")),
            "claim": entry.get("claim", ""),
            "citation_text": entry.get("citation_text", entry.get("text", "")),
            "source_id": entry.get("source_id", entry.get("source", "")),
            "referenced": True,
        })
    return items


def load_items(path: str | Path) -> list[dict]:
    """Load audit items from a JSON file, auto-detecting the format."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "answer" in data and "citations" in data:
        return items_from_answer_doc(data)
    return items_from_list(data)


def audit_items(items: list[dict], threshold: float = 0.6) -> dict:
    """Score every item; return {"results": [...], "summary": {...}}."""
    results = []
    for item in items:
        claim = item.get("claim", "")
        passage = item.get("citation_text", "")
        if not claim.strip():
            res = score_entailment("unreferenced placeholder claim", passage or "")
            verdict, score = "UNSUPPORTED", 0.0
            flags = ["citation_not_referenced"]
            evidence = ["citation is never referenced by the answer text"]
        elif not passage.strip():
            verdict, score, flags = "UNSUPPORTED", 0.0, ["empty_citation_text"]
            evidence = ["citation text is empty"]
            res = None
        else:
            scored = score_entailment(claim, passage)
            verdict, score, flags, evidence = (
                scored.verdict, scored.score, scored.flags, scored.evidence,
            )
            res = scored
        results.append({
            "id": item.get("id"),
            "claim": claim,
            "source_id": item.get("source_id", ""),
            "citation_excerpt": (passage[:400] + "...") if len(passage) > 400 else passage,
            "score": round(score, 4),
            "verdict": verdict,
            "supported": score >= threshold,
            "flags": flags,
            "evidence": evidence,
            "signals": res.signals if res else {},
        })
    counts = {"SUPPORTED": 0, "WEAK": 0, "UNSUPPORTED": 0, "CONTRADICTED": 0}
    for r in results:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    n = len(results)
    n_supported = sum(1 for r in results if r["supported"])
    worst = sorted(results, key=lambda r: r["score"])[:3]
    summary = {
        "total": n,
        "threshold": threshold,
        "supported": n_supported,
        "pct_supported": round(100.0 * n_supported / n, 1) if n else 0.0,
        "verdict_counts": counts,
        "avg_score": round(sum(r["score"] for r in results) / n, 4) if n else 0.0,
        "worst_offenders": [
            {"id": w["id"], "verdict": w["verdict"], "score": w["score"],
             "claim": w["claim"][:120]}
            for w in worst
        ],
    }
    return {"results": results, "summary": summary}
