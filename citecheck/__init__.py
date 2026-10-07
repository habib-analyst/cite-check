"""cite-check: claim <-> citation entailment verifier for RAG answers."""

from .entailment import EntailmentResult, score_entailment
from .audit import audit_items, items_from_answer_doc, load_items
from .report import to_json, to_markdown, to_text

__version__ = "0.1.0"

__all__ = [
    "EntailmentResult",
    "score_entailment",
    "audit_items",
    "items_from_answer_doc",
    "load_items",
    "to_json",
    "to_markdown",
    "to_text",
    "__version__",
]
