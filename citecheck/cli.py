"""Command-line interface for cite-check."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .audit import audit_items, items_from_answer_doc, load_items
from .fixtures import DEMO_DOC
from .report import to_json, to_markdown, to_text


def _render(audit_result: dict, fmt: str) -> str:
    if fmt == "json":
        return to_json(audit_result)
    if fmt == "text":
        return to_text(audit_result)
    return to_markdown(audit_result)


def cmd_demo(args: argparse.Namespace) -> int:
    items = items_from_answer_doc(DEMO_DOC)
    result = audit_items(items, threshold=args.threshold)
    out = _render(result, args.format)
    if args.out:
        Path(args.out).write_text(out, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(out, end="" if out.endswith("\n") else "\n")
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    path = Path(args.input)
    if not path.exists():
        print(f"cite-check: error: file not found: {path}", file=sys.stderr)
        return 2
    try:
        items = load_items(path)
    except Exception as exc:  # malformed JSON etc.
        print(f"cite-check: error: could not parse {path}: {exc}", file=sys.stderr)
        return 2
    if not items:
        print(f"cite-check: error: no audit items found in {path}", file=sys.stderr)
        return 2

    nli_scores = None
    if args.nli:
        from .nli_backend import HuggingFaceNLIBackend, NLINotAvailable

        backend = HuggingFaceNLIBackend()
        if not backend.available():
            print(
                "cite-check: warning: --nli requested but transformers is not "
                "installed; continuing with the offline scorer only.",
                file=sys.stderr,
            )
        else:
            try:
                nli_scores = {}
                for item in items:
                    if item.get("claim") and item.get("citation_text"):
                        nli_scores[item["id"]] = backend.score(
                            item["citation_text"], item["claim"]
                        )
            except NLINotAvailable as exc:
                print(f"cite-check: warning: {exc}", file=sys.stderr)

    result = audit_items(items, threshold=args.threshold)
    if nli_scores:
        for r in result["results"]:
            if r["id"] in nli_scores:
                r["signals"]["nli_entailment_p"] = round(nli_scores[r["id"]], 4)
    out = _render(result, args.format)
    if args.out:
        Path(args.out).write_text(out, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(out, end="" if out.endswith("\n") else "\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cite-check",
        description="Verify that RAG citations actually support their claims.",
    )
    p.add_argument("--version", action="version", version=f"cite-check {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("demo", help="run the bundled 4-claim demo fixture")
    d.add_argument("--threshold", type=float, default=0.6)
    d.add_argument("--format", choices=["md", "json", "text"], default="md")
    d.add_argument("--out", default=None, help="write report to file instead of stdout")
    d.set_defaults(func=cmd_demo)

    a = sub.add_parser("audit", help="audit a JSON answers/citations file")
    a.add_argument("input", help="JSON file (item list or RAG answer doc)")
    a.add_argument("--threshold", type=float, default=0.6,
                   help="score >= threshold counts as supported (default 0.6)")
    a.add_argument("--format", choices=["md", "json", "text"], default="md")
    a.add_argument("--out", default=None, help="write report to file instead of stdout")
    a.add_argument("--nli", action="store_true",
                   help="also score with the optional HuggingFace NLI backend")
    a.set_defaults(func=cmd_audit)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
