"""Report rendering: Markdown and JSON."""

from __future__ import annotations

import datetime as _dt
import json

_VERDICT_BADGE = {
    "SUPPORTED": "SUPPORTED",
    "WEAK": "WEAK",
    "UNSUPPORTED": "UNSUPPORTED",
    "CONTRADICTED": "CONTRADICTED",
}


def to_json(audit_result: dict) -> str:
    return json.dumps(audit_result, indent=2, ensure_ascii=False)


def to_text(audit_result: dict) -> str:
    """Compact one-line-per-claim summary."""
    lines = []
    s = audit_result["summary"]
    lines.append(
        f"supported {s['supported']}/{s['total']} "
        f"({s['pct_supported']}%) @ threshold {s['threshold']}"
    )
    for r in audit_result["results"]:
        flag_str = f" [{','.join(r['flags'])}]" if r["flags"] else ""
        lines.append(
            f"{r['id']}: {r['verdict']} ({r['score']:.2f}){flag_str} :: {r['claim'][:100]}"
        )
    return "\n".join(lines)


def to_markdown(audit_result: dict) -> str:
    s = audit_result["summary"]
    now = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# cite-check audit report",
        "",
        f"_Generated {now} | threshold {s['threshold']} | "
        f"{s['supported']}/{s['total']} claims supported ({s['pct_supported']}%)_",
        "",
        "## Summary",
        "",
        "| Verdict | Count |",
        "|---|---|",
    ]
    for verdict in ("SUPPORTED", "WEAK", "UNSUPPORTED", "CONTRADICTED"):
        lines.append(f"| {_VERDICT_BADGE[verdict]} | {s['verdict_counts'].get(verdict, 0)} |")
    lines += [
        "",
        f"**Mean support score:** {s['avg_score']:.2f}",
        "",
    ]
    if s["worst_offenders"]:
        lines += ["### Worst offenders", ""]
        for w in s["worst_offenders"]:
            lines.append(
                f"- `{w['id']}` **{w['verdict']}** ({w['score']:.2f}): {w['claim']}"
            )
        lines.append("")
    lines += ["## Per-claim findings", ""]
    for i, r in enumerate(audit_result["results"], 1):
        badge = _VERDICT_BADGE.get(r["verdict"], r["verdict"])
        lines += [
            f"### {i}. [{badge}] `{r['id']}` — score {r['score']:.2f}",
            "",
            f"**Claim:** {r['claim'] or '(no claim text)'}",
            "",
            f"**Source:** {r['source_id'] or '(unknown)'}",
            "",
            "**Signals:**",
            "",
        ]
        for k, v in r["signals"].items():
            if isinstance(v, float):
                lines.append(f"- {k}: {v:.2f}")
            elif isinstance(v, bool):
                lines.append(f"- {k}: {'yes' if v else 'no'}")
            elif isinstance(v, (list, tuple)):
                lines.append(f"- {k}: {', '.join(map(str, v)) or '—'}")
            else:
                lines.append(f"- {k}: {v}")
        lines += ["", "**Evidence:**", ""]
        for e in r["evidence"]:
            lines.append(f"- {e}")
        if r["flags"]:
            lines += ["", f"**Flags:** {', '.join(r['flags'])}"]
        lines += [
            "",
            "> " + r["citation_excerpt"].replace("\n", " ") if r["citation_excerpt"] else "> (empty)",
            "",
            "---",
            "",
        ]
    return "\n".join(lines).rstrip() + "\n"
