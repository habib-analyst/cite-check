"""Offline claim <-> citation entailment scoring.

A transparent, dependency-free baseline for auditing whether a cited passage
actually supports a claim. Combines five lexical signals:

  1. key-phrase coverage  - fraction of the claim's key phrases (chunked runs
     of content words) whose terms appear in the passage
  2. lexical overlap       - TF-weighted token overlap between claim and passage
  3. negation-flip         - claim negated while the aligned passage affirms
     (or vice versa) -> contradiction
  4. numeric consistency   - every (number, unit) in the claim must be grounded
     in the passage; a same-unit different-value number is a mismatch
  5. antonym pairs         - claim term vs. its antonym in the passage, in
     aligned context -> contradiction

Verdicts: SUPPORTED / WEAK / UNSUPPORTED / CONTRADICTED.

This is intentionally a *strong transparent baseline*, not a full NLI model:
it cannot do coreference, deep paraphrase, or multi-hop reasoning. See the
README's "Limitations" section for an honest accounting.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

__all__ = ["EntailmentResult", "score_entailment"]

# --------------------------------------------------------------------------
# Tokenization / normalization
# --------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    """
    a an the and or but if then than so as at by for of in on to with from into
    over under between through during before after about against among per via
    is are was were be been being have has had having do does did doing will
    would shall should may might must can could ought it its this that these
    those they them their theirs he him his she her hers we us our ours you
    your yours i me my mine what which who whom whose when where why how all
    any both each few more most other some such only own same too very also
    just even still however although though because while whereas across within
    the a an s t d ll ve re m
    """.split()
)

# Negation markers matched against the raw text (not stemmed tokens), so
# contractions like "don't" are caught.
_NEG_RE = re.compile(
    r"\b(?:not|no|never|neither|nor|none|nothing|nobody|noone|without|cannot)\b"
    r"|\b(?:can't|won't|don't|doesn't|isn't|aren't|wasn't|weren't|hasn't|haven't"
    r"|hadn't|couldn't|shouldn't|wouldn't|mustn't|needn't)\b"
    r"|n't\b",
    re.IGNORECASE,
)


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens."""
    return _TOKEN_RE.findall(text.lower())


def stem(word: str) -> str:
    """Tiny consistent stemmer (NOT Porter): strips common English suffixes.

    Consistency between the two sides being compared matters more than
    linguistic correctness here; variant expansion in ``_variants`` recovers
    the common silent-e cases (increas <-> increase).
    """
    w = word.lower()
    if len(w) <= 3:
        return w
    if w.endswith("ies") and len(w) > 5:
        return w[:-3] + "y"  # studies -> study
    if w.endswith(("sses", "shes", "ches", "xes", "zes")):
        return w[:-2]  # classes -> class
    if w.endswith("es") and len(w) > 4 and w[-3] in ("s", "x", "z"):
        return w[:-2]  # boxes -> box
    if w.endswith("s") and not w.endswith(("ss", "us", "is")) and len(w) > 3:
        w = w[:-1]  # trials -> trial
    if w.endswith("eed"):
        return w
    if w.endswith("ed") and len(w) > 4:
        b = w[:-2]
        # restore silent e: reduced -> reduce, increased -> increase
        if len(b) > 2 and b[-1] not in "aeiou" and b[-2] in "aeiou":
            return b + "e"
        if len(b) > 3 and b[-1] == b[-2] and b[-1] not in "aeiou":
            return b[:-1]  # stopped -> stop
        return b
    if w.endswith("ing") and len(w) > 5:
        b = w[:-3]
        if len(b) > 3 and b[-1] == b[-2] and b[-1] not in "aeiou":
            b = b[:-1]
        return b
    return w


def _variants(stemmed: str) -> set[str]:
    """Plausible surface variants of a stem, for fuzzy matching."""
    out = {stemmed, stemmed + "e"}
    if stemmed.endswith("e") and len(stemmed) > 3:
        out.add(stemmed[:-1])
    return out


def _is_content(token: str) -> bool:
    return (
        token not in STOPWORDS
        and len(token) > 1
        and not token.isdigit()
    )


# --------------------------------------------------------------------------
# Lexical resources (stemmed at load)
# --------------------------------------------------------------------------

_SYNONYM_PAIRS = [
    ("reduce", "lower"), ("reduce", "decrease"), ("reduce", "diminish"),
    ("reduce", "cut"), ("reduce", "lessen"),
    ("increase", "raise"), ("increase", "boost"), ("increase", "elevate"),
    ("increase", "grow"),
    ("improve", "enhance"), ("improve", "ameliorate"),
    ("show", "demonstrate"), ("show", "indicate"), ("show", "reveal"),
    ("show", "report"), ("show", "suggest"), ("show", "find"),
    ("use", "utilize"), ("use", "employ"),
    ("help", "aid"), ("help", "assist"),
    ("associated", "linked"), ("associated", "related"), ("associated", "correlated"),
    ("cause", "induce"), ("cause", "trigger"), ("cause", "produce"),
    ("prevent", "avert"), ("prevent", "avoid"),
    ("treatment", "therapy"),
    ("loss", "reduction"),
    ("common", "frequent"), ("common", "prevalent"),
    ("significant", "substantial"), ("significant", "considerable"),
    ("effective", "efficacious"),
    ("risk", "hazard"),
    ("death", "mortality"),
    ("trial", "study"),
]

_ANTONYM_PAIRS = [
    ("increase", "decrease"), ("rise", "fall"), ("grow", "shrink"),
    ("improve", "worsen"), ("improvement", "decline"), ("enhance", "diminish"),
    ("effective", "ineffective"), ("safe", "unsafe"),
    ("higher", "lower"), ("high", "low"), ("greater", "lesser"),
    ("more", "less"), ("most", "least"),
    ("gain", "loss"), ("benefit", "harm"), ("beneficial", "harmful"),
    ("positive", "negative"), ("present", "absent"), ("presence", "absence"),
    ("significant", "insignificant"), ("common", "rare"),
    ("frequent", "infrequent"), ("strong", "weak"), ("large", "small"),
    ("long", "short"), ("early", "late"), ("fast", "slow"),
    ("success", "failure"), ("accurate", "inaccurate"),
    ("correct", "incorrect"), ("true", "false"),
    ("possible", "impossible"), ("likely", "unlikely"),
    ("always", "never"),
    ("activate", "inhibit"), ("stimulate", "suppress"),
    ("upregulate", "downregulate"), ("sensitive", "resistant"),
    ("superior", "inferior"), ("better", "worse"),
    ("advantage", "disadvantage"),
    ("elevated", "reduced"), ("prolonged", "shortened"),
    ("alive", "dead"),
]

# stemmed, bidirectional lookup tables
_SYNONYMS: dict[str, set[str]] = {}
for _a, _b in _SYNONYM_PAIRS:
    _sa, _sb = stem(_a), stem(_b)
    _SYNONYMS.setdefault(_sa, set()).add(_sb)
    _SYNONYMS.setdefault(_sb, set()).add(_sa)

_ANTONYMS: dict[str, set[str]] = {}
_ANTONYM_DISPLAY: dict[str, str] = {}  # stem -> canonical display pair
for _a, _b in _ANTONYM_PAIRS:
    _sa, _sb = stem(_a), stem(_b)
    _ANTONYMS.setdefault(_sa, set()).add(_sb)
    _ANTONYMS.setdefault(_sb, set()).add(_sa)
    _ANTONYM_DISPLAY[_sa] = f"{_a} <-> {_b}"
    _ANTONYM_DISPLAY[_sb] = f"{_a} <-> {_b}"


# --------------------------------------------------------------------------
# Numbers
# --------------------------------------------------------------------------

_WORD_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}
_MULT = {"hundred": 100, "thousand": 1_000, "million": 1_000_000,
         "billion": 1_000_000_000}

_UNIT_RE = re.compile(
    r"\s*(%|percent(?:age)?(?:\s+points?)?|milligrams?|grams?|kilograms?|"
    r"mgs?|gs?|kgs?|mm|cm|km|years?|months?|weeks?|days?|hours?|fold)(?![a-z0-9])",
    re.IGNORECASE,
)
_RANGE_GAP_RE = re.compile(r"^[\s\u2013\u2014\-]*(?:to)?[\s\u2013\u2014\-]*$")


def _norm_unit(unit: str | None) -> str | None:
    if unit is None:
        return None
    u = unit.lower().strip()
    if u == "%" or u.startswith("percent"):
        return "pct"
    if u.startswith("milligram"):
        return "mg"
    if u.startswith("kilogram"):
        return "kg"
    if u.startswith("gram"):
        return "g"
    if u.endswith("s") and len(u) > 2 and u not in ("gs", "ms"):
        u = u[:-1]  # years -> year
    return u


def extract_numbers(text: str) -> list[tuple[float, str | None, str]]:
    """Extract (value, unit, raw) numbers, incl. word numbers and ranges."""
    found: list[tuple[int, float, str | None, str]] = []  # (pos, val, unit, raw)
    for m in re.finditer(r"\b\d+(?:\.\d+)?\b", text):
        um = _UNIT_RE.match(text, m.end())
        unit = _norm_unit(um.group(1)) if um else None
        found.append((m.start(), float(m.group()), unit, m.group()))
    toks = tokenize(text)
    for i, t in enumerate(toks):
        if t in _WORD_NUM:
            v = float(_WORD_NUM[t])
            if i + 1 < len(toks) and toks[i + 1] in _MULT:
                v *= _MULT[toks[i + 1]]
            found.append((len(text) + i, v, None, t))  # word nums sort last
    found.sort(key=lambda x: x[0])
    # propagate units across ranges: "1.0 to 1.5 percentage points"
    out: list[tuple[float, str | None, str]] = [
        (v, u, r) for _, v, u, r in found
    ]
    spans = [(p, p + len(r)) for p, _, _, r in found]
    for i in range(len(out) - 1):
        v, u, r = out[i]
        if u is None and out[i + 1][1] is not None:
            gap = text[spans[i][1]:spans[i + 1][0]]
            if _RANGE_GAP_RE.match(gap):
                out[i] = (v, out[i + 1][1], r)
    return out


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=0.02, abs_tol=1e-9)


# --------------------------------------------------------------------------
# Result
# --------------------------------------------------------------------------

SUPPORTED, WEAK, UNSUPPORTED, CONTRADICTED = (
    "SUPPORTED", "WEAK", "UNSUPPORTED", "CONTRADICTED",
)

_SUPPORT_THRESHOLD = 0.60
_WEAK_THRESHOLD = 0.30


@dataclass
class EntailmentResult:
    score: float
    verdict: str
    signals: dict = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 4),
            "verdict": self.verdict,
            "signals": self.signals,
            "flags": self.flags,
            "evidence": self.evidence,
        }


# --------------------------------------------------------------------------
# Scorer
# --------------------------------------------------------------------------

def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def _content_stems(text: str) -> list[str]:
    """Ordered, deduplicated content-word stems."""
    seen: dict[str, None] = {}
    for tok in tokenize(text):
        if _is_content(tok):
            s = stem(tok)
            if s not in seen:
                seen[s] = None
    return list(seen)


def _key_phrases(text: str) -> list[list[str]]:
    """Chunked runs of content stems (numbers don't break a run), <= 4 tokens."""
    runs: list[list[str]] = []
    current: list[str] = []
    for tok in tokenize(text):
        if tok.isdigit() or re.fullmatch(r"\d+\.\d+", tok):
            continue  # numbers handled by the numeric signal; don't split phrases
        if _is_content(tok):
            current.append(stem(tok))
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    phrases: list[list[str]] = []
    for run in runs:
        for i in range(0, len(run), 4):
            phrases.append(run[i:i + 4])
    return phrases


class _PassageIndex:
    """Precomputed, variant-expanded view of a passage for fast matching."""

    def __init__(self, passage: str):
        self.sentences = _sentences(passage)
        self.stems = _content_stems(passage)
        self.expanded: set[str] = set()
        for s in self.stems:
            self.expanded.update(_variants(s))
        self.sent_expanded = [self._expand_sentence(s) for s in self.sentences]

    def _expand_sentence(self, sent: str) -> set[str]:
        out: set[str] = set()
        for tok in tokenize(sent):
            if _is_content(tok):
                out.update(_variants(stem(tok)))
        return out

    def claim_variants(self, stemmed_term: str) -> set[str]:
        out = set(_variants(stemmed_term))
        for syn in _SYNONYMS.get(stemmed_term, ()):
            out.update(_variants(syn))
        return out

    def present(self, stemmed_term: str) -> bool:
        """Is a claim term (or a synonym) present in the passage?"""
        return bool(self.claim_variants(stemmed_term) & self.expanded)

    def present_in(self, stemmed_term: str, sent_idx: int) -> bool:
        return bool(self.claim_variants(stemmed_term) & self.sent_expanded[sent_idx])


def score_entailment(claim: str, passage: str) -> EntailmentResult:
    """Score how well `passage` supports `claim`. Returns EntailmentResult."""
    claim, passage = claim.strip(), passage.strip()
    if not claim or not passage:
        return EntailmentResult(
            score=0.0, verdict=UNSUPPORTED,
            signals={"coverage": 0.0, "overlap": 0.0},
            flags=["empty_input"],
            evidence=["claim or passage is empty"],
        )

    claim_terms = _content_stems(claim)
    claim_phrases = _key_phrases(claim)
    pidx = _PassageIndex(passage)

    # --- signal 1: key-phrase coverage -------------------------------------
    matched_w = 0.0
    total_w = 0.0
    for phrase in claim_phrases:
        hits = sum(1 for t in phrase if pidx.present(t))
        matched_w += hits
        total_w += len(phrase)
    coverage = (matched_w / total_w) if total_w else 0.0

    # --- signal 2: TF-weighted lexical overlap ------------------------------
    tf = Counter(claim_terms)
    total_weight = sum(1.0 + math.log(1.0 + c) for c in tf.values())
    overlap_weight = sum(
        1.0 + math.log(1.0 + c) for t, c in tf.items() if pidx.present(t)
    )
    overlap = (overlap_weight / total_weight) if total_weight else 0.0

    n_matched = sum(1 for t in claim_terms if pidx.present(t))
    aligned_frac = (n_matched / len(claim_terms)) if claim_terms else 0.0
    aligned = n_matched >= 2 and aligned_frac >= 0.25

    # --- signal 3: negation flip --------------------------------------------
    claim_neg = bool(_NEG_RE.search(claim))
    aligned_sents = [
        i for i in range(len(pidx.sentences))
        if sum(1 for t in claim_terms if pidx.present_in(t, i)) >= 2
        or (claim_terms and sum(1 for t in claim_terms if pidx.present_in(t, i)) >= 1
            and len(claim_terms) <= 2)
    ]
    passage_neg = any(_NEG_RE.search(pidx.sentences[i]) for i in aligned_sents)
    neg_flip = claim_neg != passage_neg

    # --- signal 4: numeric consistency ---------------------------------------
    claim_nums = extract_numbers(claim)
    passage_nums = extract_numbers(passage)
    mismatch = False
    ungrounded = 0
    grounded = 0
    for v, u, raw in claim_nums:
        same_unit = [(pv, pu) for pv, pu, _ in passage_nums if pu == u]
        if same_unit and any(_close(v, pv) for pv, _ in same_unit):
            grounded += 1
        elif u is not None and any(
            _close(v, pv) for pv, pu, _ in passage_nums if pu is None
        ):
            grounded += 1  # value matches a unit-less passage number
        elif u is None and any(_close(v, pv) for pv, _, _ in passage_nums):
            grounded += 1
        elif same_unit:
            mismatch = True
        else:
            ungrounded += 1

    # --- signal 5: antonym pairs ---------------------------------------------
    antonym_hits: list[str] = []
    for t in claim_terms:
        for ant in _ANTONYMS.get(t, ()):
            ant_present = bool(set(_variants(ant)) & pidx.expanded)
            term_present = pidx.present(t)
            if ant_present and not term_present:
                antonym_hits.append(_ANTONYM_DISPLAY.get(t, f"{t} <-> ?"))

    contradiction = (neg_flip and aligned) or (
        bool(antonym_hits) and n_matched >= 1
    )

    # --- combine --------------------------------------------------------------
    base = 0.55 * coverage + 0.45 * overlap
    score = base
    if mismatch:
        score = min(base - 0.40, 0.25)
    elif ungrounded:
        score = base - min(0.25, 0.12 * ungrounded)
    score = max(0.0, min(1.0, score))

    flags: list[str] = []
    if neg_flip and aligned_sents:
        flags.append("negation_flip")
    if mismatch:
        flags.append("numeric_mismatch")
    if ungrounded:
        flags.append("ungrounded_numbers")
    if antonym_hits and (aligned or n_matched >= 1):
        flags.append("antonym_pair")
    if not aligned and neg_flip and not aligned_sents:
        flags.append("low_alignment")

    if contradiction:
        verdict = CONTRADICTED
        score = min(score, 0.25)
    elif score >= _SUPPORT_THRESHOLD:
        verdict = SUPPORTED
    elif score >= _WEAK_THRESHOLD:
        verdict = WEAK
    else:
        verdict = UNSUPPORTED

    evidence = [
        f"key-phrase coverage {coverage:.0%} "
        f"({int(round(matched_w))}/{int(total_w)} phrase tokens)",
        f"TF-weighted lexical overlap {overlap:.0%}",
    ]
    if claim_neg and not passage_neg and neg_flip and aligned_sents:
        evidence.append("claim is negated but the aligned passage affirms it")
    elif passage_neg and not claim_neg and neg_flip and aligned_sents:
        evidence.append("passage is negated but the claim affirms it")
    if mismatch:
        bad = [raw for v, u, raw in claim_nums]
        evidence.append(f"claim numbers not matched in passage: {', '.join(bad)}")
    if ungrounded:
        evidence.append(f"{ungrounded} claim number(s) have no grounding in passage")
    for hit in antonym_hits:
        evidence.append(f"antonym pair across claim/passage: {hit}")
    if not claim_nums:
        evidence.append("claim contains no numbers to verify")

    signals = {
        "coverage": round(coverage, 4),
        "overlap": round(overlap, 4),
        "base_score": round(base, 4),
        "aligned_terms": f"{n_matched}/{len(claim_terms)}",
        "claim_negated": claim_neg,
        "passage_negated": passage_neg,
        "negation_flip": neg_flip,
        "numeric_mismatch": mismatch,
        "ungrounded_numbers": ungrounded,
        "grounded_numbers": grounded,
        "antonym_hits": antonym_hits,
        "weights": "0.55*coverage + 0.45*overlap; -0.40 & cap 0.25 on numeric "
                   "mismatch; contradiction caps score at 0.25",
    }
    return EntailmentResult(
        score=round(score, 4), verdict=verdict,
        signals=signals, flags=flags, evidence=evidence,
    )
