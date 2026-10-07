# cite-check

[![CI](https://github.com/habib-analyst/cite-check/actions/workflows/ci.yml/badge.svg)](https://github.com/habib-analyst/cite-check/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/habib-analyst/cite-check/blob/main/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://github.com/habib-analyst/cite-check)

**Do your RAG citations actually support the claims?** cite-check audits
(claim, cited-passage) pairs and flags hallucinated or contradicted citations —
offline, dependency-free, and fully transparent about how each score was computed.

## The problem

RAG systems routinely produce answers with citations that *look* authoritative
but don't support the claim: wrong numbers, negated findings, or passages that
are merely topically related. Manual spot-checking doesn't scale, and
model-based judges are opaque, expensive, and themselves prone to hallucination.

cite-check is a **strong transparent baseline** for citation faithfulness:
a fast, deterministic, explainable first pass that catches the obvious failures
(hallucinated numbers, negation flips, antonym contradictions, ungrounded
claims) before you spend GPU time on a heavier judge.

## Architecture

```
RAG answer JSON ──► audit.py ──► entailment.py ──► report.py ──► Markdown/JSON
  {answer with [n]      parse [n]     ┌─ key-phrase coverage (0.55)
   markers, citations}  markers       ├─ TF-weighted lexical overlap (0.45)
                                      ├─ negation-flip detection ──► CONTRADICTED
                                      ├─ numeric consistency ──────► penalty/cap
                                      └─ antonym-pair lexicon ─────► CONTRADICTED
                                                        │
                                              optional: nli_backend.py
                                              (HuggingFace NLI, via
                                               CITECHECK_NLI_MODEL)
```

Each claim gets a 0–1 support score, a per-signal breakdown, human-readable
evidence strings, and one of four verdicts:
`SUPPORTED` (≥ 0.60), `WEAK` (≥ 0.30), `UNSUPPORTED`, `CONTRADICTED`.

## Quickstart (< 5 min)

```bash
git clone https://github.com/habib-analyst/cite-check && cd cite-check
pip install .                       # stdlib-only at runtime; no heavy deps

# 1. See it catch bad citations on the bundled fixture:
python -m citecheck demo
#    or: cite-check demo

# 2. Audit your own RAG output:
cite-check audit answers.json --threshold 0.6 --format md --out report.md
```

`answers.json` accepts two formats — a RAG answer doc with inline `[n]`
markers:

```json
{"question": "...",
 "answer": "Metformin lowers HbA1c [1]. Statins do not raise diabetes risk [2].",
 "citations": [{"id": "1", "source": "...", "text": "..."}, ...]}
```

or a plain list of `{id, claim, citation_text, source_id}` items.

Run the tests: `pip install -r requirements.txt && pytest tests/`

## Real example output

Actual output of `python -m citecheck demo` (bundled fixture: one clean, one
weak, one hallucinated, one contradicted citation):

```
supported 1/4 (25.0%) @ threshold 0.6
cite-1: SUPPORTED (1.00) :: Metformin lowers HbA1c by 1.0 to 1.5 percentage points in most patients with type 2 diabetes.
cite-2: WEAK (0.30) :: The treatment may also support modest weight loss and improve long-term cardiovascular outcomes.
cite-3: UNSUPPORTED (0.00) [numeric_mismatch] :: A 2024 trial found the drug cut stroke risk by 40% in high-risk patients.
cite-4: CONTRADICTED (0.25) [negation_flip] :: Statins do not increase the risk of diabetes.
```

<details>
<summary>Full Markdown report (click to expand)</summary>

```markdown
# cite-check audit report

_Generated 2026-10-07 16:38 UTC | threshold 0.6 | 1/4 claims supported (25.0%)_

## Summary

| Verdict | Count |
|---|---|
| SUPPORTED | 1 |
| WEAK | 1 |
| UNSUPPORTED | 1 |
| CONTRADICTED | 1 |

**Mean support score:** 0.39

### Worst offenders

- `cite-3` **UNSUPPORTED** (0.00): A 2024 trial found the drug cut stroke risk by 40% in high-risk patients.
- `cite-4` **CONTRADICTED** (0.25): Statins do not increase the risk of diabetes.
- `cite-2` **WEAK** (0.30): The treatment may also support modest weight loss and improve long-term cardiovascular outcomes.

## Per-claim findings

### 1. [SUPPORTED] `cite-1` — score 1.00

**Claim:** Metformin lowers HbA1c by 1.0 to 1.5 percentage points in most patients with type 2 diabetes.

**Source:** ADA Standards of Care, 2024

**Signals:**

- coverage: 1.00
- overlap: 1.00
- base_score: 1.00
- aligned_terms: 8/8
- claim_negated: no
- passage_negated: no
- negation_flip: no
- numeric_mismatch: no
- ungrounded_numbers: 0
- grounded_numbers: 3
- antonym_hits: —
- weights: 0.55*coverage + 0.45*overlap; -0.40 & cap 0.25 on numeric mismatch; contradiction caps score at 0.25

**Evidence:**

- key-phrase coverage 100% (8/8 phrase tokens)
- TF-weighted lexical overlap 100%

> In adults with type 2 diabetes, metformin therapy typically reduces HbA1c by 1.0–1.5 percentage points. Most patients tolerate it well.

---

### 2. [WEAK] `cite-2` — score 0.30

**Claim:** The treatment may also support modest weight loss and improve long-term cardiovascular outcomes.

**Source:** Metformin outcomes review

**Signals:**

- coverage: 0.30
- overlap: 0.30
- base_score: 0.30
- aligned_terms: 3/10
- claim_negated: no
- passage_negated: no
- negation_flip: no
- numeric_mismatch: no
- ungrounded_numbers: 0
- grounded_numbers: 0
- antonym_hits: —
- weights: 0.55*coverage + 0.45*overlap; -0.40 & cap 0.25 on numeric mismatch; contradiction caps score at 0.25

**Evidence:**

- key-phrase coverage 30% (3/10 phrase tokens)
- TF-weighted lexical overlap 30%
- claim contains no numbers to verify

> Patients on metformin experienced modest weight reduction averaging 2 kg over one year. Gastrointestinal side effects were the most common adverse events.

---

### 3. [UNSUPPORTED] `cite-3` — score 0.00

**Claim:** A 2024 trial found the drug cut stroke risk by 40% in high-risk patients.

**Source:** Unrelated formulary note

**Signals:**

- coverage: 0.00
- overlap: 0.00
- base_score: 0.00
- aligned_terms: 0/8
- claim_negated: no
- passage_negated: no
- negation_flip: no
- numeric_mismatch: yes
- ungrounded_numbers: 0
- grounded_numbers: 0
- antonym_hits: —
- weights: 0.55*coverage + 0.45*overlap; -0.40 & cap 0.25 on numeric mismatch; contradiction caps score at 0.25

**Evidence:**

- key-phrase coverage 0% (0/9 phrase tokens)
- TF-weighted lexical overlap 0%
- claim numbers not matched in passage: 2024, 40

**Flags:** numeric_mismatch

> Metformin is first-line therapy for type 2 diabetes. It lowers HbA1c by 1.0 to 1.5 percentage points and is generally well tolerated.

---

### 4. [CONTRADICTED] `cite-4` — score 0.25

**Claim:** Statins do not increase the risk of diabetes.

**Source:** Statin safety meta-analysis

**Signals:**

- coverage: 0.80
- overlap: 0.80
- base_score: 0.80
- aligned_terms: 4/5
- claim_negated: yes
- passage_negated: no
- negation_flip: yes
- numeric_mismatch: no
- ungrounded_numbers: 0
- grounded_numbers: 0
- antonym_hits: —
- weights: 0.55*coverage + 0.45*overlap; -0.40 & cap 0.25 on numeric mismatch; contradiction caps score at 0.25

**Evidence:**

- key-phrase coverage 80% (4/5 phrase tokens)
- TF-weighted lexical overlap 80%
- claim is negated but the aligned passage affirms it
- claim contains no numbers to verify

**Flags:** negation_flip

> Statin therapy is associated with a small increased risk of new-onset diabetes.

---
```

</details>

## How the score works

`score = 0.55 × key-phrase coverage + 0.45 × TF-weighted overlap`, then:

- **numeric mismatch** (same unit, different value, e.g. claim "40%" vs passage
  "1.5%"): −0.40 and score capped at 0.25 → `UNSUPPORTED`
- **ungrounded number** (claim number appears nowhere in the passage): −0.12 each
- **negation flip** (claim negated while the aligned passage affirms it, or vice
  versa) or **antonym pair** (`safe`/`unsafe`) in aligned context:
  verdict `CONTRADICTED`, score capped at 0.25

Numbers understand units (`%` ≡ `percent`, `mg/g/kg`, time units), ranges
("1.0 to 1.5" / "1.0–1.5"), and word numbers ("twenty"). A small bundled
synonym map (`reduce`↔`lower`, `trial`↔`study`, …) plus a lightweight stemmer
handles routine morphological variation.

Optional second opinion: `cite-check audit answers.json --nli` also scores each
pair with a HuggingFace NLI model (`CITECHECK_NLI_MODEL`, default
`MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`). Requires
`transformers` + `torch`; never needed for the offline path or the tests.

## Honest limitations

This is a **lexical baseline, not a language-understanding system**. Know what
it can't do:

- **Paraphrase beyond the synonym list.** "Improves cardiovascular health" vs
  "enhances heart health" scores WEAK, not SUPPORTED — `cardiovascular`↔`heart`
  isn't in the lexicon and there's no embedding similarity.
- **No coreference or multi-hop reasoning.** "It reduced events" won't resolve
  "it"; claims needing two passages combined aren't supported.
- **Crude stemming.** A hand-rolled suffix stripper, not Porter/Snowball —
  consistent between the two sides, but occasionally wrong.
- **Negation scope is heuristic.** "Not only effective but safe" can misfire;
  double negatives aren't handled.
- **Numbers need explicit digits or simple word numbers.** "°C" vs
  "degrees Celsius" isn't normalized; currencies and complex units aren't.
- **English only.** The stopword list, negation markers, and lexicons are
  English-specific.
- **High lexical overlap ≠ truth.** A passage can support a claim that's still
  factually wrong — cite-check verifies *attribution*, not *veracity*.

For production RAG eval, use cite-check as the cheap first-pass filter and
route `WEAK`/`UNSUPPORTED`/`CONTRADICTED` claims to a stronger (model-based)
judge or human review.

## Roadmap

- [ ] Embedding-based semantic overlap as an optional signal (still offline via a
      tiny bundled model)
- [ ] Claim decomposition: split compound sentences into atomic claims before scoring
- [ ] Multi-citation support: one claim supported by the *union* of passages
- [ ] Precision/recall-style citation metrics (à la ALCE) over full answers
- [ ] HTML report with highlighted evidence spans

## Citations

- Gao et al., "Enabling Large Language Models to Generate Text with Citations",
  EMNLP 2023 (ALCE: citation precision/recall for RAG).
- Rashkin et al., "Measuring Attribution in Natural Language Generation Models",
  2021 (AIS: the attributed-to-identified-sources framework).
- Honovich et al., "TRUE: Re-evaluating Factual Consistency Evaluation", NAACL 2022.
- Manakul et al., "SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection
  for Generative Large Language Models", EMNLP 2023.
- Es et al., "RAGAs: Automated Evaluation of Retrieval Augmented Generation", 2023.
- Maynez et al., "On Faithfulness and Factuality in Abstractive Summarization",
  ACL 2020.

## License

MIT © 2026 Habib Ur Rehman
