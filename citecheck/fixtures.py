"""Bundled demo fixture: one RAG answer with four citations.

- [1] clean:        passage fully supports the claim        -> SUPPORTED
- [2] weak:         passage covers weight loss, not cardio  -> WEAK
- [3] hallucinated: 40% stroke-risk cut appears nowhere     -> UNSUPPORTED
- [4] contradicted: claim negates what the passage affirms  -> CONTRADICTED
"""

DEMO_DOC = {
    "question": "What are the effects of metformin and statins?",
    "answer": (
        "Metformin lowers HbA1c by 1.0 to 1.5 percentage points in most "
        "patients with type 2 diabetes [1]. The treatment may also support "
        "modest weight loss and improve long-term cardiovascular outcomes [2]. "
        "A 2024 trial found the drug cut stroke risk by 40% in high-risk "
        "patients [3]. Statins do not increase the risk of diabetes [4]."
    ),
    "citations": [
        {
            "id": "1",
            "source": "ADA Standards of Care, 2024",
            "text": (
                "In adults with type 2 diabetes, metformin therapy typically "
                "reduces HbA1c by 1.0\u20131.5 percentage points. "
                "Most patients tolerate it well."
            ),
        },
        {
            "id": "2",
            "source": "Metformin outcomes review",
            "text": (
                "Patients on metformin experienced modest weight reduction "
                "averaging 2 kg over one year. Gastrointestinal side effects "
                "were the most common adverse events."
            ),
        },
        {
            "id": "3",
            "source": "Unrelated formulary note",
            "text": (
                "Metformin is first-line therapy for type 2 diabetes. "
                "It lowers HbA1c by 1.0 to 1.5 percentage points and is "
                "generally well tolerated."
            ),
        },
        {
            "id": "4",
            "source": "Statin safety meta-analysis",
            "text": (
                "Statin therapy is associated with a small increased risk "
                "of new-onset diabetes."
            ),
        },
    ],
}
