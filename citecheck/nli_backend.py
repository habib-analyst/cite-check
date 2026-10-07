"""Optional HuggingFace NLI backend.

Pluggable cross-encoder NLI scorer for users who *do* want a model-based
second opinion. Never required: everything else in cite-check works offline.

Usage:
    export CITECHECK_NLI_MODEL="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    cite-check audit answers.json --nli

Requires: transformers + torch (or another backend transformers supports).
"""

from __future__ import annotations

import os

DEFAULT_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"


class NLINotAvailable(RuntimeError):
    """Raised when the NLI backend is requested but unusable."""


class HuggingFaceNLIBackend:
    """Zero-shot NLI scorer returning P(entailment) in [0, 1]."""

    def __init__(self, model: str | None = None):
        self.model_name = model or os.environ.get("CITECHECK_NLI_MODEL", DEFAULT_MODEL)
        self._pipe = None

    @staticmethod
    def available() -> bool:
        try:
            import transformers  # noqa: F401
            return True
        except ImportError:
            return False

    def _ensure(self):
        if self._pipe is None:
            if not self.available():
                raise NLINotAvailable(
                    "transformers is not installed; the NLI backend is unavailable. "
                    "Install it with: pip install transformers torch"
                )
            from transformers import pipeline

            self._pipe = pipeline(
                "zero-shot-classification",
                model=self.model_name,
            )
        return self._pipe

    def score(self, premise: str, hypothesis: str) -> float:
        """Probability that the premise entails the hypothesis."""
        pipe = self._ensure()
        out = pipe(
            premise,
            candidate_labels=["entailment", "neutral", "contradiction"],
            hypothesis_template="{}",
        )
        # zero-shot pipeline returns labels/scores aligned by index
        labels = out["labels"] if isinstance(out, dict) else out[0]["labels"]
        scores = out["scores"] if isinstance(out, dict) else out[0]["scores"]
        for label, s in zip(labels, scores):
            if label == "entailment":
                return float(s)
        return 0.0
