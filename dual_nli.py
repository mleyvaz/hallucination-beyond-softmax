"""
Dual-NLI protocol — reference implementation.

Computes T from Model A (MNLI-trained) and F from Model B (FEVER+sci-trained)
INDEPENDENTLY, removing the softmax T+F<=1 constraint that current detectors
inherit from single-NLI pipelines.

This is the operational core of:
    Leyva-Vázquez & Smarandache (2026, in preparation).
    Hallucination Detection Beyond Softmax: Why Current Detectors
    Structurally Cannot Measure Paraconsistency.

Usage:
    from dual_nli import DualNLI
    extractor = DualNLI(backend="huggingface")
    t, f, paraconsistent = extractor.score(response, ground_truth)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class DualNLIScore:
    T: float          # entailment from Model A
    F: float          # contradiction from Model B
    sum_TF: float     # T + F (paraconsistent if > 1)
    paraconsistent: bool
    model_A: str
    model_B: str
    raw_A: dict       # full raw output from Model A
    raw_B: dict       # full raw output from Model B


class DualNLI:
    """
    Dual independent NLI extractor for hallucination detection beyond
    the softmax constraint.

    Backends:
      - "huggingface"  — uses two transformers pipelines
      - "stub"         — deterministic placeholder for offline development
    """

    DEFAULT_MODEL_A = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    DEFAULT_MODEL_B = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"

    def __init__(self,
                 backend: str = "stub",
                 model_a: Optional[str] = None,
                 model_b: Optional[str] = None,
                 paraconsistent_threshold: float = 1.0):
        self.backend = backend
        self.model_a_name = model_a or self.DEFAULT_MODEL_A
        self.model_b_name = model_b or self.DEFAULT_MODEL_B
        self.paraconsistent_threshold = paraconsistent_threshold
        self.pipe_a = None
        self.pipe_b = None
        if backend == "huggingface":
            self._init_hf()

    def _init_hf(self):
        try:
            from transformers import pipeline
            self.pipe_a = pipeline("text-classification",
                                    model=self.model_a_name,
                                    return_all_scores=True)
            self.pipe_b = pipeline("text-classification",
                                    model=self.model_b_name,
                                    return_all_scores=True)
        except Exception as e:
            raise RuntimeError(
                "huggingface backend requires `transformers` + `torch`. "
                f"Install: pip install transformers torch. Original error: {e}"
            )

    def _score_one(self, pipe, premise: str, hypothesis: str,
                   target: str) -> tuple[float, dict]:
        """Run NLI inference and extract score for target label."""
        if pipe is None:
            # Stub: shared-token heuristic for offline dev
            p = set(premise.lower().split())
            h = set(hypothesis.lower().split())
            shared = len(p & h) / max(len(h), 1)
            scores = {"entailment": shared,
                      "neutral": max(0.0, 1.0 - shared * 1.3),
                      "contradiction": max(0.0, 0.3 - shared)}
            return scores[target], scores
        out = pipe(f"{premise} </s></s> {hypothesis}")[0]
        labels = {d["label"].lower(): d["score"] for d in out}
        return labels.get(target, 0.0), labels

    def score(self, response: str, ground_truth: str) -> DualNLIScore:
        """
        Returns a DualNLIScore for (response, ground_truth) pair.

        T comes from Model A entailment(response → ground_truth).
        F comes from Model B contradiction(response → ground_truth).
        These are NOT softmax-coupled, so T + F can exceed 1.
        """
        T, raw_A = self._score_one(self.pipe_a, response, ground_truth, "entailment")
        F, raw_B = self._score_one(self.pipe_b, response, ground_truth, "contradiction")
        sum_TF = T + F
        paraconsistent = sum_TF > self.paraconsistent_threshold
        return DualNLIScore(
            T=T, F=F,
            sum_TF=sum_TF,
            paraconsistent=paraconsistent,
            model_A=self.model_a_name,
            model_B=self.model_b_name,
            raw_A=raw_A,
            raw_B=raw_B,
        )

    def batch_score(self, pairs: list[tuple[str, str]]) -> list[DualNLIScore]:
        return [self.score(r, g) for r, g in pairs]


# ===== Verification of Theorem 1 (in stub mode) =====
def demonstrate_softmax_trap():
    """
    Demonstrates Theorem 1: under single-NLI softmax, T + F <= 1 always.
    Under dual-NLI, T + F > 1 is possible.
    """
    print("=" * 70)
    print("Demonstration of Theorem 1 (Softmax-Paraconsistency Impossibility)")
    print("=" * 70)

    # Single-NLI: stub a softmax-normalized output
    print("\n[Single-NLI softmax simulation]")
    print(f"{'Test case':<50} {'T':>6} {'N':>6} {'F':>6} {'T+F':>8}")
    print("-" * 80)
    test_cases = [
        ("Strong entailment", 0.85, 0.10, 0.05),
        ("Strong contradiction", 0.05, 0.10, 0.85),
        ("Pure neutral", 0.10, 0.80, 0.10),
        ("Mixed (boundary)", 0.40, 0.20, 0.40),
        ("Hypothetical paraconsistent", 0.70, 0.10, 0.65),  # IMPOSSIBLE under softmax
    ]
    for name, t, n, f in test_cases:
        s = t + n + f
        feasible = abs(s - 1.0) < 0.001
        marker = " " if feasible else "  <-- VIOLATES SOFTMAX (T+N+F != 1)"
        print(f"{name:<50} {t:>6.2f} {n:>6.2f} {f:>6.2f} {t+f:>8.2f}{marker}")

    print("\n[Dual-NLI: T from Model A, F from Model B independently]")
    print("Under dual-NLI, the same paraconsistent case becomes feasible:")
    print(f"  T_A = 0.70 (entailment per Model A)")
    print(f"  F_B = 0.65 (contradiction per Model B)")
    print(f"  T + F = 1.35 > 1.0  -->  PARACONSISTENT REGIME")
    print("\nThis is the regime that single-NLI detectors structurally cannot measure.")


if __name__ == "__main__":
    demonstrate_softmax_trap()

    print("\n" + "=" * 70)
    print("Operational test (stub backend):")
    print("=" * 70)
    extractor = DualNLI(backend="stub")
    pairs = [
        ("Paris is the capital of France.",
         "Paris is the capital of France."),
        ("The capital of France is Berlin, but also Paris in some sense.",
         "Paris is the capital of France."),
        ("I'm not sure, perhaps Lyon or Paris.",
         "Paris is the capital of France."),
    ]
    for response, gt in pairs:
        s = extractor.score(response, gt)
        marker = "  <-- PARACONSISTENT" if s.paraconsistent else ""
        print(f"\n  Response: {response[:60]}...")
        print(f"  GT:       {gt[:60]}...")
        print(f"  T = {s.T:.3f}  |  F = {s.F:.3f}  |  T+F = {s.sum_TF:.3f}{marker}")
