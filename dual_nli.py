"""
Dual-NLI protocol -- reference implementation (v1.8, 2026-09-09).

Reads the support coordinate T from one NLI call and the contradiction coordinate F
from a separate NLI call, so that T and F do not share one softmax normalization
(Theorem 1 of the paper binds a single call: T + neutral + F = 1 => T + F <= 1).

Published protocol (Section 4 of the paper):
  * holistic scoring      : T = entailment_A(premise, claim), F = contradiction_B(premise, claim)
  * decomposed scoring    : T = entailment_A(support_span, claim), F = contradiction_B(refute_span, claim)
  * lexical topicality gate: F is kept only if Jaccard(refute_span, claim) >= theta (default 0.3),
                             computed over stemmed content words
  * margin rule           : flag iff T + F > 1 and min(T, F) >= tau (default 0.15)
  * bilateral rule        : flag iff min(T, F) >= tau          (no diagonal; Table 6)
  * polarity-blind scoring: T = max entailment over spans, F = max contradiction over spans
                             (Experiment 7; no oracle span roles)

All experiment scripts call `nli_scores` and `jaccard` from this module, so the
reference implementation and the reported numbers use one code path. Inputs are
passed to the Hugging Face pipeline as a (text, text_pair) pair; no model-specific
separator tokens are inserted by hand.

Usage:
    from dual_nli import DualNLI
    d = DualNLI(backend="huggingface")
    s = d.score_decomposed(support_span, refute_span, claim)
    print(s.T, s.F, s.flag_margin, s.flag_bilateral)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import re

DEFAULT_MODEL_A = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
DEFAULT_MODEL_B = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"

# ---------------------------------------------------------------------------
# NLI scoring helper (shared by every experiment script)
# ---------------------------------------------------------------------------

def build_pipe(model_name: str):
    from transformers import pipeline
    return pipeline("text-classification", model=model_name, top_k=None)


def nli_scores(pipe, premise: str, hypothesis: str) -> dict:
    """Full {label: probability} for one (premise, hypothesis) call.

    Passing the pair as {"text", "text_pair"} lets each tokenizer insert its own
    separator tokens; hand-written separators (e.g. RoBERTa's </s></s>) are not
    portable across models and are not used.
    """
    if pipe is None:  # stub backend: deterministic token-overlap heuristic for offline tests
        p = set(premise.lower().split()); h = set(hypothesis.lower().split())
        shared = len(p & h) / max(len(h), 1)
        return {"entailment": shared, "neutral": max(0.0, 1.0 - shared * 1.3),
                "contradiction": max(0.0, 0.3 - shared)}
    out = pipe({"text": premise, "text_pair": hypothesis})
    if out and isinstance(out[0], list):
        out = out[0]
    return {d["label"].lower(): float(d["score"]) for d in out}


# ---------------------------------------------------------------------------
# Lexical topicality gate (Experiment 4b / Section 4.4 of the paper)
# ---------------------------------------------------------------------------

STOP = set("""a an the of in on at to for from by with and or but however that this these those is are was were be been
being has have had does do did not no nor it its as than then there here into over under about which who whom whose
also states state claims claim reports report concludes conclude asserts assert argues argue insists insist maintains
maintain holds hold contends contend denies deny declares declare confirms confirm documents document certifies certify
teaches teach indicates indicate finds find shows show explains explain describes describe lists list attributes
attribute establishes establish affirms affirm verifies verify warns warn suggests suggest places place records record
according official disputed pamphlet blog post essay site treatise almanac newsletter theory forum flyer worksheet
leaflet op-ed study trial memo ledger textbook manual survey institute archive handbook entry bureau reference
guideline course curriculum agency audit biography atlas encyclopedia revisionist pseudoscientific outdated rival
amateur fringe confused erroneous competing flawed conspiracy geocentric retracted whistleblower denialist follow-up
leaked physics city lab""".split())


def content(s: str) -> set:
    toks = re.findall(r"[a-z]+", s.lower())
    return {t[:6] for t in toks if t not in STOP and len(t) > 2}


def jaccard(a: str, b: str) -> float:
    A, B = content(a), content(b)
    return len(A & B) / len(A | B) if (A | B) else 0.0


# ---------------------------------------------------------------------------
# Decision rules (Section 4.4 and Table 6)
# ---------------------------------------------------------------------------

def rule_margin(t: float, f: float, tau: float = 0.15) -> bool:
    """Published rule R_diag: T + F > 1 and min(T, F) >= tau."""
    return (t + f > 1.0) and (min(t, f) >= tau)


def rule_bilateral(t: float, f: float, tau: float = 0.15) -> bool:
    """R_bil: min(T, F) >= tau, no diagonal requirement."""
    return min(t, f) >= tau


def rule_f_only(t: float, f: float, tau: float = 0.15) -> bool:
    """R_F: single-channel rule F >= tau."""
    return f >= tau


# ---------------------------------------------------------------------------
# Protocol object
# ---------------------------------------------------------------------------

@dataclass
class DualNLIScore:
    T: float
    F: float
    F_ungated: float
    gated: bool
    jaccard: float
    flag_margin: bool
    flag_bilateral: bool
    raw_A: dict = field(default_factory=dict)
    raw_B: dict = field(default_factory=dict)

    @property
    def sum_TF(self) -> float:
        return self.T + self.F


class DualNLI:
    """Two NLI heads, one coordinate each.

    backend = "huggingface" loads the two public checkpoints (CPU is enough);
    backend = "stub" uses the offline heuristic in `nli_scores` for smoke tests.
    """

    def __init__(self, backend: str = "stub", model_a: Optional[str] = None,
                 model_b: Optional[str] = None, tau: float = 0.15, theta: float = 0.3):
        self.model_a_name = model_a or DEFAULT_MODEL_A
        self.model_b_name = model_b or DEFAULT_MODEL_B
        self.tau, self.theta = tau, theta
        self.pipe_a = self.pipe_b = None
        if backend == "huggingface":
            self.pipe_a = build_pipe(self.model_a_name)
            self.pipe_b = build_pipe(self.model_b_name)
        elif backend != "stub":
            raise ValueError("backend must be 'huggingface' or 'stub'")

    def _pack(self, raw_a: dict, raw_b: dict, gate_span: str, claim: str, gate: bool) -> DualNLIScore:
        t = raw_a.get("entailment", 0.0); f_raw = raw_b.get("contradiction", 0.0)
        j = jaccard(gate_span, claim)
        f = f_raw if (not gate or j >= self.theta) else 0.0
        return DualNLIScore(T=t, F=f, F_ungated=f_raw, gated=gate, jaccard=j,
                            flag_margin=rule_margin(t, f, self.tau),
                            flag_bilateral=rule_bilateral(t, f, self.tau),
                            raw_A=raw_a, raw_B=raw_b)

    def score_holistic(self, premise: str, claim: str, gate: bool = False) -> DualNLIScore:
        """Both heads read the whole premise (Section 5.2, negative result)."""
        return self._pack(nli_scores(self.pipe_a, premise, claim),
                          nli_scores(self.pipe_b, premise, claim), premise, claim, gate)

    def score_decomposed(self, support_span: str, refute_span: str, claim: str,
                         gate: bool = True) -> DualNLIScore:
        """Oracle-span protocol: A reads the supporting span, B the refuting span."""
        return self._pack(nli_scores(self.pipe_a, support_span, claim),
                          nli_scores(self.pipe_b, refute_span, claim), refute_span, claim, gate)

    def score_polarity_blind(self, spans: list, claim: str, gate: bool = True) -> DualNLIScore:
        """Experiment 7: max entailment (A) and max contradiction (B) over all spans, no roles."""
        a = [nli_scores(self.pipe_a, s, claim) for s in spans]
        b = [nli_scores(self.pipe_b, s, claim) for s in spans]
        t = max(x.get("entailment", 0.0) for x in a)
        cands = [(x.get("contradiction", 0.0), s) for x, s in zip(b, spans)]
        f_raw, span_f = max(cands)
        if gate:
            ok = [(c, s) for c, s in cands if jaccard(s, claim) >= self.theta]
            f, span_f = max(ok) if ok else (0.0, span_f)
        else:
            f = f_raw
        j = jaccard(span_f, claim)
        return DualNLIScore(T=t, F=f, F_ungated=f_raw, gated=gate, jaccard=j,
                            flag_margin=rule_margin(t, f, self.tau),
                            flag_bilateral=rule_bilateral(t, f, self.tau),
                            raw_A={"per_span": a}, raw_B={"per_span": b})


if __name__ == "__main__":
    # Offline smoke test with the stub backend; numbers are heuristic, not model outputs.
    d = DualNLI(backend="stub")
    s = d.score_decomposed("The physics textbook states that water boils at 100 degrees Celsius at sea level.",
                           "The lab report concludes that water does not boil at 100 degrees Celsius at sea level.",
                           "Water boils at 100 degrees Celsius at sea level.")
    print(f"stub: T={s.T:.3f} F={s.F:.3f} J={s.jaccard:.2f} margin={s.flag_margin} bilateral={s.flag_bilateral}")
