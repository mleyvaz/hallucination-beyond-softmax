"""
Experiment A — Dual-NLI with REAL NLI models (replaces illustrative heuristics).

Runs the SAME 50 hand-crafted pairs from synthetic_validation.py, but scores them
with two independent, state-of-the-art NLI models instead of the token-overlap /
negation-cue heuristics:

  - Model A (T = entailment): MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli
  - Model B (F = contradiction): ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli

Two regimes are compared on the identical pairs:

  DUAL-NLI   : T = entailment(Model A),  F = contradiction(Model B)
               -> T and F are NOT coupled by a shared softmax; T + F can exceed 1.

  SINGLE-NLI : T = entailment(Model A),  F = contradiction(Model A)
               -> both read from the SAME softmax output, so T + F <= 1 ALWAYS
                  (empirical confirmation of Theorem 1 with real models).

Output:
  - validation_results_real.csv  : raw per-pair scores with real models
  - validation_summary_real.txt  : aggregated statistics
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, asdict

from synthetic_validation import PAIRS  # reuse the exact 50 pairs

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE  # scripts and data live flat in this repo; do not assume a parent directory
OUT_CSV = os.path.join(ROOT, "validation_results_real.csv")
OUT_TXT = os.path.join(ROOT, "validation_summary_real.txt")

MODEL_A = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
MODEL_B = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"


def build_pipe(model_name):
    from transformers import pipeline
    return pipeline("text-classification", model=model_name, top_k=None)


def label_scores(pipe, premise: str, hypothesis: str) -> dict:
    """Return {label_lower: score} for an NLI (premise, hypothesis) pair."""
    out = pipe({"text": premise, "text_pair": hypothesis})
    # transformers may return [[{...}]] or [{...}] depending on version
    if out and isinstance(out[0], list):
        out = out[0]
    return {d["label"].lower(): float(d["score"]) for d in out}


@dataclass
class PairResult:
    pair_id: int
    cls: str
    premise: str
    hypothesis: str
    # Dual-NLI
    T_dual: float
    F_dual: float
    sum_TF_dual: float
    paraconsistent_dual: bool
    # Single-NLI (Model A softmax only)
    T_single: float
    N_single: float
    F_single: float
    sum_TF_single: float
    paraconsistent_single: bool


def run():
    print(f"Loading Model A: {MODEL_A} ...", flush=True)
    pipe_a = build_pipe(MODEL_A)
    print(f"Loading Model B: {MODEL_B} ...", flush=True)
    pipe_b = build_pipe(MODEL_B)
    print("Both models loaded. Scoring 50 pairs...\n", flush=True)

    results = []
    for i, (cls, premise, hypothesis) in enumerate(PAIRS, 1):
        a = label_scores(pipe_a, premise, hypothesis)
        b = label_scores(pipe_b, premise, hypothesis)

        t_a = a.get("entailment", 0.0)
        c_a = a.get("contradiction", 0.0)
        n_a = a.get("neutral", 0.0)
        f_b = b.get("contradiction", 0.0)

        # Dual-NLI: T from A, F from B (independent softmaxes)
        sum_dual = t_a + f_b
        # Single-NLI: T and F both from Model A's single softmax
        sum_single = t_a + c_a

        results.append(PairResult(
            pair_id=i, cls=cls, premise=premise, hypothesis=hypothesis,
            T_dual=round(t_a, 4), F_dual=round(f_b, 4),
            sum_TF_dual=round(sum_dual, 4),
            paraconsistent_dual=sum_dual > 1.0,
            T_single=round(t_a, 4), N_single=round(n_a, 4), F_single=round(c_a, 4),
            sum_TF_single=round(sum_single, 4),
            paraconsistent_single=sum_single > 1.0,
        ))
        print(f"  [{i:>2}/{len(PAIRS)}] {cls}  T={t_a:.3f}  F_B={f_b:.3f}  "
              f"T+F_dual={sum_dual:.3f}  | F_A={c_a:.3f}  T+F_single={sum_single:.3f}",
              flush=True)
    return results


def write_csv(results):
    fieldnames = list(asdict(results[0]).keys())
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow(asdict(r))


def summarize(results) -> str:
    classes = ["ENT", "CON", "NEU", "PAR"]
    L = []
    L.append("=" * 84)
    L.append("EXPERIMENT A — Dual-NLI vs Single-NLI with REAL NLI models")
    L.append("=" * 84)
    L.append(f"Model A (T): {MODEL_A}")
    L.append(f"Model B (F): {MODEL_B}")
    L.append(f"Total pairs: {len(results)}")
    L.append("")
    L.append(f"{'Class':<6} {'N':>4} {'mean(T)':>9} {'mean(F_B)':>10} "
             f"{'mean(T+F)':>11} {'%para_dual':>12} {'%para_single':>14}")
    L.append("-" * 84)
    for cls in classes:
        sub = [r for r in results if r.cls == cls]
        if not sub:
            continue
        n = len(sub)
        mt = sum(r.T_dual for r in sub) / n
        mf = sum(r.F_dual for r in sub) / n
        ms = sum(r.sum_TF_dual for r in sub) / n
        pd = 100 * sum(1 for r in sub if r.paraconsistent_dual) / n
        ps = 100 * sum(1 for r in sub if r.paraconsistent_single) / n
        L.append(f"{cls:<6} {n:>4} {mt:>9.3f} {mf:>10.3f} {ms:>11.3f} "
                 f"{pd:>11.1f}% {ps:>13.1f}%")
    L.append("")

    n_total = len(results)
    npd = sum(1 for r in results if r.paraconsistent_dual)
    nps = sum(1 for r in results if r.paraconsistent_single)
    L.append("Aggregate:")
    L.append(f"  Paraconsistent (T+F>1) under DUAL-NLI:    {npd}/{n_total} "
             f"({100*npd/n_total:.1f}%)")
    L.append(f"  Paraconsistent (T+F>1) under SINGLE-NLI:  {nps}/{n_total} "
             f"({100*nps/n_total:.1f}%)")
    L.append("")
    L.append("Theorem 1 verification (real models):")
    if nps == 0:
        L.append("  CONFIRMED. Single-NLI (Model A softmax) never produces T + F > 1.")
    else:
        L.append(f"  NOTE: single-NLI produced {nps} cases with T+F>1 "
                 f"(check label normalization / rounding).")
    L.append("")

    # Independence diagnostic: Pearson(T_A, 1 - F_B)
    n = len(results)
    t_vals = [r.T_dual for r in results]
    omf = [1 - r.F_dual for r in results]
    mt = sum(t_vals) / n
    mo = sum(omf) / n
    cov = sum((t_vals[i] - mt) * (omf[i] - mo) for i in range(n)) / n
    vt = sum((x - mt) ** 2 for x in t_vals) / n
    vo = sum((x - mo) ** 2 for x in omf) / n
    corr = cov / ((vt ** 0.5) * (vo ** 0.5)) if vt > 0 and vo > 0 else float("nan")
    L.append("Independence diagnostic:")
    L.append(f"  Pearson(T_A, 1 - F_B) = {corr:.3f}")
    L.append("  (close to 0 => independent; close to 1 => softmax-like coupling)")
    L.append("")

    par = [r for r in results if r.cls == "PAR"]
    npar = len(par)
    rec = sum(1 for r in par if r.paraconsistent_dual)
    L.append("Recovery of paraconsistent-class items:")
    L.append(f"  Of {npar} hand-crafted paraconsistent pairs, dual-NLI flagged "
             f"{rec}/{npar} ({100*rec/max(npar,1):.1f}%) as T+F>1.")
    L.append(f"  Single-NLI flagged "
             f"{sum(1 for r in par if r.paraconsistent_single)}/{npar} as T+F>1.")
    L.append("")
    return "\n".join(L)


def main():
    results = run()
    write_csv(results)
    summary = summarize(results)
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(summary)
    print("\n" + summary)
    print(f"\nWrote: {OUT_CSV}")
    print(f"Wrote: {OUT_TXT}")


if __name__ == "__main__":
    main()
