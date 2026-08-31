"""
Experiment A3 - Controls for the evidence-decomposed protocol (added 2026-08-31).

Answers the reviewer objection "decomposition recovers T+F>1 by construction":
if any two clear segments handed to Model A and Model B yield T high and F high,
the 75% recovery of Experiment A2 would say nothing about conflict.

Three controls, all scored with the SAME decomposed protocol as A2:

  AGREE-SUP : two independent sources that BOTH support the claim
              (segment 1 -> Model A for T, segment 2 -> Model B for F).
              Expected: T high, F low  -> not flagged.
  AGREE-REF : two independent sources that BOTH refute the claim
              (segment 1 -> Model A for T, segment 2 -> Model B for F).
              Expected: T low, F high  -> not flagged.
  CROSS     : supporting segment of item i with refuting segment of item j
              (j != i, different claim), scored against claim i.
              Expected: F low (segment j is about another claim) -> not flagged.

Plus:
  - threshold sweep of the margin rule min(T,F) >= tau, tau in {0.00 .. 0.40},
    reporting recall on A2 decomposed conflict items and false-positive rate on
    all controls (A2 ENT/CON single-segment controls + the three new controls);
  - Pearson(T_A, 1 - F_B) restricted to NEU + PAR pairs of Experiment A
    (removing the class-driven correlation of ENT/CON pairs).

Outputs: validation_results_a3.csv, validation_summary_a3.txt (repo root).
"""
from __future__ import annotations

import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
OUT_CSV = os.path.join(ROOT, "validation_results_a3.csv")
OUT_TXT = os.path.join(ROOT, "validation_summary_a3.txt")
A2_CSV = os.path.join(ROOT, "validation_results_a2.csv")
REAL_CSV = os.path.join(ROOT, "validation_results_real.csv")

from experiment_a2_conflicting_evidence import (  # noqa: E402
    MODEL_A, MODEL_B, CONFLICT_ITEMS, CONTROLS, build_pipe, label_scores)

# Second supporting source for each A2 claim (same claim, different source).
AGREE_SUP_2 = [
    "The national gazetteer confirms that Paris is the capital of France.",
    "The chemistry handbook confirms that water boils at 100 degrees Celsius at sea level.",
    "The satellite survey confirms that the Amazon rainforest is located in South America.",
    "The physics archive confirms that Einstein developed the theory of general relativity.",
    "The botany textbook confirms that photosynthesis converts sunlight into chemical energy.",
    "The cardiology reference confirms that the human heart has four chambers.",
    "The mountaineering federation confirms that Mount Everest is the tallest mountain on Earth.",
    "The maritime atlas confirms that the Pacific is the largest ocean on Earth.",
    "The First Folio records that Shakespeare wrote Hamlet.",
    "The physics primer confirms that light travels faster than sound in air.",
    "The Caribbean gazetteer confirms that Cuba is an island nation in the Caribbean.",
    "The United Nations report confirms that Tokyo is the most populous metropolitan area.",
    "The materials database confirms that gold is denser than iron.",
    "The public health agency confirms that vaccines prevent infectious diseases.",
    "The planetarium guide confirms that the Earth orbits the Sun.",
    "The dietary reference confirms that oranges contain vitamin C.",
    "The independent inspection confirms that the bridge can support heavy trucks.",
    "The meteorological service confirms that global temperatures rose over the last decade.",
    "The replication study confirms that the drug reduces blood pressure.",
    "The tax authority confirms that the company paid its taxes in full.",
]

# Second refuting source for each A2 claim.
AGREE_REF_2 = [
    "The travel forum insists that Paris is not the capital of France.",
    "The student worksheet concludes that water does not boil at 100 degrees Celsius at sea level.",
    "The mislabeled map places the Amazon rainforest outside South America.",
    "The alternative history blog denies that Einstein developed the theory of general relativity.",
    "The discredited manual claims that photosynthesis does not convert sunlight into chemical energy.",
    "The folk anatomy pamphlet states that the human heart does not have four chambers.",
    "The rival club newsletter maintains that Mount Everest is not the tallest mountain on Earth.",
    "The trivia card asserts that the Pacific is not the largest ocean on Earth.",
    "The anonymous essay claims that Shakespeare did not write Hamlet.",
    "The misprinted textbook states that light does not travel faster than sound in air.",
    "The tourist brochure claims that Cuba is not an island nation in the Caribbean.",
    "The outdated almanac states that Tokyo is not the most populous metropolitan area.",
    "The erroneous lab sheet reports that gold is not denser than iron.",
    "The anti-vaccine flyer claims that vaccines do not prevent infectious diseases.",
    "The flat-earth video asserts that the Earth does not orbit the Sun.",
    "The mistaken nutrition post claims that oranges do not contain vitamin C.",
    "The anonymous complaint warns that the bridge cannot support heavy trucks.",
    "The contrarian column argues that global temperatures did not rise over the last decade.",
    "The negative trial reports that the drug does not reduce blood pressure.",
    "The rival auditor alleges that the company did not pay its taxes in full.",
]

TAUS = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]


def flag(t, f, tau):
    return (t + f > 1.0) and (min(t, f) >= tau)


def pearson(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    return sxy / (sxx * syy) ** 0.5 if sxx > 0 and syy > 0 else float("nan")


def main():
    print(f"Loading Model A: {MODEL_A} ...", flush=True)
    pipe_a = build_pipe(MODEL_A)
    print(f"Loading Model B: {MODEL_B} ...", flush=True)
    pipe_b = build_pipe(MODEL_B)
    rows = []

    def score(cls, i, seg_t, seg_f, claim):
        a = label_scores(pipe_a, seg_t, claim)
        b = label_scores(pipe_b, seg_f, claim)
        t, f = a.get("entailment", 0.0), b.get("contradiction", 0.0)
        rows.append({"cls": cls, "item": i, "T": round(t, 4), "F": round(f, 4),
                     "sum": round(t + f, 4), "claim": claim})
        print(f"  {cls:<9} [{i:>2}] T={t:.3f} F={f:.3f} sum={t+f:.3f}", flush=True)

    print("AGREE-SUP (both sources support)...", flush=True)
    for i, ((sup, _ref, claim), sup2) in enumerate(zip(CONFLICT_ITEMS, AGREE_SUP_2), 1):
        score("AGREE-SUP", i, sup, sup2, claim)
    print("AGREE-REF (both sources refute)...", flush=True)
    for i, ((_sup, ref, claim), ref2) in enumerate(zip(CONFLICT_ITEMS, AGREE_REF_2), 1):
        score("AGREE-REF", i, ref, ref2, claim)
    print("CROSS (support i + refute j!=i, scored on claim i)...", flush=True)
    n = len(CONFLICT_ITEMS)
    for i, (sup, _ref, claim) in enumerate(CONFLICT_ITEMS, 1):
        j = i % n  # next item's refuting segment (1-based: item i uses refute of item i+1)
        ref_j = CONFLICT_ITEMS[j][1]
        score("CROSS", i, sup, ref_j, claim)

    with open(OUT_CSV, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # ---- pull A2 decomposed conflict scores and A2 single-segment controls
    a2 = list(csv.DictReader(open(A2_CSV, encoding="utf-8")))
    conf = [(float(r["T_decomp"]), float(r["F_decomp"])) for r in a2]
    a2_ctrl = []
    for cls, prem, hyp in CONTROLS:
        a = label_scores(pipe_a, prem, hyp)
        b = label_scores(pipe_b, prem, hyp)
        a2_ctrl.append((cls, a.get("entailment", 0.0), b.get("contradiction", 0.0)))

    # ---- Pearson on NEU + PAR subset of Experiment A
    real = list(csv.DictReader(open(REAL_CSV, encoding="utf-8")))
    cols = real[0].keys()
    tcol = next((c for c in cols if c.lower() in ("t_a", "t", "t_dual", "t_real")), None)
    fcol = next((c for c in cols if c.lower() in ("f_b", "f", "f_dual", "f_real")), None)
    ccol = next((c for c in cols if c.lower() in ("cls", "class", "label")), None)
    pear_all = pear_sub = float("nan")
    if tcol and fcol and ccol:
        xs = [float(r[tcol]) for r in real]
        ys = [1 - float(r[fcol]) for r in real]
        pear_all = pearson(xs, ys)
        sub = [r for r in real if r[ccol].upper() in ("NEU", "PAR")]
        pear_sub = pearson([float(r[tcol]) for r in sub], [1 - float(r[fcol]) for r in sub])

    # ---- summary
    L = ["=" * 80, "EXPERIMENT A3 - Controls for evidence decomposition + threshold sweep",
         "=" * 80, f"Model A (T): {MODEL_A}", f"Model B (F): {MODEL_B}", ""]
    for cls in ("AGREE-SUP", "AGREE-REF", "CROSS"):
        sub = [r for r in rows if r["cls"] == cls]
        mt = sum(r["T"] for r in sub) / len(sub)
        mf = sum(r["F"] for r in sub) / len(sub)
        ms = sum(r["sum"] for r in sub) / len(sub)
        raw = sum(1 for r in sub if r["sum"] > 1)
        mar = sum(1 for r in sub if flag(r["T"], r["F"], 0.15))
        L.append(f"{cls:<9} n={len(sub)}  mean T={mt:.3f}  mean F={mf:.3f}  mean T+F={ms:.3f}  "
                 f"raw T+F>1: {raw}/{len(sub)}  margin(0.15): {mar}/{len(sub)}")
    L.append("")
    L.append("Threshold sweep (margin rule: T+F>1 and min(T,F)>=tau)")
    L.append(f"{'tau':>5} | {'recall A2-decomp (20)':>22} | {'FP ENT/CON (20)':>15} | "
             f"{'FP AGREE-SUP (20)':>17} | {'FP AGREE-REF (20)':>17} | {'FP CROSS (20)':>13}")
    all_ctrl = {
        "ENT/CON": [(t, f) for _c, t, f in a2_ctrl],
        "AGREE-SUP": [(r["T"], r["F"]) for r in rows if r["cls"] == "AGREE-SUP"],
        "AGREE-REF": [(r["T"], r["F"]) for r in rows if r["cls"] == "AGREE-REF"],
        "CROSS": [(r["T"], r["F"]) for r in rows if r["cls"] == "CROSS"],
    }
    for tau in TAUS:
        rec = sum(1 for t, f in conf if flag(t, f, tau))
        fps = {k: sum(1 for t, f in v if flag(t, f, tau)) for k, v in all_ctrl.items()}
        L.append(f"{tau:>5.2f} | {rec:>19}/20 | {fps['ENT/CON']:>12}/20 | "
                 f"{fps['AGREE-SUP']:>14}/20 | {fps['AGREE-REF']:>14}/20 | {fps['CROSS']:>10}/20")
    L.append("")
    L.append(f"Pearson(T_A, 1-F_B) all 50 pairs of Experiment A: r = {pear_all:.3f}")
    L.append(f"Pearson(T_A, 1-F_B) NEU+PAR pairs only (n=30):  r = {pear_sub:.3f}")
    txt = "\n".join(L)
    open(OUT_TXT, "w", encoding="utf-8").write(txt)
    print("\n" + txt)
    print(f"Wrote: {OUT_CSV}\nWrote: {OUT_TXT}")


if __name__ == "__main__":
    main()
