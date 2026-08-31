"""
Experiment A2 — Dual-NLI over CONFLICTING-EVIDENCE stimuli (corrected design).

Diagnosis from Experiment A (real models): the original PAR pairs placed the
contradiction INSIDE the hypothesis ("X, but also not X"). Real NLI models
correctly read such hypotheses as neither entailed nor contradicted by a
simple premise -> neutral collapse (0/20 recovery). That is a stimulus-design
artifact, not a test of the dual-NLI protocol.

The protocol's intended regime is CONFLICTING EVIDENCE about an atomic claim:
  premise    = "Source A reports X. However, Source B reports not-X."
  hypothesis = "X"

Two scoring modes are compared on the same 20 corrected items:

  HOLISTIC   : T = entail(A: full premise -> claim)
               F = contradict(B: full premise -> claim)
  DECOMPOSED : T = entail(A: supporting segment -> claim)
               F = contradict(B: refuting segment -> claim)

Controls: the 10 ENT and 10 CON pairs from Experiment A, scored holistically.

Output: validation_results_a2.csv / validation_summary_a2.txt
"""
from __future__ import annotations

import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_CSV = os.path.join(ROOT, "validation_results_a2.csv")
OUT_TXT = os.path.join(ROOT, "validation_summary_a2.txt")

MODEL_A = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
MODEL_B = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"

# 20 conflicting-evidence items: (support_segment, refute_segment, claim)
CONFLICT_ITEMS = [
    ("The city's official records state that Paris is the capital of France.",
     "A disputed 2020 pamphlet claims that Paris is not the capital of France.",
     "Paris is the capital of France."),
    ("The physics textbook states that water boils at 100 degrees Celsius at sea level.",
     "The lab report concludes that water does not boil at 100 degrees Celsius at sea level.",
     "Water boils at 100 degrees Celsius at sea level."),
    ("The atlas indicates that the Amazon rainforest is located in South America.",
     "The blog post asserts that the Amazon rainforest is not located in South America.",
     "The Amazon rainforest is located in South America."),
    ("The biography documents that Einstein developed the theory of general relativity.",
     "The revisionist essay argues that Einstein did not develop the theory of general relativity.",
     "Einstein developed the theory of general relativity."),
    ("The biology curriculum teaches that photosynthesis converts sunlight into chemical energy.",
     "The pseudoscientific site denies that photosynthesis converts sunlight into chemical energy.",
     "Photosynthesis converts sunlight into chemical energy."),
    ("The anatomy manual confirms that the human heart has four chambers.",
     "The outdated treatise maintains that the human heart does not have four chambers.",
     "The human heart has four chambers."),
    ("The geographic survey certifies that Mount Everest is the tallest mountain on Earth.",
     "The rival almanac insists that Mount Everest is not the tallest mountain on Earth.",
     "Mount Everest is the tallest mountain on Earth."),
    ("The oceanographic institute reports that the Pacific is the largest ocean on Earth.",
     "The amateur newsletter contends that the Pacific is not the largest ocean on Earth.",
     "The Pacific is the largest ocean on Earth."),
    ("The literary archive attributes the play Hamlet to Shakespeare.",
     "The fringe theory holds that Shakespeare did not write Hamlet.",
     "Shakespeare wrote Hamlet."),
    ("The acoustics handbook establishes that light travels faster than sound in air.",
     "The confused forum post declares that light does not travel faster than sound in air.",
     "Light travels faster than sound in air."),
    ("The encyclopedia entry describes Cuba as an island nation in the Caribbean.",
     "The erroneous flyer states that Cuba is not an island nation in the Caribbean.",
     "Cuba is an island nation in the Caribbean."),
    ("The census bureau reports that Tokyo is the most populous metropolitan area.",
     "The competing report concludes that Tokyo is not the most populous metropolitan area.",
     "Tokyo is the most populous metropolitan area."),
    ("The chemistry reference lists gold as denser than iron.",
     "The flawed worksheet answers that gold is not denser than iron.",
     "Gold is denser than iron."),
    ("The medical guideline affirms that vaccines prevent infectious diseases.",
     "The conspiracy leaflet claims that vaccines do not prevent infectious diseases.",
     "Vaccines prevent infectious diseases."),
    ("The astronomy course explains that the Earth orbits the Sun.",
     "The geocentric pamphlet asserts that the Earth does not orbit the Sun.",
     "The Earth orbits the Sun."),
    ("The nutrition study finds that oranges contain vitamin C.",
     "The retracted article claims that oranges do not contain vitamin C.",
     "Oranges contain vitamin C."),
    ("The engineering report verifies that the bridge can support heavy trucks.",
     "The whistleblower memo warns that the bridge cannot support heavy trucks.",
     "The bridge can support heavy trucks."),
    ("The climate agency documents that global temperatures rose over the last decade.",
     "The denialist op-ed argues that global temperatures did not rise over the last decade.",
     "Global temperatures rose over the last decade."),
    ("The clinical trial shows that the drug reduces blood pressure.",
     "The follow-up study reports that the drug does not reduce blood pressure.",
     "The drug reduces blood pressure."),
    ("The audit certifies that the company paid its taxes in full.",
     "The leaked ledger suggests that the company did not pay its taxes in full.",
     "The company paid its taxes in full."),
]

from synthetic_validation import PAIRS
CONTROLS = [(c, p, h) for (c, p, h) in PAIRS if c in ("ENT", "CON")]


def build_pipe(model_name):
    from transformers import pipeline
    return pipeline("text-classification", model=model_name, top_k=None)


def label_scores(pipe, premise, hypothesis):
    out = pipe({"text": premise, "text_pair": hypothesis})
    if out and isinstance(out[0], list):
        out = out[0]
    return {d["label"].lower(): float(d["score"]) for d in out}


def main():
    print(f"Loading Model A: {MODEL_A} ...", flush=True)
    pipe_a = build_pipe(MODEL_A)
    print(f"Loading Model B: {MODEL_B} ...", flush=True)
    pipe_b = build_pipe(MODEL_B)
    print("Scoring...", flush=True)

    rows = []

    # --- conflicting-evidence items, holistic + decomposed
    for i, (sup, ref, claim) in enumerate(CONFLICT_ITEMS, 1):
        full = sup + " However, " + ref
        a_h = label_scores(pipe_a, full, claim)
        b_h = label_scores(pipe_b, full, claim)
        t_h, f_h = a_h.get("entailment", 0.0), b_h.get("contradiction", 0.0)
        a_d = label_scores(pipe_a, sup, claim)
        b_d = label_scores(pipe_b, ref, claim)
        t_d, f_d = a_d.get("entailment", 0.0), b_d.get("contradiction", 0.0)
        rows.append({
            "item": i, "cls": "PAR-EVID",
            "T_holistic": round(t_h, 4), "F_holistic": round(f_h, 4),
            "sum_holistic": round(t_h + f_h, 4), "para_holistic": t_h + f_h > 1,
            "T_decomp": round(t_d, 4), "F_decomp": round(f_d, 4),
            "sum_decomp": round(t_d + f_d, 4), "para_decomp": t_d + f_d > 1,
            "claim": claim,
        })
        print(f"  [{i:>2}/20] holistic T={t_h:.3f} F={f_h:.3f} sum={t_h+f_h:.3f} | "
              f"decomp T={t_d:.3f} F={f_d:.3f} sum={t_d+f_d:.3f}", flush=True)

    # --- controls (ENT/CON), holistic only
    ctrl = []
    for i, (cls, prem, hyp) in enumerate(CONTROLS, 1):
        a = label_scores(pipe_a, prem, hyp)
        b = label_scores(pipe_b, prem, hyp)
        t, f = a.get("entailment", 0.0), b.get("contradiction", 0.0)
        ctrl.append({"cls": cls, "T": t, "F": f, "para": t + f > 1})

    with open(OUT_CSV, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # --- summary
    L = []
    L.append("=" * 80)
    L.append("EXPERIMENT A2 — Conflicting-evidence stimuli (corrected design)")
    L.append("=" * 80)
    L.append(f"Model A (T): {MODEL_A}")
    L.append(f"Model B (F): {MODEL_B}")
    L.append("")
    n = len(rows)
    for mode in ("holistic", "decomp"):
        para = sum(1 for r in rows if r[f"para_{mode}"])
        mt = sum(r[f"T_{mode}"] for r in rows) / n
        mf = sum(r[f"F_{mode}"] for r in rows) / n
        ms = sum(r[f"sum_{mode}"] for r in rows) / n
        L.append(f"{mode.upper():<10} mean T={mt:.3f}  mean F={mf:.3f}  "
                 f"mean T+F={ms:.3f}  recovered T+F>1: {para}/{n} ({100*para/n:.0f}%)")
    L.append("")
    for cls in ("ENT", "CON"):
        sub = [c for c in ctrl if c["cls"] == cls]
        fp = sum(1 for c in sub if c["para"])
        L.append(f"Control {cls}: false-paraconsistent {fp}/{len(sub)} "
                 f"(mean T={sum(c['T'] for c in sub)/len(sub):.3f}, "
                 f"mean F={sum(c['F'] for c in sub)/len(sub):.3f})")
    L.append("")
    # margin rule evaluated PER ITEM (2026-08-31: an earlier summary evaluated it on class means)
    def margin(t, f):
        return t + f > 1 and min(t, f) >= 0.15
    L.append("Margin decision rule (paraconsistent iff T+F>1 AND min(T,F)>=0.15), evaluated PER ITEM:")
    L.append(f"  PAR-EVID decomposed : {sum(1 for r in rows if margin(r['T_decomp'], r['F_decomp']))}/{n} recovered")
    L.append(f"  PAR-EVID holistic   : {sum(1 for r in rows if margin(r['T_holistic'], r['F_holistic']))}/{n} recovered")
    L.append(f"  Controls ENT/CON    : {sum(1 for c in ctrl if margin(c['T'], c['F']))}/{len(ctrl)} false positives")
    L.append("")
    txt = "\n".join(L)
    with open(OUT_TXT, "w", encoding="utf-8") as fh:
        fh.write(txt)
    print("\n" + txt)
    print(f"Wrote: {OUT_CSV}\nWrote: {OUT_TXT}")


if __name__ == "__main__":
    main()
