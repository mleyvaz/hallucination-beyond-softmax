"""
Experiment A11 - Intervention on the released stimuli: neutralise the source description
(added 2026-09-10, third adversarial review, blocker B1). Pre-registered in
PREREGISTRO_A10_A11_2026-09-10.md before the first inference.

Section 5.4 of v19 claimed that the credibility marker "drove F below the margin" on the
five items the decomposed protocol misses, and that the holistic failure had "two
compounding reasons". Neither claim was tested on the original texts: A9 used synthetic
frames. Here the original spans are intervened directly. Only the SOURCE NOUN PHRASE is
replaced, by "The document"; the reporting verb and the propositional content are kept
verbatim (single agreement fix: item 1, "records state" -> "document states").

Arms:
  original      supporting span verbatim        refuting span verbatim
  ref_neutral   supporting span verbatim        refuting source -> "The document"
  sup_neutral   supporting source -> "The document"   refuting span verbatim
  both_neutral  both sources -> "The document"

Measures per arm, with the published protocol:
  T = entailment_A(supporting span -> claim)
  F = contradiction_B(refuting span -> claim), plus the released lexical gate (Jaccard)
  margin rule: T + F > 1 and min(T, F) >= 0.15
  holistic: both heads on "sup However, ref" (the concatenation of experiment_a2)

Registered predictions: P1 (F rises above margin on >= 2 of items 5, 11, 16 under
ref_neutral), P2 (margin rule >= 18/20 under ref_neutral or both_neutral), P3 (items 1
and 8 recovered by sup_neutral, not by ref_neutral), P4 (holistic > 4/20 when
neutralised). The substitution table is written to the CSV so that it can be audited.

Output: validation_results_a11.csv, validation_summary_a11.txt
"""
from __future__ import annotations
import csv, os, sys

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = HERE
sys.path.insert(0, HERE)
from experiment_a2_conflicting_evidence import MODEL_A, MODEL_B, CONFLICT_ITEMS
from dual_nli import jaccard  # released gate, model-free
from dual_nli import build_pipe, nli_scores

OUT_CSV = os.path.join(ROOT, "validation_results_a11.csv")
OUT_TXT = os.path.join(ROOT, "validation_summary_a11.txt")
TAU, THETA = 0.15, 0.30  # margin and gate as published

# Source noun phrases, transcribed from the released CONFLICT_ITEMS (1-indexed).
SUP_NP = [
    "The city's official records", "The physics textbook", "The atlas", "The biography",
    "The biology curriculum", "The anatomy manual", "The geographic survey",
    "The oceanographic institute", "The literary archive", "The acoustics handbook",
    "The encyclopedia entry", "The census bureau", "The chemistry reference",
    "The medical guideline", "The astronomy course", "The nutrition study",
    "The engineering report", "The climate agency", "The clinical trial", "The audit",
]
REF_NP = [
    "A disputed 2020 pamphlet", "The lab report", "The blog post", "The revisionist essay",
    "The pseudoscientific site", "The outdated treatise", "The rival almanac",
    "The amateur newsletter", "The fringe theory", "The confused forum post",
    "The erroneous flyer", "The competing report", "The flawed worksheet",
    "The conspiracy leaflet", "The geocentric pamphlet", "The retracted article",
    "The whistleblower memo", "The denialist op-ed", "The follow-up study", "The leaked ledger",
]
NEUTRAL = "The document"


def neutralise(span: str, np: str, fix_agreement: bool = False) -> str:
    assert span.startswith(np), f"source NP not at the start of: {span}"
    out = NEUTRAL + span[len(np):]
    if fix_agreement:  # item 1: plural "records state" -> singular "document states"
        out = out.replace("The document state that", "The document states that", 1)
    return out


def margin(t: float, f: float) -> bool:
    return (t + f > 1.0) and (min(t, f) >= TAU)


def main() -> None:
    print("Loading models...", flush=True)
    pa, pb = build_pipe(MODEL_A), build_pipe(MODEL_B)
    sup_var, ref_var, rows = {}, {}, []
    for i, (sup, ref, claim) in enumerate(CONFLICT_ITEMS, 1):
        sup_var[i] = {"orig": sup, "neutral": neutralise(sup, SUP_NP[i - 1], fix_agreement=(i == 1))}
        ref_var[i] = {"orig": ref, "neutral": neutralise(ref, REF_NP[i - 1])}

    # T only depends on the supporting variant; F only on the refuting variant. Score each once.
    T = {(i, v): nli_scores(pa, sup_var[i][v], CONFLICT_ITEMS[i - 1][2])["entailment"]
         for i in range(1, 21) for v in ("orig", "neutral")}
    F = {(i, v): nli_scores(pb, ref_var[i][v], CONFLICT_ITEMS[i - 1][2])["contradiction"]
         for i in range(1, 21) for v in ("orig", "neutral")}
    print("  spans scored", flush=True)

    ARMS = {"original": ("orig", "orig"), "ref_neutral": ("orig", "neutral"),
            "sup_neutral": ("neutral", "orig"), "both_neutral": ("neutral", "neutral")}
    for i in range(1, 21):
        sup0, ref0, claim = CONFLICT_ITEMS[i - 1]
        for arm, (sv, rv) in ARMS.items():
            s, r = sup_var[i][sv], ref_var[i][rv]
            full = s + " However, " + r
            ha = nli_scores(pa, full, claim); hb = nli_scores(pb, full, claim)
            t, f_raw = T[(i, sv)], F[(i, rv)]
            j = jaccard(r, claim)
            f_gated = f_raw if j >= THETA else 0.0
            rows.append({
                "item": i, "arm": arm, "support_span": s, "refute_span": r, "claim": claim,
                "T": round(t, 4), "F_raw": round(f_raw, 4), "jaccard": round(j, 3),
                "F_gated": round(f_gated, 4), "margin": int(margin(t, f_gated)),
                "T_holistic": round(ha["entailment"], 4), "F_holistic": round(hb["contradiction"], 4),
                "margin_holistic": int(margin(ha["entailment"], hb["contradiction"])),
            })
        print(f"  item {i}/20 done", flush=True)

    with open(OUT_CSV, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows: w.writerow(r)

    idx = {(r["item"], r["arm"]): r for r in rows}
    L = ["Experiment A11 - source-description intervention on the released stimuli (pre-registered 2026-09-10)", ""]
    L.append(f"{'arm':<13} {'meanT':>7} {'meanF':>7} {'margin':>8} {'holistic':>9}")
    for arm in ARMS:
        cell = [idx[(i, arm)] for i in range(1, 21)]
        L.append(f"{arm:<13} {sum(r['T'] for r in cell)/20:7.3f} {sum(r['F_gated'] for r in cell)/20:7.3f} "
                 f"{sum(r['margin'] for r in cell):>5}/20 {sum(r['margin_holistic'] for r in cell):>6}/20")
    L.append("")
    L.append("P1  F above the margin under ref_neutral on items 5, 11, 16 (the three the paper attributes to the marker):")
    p1 = 0
    for i in (5, 11, 16):
        o, n = idx[(i, "original")], idx[(i, "ref_neutral")]
        ok = n["F_gated"] >= TAU; p1 += ok
        L.append(f"   item {i:>2}: F {o['F_gated']:.4f} -> {n['F_gated']:.4f}  (gate J {o['jaccard']:.2f} -> {n['jaccard']:.2f})  {'above' if ok else 'still below'} tau")
    L.append(f"   -> P1 {'SUPPORTED' if p1 >= 2 else 'NOT SUPPORTED'} ({p1}/3)")
    m_ref = sum(idx[(i, 'ref_neutral')]['margin'] for i in range(1, 21))
    m_both = sum(idx[(i, 'both_neutral')]['margin'] for i in range(1, 21))
    m_orig = sum(idx[(i, 'original')]['margin'] for i in range(1, 21))
    L.append(f"P2  margin rule: original {m_orig}/20 -> ref_neutral {m_ref}/20, both_neutral {m_both}/20"
             f"  -> P2 {'SUPPORTED' if max(m_ref, m_both) >= 18 else 'NOT SUPPORTED'}")
    L.append("P3  items 1 and 8 (attributed to a weak supporting span):")
    p3 = True
    for i in (1, 8):
        r_, s_, o_ = idx[(i, "ref_neutral")], idx[(i, "sup_neutral")], idx[(i, "original")]
        L.append(f"   item {i}: T {o_['T']:.4f} -> {s_['T']:.4f} (sup_neutral) | margin orig {o_['margin']} ref_neutral {r_['margin']} sup_neutral {s_['margin']} both {idx[(i,'both_neutral')]['margin']}")
        if r_["margin"] or not s_["margin"]: p3 = False
    L.append(f"   -> P3 {'SUPPORTED' if p3 else 'NOT SUPPORTED'}")
    h_orig = sum(idx[(i, 'original')]['margin_holistic'] for i in range(1, 21))
    h_both = sum(idx[(i, 'both_neutral')]['margin_holistic'] for i in range(1, 21))
    h_ref = sum(idx[(i, 'ref_neutral')]['margin_holistic'] for i in range(1, 21))
    L.append(f"P4  holistic: original {h_orig}/20 -> ref_neutral {h_ref}/20, both_neutral {h_both}/20"
             f"  -> P4 {'SUPPORTED' if max(h_ref, h_both) > 4 else 'NOT SUPPORTED'}")
    L.append("")
    L.append("Per-item F under each refuting variant (gated):")
    L.append(f"{'item':>4} {'F orig':>8} {'F neutral':>10} {'delta':>8}")
    for i in range(1, 21):
        o, n = idx[(i, "original")], idx[(i, "ref_neutral")]
        L.append(f"{i:>4} {o['F_gated']:8.4f} {n['F_gated']:10.4f} {n['F_gated']-o['F_gated']:+8.4f}")
    txt = "\n".join(L)
    open(OUT_TXT, "w", encoding="utf-8").write(txt)
    print(txt)
    print("Wrote", OUT_CSV, OUT_TXT)


if __name__ == "__main__":
    main()
