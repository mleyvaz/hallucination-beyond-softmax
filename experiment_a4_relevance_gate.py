"""
Experiment A4 - Relevance gate for the F channel (added 2026-08-31).

Experiment A3 showed that the decomposed protocol produces false positives on
CROSS controls (supporting segment of claim i + refuting segment of a DIFFERENT
claim j): Model B emits high contradiction for a negated sentence even when it
is about another topic. The F channel therefore partly measures "presence of a
negation", not conflict about the claim.

Fix tested here: a relevance gate on the refuting segment. Model A is asked
about (refuting segment, claim); its NEUTRAL probability is a topicality score
(high neutral = off-topic). F is kept only if the segment is on-topic:

    soft gate : F_gated = F_B(ref, claim) * (1 - neutral_A(ref, claim))
    hard gate : F_gated = F_B(ref, claim) if neutral_A(ref, claim) < 0.5 else 0

T is unchanged: entailment_A(supporting segment, claim). (T, F_gated) are still
computed on different inputs by different models, so Theorem 1 does not bind.

Evaluated on: A2 conflicting items (recall), A3 AGREE-SUP / AGREE-REF / CROSS,
and the ENT/CON single-segment controls, with the margin rule (T+F>1, min>=0.15)
and a tau sweep. Outputs: validation_results_a4.csv, validation_summary_a4.txt.
"""
from __future__ import annotations
import csv, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
OUT_CSV = os.path.join(ROOT, "validation_results_a4.csv")
OUT_TXT = os.path.join(ROOT, "validation_summary_a4.txt")
from experiment_a2_conflicting_evidence import (MODEL_A, MODEL_B, CONFLICT_ITEMS, CONTROLS,
                                                build_pipe, label_scores)
from experiment_a3_controls import AGREE_SUP_2, AGREE_REF_2, TAUS, flag

def main():
    print(f"Loading Model A: {MODEL_A} ...", flush=True); pa = build_pipe(MODEL_A)
    print(f"Loading Model B: {MODEL_B} ...", flush=True); pb = build_pipe(MODEL_B)
    rows = []
    def score(cls, i, seg_t, seg_f, claim):
        a_t = label_scores(pa, seg_t, claim)
        b_f = label_scores(pb, seg_f, claim)
        a_f = label_scores(pa, seg_f, claim)          # relevance probe on the refuting segment
        t = a_t.get("entailment", 0.0); f = b_f.get("contradiction", 0.0)
        neu = a_f.get("neutral", 0.0)
        f_soft = f * (1 - neu); f_hard = f if neu < 0.5 else 0.0
        rows.append({"cls": cls, "item": i, "T": round(t, 4), "F_raw": round(f, 4),
                     "neutral_A_ref": round(neu, 4), "F_soft": round(f_soft, 4),
                     "F_hard": round(f_hard, 4), "claim": claim})
        print(f"  {cls:<9} [{i:>2}] T={t:.3f} F={f:.3f} neuA={neu:.3f} Fsoft={f_soft:.3f}", flush=True)
    n = len(CONFLICT_ITEMS)
    print("CONFLICT (decomposed)...", flush=True)
    for i, (sup, ref, claim) in enumerate(CONFLICT_ITEMS, 1): score("CONFLICT", i, sup, ref, claim)
    print("AGREE-SUP...", flush=True)
    for i, ((sup, _r, claim), sup2) in enumerate(zip(CONFLICT_ITEMS, AGREE_SUP_2), 1): score("AGREE-SUP", i, sup, sup2, claim)
    print("AGREE-REF...", flush=True)
    for i, ((_s, ref, claim), ref2) in enumerate(zip(CONFLICT_ITEMS, AGREE_REF_2), 1): score("AGREE-REF", i, ref, ref2, claim)
    print("CROSS...", flush=True)
    for i, (sup, _r, claim) in enumerate(CONFLICT_ITEMS, 1):
        ref_j = CONFLICT_ITEMS[i % n][1]; score("CROSS", i, sup, ref_j, claim)
    print("ENT/CON single-segment controls...", flush=True)
    for i, (cls, prem, hyp) in enumerate(CONTROLS, 1): score(cls, i, prem, prem, hyp)

    with open(OUT_CSV, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows: w.writerow(r)

    L = ["=" * 80, "EXPERIMENT A4 - Relevance gate on the F channel", "=" * 80,
         f"Model A (T, gate): {MODEL_A}", f"Model B (F): {MODEL_B}", ""]
    fams = ["CONFLICT", "AGREE-SUP", "AGREE-REF", "CROSS", "ENT", "CON"]
    for variant in ("F_raw", "F_soft", "F_hard"):
        L.append(f"--- variant {variant} (margin rule tau=0.15)")
        for cls in fams:
            sub = [r for r in rows if r["cls"] == cls]
            k = sum(1 for r in sub if flag(r["T"], r[variant], 0.15))
            mf = sum(r[variant] for r in sub) / len(sub); mt = sum(r["T"] for r in sub) / len(sub)
            L.append(f"  {cls:<9} n={len(sub):<2} mean T={mt:.3f} mean F={mf:.3f} flagged {k}/{len(sub)}")
        L.append("")
    L.append("Mean neutral_A on the refuting segment (topicality probe):")
    for cls in fams:
        sub = [r for r in rows if r["cls"] == cls]
        L.append(f"  {cls:<9} mean neutral_A(ref, claim) = {sum(r['neutral_A_ref'] for r in sub)/len(sub):.3f}")
    L.append("")
    L.append("Threshold sweep with F_soft (recall CONFLICT | FP ENT/CON | FP AGREE-SUP | FP AGREE-REF | FP CROSS)")
    for tau in TAUS:
        def cnt(cls_list):
            sub = [r for r in rows if r["cls"] in cls_list]
            return sum(1 for r in sub if flag(r["T"], r["F_soft"], tau)), len(sub)
        rc, rn = cnt(["CONFLICT"]); e, en = cnt(["ENT", "CON"]); s, sn = cnt(["AGREE-SUP"]); g, gn = cnt(["AGREE-REF"]); c, cn = cnt(["CROSS"])
        L.append(f" {tau:>5.2f} | {rc:>2}/{rn} | {e:>2}/{en} | {s:>2}/{sn} | {g:>2}/{gn} | {c:>2}/{cn}")
    txt = "\n".join(L); open(OUT_TXT, "w", encoding="utf-8").write(txt); print("\n" + txt)

if __name__ == "__main__":
    main()
