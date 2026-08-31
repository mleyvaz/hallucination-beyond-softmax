"""
Experiment A6 - Model-assignment swap (added 2026-08-31, requested by reviewers).

Counterbalances which model serves which channel. The main protocol uses
T = entailment_A(support, claim), F = contradiction_B(refute, claim). Here:
    SWAP: T = entailment_B(support, claim), F = contradiction_A(refute, claim)
on the 20 conflicting items (decomposed) and on the three A3 control families,
with the lexical gate (theta = 0.3, from experiment_a5) and the margin rule.
If the recovery of Experiment 2 depends on the specific pairing, the swap will
show it. Output: validation_results_a6.csv, validation_summary_a6.txt.
"""
from __future__ import annotations
import csv, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from experiment_a2_conflicting_evidence import MODEL_A, MODEL_B, CONFLICT_ITEMS, build_pipe, label_scores
from experiment_a3_controls import AGREE_SUP_2, AGREE_REF_2, flag
from experiment_a5_lexical_gate import jacc
OUT_CSV = os.path.join(ROOT, "validation_results_a6.csv"); OUT_TXT = os.path.join(ROOT, "validation_summary_a6.txt")

def main():
    print("Loading models...", flush=True); pa = build_pipe(MODEL_A); pb = build_pipe(MODEL_B)
    n = len(CONFLICT_ITEMS); rows = []
    def score(cls, i, seg_t, seg_f, claim):
        t = label_scores(pb, seg_t, claim).get("entailment", 0.0)      # swapped: B gives T
        f = label_scores(pa, seg_f, claim).get("contradiction", 0.0)   # swapped: A gives F
        j = jacc(seg_f, claim); f_lex = f if j >= 0.3 else 0.0
        rows.append({"cls": cls, "item": i, "T_swap": round(t, 4), "F_swap": round(f, 4),
                     "jaccard": round(j, 3), "F_swap_lex": round(f_lex, 4), "claim": claim})
        print(f"  {cls:<9} [{i:>2}] T={t:.3f} F={f:.3f}", flush=True)
    for i, (sup, ref, claim) in enumerate(CONFLICT_ITEMS, 1): score("CONFLICT", i, sup, ref, claim)
    for i, ((sup, _r, claim), sup2) in enumerate(zip(CONFLICT_ITEMS, AGREE_SUP_2), 1): score("AGREE-SUP", i, sup, sup2, claim)
    for i, ((_s, ref, claim), ref2) in enumerate(zip(CONFLICT_ITEMS, AGREE_REF_2), 1): score("AGREE-REF", i, ref, ref2, claim)
    for i, (sup, _r, claim) in enumerate(CONFLICT_ITEMS, 1): score("CROSS", i, sup, CONFLICT_ITEMS[i % n][1], claim)
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(r) for r in rows]
    L = ["=" * 80, "EXPERIMENT A6 - Model-assignment swap (T from Model B, F from Model A)", "=" * 80, ""]
    for cls in ("CONFLICT", "AGREE-SUP", "AGREE-REF", "CROSS"):
        sub = [r for r in rows if r["cls"] == cls]
        raw = sum(1 for r in sub if flag(r["T_swap"], r["F_swap"], 0.15))
        lex = sum(1 for r in sub if flag(r["T_swap"], r["F_swap_lex"], 0.15))
        L.append(f"{cls:<9} n={len(sub)} mean T={sum(r['T_swap'] for r in sub)/len(sub):.3f} "
                 f"mean F={sum(r['F_swap'] for r in sub)/len(sub):.3f} flagged ungated {raw}/{len(sub)} | lexical gate {lex}/{len(sub)}")
    txt = "\n".join(L); open(OUT_TXT, "w", encoding="utf-8").write(txt); print("\n" + txt)

if __name__ == "__main__":
    main()
