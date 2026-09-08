"""
Experiment A7 - Single-head decomposed baselines (added 2026-08-31, reviewer request).

Theorem 1 constrains ONE softmax call on ONE (premise, claim) pair. Once the
premise is decomposed into two spans, two calls of the SAME head can already
produce T + F > 1:
    A-only : T = entailment_A(support, claim), F = contradiction_A(refute, claim)
    B-only : T = entailment_B(support, claim), F = contradiction_B(refute, claim)
This experiment reports both single-head decomposed baselines on the conflicting
items and on the three decomposition control families, with and without the
lexical gate, under the margin rule. It also reports the single-channel baseline
"F >= 0.15 on the refuting span" for every configuration, since with oracle
spans T is nearly constant and the two-dimensional rule may reduce to it.
Output: validation_results_a7.csv, validation_summary_a7.txt.
"""
from __future__ import annotations
import csv, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = HERE  # scripts and data live flat in this repo; do not assume a parent directory
sys.path.insert(0, HERE)
from experiment_a2_conflicting_evidence import MODEL_A, MODEL_B, CONFLICT_ITEMS, build_pipe, label_scores
from experiment_a3_controls import AGREE_SUP_2, AGREE_REF_2, flag
from experiment_a5_lexical_gate import jacc
OUT_CSV = os.path.join(ROOT, "validation_results_a7.csv"); OUT_TXT = os.path.join(ROOT, "validation_summary_a7.txt")

def main():
    print("Loading models...", flush=True); pa = build_pipe(MODEL_A); pb = build_pipe(MODEL_B)
    n = len(CONFLICT_ITEMS); rows = []
    def score(cls, i, sup, ref, claim):
        a_s = label_scores(pa, sup, claim); a_r = label_scores(pa, ref, claim)
        b_s = label_scores(pb, sup, claim); b_r = label_scores(pb, ref, claim)
        rows.append({"cls": cls, "item": i, "jaccard": round(jacc(ref, claim), 3),
                     "T_A": round(a_s.get("entailment", 0.0), 4), "F_A": round(a_r.get("contradiction", 0.0), 4),
                     "T_B": round(b_s.get("entailment", 0.0), 4), "F_B": round(b_r.get("contradiction", 0.0), 4),
                     "claim": claim})
        print(f"  {cls:<9} [{i:>2}] TA={rows[-1]['T_A']:.3f} FA={rows[-1]['F_A']:.3f} TB={rows[-1]['T_B']:.3f} FB={rows[-1]['F_B']:.3f}", flush=True)
    for i, (sup, ref, claim) in enumerate(CONFLICT_ITEMS, 1): score("CONFLICT", i, sup, ref, claim)
    for i, ((sup, _r, claim), sup2) in enumerate(zip(CONFLICT_ITEMS, AGREE_SUP_2), 1): score("AGREE-SUP", i, sup, sup2, claim)
    for i, ((_s, ref, claim), ref2) in enumerate(zip(CONFLICT_ITEMS, AGREE_REF_2), 1): score("AGREE-REF", i, ref, ref2, claim)
    for i, (sup, _r, claim) in enumerate(CONFLICT_ITEMS, 1): score("CROSS", i, sup, CONFLICT_ITEMS[i % n][1], claim)
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(r) for r in rows]
    configs = {"A-only (T_A,F_A)": ("T_A", "F_A"), "B-only (T_B,F_B)": ("T_B", "F_B"),
               "dual A->T, B->F": ("T_A", "F_B"), "dual B->T, A->F": ("T_B", "F_A")}
    L = ["=" * 80, "EXPERIMENT A7 - Single-head decomposed baselines and F-only rule", "=" * 80, "",
         "Counts flagged under margin rule (T+F>1 and min(T,F)>=0.15) | with lexical gate (J>=0.3) | F-only rule (F>=0.15, gated)"]
    for name, (tk, fk) in configs.items():
        L.append(f"--- {name}")
        for cls in ("CONFLICT", "AGREE-SUP", "AGREE-REF", "CROSS"):
            sub = [r for r in rows if r["cls"] == cls]
            raw = sum(1 for r in sub if flag(r[tk], r[fk], 0.15))
            lex = sum(1 for r in sub if flag(r[tk], r[fk] if r["jaccard"] >= 0.3 else 0.0, 0.15))
            fonly = sum(1 for r in sub if (r[fk] if r["jaccard"] >= 0.3 else 0.0) >= 0.15)
            mt = sum(r[tk] for r in sub) / len(sub); mf = sum(r[fk] for r in sub) / len(sub)
            L.append(f"  {cls:<9} meanT={mt:.3f} meanF={mf:.3f} | margin {raw:>2}/20 | gated {lex:>2}/20 | F-only {fonly:>2}/20")
    txt = "\n".join(L); open(OUT_TXT, "w", encoding="utf-8").write(txt); print("\n" + txt)

if __name__ == "__main__":
    main()
