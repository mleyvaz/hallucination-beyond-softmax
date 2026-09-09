"""
Experiment A5 - Lexical topicality gate for the F channel (added 2026-08-31).

Experiment A4 showed that gating F with Model A's NEUTRAL probability removes the
CROSS false positives but collapses recall (15/20 -> 5/20): Model A reads a
refutation attributed to a low-credibility source as NEUTRAL, so the gate
re-imports exactly the source-credibility adjudication the protocol tries to
avoid. Here the gate uses no NLI model at all: content-word overlap between the
refuting segment and the claim (Jaccard over lemmatized-by-stem content words,
stopwords and negation markers removed).

    F_lex = F_B(ref, claim) if J(ref, claim) >= theta else 0

Recomputed from validation_results_a4.csv (T, F_raw) and the stimuli in the
experiment scripts. Reports recall / false positives for several theta with the
margin rule (T+F>1, min>=0.15). Output: validation_summary_a5.txt,
validation_results_a5.csv.
"""
from __future__ import annotations
import csv, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = HERE  # scripts and data live flat in this repo; do not assume a parent directory
sys.path.insert(0, HERE)
from experiment_a2_conflicting_evidence import CONFLICT_ITEMS, CONTROLS
from experiment_a3_controls import AGREE_SUP_2, AGREE_REF_2, flag

A4 = os.path.join(ROOT, "validation_results_a4.csv")
OUT_TXT = os.path.join(ROOT, "validation_summary_a5.txt")
OUT_CSV = os.path.join(ROOT, "validation_results_a5.csv")

from dual_nli import STOP, content, jaccard as jacc  # single code path with the reference implementation


def main():
    # rebuild (cls, item) -> refuting segment, claim
    n = len(CONFLICT_ITEMS)
    seg = {}
    for i, (sup, ref, claim) in enumerate(CONFLICT_ITEMS, 1):
        seg[("CONFLICT", i)] = (ref, claim)
        seg[("AGREE-SUP", i)] = (AGREE_SUP_2[i - 1], claim)
        seg[("AGREE-REF", i)] = (AGREE_REF_2[i - 1], claim)
        seg[("CROSS", i)] = (CONFLICT_ITEMS[i % n][1], claim)
    for i, (cls, prem, hyp) in enumerate(CONTROLS, 1):
        seg[(cls, i)] = (prem, hyp)

    rows = list(csv.DictReader(open(A4, encoding="utf-8")))
    for r in rows:
        ref, claim = seg[(r["cls"], int(r["item"]))]
        r["jaccard"] = round(jacc(ref, claim), 3)
        r["T"] = float(r["T"]); r["F_raw"] = float(r["F_raw"])

    THETAS = [0.0, 0.2, 0.3, 0.4, 0.5]
    fams = ["CONFLICT", "AGREE-SUP", "AGREE-REF", "CROSS", "ENT", "CON"]
    L = ["=" * 80, "EXPERIMENT A5 - Lexical topicality gate on the F channel", "=" * 80, ""]
    L.append("Mean Jaccard(content words of refuting segment, claim) per family:")
    for cls in fams:
        sub = [r for r in rows if r["cls"] == cls]
        L.append(f"  {cls:<9} mean J = {sum(r['jaccard'] for r in sub)/len(sub):.3f}  "
                 f"min J = {min(r['jaccard'] for r in sub):.3f}  max J = {max(r['jaccard'] for r in sub):.3f}")
    L.append("")
    L.append("Margin rule tau=0.15 with F_lex = F_raw if J >= theta else 0")
    L.append(f"{'theta':>6} | {'recall CONFLICT':>15} | {'FP ENT/CON':>10} | {'FP AGREE-SUP':>12} | {'FP AGREE-REF':>12} | {'FP CROSS':>8}")
    best = None
    for th in THETAS:
        def cnt(cl):
            sub = [r for r in rows if r["cls"] in cl]
            return sum(1 for r in sub if flag(r["T"], r["F_raw"] if r["jaccard"] >= th else 0.0, 0.15)), len(sub)
        rc, rn = cnt(["CONFLICT"]); e, en = cnt(["ENT", "CON"]); s, sn = cnt(["AGREE-SUP"]); g, gn = cnt(["AGREE-REF"]); c, cn = cnt(["CROSS"])
        L.append(f"{th:>6.2f} | {rc:>12}/{rn} | {e:>7}/{en} | {s:>9}/{sn} | {g:>9}/{gn} | {c:>5}/{cn}")
        if th == 0.3: best = (rc, e, s, g, c)
    L.append("")
    L.append("Note: the gate is deliberately model-free. It separates 'refutation of another claim' from")
    L.append("'refutation of this claim' by topical overlap alone; it does not and cannot judge credibility.")
    txt = "\n".join(L); open(OUT_TXT, "w", encoding="utf-8").write(txt); print(txt)
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows: w.writerow(r)
    print("Wrote", OUT_TXT, OUT_CSV)


if __name__ == "__main__":
    main()
