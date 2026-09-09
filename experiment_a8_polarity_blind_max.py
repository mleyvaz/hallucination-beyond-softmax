"""
Experiment A8 - Polarity-blind span selection and collapsed comparators
(added 2026-09-09 in response to an adversarial review for Computacion y Sistemas).

Two comparators that the earlier experiments did not run:

1. Polarity-blind max-max selection. Every segment of a two-source premise is
   scored by both heads; the support coordinate is the MAXIMUM entailment over
   segments and the contradiction coordinate is the MAXIMUM contradiction over
   segments. No oracle assignment of "supporting" vs "refuting" span is used.
   This is the selection step HallDetect (Oukelmoun et al. 2026, Sec. 4.4) performs
   with a single DeBERTa head; we run it (a) with Model A for both maxima
   (HallDetect-style, one head) and (b) with Model A for entailment and Model B
   for contradiction (our dual assignment, but without oracle polarity).
   The lexical gate, when applied, restricts the contradiction maximum to
   segments with Jaccard(segment, claim) >= 0.3.

2. Collapsed difference score D = E_max - C_max (one scalar), which is how a
   HallDetect-style pipeline ends. We report its distribution per family to show
   which families the scalar keeps apart and which it merges: (high, high) and
   (low, low) both map to D ~ 0.

Families: CONFLICT, AGREE-SUP, AGREE-REF, CROSS (two segments each), plus the
single-segment ENT, CON and NEU pairs of Experiment 1 (one segment: the
premise itself, so max = the single score).

Output: validation_results_a8.csv, validation_summary_a8.txt.
"""
from __future__ import annotations
import csv, os, sys, statistics
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = HERE
sys.path.insert(0, HERE)
from experiment_a2_conflicting_evidence import MODEL_A, MODEL_B, CONFLICT_ITEMS, PAIRS, build_pipe, label_scores
from experiment_a3_controls import AGREE_SUP_2, AGREE_REF_2, flag
from experiment_a5_lexical_gate import jacc
OUT_CSV = os.path.join(ROOT, "validation_results_a8.csv"); OUT_TXT = os.path.join(ROOT, "validation_summary_a8.txt")
TAU, THETA = 0.15, 0.3


def main():
    print("Loading models...", flush=True); pa = build_pipe(MODEL_A); pb = build_pipe(MODEL_B)
    n = len(CONFLICT_ITEMS); rows = []

    def score(cls, i, segs, claim):
        rec = {"cls": cls, "item": i, "claim": claim, "n_segments": len(segs)}
        for k, s in enumerate(segs, 1):
            a = label_scores(pa, s, claim); b = label_scores(pb, s, claim)
            rec[f"seg{k}_J"] = round(jacc(s, claim), 3)
            rec[f"seg{k}_entA"] = round(a.get("entailment", 0.0), 4); rec[f"seg{k}_conA"] = round(a.get("contradiction", 0.0), 4)
            rec[f"seg{k}_entB"] = round(b.get("entailment", 0.0), 4); rec[f"seg{k}_conB"] = round(b.get("contradiction", 0.0), 4)
        if len(segs) == 1:
            for key in ("J", "entA", "conA", "entB", "conB"):
                rec[f"seg2_{key}"] = ""
        # polarity-blind maxima
        ks = range(1, len(segs) + 1)
        rec["Emax_A"] = max(rec[f"seg{k}_entA"] for k in ks)
        rec["Cmax_A"] = max(rec[f"seg{k}_conA"] for k in ks)
        rec["Cmax_B"] = max(rec[f"seg{k}_conB"] for k in ks)
        gated = [rec[f"seg{k}_conB"] for k in ks if rec[f"seg{k}_J"] >= THETA]
        rec["Cmax_B_gated"] = max(gated) if gated else 0.0
        gatedA = [rec[f"seg{k}_conA"] for k in ks if rec[f"seg{k}_J"] >= THETA]
        rec["Cmax_A_gated"] = max(gatedA) if gatedA else 0.0
        rec["D_A"] = round(rec["Emax_A"] - rec["Cmax_A"], 4)          # HallDetect-style collapsed scalar, one head
        rec["D_AB"] = round(rec["Emax_A"] - rec["Cmax_B"], 4)         # collapsed scalar, dual heads
        rows.append(rec)
        print(f"  {cls:<9} [{i:>2}] EmaxA={rec['Emax_A']:.3f} CmaxA={rec['Cmax_A']:.3f} CmaxB={rec['Cmax_B']:.3f} D_A={rec['D_A']:+.3f}", flush=True)

    for i, (sup, ref, claim) in enumerate(CONFLICT_ITEMS, 1): score("CONFLICT", i, [sup, ref], claim)
    for i, ((sup, _r, claim), sup2) in enumerate(zip(CONFLICT_ITEMS, AGREE_SUP_2), 1): score("AGREE-SUP", i, [sup, sup2], claim)
    for i, ((_s, ref, claim), ref2) in enumerate(zip(CONFLICT_ITEMS, AGREE_REF_2), 1): score("AGREE-REF", i, [ref, ref2], claim)
    for i, (sup, _r, claim) in enumerate(CONFLICT_ITEMS, 1): score("CROSS", i, [sup, CONFLICT_ITEMS[i % n][1]], claim)
    for cls in ("ENT", "CON", "NEU"):
        for i, (_c, prem, hyp) in enumerate([x for x in PAIRS if x[0] == cls], 1): score(cls, i, [prem], hyp)

    fields = list(rows[0].keys())
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); [w.writerow(r) for r in rows]

    fams = ("CONFLICT", "AGREE-SUP", "AGREE-REF", "CROSS", "ENT", "CON", "NEU")
    L = ["=" * 88, "EXPERIMENT A8 - Polarity-blind max-max selection and collapsed difference score", "=" * 88, "",
         f"Rules: diag = T+F>1 and min(T,F)>={TAU}; bil = min(T,F)>={TAU}. Gate: contradiction max restricted to segments with J>={THETA}.", ""]
    configs = {
        "A-only max-max, ungated (HallDetect-style selection)": ("Emax_A", "Cmax_A"),
        "A-only max-max, gated": ("Emax_A", "Cmax_A_gated"),
        "dual max-max (E from A, C from B), ungated": ("Emax_A", "Cmax_B"),
        "dual max-max (E from A, C from B), gated": ("Emax_A", "Cmax_B_gated"),
    }
    for name, (ek, ck) in configs.items():
        L.append(f"--- {name}")
        for cls in fams:
            sub = [r for r in rows if r["cls"] == cls]
            d = sum(1 for r in sub if flag(r[ek], r[ck], TAU)); b = sum(1 for r in sub if min(r[ek], r[ck]) >= TAU)
            me = statistics.mean(r[ek] for r in sub); mc = statistics.mean(r[ck] for r in sub)
            L.append(f"  {cls:<9} n={len(sub):>2} meanE={me:.3f} meanC={mc:.3f} | diag {d:>2}/{len(sub)} | bil {b:>2}/{len(sub)}")
        L.append("")
    for dk, label in (("D_A", "collapsed D = Emax_A - Cmax_A (one head, HallDetect-style)"), ("D_AB", "collapsed D = Emax_A - Cmax_B (dual heads)")):
        L.append(f"--- {label}: per-family mean, min, max")
        for cls in fams:
            v = [r[dk] for r in rows if r["cls"] == cls]
            L.append(f"  {cls:<9} n={len(v):>2} mean={statistics.mean(v):+.3f} min={min(v):+.3f} max={max(v):+.3f}")
        # overlap of ranges: share of CONFLICT items whose D lies inside the NEU range
        neu = [r[dk] for r in rows if r["cls"] == "NEU"]; con = [r[dk] for r in rows if r["cls"] == "CONFLICT"]
        lo, hi = min(neu), max(neu)
        L.append(f"  CONFLICT items with D inside the NEU range [{lo:+.3f}, {hi:+.3f}]: {sum(1 for x in con if lo <= x <= hi)}/{len(con)}")
        L.append("")
    txt = "\n".join(L); open(OUT_TXT, "w", encoding="utf-8").write(txt); print("\n" + txt)


if __name__ == "__main__":
    main()
