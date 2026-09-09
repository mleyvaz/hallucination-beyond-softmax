"""
Interval search over the collapsed difference score D (Experiment 7 / A8).

Added 2026-09-09 after the second adversarial review, which pointed out that the
sentence "caps recall at 8/20 even under the best post-hoc interval" did not state
its objective or false-positive budget, and that the published D was computed from
the UNGATED contradiction maximum while the two-coordinate rule used the GATED one.

This script makes both explicit. For each scalar it enumerates every closed
interval whose endpoints are observed values (ties grouped, so no interval can
split equal observations) and reports:
  * max TP subject to FP <= b, for b in {0, 1, 2, 4, 90}
  * max F1 (CONFLICT = positive, the 90 control items = negative)
  * the TP needed to reach full recall and the FP that costs
for three scalars: D_AB as published (ungated), D_AB recomputed from the gated
contradiction maximum (the pair the two-coordinate rule actually uses), and D_A
(one head). It also reports the two-coordinate margin rule on the same pair,
gated and ungated, so every comparison is made on one common treatment.

All numbers are retrospective on the 110 published items (20 CONFLICT + 90
controls); nothing here estimates generalization. Standard library only.
Usage: python audit_a8_intervals.py [validation_results_a8.csv]
"""
from __future__ import annotations
import csv, os, sys
from decimal import Decimal as D
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "validation_results_a8.csv")
TAU = D("0.15")

with open(PATH, encoding="utf-8-sig", newline="") as fh:
    rows = list(csv.DictReader(fh))
assert len(rows) == 110 and sum(r["cls"] == "CONFLICT" for r in rows) == 20
NPOS = 20


def search(points):
    grouped = {}
    for x, positive in points:
        c = grouped.setdefault(x, [0, 0]); c[0 if positive else 1] += 1
    values = sorted(grouped); out = []
    for i, lo in enumerate(values):
        tp = fp = 0
        for hi in values[i:]:
            a, b = grouped[hi]; tp += a; fp += b
            out.append((tp, fp, Fraction(2 * tp, NPOS + tp + fp), lo, hi))
    return out


def fmt(z):
    tp, fp, f1, lo, hi = z
    return f"TP={tp:>2}/20 FP={fp:>2}/90 F1={float(f1):.4f} interval=[{lo}, {hi}]"


L = ["Interval search on the collapsed scalar D (A8, 110 items: 20 CONFLICT vs 90 controls)", ""]
scalars = [
    ("D_AB published = Emax_A - Cmax_B (UNGATED)", lambda r: D(r["D_AB"])),
    ("D_AB gated     = Emax_A - Cmax_B_gated (same pair as the 2-D rule)", lambda r: D(r["Emax_A"]) - D(r["Cmax_B_gated"])),
    ("D_A  published = Emax_A - Cmax_A (one head, UNGATED)", lambda r: D(r["D_A"])),
    ("D_A  gated     = Emax_A - Cmax_A_gated", lambda r: D(r["Emax_A"]) - D(r["Cmax_A_gated"])),
]
for label, get in scalars:
    z = search([(get(r), r["cls"] == "CONFLICT") for r in rows])
    L.append("--- " + label)
    L.append("  max F1                 : " + fmt(max(z, key=lambda q: (q[2], -q[1]))))
    for b in (0, 1, 2, 4, 90):
        el = [q for q in z if q[1] <= b]
        L.append(f"  max TP with FP <= {b:>2}   : " + fmt(max(el, key=lambda q: (q[0], -q[1]))))
    full = [q for q in z if q[0] == NPOS]
    L.append("  full recall, min FP    : " + fmt(min(full, key=lambda q: q[1])))
    L.append("")

L.append("--- two-coordinate rules on the same pairs (for reference at a common treatment)")
for ek, ck, name in (("Emax_A", "Cmax_B", "dual ungated"), ("Emax_A", "Cmax_B_gated", "dual gated"),
                     ("Emax_A", "Cmax_A", "A-only ungated"), ("Emax_A", "Cmax_A_gated", "A-only gated")):
    for rule, rn in ((lambda t, f: t + f > 1 and min(t, f) >= TAU, "diag"), (lambda t, f: min(t, f) >= TAU, "bil")):
        tp = sum(rule(D(r[ek]), D(r[ck])) and r["cls"] == "CONFLICT" for r in rows)
        fp = sum(rule(D(r[ek]), D(r[ck])) and r["cls"] != "CONFLICT" for r in rows)
        L.append(f"  {name:<15} {rn:<5} TP={tp:>2}/20 FP={fp:>2}/90")
txt = "\n".join(L)
open(os.path.join(HERE, "audit_a8_intervals.txt"), "w", encoding="utf-8").write(txt)
print(txt)
