"""
Rebuild Table 6 (decision-rule comparison) and Figure 1 (per-family (T, F) scatter)
from the released per-item CSV files. No model is run: this script uses only the
standard library for the table and matplotlib for the figure.

Inputs : validation_results_a5.csv  (main dual assignment, lexical gate; columns T, F_raw, jaccard;
         all six families including the single-segment ENT / CON controls)
Outputs: table6_counts.txt, figures/fig_tf_scatter.pdf, figures/fig_tf_scatter.png
"""
from __future__ import annotations
import csv, os
from decimal import Decimal as D

HERE = os.path.dirname(os.path.abspath(__file__))
TAU, THETA = D("0.15"), D("0.3")

RULES = {
    "R_diag (published)": lambda t, f: t + f > 1 and min(t, f) >= TAU,
    "R_bil (no diagonal)": lambda t, f: min(t, f) >= TAU,
    "R_F (single-channel)": lambda t, f: f >= TAU,
    "R_rect (post hoc: T>=0.60, F>=0.20)": lambda t, f: t >= D("0.60") and f >= D("0.20"),
}


def load_points():
    """Return {family: [(T, F_gated, item), ...]} for the six families of Table 6."""
    pts = {}
    with open(os.path.join(HERE, "validation_results_a5.csv"), encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            t = D(r["T"]); f = D(r["F_raw"]) if D(r["jaccard"]) >= THETA else D(0)
            pts.setdefault(r["cls"], []).append((t, f, int(r["item"])))
    return pts


def table6(pts):
    order = ["CONFLICT", "AGREE-SUP", "AGREE-REF", "CROSS", "ENT", "CON"]
    lines = ["Table 6 rebuilt from CSV (lexical gate theta=0.3, tau=0.15)", ""]
    head = f"{'family':<10}{'n':>4}" + "".join(f"{name:>40}" for name in RULES)
    lines.append(head)
    for fam in order:
        sub = pts[fam]
        cnt = [sum(1 for t, f, _ in sub if rule(t, f)) for rule in RULES.values()]
        lines.append(f"{fam:<10}{len(sub):>4}" + "".join(f"{c:>37}/{len(sub):<2}" for c in cnt))
    # descriptive precision / recall on the 100 items (CONFLICT positive, all others negative)
    lines.append("")
    for name, rule in RULES.items():
        tp = sum(1 for t, f, _ in pts["CONFLICT"] if rule(t, f))
        fp = sum(1 for fam in order[1:] for t, f, _ in pts[fam] if rule(t, f))
        lines.append(f"{name:<40} TP={tp:>2}/20  FP={fp:>2}/80  precision={tp}/{tp+fp}  recall={tp}/20")
    return "\n".join(lines)


def figure(pts):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; table only"); return
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    style = {"CONFLICT": ("o", "tab:red", "CONFLICT (contested)"),
             "AGREE-SUP": ("s", "tab:green", "AGREE-SUP"),
             "AGREE-REF": ("^", "tab:blue", "AGREE-REF"),
             "CROSS": ("x", "tab:gray", "CROSS"),
             "ENT": ("D", "tab:olive", "ENT (single segment)"),
             "CON": ("v", "tab:purple", "CON (single segment)")}
    for fam, (m, c, lab) in style.items():
        xs = [float(t) for t, f, _ in pts[fam]]; ys = [float(f) for t, f, _ in pts[fam]]
        ax.scatter(xs, ys, marker=m, c=c, label=lab, alpha=0.8, s=38, linewidths=1)
    ax.plot([0, 1], [1, 0], "k--", lw=1, label="$T+F=1$ (single-call bound)")
    ax.axvline(0.15, color="k", lw=0.6, ls=":"); ax.axhline(0.15, color="k", lw=0.6, ls=":")
    ax.text(0.16, 0.97, r"$\tau=0.15$", fontsize=8, va="top")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("$T$ = entailment, Model A (supporting span)")
    ax.set_ylabel("$F$ = contradiction, Model B (refuting span, gated)")
    ax.set_aspect("equal"); ax.legend(fontsize=7, loc="upper right", framealpha=0.9)
    ax.set_title("Per-item scores of the main dual assignment", fontsize=9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, "figures", f"fig_tf_scatter.{ext}"), dpi=200)
    print("figure written to figures/fig_tf_scatter.{pdf,png}")


if __name__ == "__main__":
    pts = load_points()
    txt = table6(pts)
    open(os.path.join(HERE, "table6_counts.txt"), "w", encoding="utf-8").write(txt)
    print(txt)
    figure(pts)
