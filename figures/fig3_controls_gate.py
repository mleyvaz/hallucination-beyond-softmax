"""Figure 3: decomposition controls (Experiment 3) ungated, with the NLI gate and with the lexical gate (Experiment 4).
Reads ../validation_results_a4.csv. Two panels in the (T, F) unit square: left = raw F_B,
right = gated F_soft. Margin region T+F>1 & min(T,F)>=0.15 shaded."""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
rows = list(csv.DictReader(open(os.path.join(ROOT, "validation_results_a5.csv"), encoding="utf-8")))
for r in rows:
    r["F_lex"] = r["F_raw"] if float(r["jaccard"]) >= 0.3 else "0.0"
STY = {"CONFLICT": ("o", "#8C1A1A", "conflicting evidence (decomposed)"),
       "CROSS": ("x", "#1F1F3D", "CROSS: refuting segment of another claim"),
       "AGREE-SUP": ("^", "#2E7D32", "AGREE-SUP: both sources support"),
       "AGREE-REF": ("v", "#EF6C00", "AGREE-REF: both sources refute"),
       "ENT": ("s", "#9E9E9E", "entailment / contradiction controls"),
       "CON": ("s", "#9E9E9E", None)}
fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharey=True)
for ax, fcol, title in [(axes[0], "F_raw", "(a) F from Model B, ungated"),
                        (axes[1], "F_soft", "(b) NLI gate: F · (1 − neutral_A)"),
                        (axes[2], "F_lex", "(c) lexical topicality gate (J ≥ 0.3)")]:
    tau = 0.15
    xs = np.linspace(0, 1, 200)
    ax.fill_between(xs, np.maximum(1 - xs, tau), 1, where=(xs >= tau), color="#8C1A1A", alpha=0.07)
    ax.plot([0, 1], [1, 0], "--", color="black", lw=1)
    for cls, (m, c, lab) in STY.items():
        sub = [r for r in rows if r["cls"] == cls]
        ax.scatter([float(r["T"]) for r in sub], [float(r[fcol]) for r in sub], marker=m, s=42,
                   c=c, label=lab, alpha=0.85, edgecolors="white" if m == "o" else None, linewidths=0.5)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02); ax.set_xlabel("T (Model A, supporting segment)")
    ax.set_title(title, fontsize=10); ax.grid(alpha=0.2)
axes[0].set_ylabel("F (Model B, refuting segment)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.06, 1, 1))
out = os.path.join(HERE, "fig3_controls_gate.png"); fig.savefig(out, dpi=200); print("Wrote", out)
