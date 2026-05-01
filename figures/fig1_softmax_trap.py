"""
Figure 1 — The Softmax Trap.
Shows the geometric region of (T, F) reachable under softmax constraint
vs. the unreachable paraconsistent region T + F > 1.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "fig1_softmax_trap.png")

fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
fig.suptitle("The Softmax Trap: why current hallucination detectors structurally cannot measure paraconsistency",
             fontsize=13, fontweight='bold')

# ============== LEFT: Softmax-constrained region ==============
ax = axes[0]
ax.set_title("(a) Softmax-normalized NLI:\nT + N + F = 1, so T + F ≤ 1 always",
             fontsize=11)
ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
ax.set_xlabel("T  (entailment score)", fontsize=10)
ax.set_ylabel("F  (contradiction score)", fontsize=10)

# Reachable triangle (T + F ≤ 1)
ax.fill_between([0, 1], [0, 0], [1, 0], color='#A5D6A7', alpha=0.4)
ax.plot([0, 1], [1, 0], 'k--', linewidth=1.5)
ax.text(0.15, 0.15, "REACHABLE\n(T + F ≤ 1)", fontsize=11, ha='center', color='#1B5E20', fontweight='bold')

# Unreachable region (T + F > 1)
ax.fill_between([0, 1], [1, 0], 1, color='#EF9A9A', alpha=0.5)
ax.text(0.65, 0.65, "UNREACHABLE\n(T + F > 1)\nparaconsistent\nregime", fontsize=10, ha='center',
        color='#B71C1C', fontweight='bold')

ax.text(0.5, 0.05, "T + F = 1", fontsize=9, ha='center', style='italic', color='gray')
ax.set_aspect('equal'); ax.grid(alpha=0.25)

# ============== RIGHT: Dual-NLI proposal ==============
ax = axes[1]
ax.set_title("(b) Dual-NLI protocol (proposed):\nT and F from independent models — full square reachable",
             fontsize=11)
ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
ax.set_xlabel("T  (NLI-A entailment, MNLI-trained)", fontsize=10)
ax.set_ylabel("F  (NLI-B contradiction, FEVER+sci-trained)", fontsize=10)

# Whole square reachable
ax.fill_between([0, 1], 0, 1, color='#90CAF9', alpha=0.3)

# Sub-regions
ax.add_patch(patches.Rectangle((0.6, 0.6), 0.4, 0.4, color='#CE93D8', alpha=0.7))
ax.text(0.80, 0.80, "PARACONSISTENT\nT + F > 1\n(now measurable)",
        fontsize=10, ha='center', fontweight='bold', color='#4A148C')

ax.plot([0, 1], [1, 0], 'k:', linewidth=1, alpha=0.5)
ax.text(0.97, 0.05, "T + F = 1", fontsize=9, ha='right', style='italic', color='gray')

# Example points
examples = [
    (0.85, 0.10, '#2E7D32', 'C₁ confident correct'),
    (0.10, 0.85, '#C62828', 'C₂ confident hallucination'),
    (0.20, 0.20, '#FF8F00', 'C₃ admitted uncertainty'),
    (0.45, 0.45, '#1976D2', 'C₄ epistemic gap'),
    (0.75, 0.70, '#6A1B9A', 'C₅ paraconsistent'),
]
for x, y, c, lab in examples:
    ax.plot(x, y, 'o', markersize=11, color=c, markeredgecolor='black', linewidth=1.2, zorder=10)
    if "C₅" in lab or "C₄" in lab:
        ax.annotate(lab, xy=(x, y), xytext=(-90, 8), textcoords='offset points',
                     fontsize=8, color=c, fontweight='bold')
    else:
        ax.annotate(lab, xy=(x, y), xytext=(8, 8), textcoords='offset points',
                     fontsize=8, color=c)

ax.set_aspect('equal'); ax.grid(alpha=0.25)

plt.subplots_adjust(top=0.85, wspace=0.25)
plt.savefig(OUT, dpi=120)
print(f"Saved: {OUT}")
