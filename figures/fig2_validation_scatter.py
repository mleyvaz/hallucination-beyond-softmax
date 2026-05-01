"""
Figure 2 of the Trichotomy Refactored paper.

Scatter plot of the 50 synthetic-validation pairs in the (T, F) unit
square, colored by hand-assigned class. Shows that:
  - ENT pairs cluster on the right edge (high T, low F)
  - CON pairs cluster on the bottom edge (low T, high F)
  - NEU pairs cluster near the origin
  - PAR pairs occupy the upper triangle T + F > 1 (the regime that
    single-NLI softmax-normalized models cannot reach)
"""
import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CSV_PATH = os.path.join(ROOT, "validation_results.csv")
OUT_PATH = os.path.join(HERE, "fig2_validation_scatter.png")


def load():
    pairs = []
    with open(CSV_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pairs.append({
                "cls": row["cls"],
                "T": float(row["T_dual"]),
                "F": float(row["F_dual"]),
            })
    return pairs


def main():
    pairs = load()
    fig, ax = plt.subplots(figsize=(7.0, 6.5))

    # Shaded paraconsistent region (upper triangle T+F > 1)
    para_triangle = Polygon(
        [(0, 1), (1, 1), (1, 0)],
        closed=True, alpha=0.10, color="#cc3333",
        label=r"Paraconsistent regime $T + F > 1$",
    )
    ax.add_patch(para_triangle)

    # Reachable region under softmax (lower triangle)
    softmax_triangle = Polygon(
        [(0, 0), (1, 0), (0, 1)],
        closed=True, alpha=0.06, color="#3366cc",
        label=r"Softmax-reachable region $T + F \leq 1$",
    )
    ax.add_patch(softmax_triangle)

    # Diagonal T+F=1
    ax.plot([0, 1], [1, 0], "k--", linewidth=1.2, alpha=0.7,
            label=r"Boundary $T + F = 1$")

    style = {
        "ENT": dict(marker="o", color="#1f77b4", label="Entailment (n=10)"),
        "CON": dict(marker="s", color="#d62728", label="Contradiction (n=10)"),
        "NEU": dict(marker="^", color="#7f7f7f", label="Neutral (n=10)"),
        "PAR": dict(marker="D", color="#2ca02c", label="Paraconsistent (n=20)"),
    }
    for cls in ["ENT", "CON", "NEU", "PAR"]:
        sub = [p for p in pairs if p["cls"] == cls]
        xs = [p["T"] for p in sub]
        ys = [p["F"] for p in sub]
        ax.scatter(xs, ys, s=70, edgecolor="black", linewidth=0.7,
                   alpha=0.85, **style[cls])

    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel(r"$T$ — entailment from Model A (token overlap)",
                  fontsize=11)
    ax.set_ylabel(r"$F$ — contradiction from Model B (negation cue)",
                  fontsize=11)
    ax.set_title("Figure 2. Synthetic validation: 50 pairs in the (T, F) "
                 "unit square\nunder the dual-NLI protocol",
                 fontsize=11.5, pad=12)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", alpha=0.5)

    # Custom legend with both regions and class markers
    legend_handles = [
        Line2D([0], [0], color="k", linestyle="--", linewidth=1.2,
               label=r"Boundary $T + F = 1$"),
        Polygon([(0, 0), (0, 0)], color="#cc3333", alpha=0.10,
                label=r"Paraconsistent regime $T + F > 1$"),
        Polygon([(0, 0), (0, 0)], color="#3366cc", alpha=0.06,
                label=r"Softmax-reachable $T + F \leq 1$"),
    ]
    for cls in ["ENT", "CON", "NEU", "PAR"]:
        legend_handles.append(
            Line2D([0], [0], marker=style[cls]["marker"],
                   color="w", markerfacecolor=style[cls]["color"],
                   markeredgecolor="black", markersize=9,
                   label=style[cls]["label"]))
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8.5,
              framealpha=0.95, edgecolor="gray")

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Wrote: {OUT_PATH}")


if __name__ == "__main__":
    main()
