"""
Figure 2 of the paper: per-item paired differences behind the two template contrasts of
Experiment 8, plus the polarity-blind contradiction firing.

Motivation: Table 8 reports means, and a mean of 0.93 or 0.73 hides how much the 20 items vary.
This figure shows every item, so a reader can see that both contrasts are positive on all 20
items but differ in spread by a factor of five.

Input : validation_results_a9.csv (released)      Output: figures/fig_frames.{pdf,png}
No model is loaded; matplotlib only.
Usage: python make_fig_frames.py
"""
from __future__ import annotations
import csv, os, statistics

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "validation_results_a9.csv")


def load():
    with open(CSV, encoding="utf-8") as fh:
        return {(r["item"], r["polarity"], r["frame"]): r for r in csv.DictReader(fh)}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    idx = load()
    items = [str(i) for i in range(1, 21)]
    verb = [float(idx[(i, "X", "confirms")]["entA"]) - float(idx[(i, "X", "claims")]["entA"]) for i in items]
    source = [float(idx[(i, "X", "official")]["entA"]) - float(idx[(i, "X", "pamphlet")]["entA"]) for i in items]
    con_x = [float(idx[(i, "X", "pamphlet")]["conB"]) for i in items]

    order = sorted(range(20), key=lambda k: source[k])
    y = range(20)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.2), gridspec_kw={"width_ratios": [1.55, 1]})

    ax = axes[0]
    for k, o in enumerate(order):
        ax.plot([verb[o], source[o]], [k, k], color="0.85", lw=1, zorder=1)
    ax.scatter([verb[o] for o in order], list(y), s=26, marker="o", color="tab:blue", zorder=2,
               label=r"verb: $\it{confirms}-\it{claims}$ (source fixed)")
    ax.scatter([source[o] for o in order], list(y), s=26, marker="D", color="tab:red", zorder=2,
               label=r"source: $\it{official\ record}-\it{disputed\ pamphlet}$ (verb fixed)")
    ax.axvline(statistics.mean(verb), color="tab:blue", ls="--", lw=1)
    ax.axvline(statistics.mean(source), color="tab:red", ls=":", lw=1.2)
    ax.axvline(0, color="k", lw=0.7)
    ax.set_yticks(list(y)); ax.set_yticklabels([f"{order[k] + 1}" for k in y], fontsize=6)
    ax.set_ylabel("claim (sorted by the source contrast)", fontsize=8)
    ax.set_xlabel(r"paired difference in Model $A$ entailment, supporting content", fontsize=8)
    ax.set_xlim(-0.05, 1.02); ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.5, loc="upper left", framealpha=0.95, borderpad=0.3)
    ax.set_title("(a) every item, not just the mean", fontsize=8.5)

    ax = axes[1]
    ax.scatter(con_x, list(y), s=26, marker="s", color="tab:purple")
    ax.axvline(0.15, color="k", ls=":", lw=1)
    ax.text(0.155, 19.4, r"$\tau=0.15$", fontsize=7, va="top")
    ax.set_yticks(list(y)); ax.set_yticklabels([])
    lab = (r"Model $B$ contradiction on $\it{supporting}$ content" + "\n"
           + "“A disputed pamphlet states that $X$”")
    ax.set_xlabel(lab, fontsize=8)
    ax.set_xlim(-0.02, 0.7); ax.tick_params(labelsize=7)
    ax.set_title("(b) firing without a refutation", fontsize=8.5)

    fig.tight_layout()
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, "figures", f"fig_frames.{ext}"), dpi=200)
    print("means: verb %.4f (min %.4f max %.4f), source %.4f (min %.4f max %.4f); "
          "supporting-content contradiction >= 0.15 on %d/20"
          % (statistics.mean(verb), min(verb), max(verb),
             statistics.mean(source), min(source), max(source),
             sum(v >= 0.15 for v in con_x)))
    print("figures/fig_frames.{pdf,png} written")


if __name__ == "__main__":
    main()
