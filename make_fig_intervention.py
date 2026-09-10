"""
Optional extra figure (not used in the manuscript): the source-description intervention
(Experiment 10, A11) and the
crossed factorial design (Experiment 9, A10).

(a) Per-item gated F under the released refuting span and under the same span with its source
    noun phrase replaced by "The document"; the margin tau = 0.15 is drawn. Every other word,
    including the reporting verb, is verbatim.
(b) Paired adjective and verb effects from the crossed design, with 95% bootstrap intervals
    over items: the adjective contrast under each noun, and the verb contrast under each source.

Input : validation_results_a11.csv, validation_results_a10.csv (released)
Output: figures/fig_intervention.{pdf,png}. No model is loaded; matplotlib only.
Usage : python make_fig_intervention.py
"""
from __future__ import annotations
import csv, os, random, statistics

HERE = os.path.dirname(os.path.abspath(__file__))
A11 = os.path.join(HERE, "validation_results_a11.csv")
A10 = os.path.join(HERE, "validation_results_a10.csv")
TAU = 0.15


def rows(path):
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def boot(d, n=5000, seed=7):
    rng = random.Random(seed)
    ms = sorted(statistics.mean(rng.choice(d) for _ in d) for _ in range(n))
    return ms[int(0.025 * n)], ms[int(0.975 * n)]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a11 = {(int(r["item"]), r["arm"]): r for r in rows(A11)}
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))

    # ---- (a) intervention on the released refuting spans
    items = list(range(1, 21))
    fo = [float(a11[(i, "original")]["F_gated"]) for i in items]
    fn = [float(a11[(i, "ref_neutral")]["F_gated"]) for i in items]
    for i, o, n in zip(items, fo, fn):
        ax[0].plot([i, i], [o, n], color="0.75", lw=1, zorder=1)
    ax[0].scatter(items, fo, s=26, color="#d62728", label="released refuting span", zorder=2)
    ax[0].scatter(items, fn, s=26, color="#1f77b4", marker="^", label='source $\\to$ "The document"', zorder=2)
    ax[0].axhline(TAU, color="0.35", ls="--", lw=1)
    ax[0].text(20.4, TAU + 0.02, r"$\tau=0.15$", fontsize=7, ha="right", color="0.35")
    ax[0].set_xlabel("item"); ax[0].set_ylabel("gated $F$"); ax[0].set_xlim(0.3, 20.7); ax[0].set_ylim(-0.03, 1.05)
    ax[0].set_xticks([1, 5, 10, 15, 20]); ax[0].legend(fontsize=7, loc="lower right")
    ax[0].set_title("(a) Intervening on the source description of the released spans", fontsize=9)

    # ---- (b) crossed design
    if os.path.exists(A10):
        a10 = {(int(r["item"]), r["polarity"], r["frame"]): r for r in rows(A10)}

        def diff(fa, fb, key="entA"):
            return [float(a10[(i, "X", fa)][key]) - float(a10[(i, "X", fb)][key]) for i in items]

        contrasts = [
            ("adjective\n(document)", diff("plain_document_states", "disputed_document_states")),
            ("adjective\n(pamphlet)", diff("plain_pamphlet_states", "disputed_pamphlet_states")),
            ("noun\n(no adjective)", diff("plain_document_states", "plain_pamphlet_states")),
            ("verb\n(document)", diff("plain_document_confirms", "plain_document_claims")),
            ("verb\n(disputed pamphlet)", diff("disputed_pamphlet_confirms", "disputed_pamphlet_claims")),
        ]
        xs = range(len(contrasts))
        means = [statistics.mean(d) for _, d in contrasts]
        cis = [boot(d) for _, d in contrasts]
        err = [[m - lo for m, (lo, _) in zip(means, cis)], [hi - m for m, (_, hi) in zip(means, cis)]]
        ax[1].bar(xs, means, yerr=err, capsize=3, color=["#4c72b0", "#4c72b0", "#55a868", "#dd8452", "#dd8452"], width=0.62)
        for x, (_, d) in zip(xs, contrasts):
            ax[1].scatter([x] * len(d), d, s=7, color="0.25", alpha=0.55, zorder=3)
        ax[1].axhline(0, color="0.3", lw=0.8)
        ax[1].set_xticks(list(xs)); ax[1].set_xticklabels([c[0] for c in contrasts], fontsize=7)
        ax[1].set_ylabel("paired difference in entailment $A$")
        ax[1].set_title("(b) Crossed design: each factor with the others held fixed", fontsize=9)
    else:
        ax[1].text(0.5, 0.5, "run experiment_a10_crossed_frames.py", ha="center", va="center", fontsize=8)
        ax[1].set_axis_off()

    plt.tight_layout()
    out = os.path.join(HERE, "figures")
    os.makedirs(out, exist_ok=True)
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(out, f"fig_intervention.{ext}"), dpi=200, bbox_inches="tight")
    print("Wrote", os.path.join(out, "fig_intervention.pdf"))


if __name__ == "__main__":
    main()
