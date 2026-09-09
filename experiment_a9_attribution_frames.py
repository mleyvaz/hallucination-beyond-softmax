"""
Experiment A9 - Attribution-frame control (added 2026-09-09, second adversarial review, C1/B2).

Question: when a refuting span is wrapped in a reporting frame ("the pamphlet claims
that not-X"), is the low contradiction score a correct neutral reading of a non-factive
report, a credibility discount, or neither? The earlier stimuli confound content, source,
verb and polarity. Here the CONTENT is held fixed (the 20 claims and their negations,
taken verbatim from the "that"-clauses of the released CONFLICT_ITEMS) and only the
frame varies:

  bare        : "<content>"
  claims      : "The document claims that <content>."          (non-factive verb, neutral source)
  confirms    : "The document confirms that <content>."        (factive verb, neutral source)
  according   : "According to the document, <content>."        (neutral attribution, no verb)
  official    : "The official record states that <content>."  (high-credibility source, same verb)
  pamphlet    : "A disputed pamphlet states that <content>."   (low-credibility source, same verb)

Each of the 6 frames x 2 polarities (X, not-X) x 20 items = 240 sentences is scored by
both heads against the bare claim X. Reported per cell: mean entailment_A, mean
neutral_A, mean contradiction_A, mean entailment_B, mean contradiction_B, and the
count of not-X sentences with contradiction_B >= 0.15 (the F channel's margin).

Output: validation_results_a9.csv, validation_summary_a9.txt.
"""
from __future__ import annotations
import csv, os, re, sys, statistics
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = HERE
sys.path.insert(0, HERE)
from experiment_a2_conflicting_evidence import MODEL_A, MODEL_B, CONFLICT_ITEMS
from dual_nli import build_pipe, nli_scores
OUT_CSV = os.path.join(ROOT, "validation_results_a9.csv"); OUT_TXT = os.path.join(ROOT, "validation_summary_a9.txt")

FRAMES = {
    "bare": "{c}",
    "claims": "The document claims that {c}",
    "confirms": "The document confirms that {c}",
    "according": "According to the document, {c}",
    "official": "The official record states that {c}",
    "pamphlet": "A disputed pamphlet states that {c}",
}


def that_clause(span: str) -> str:
    """Content after the reporting verb's 'that'; falls back to the span itself."""
    m = re.search(r"\bthat\s+(.*)$", span)
    c = (m.group(1) if m else span).strip()
    return c[0].lower() + c[1:] if not c.startswith(("Paris", "Einstein", "Mount", "Shakespeare", "Cuba", "Tokyo", "Gold", "Vaccines", "Oranges", "Water", "Light", "The", "Photosynthesis", "Global")) else c


PROPER = {"Paris", "Einstein", "Mount", "Shakespeare", "Cuba", "Tokyo", "Earth"}


def render(frame: str, content: str) -> str:
    c = content
    if frame != "bare" and c[:1].isupper() and not c.split()[0] in PROPER:
        c = c[0].lower() + c[1:]
    s = FRAMES[frame].format(c=c)
    s = s[0].upper() + s[1:]
    return s if s.endswith(".") else s + "."


def main():
    print("Loading models...", flush=True); pa = build_pipe(MODEL_A); pb = build_pipe(MODEL_B)
    rows = []
    for i, (sup, ref, claim) in enumerate(CONFLICT_ITEMS, 1):
        # X = the bare claim itself; not-X = the "that"-clause of the released refuting span
        # (item 5's refutation is carried by the verb "denies", so its negation is spelled out).
        notx = that_clause(ref)
        if i == 5:
            notx = "photosynthesis does not convert sunlight into chemical energy"
        contents = {"X": claim.rstrip("."), "notX": notx}
        for pol, content in contents.items():
            for frame in FRAMES:
                s = render(frame, content)
                a = nli_scores(pa, s, claim); b = nli_scores(pb, s, claim)
                rows.append({"item": i, "polarity": pol, "frame": frame, "sentence": s, "claim": claim,
                             "entA": round(a.get("entailment", 0.0), 4), "neuA": round(a.get("neutral", 0.0), 4),
                             "conA": round(a.get("contradiction", 0.0), 4),
                             "entB": round(b.get("entailment", 0.0), 4), "neuB": round(b.get("neutral", 0.0), 4),
                             "conB": round(b.get("contradiction", 0.0), 4)})
                print(f"  [{i:>2}] {pol:<4} {frame:<9} entA={rows[-1]['entA']:.3f} neuA={rows[-1]['neuA']:.3f} conB={rows[-1]['conB']:.3f}", flush=True)
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(r) for r in rows]

    L = ["=" * 96, "EXPERIMENT A9 - Attribution-frame control (content fixed, frame varied), n=20 items per cell", "=" * 96, "",
         f"{'polarity':<5} {'frame':<10} {'entA':>7} {'neuA':>7} {'conA':>7} {'entB':>7} {'conB':>7} {'conB>=.15':>10} {'entA>=.15':>10}"]
    for pol in ("X", "notX"):
        for frame in FRAMES:
            sub = [r for r in rows if r["polarity"] == pol and r["frame"] == frame]
            m = lambda k: statistics.mean(r[k] for r in sub)
            L.append(f"{pol:<5} {frame:<10} {m('entA'):7.3f} {m('neuA'):7.3f} {m('conA'):7.3f} {m('entB'):7.3f} {m('conB'):7.3f} "
                     f"{sum(r['conB'] >= 0.15 for r in sub):>7}/20 {sum(r['entA'] >= 0.15 for r in sub):>7}/20")
        L.append("")
    txt = "\n".join(L); open(OUT_TXT, "w", encoding="utf-8").write(txt); print("\n" + txt)


if __name__ == "__main__":
    main()
