"""
Experiment A9 - Attribution-frame control (added 2026-09-09, second adversarial review, C1/B2).

Question: when a refuting span is wrapped in a reporting frame ("the pamphlet claims
that not-X"), what does each head do with the frame? The earlier stimuli confound content,
source, verb and polarity. Here the CONTENT is held fixed (the 20 claims and their negations,
taken from the released CONFLICT_ITEMS) and only the template varies:

  bare        : "<content>"
  claims      : "The document claims that <content>."          (non-factive verb, neutral source)
  confirms    : "The document confirms that <content>."        (factive verb, neutral source)
  according   : "According to the document, <content>."        (neutral attribution, no verb)
  official    : "The official record states that <content>."   (high-credibility source, same verb)
  pamphlet    : "A disputed pamphlet states that <content>."   (low-credibility source, same verb)

Each of the 6 frames x 2 polarities (X, not-X) x 20 items = 240 sentences is scored by both
heads against the bare claim. NOTE ON SCOPE: the templates vary more than one element at a
time (the two source templates differ in determiner, adjective and noun), so the contrasts
below are effects of these formulations, not identified mechanisms; a crossed design
(verb x source, adjective x noun) would be needed for that and is not run here.

Usage:
    python experiment_a9_attribution_frames.py             # run both models (CPU, ~1 h)
    python experiment_a9_attribution_frames.py --from-csv  # rebuild the summary from the CSV, no model

Output: validation_results_a9.csv, validation_summary_a9.txt.
"""
from __future__ import annotations
import csv, os, re, sys, statistics
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = HERE
sys.path.insert(0, HERE)
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
    return (m.group(1) if m else span).strip().rstrip(".")


PROPER = {"Paris", "Einstein", "Mount", "Shakespeare", "Cuba", "Tokyo", "Earth"}


def render(frame: str, content: str) -> str:
    c = content
    if frame != "bare" and c[:1].isupper() and c.split()[0] not in PROPER:
        c = c[0].lower() + c[1:]
    s = FRAMES[frame].format(c=c)
    s = s[0].upper() + s[1:]
    return s if s.endswith(".") else s + "."


def summarize(rows) -> str:
    """Rebuild the Table 8 block (and the two paired contrasts) from rows. No model needed."""
    L = ["=" * 96,
         "EXPERIMENT A9 - Attribution-frame control (content fixed, template varied), n=20 items per cell",
         "=" * 96, "",
         "Scope: the templates vary more than one element at a time; these are template contrasts,",
         "not identified mechanisms.", "",
         f"{'polarity':<5} {'frame':<10} {'entA':>7} {'neuA':>7} {'conA':>7} {'entB':>7} {'conB':>7} {'conB>=.15':>10} {'entA>=.15':>10}"]
    for pol in ("X", "notX"):
        for frame in FRAMES:
            sub = [r for r in rows if r["polarity"] == pol and r["frame"] == frame]
            m = lambda k: statistics.mean(float(r[k]) for r in sub)
            L.append(f"{pol:<5} {frame:<10} {m('entA'):7.3f} {m('neuA'):7.3f} {m('conA'):7.3f} "
                     f"{m('entB'):7.3f} {m('conB'):7.3f} "
                     f"{sum(float(r['conB']) >= 0.15 for r in sub):>7}/20 "
                     f"{sum(float(r['entA']) >= 0.15 for r in sub):>7}/20")
        L.append("")
    idx = {(str(r["item"]), r["polarity"], r["frame"]): r for r in rows}
    L.append("Paired contrasts on Model A entailment, supporting content, per item:")
    for a, b in (("official", "pamphlet"), ("confirms", "claims")):
        d = [float(idx[(str(i), "X", a)]["entA"]) - float(idx[(str(i), "X", b)]["entA"]) for i in range(1, 21)]
        L.append(f"  {a:<9} - {b:<9} mean={statistics.mean(d):+.6f} min={min(d):+.4f} max={max(d):+.4f} "
                 f"positive={sum(x > 0 for x in d)}/20")
    return "\n".join(L)


def write_summary(rows):
    txt = summarize(rows)
    open(OUT_TXT, "w", encoding="utf-8").write(txt)
    print("\n" + txt)


def from_csv():
    with open(OUT_CSV, encoding="utf-8") as fh:
        write_summary(list(csv.DictReader(fh)))


def main():
    if "--from-csv" in sys.argv:
        return from_csv()
    from experiment_a2_conflicting_evidence import MODEL_A, MODEL_B, CONFLICT_ITEMS
    print("Loading models...", flush=True)
    pa = build_pipe(MODEL_A); pb = build_pipe(MODEL_B)
    rows = []
    for i, (sup, ref, claim) in enumerate(CONFLICT_ITEMS, 1):
        # X = the bare claim; not-X = the "that"-clause of the released refuting span.
        # Item 5's refutation is carried by the verb "denies", so its negation is spelled out;
        # that item is therefore not a pure frame substitution (declared in the paper).
        notx = that_clause(ref)
        if i == 5:
            notx = "photosynthesis does not convert sunlight into chemical energy"
        for pol, content in (("X", claim.rstrip(".")), ("notX", notx)):
            for frame in FRAMES:
                s = render(frame, content)
                a = nli_scores(pa, s, claim); b = nli_scores(pb, s, claim)
                rows.append({"item": i, "polarity": pol, "frame": frame, "sentence": s, "claim": claim,
                             "entA": round(a.get("entailment", 0.0), 4), "neuA": round(a.get("neutral", 0.0), 4),
                             "conA": round(a.get("contradiction", 0.0), 4),
                             "entB": round(b.get("entailment", 0.0), 4), "neuB": round(b.get("neutral", 0.0), 4),
                             "conB": round(b.get("contradiction", 0.0), 4)})
                print(f"  [{i:>2}] {pol:<4} {frame:<9} entA={rows[-1]['entA']:.3f} "
                      f"neuA={rows[-1]['neuA']:.3f} conB={rows[-1]['conB']:.3f}", flush=True)
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(r) for r in rows]
    write_summary(rows)


if __name__ == "__main__":
    main()
