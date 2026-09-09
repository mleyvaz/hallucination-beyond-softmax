# Hallucination Detection Beyond Softmax

Reference implementation, stimuli and raw scores for:

> Leyva-Vázquez, M. Y., Matheu Pérez, A. & Smarandache, F. (2026). **Measuring Conflicting
> Evidence with Decomposed NLI: A Controlled Diagnostic Study of a Measurement-Space Constraint
> in Hallucination Detection.** Manuscript v1.8 (under review).

## The claim in one line

When a detector reads both the entailment probability T and the contradiction probability F
from one softmax-normalized NLI head (or reads one and treats the other as its complement),
`T + N + F = 1` implies `T + F <= 1`. The state in which evidence both supports and contradicts a
claim is unreachable by construction (Theorem 1). Reading T and F from two separate NLI calls
removes the constraint; evidence decomposition, a lexical topicality gate and a bilateral rule
with a margin make the recovered region measure conflict rather than artifacts. The controls
(Experiments 3-7) delimit what each part of the rule contributes.

## Contents

| File | What it is |
|---|---|
| `dual_nli.py` | Reference implementation: pair input, holistic / decomposed / polarity-blind scoring, lexical gate, margin and bilateral rules. All experiment scripts import `nli_scores` and `jaccard` from here |
| `synthetic_validation.py` | Heuristic pre-study of v0.5 (kept as geometric illustration; source of the 50 Experiment 1 pairs) |
| `experiment_a_real_models.py` | Experiment 1: Theorem 1 with real models (single vs dual NLI, 50 pairs) |
| `experiment_a2_conflicting_evidence.py` | Experiment 2: conflicting-evidence stimuli, holistic vs decomposed scoring, 20 items + 20 single-segment controls |
| `experiment_a3_controls.py` | Experiment 3: controls for decomposition (AGREE-SUP, AGREE-REF, CROSS), threshold sweep, Pearson on NEU+PAR |
| `experiment_a4_relevance_gate.py` | Experiment 4a: NLI topicality gate on the F channel (removes CROSS false positives, collapses recall) |
| `experiment_a5_lexical_gate.py` | Experiment 4b: model-free lexical topicality gate |
| `experiment_a6_model_swap.py` | Experiment 5: model-assignment swap (T from B, F from A) |
| `experiment_a7_single_head_decomposed.py` | Experiment 6: single-head decomposed baselines and the F-only rule |
| `experiment_a8_polarity_blind_max.py` | Experiment 7: polarity-blind max-per-channel selection (HallDetect-style) and the collapsed difference score |
| `audit_a8_intervals.py` | Exhaustive interval search on the collapsed scalar D (gated and ungated) under stated objectives and FP budgets; output `audit_a8_intervals.txt` |
| `experiment_a9_attribution_frames.py` | Experiment 8: attribution-frame control (content fixed; bare / claims / confirms / according / official / pamphlet) |
| `make_table6.py` | Rebuilds Table 6 and the (T, F) scatter figure from the CSV files; no model needed |
| `table6_counts.txt` | Output of `make_table6.py` |
| `validation_results_*.csv`, `validation_summary_*.txt` | Per-item scores and summaries for each experiment |
| `figures/` | `fig_tf_scatter.{pdf,png}` (paper figure) plus earlier illustration scripts |

Models (public, CPU is enough): Model A `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`
(T channel and relevance probe); Model B `ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli`
(F channel).

## Order in which the scripts were run

1. `synthetic_validation.py` (heuristic pre-study; defines the 50 Experiment 1 pairs)
2. `experiment_a_real_models.py` (Experiment 1, 2026-07-13)
3. `experiment_a2_conflicting_evidence.py` (Experiment 2; the margin tau = 0.15 was fixed at this stage on the ENT/CON pairs)
4. `experiment_a3_controls.py` (Experiment 3 and the tau sweep, 2026-08-31)
5. `experiment_a4_relevance_gate.py`, `experiment_a5_lexical_gate.py` (Experiment 4)
6. `experiment_a6_model_swap.py` (Experiment 5), `experiment_a7_single_head_decomposed.py` (Experiment 6)
7. `experiment_a8_polarity_blind_max.py` (Experiment 7, 2026-09-09), then `audit_a8_intervals.py` (no model)
8. `experiment_a9_attribution_frames.py` (Experiment 8, 2026-09-09)
9. `make_table6.py` (Table 6 and Figure 1 from the CSV files; no model)

Dates: Experiments 1-2 were run on 2026-07-13, Experiments 3-6 on 2026-08-31, Experiments 7-8 on
2026-09-09, all against the Hugging Face Hub default revisions cached on the same machine at the
first download; the paper's "as retrieved on 2026-08-31" refers to the revision, not to a re-download.

Model revisions are the Hugging Face Hub defaults at the run dates; commit hashes were not pinned.

## Reproducing

```bash
pip install -r requirements.txt
python experiment_a_real_models.py            # Experiment 1
python experiment_a2_conflicting_evidence.py  # Experiment 2
python experiment_a3_controls.py              # Experiment 3
python experiment_a4_relevance_gate.py; python experiment_a5_lexical_gate.py   # Experiment 4
python experiment_a6_model_swap.py; python experiment_a7_single_head_decomposed.py  # Experiments 5-6
python experiment_a8_polarity_blind_max.py; python audit_a8_intervals.py       # Experiment 7 + interval search
python experiment_a9_attribution_frames.py    # Experiment 8
python make_table6.py                         # Table 6 + Figure 1 (no model)
python experiment_a_real_models.py            # -> validation_results_real.csv
python experiment_a2_conflicting_evidence.py  # -> validation_results_a2.csv
python experiment_a3_controls.py              # -> validation_results_a3.csv
python experiment_a4_relevance_gate.py        # -> validation_results_a4.csv
python experiment_a5_lexical_gate.py          # -> validation_results_a5.csv (no models needed)
python experiment_a6_model_swap.py            # -> validation_results_a6.csv
python experiment_a7_single_head_decomposed.py # -> validation_results_a7.csv
```

Each script prints its summary and writes it next to the CSV. All stimuli are inside the
scripts, verbatim. The `synthetic_validation.py` numbers are heuristic (token overlap) and are
not used for any headline result.

## License

MIT.
