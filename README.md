# Hallucination Detection Beyond Softmax

Reference implementation, stimuli and raw scores for:

> Leyva-Vázquez, M. Y., Piñero Pérez, P. Y., Pérez Pupo, I. & Smarandache, F. (2026).
> **Measuring Conflicting Evidence with Decomposed NLI: A Controlled Diagnostic Study of a
> Measurement-Space Constraint in Hallucination Detection.** Manuscript v1.10, submitted.

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
| `make_table6.py` | Rebuilds Table 6 and Figure 1, the (T, F) scatter, from the CSV files; no model needed |
| `experiment_a11_source_intervention.py` | Experiment 9: pre-registered intervention on the released stimuli (source noun phrase replaced by "The document"; verb and content verbatim), four arms, decomposed and holistic |
| `PREREGISTRATION_A10_A11.md` | Predictions registered before the Experiment 9 inference |
| `make_fig_frames.py` | Rebuilds Figure 2, the per-item paired differences of Experiment 8; no model needed |
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
8b. `experiment_a11_source_intervention.py` (Experiment 9, 2026-09-10; predictions registered first)
9. `make_table6.py` (Table 6 and Figure 1 from the CSV files; no model)

Dates and model revisions. The two checkpoints were downloaded once, on 2026-07-13, when
Experiments 1-2 were run. Experiments 3-6 (2026-08-31) and Experiments 7-8 (2026-09-09) reused
that local cache; no re-download or update was performed in between, and `transformers` was not
asked to check for a newer revision. Commit hashes were not pinned at download time and cannot
be reconstructed after the fact, so "the Hub default revision as of 2026-07-13" is the most
precise identification available; the paper states this as a declared reproducibility gap. A
later default revision may give different scores.

## Reproducing

```bash
pip install -r requirements.txt
python experiment_a_real_models.py            # Experiment 1
python experiment_a2_conflicting_evidence.py  # Experiment 2
python experiment_a3_controls.py              # Experiment 3
python experiment_a4_relevance_gate.py; python experiment_a5_lexical_gate.py   # Experiment 4
python experiment_a6_model_swap.py; python experiment_a7_single_head_decomposed.py  # Experiments 5-6
python experiment_a8_polarity_blind_max.py; python audit_a8_intervals.py       # Experiment 7 + interval search
python experiment_a11_source_intervention.py  # Experiment 9  (pre-registered intervention)
python experiment_a9_attribution_frames.py    # Experiment 8  (add --from-csv to rebuild its
                                              # summary from the released CSV, no model)
python make_table6.py                         # Table 6 + Figure 1 (no model)
python make_fig_frames.py                     # Figure 2 (no model)
```

Aggregation-only scripts (no model, no GPU, seconds): `make_table6.py` rebuilds Table 6 and
Figure 1 (Figure 1 needs matplotlib), `audit_a8_intervals.py` rebuilds the interval search of
Table 7, `experiment_a9_attribution_frames.py --from-csv` rebuilds Table 8, and
`make_fig_frames.py` rebuilds Figure 2. The remaining
tables are the per-experiment summary files written next to each CSV.

Each script prints its summary and writes it next to the CSV. All stimuli are inside the
scripts, verbatim. The `synthetic_validation.py` numbers are heuristic (token overlap) and are
not used for any headline result.

## License

MIT.
