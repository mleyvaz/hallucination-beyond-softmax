# Hallucination Detection Beyond Softmax

Reference implementation, stimuli and raw scores for:

> Leyva-Vázquez, M. Y. & Smarandache, F. (2026). **Hallucination Detection Beyond Softmax:
> A Measurement-Space Impossibility and a Dual-NLI Protocol.** Preprint v1.2.

## The claim in one line

When a detector reads both the entailment probability T and the contradiction probability F
from one softmax-normalized NLI head (or reads one and treats the other as its complement),
`T + N + F = 1` implies `T + F <= 1`. The state in which evidence both supports and contradicts a
claim is unreachable by construction (Theorem 1). Two independently trained NLI models, one per
channel, remove the constraint; evidence decomposition and a relevance-gated margin rule make
the recovered region measure conflict rather than artifacts.

## Contents

| File | What it is |
|---|---|
| `dual_nli.py` | Protocol implementation (stub backend + Hugging Face backend) |
| `synthetic_validation.py` | Heuristic pre-study of v0.5 (kept as geometric illustration; source of the 50 Experiment 1 pairs) |
| `experiment_a_real_models.py` | Experiment 1: Theorem 1 with real models (single vs dual NLI, 50 pairs) |
| `experiment_a2_conflicting_evidence.py` | Experiment 2: conflicting-evidence stimuli, holistic vs decomposed scoring, 20 items + 20 single-segment controls |
| `experiment_a3_controls.py` | Experiment 3: controls for decomposition (AGREE-SUP, AGREE-REF, CROSS), threshold sweep, Pearson on NEU+PAR |
| `experiment_a4_relevance_gate.py` | Experiment 4: relevance gate on the F channel (fixes the CROSS false positives of Experiment 3) |
| `validation_results_*.csv`, `validation_summary_*.txt` | Per-item scores and summaries for each experiment |
| `figures/` | Figure scripts and PNGs |

Models (public, CPU is enough): Model A `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`
(T channel and relevance probe); Model B `ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli`
(F channel).

## Reproducing

```bash
pip install transformers torch
python experiment_a_real_models.py            # -> validation_results_real.csv
python experiment_a2_conflicting_evidence.py  # -> validation_results_a2.csv
python experiment_a3_controls.py              # -> validation_results_a3.csv
python experiment_a4_relevance_gate.py        # -> validation_results_a4.csv
```

Each script prints its summary and writes it next to the CSV. All stimuli are inside the
scripts, verbatim. The `synthetic_validation.py` numbers are heuristic (token overlap) and are
not used for any headline result.

## License

MIT.
