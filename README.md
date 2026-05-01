# Hallucination Detection Beyond Softmax

> **Companion code repository for**
> Leyva-Vázquez, M. Y. & Smarandache, F. (2026).
> *Hallucination Detection Beyond Softmax: A Theoretical Note and an Open Dual-NLI Protocol.*
> Preprint v0.5, MIT License.

## What this repository contains

A reference implementation of the **dual-NLI protocol** for hallucination
detection, plus a self-contained synthetic validation that demonstrates the
core geometric claim of the paper.

The paper's central result (Theorem 1) is that hallucination detectors
deriving entailment T and contradiction F from the same softmax-normalized
NLI head satisfy `T + F ≤ 1` by construction, and therefore cannot reach
the paraconsistent regime `T + F > 1`. The dual-NLI protocol — drawing T
and F from two independently trained NLI models — removes the constraint
and makes the upper triangle of the unit square reachable.

## Structure

```
hallucination-beyond-softmax/
├── README.md                       — this file
├── LICENSE                         — MIT
├── dual_nli.py                     — protocol implementation (stub + HF backends)
├── synthetic_validation.py         — 50-pair validation, fully reproducible
├── validation_results.csv          — per-pair scores from the 50 pairs
├── validation_summary.txt          — aggregate statistics
└── figures/
    ├── fig1_softmax_trap.py        — geometric figure for Theorem 1
    ├── fig1_softmax_trap.png
    ├── fig2_validation_scatter.py  — scatter of the 50 validation pairs
    └── fig2_validation_scatter.png
```

## Quick start (no API keys required)

```bash
# 1. Theorem 1 demonstration (stub backend, instant)
python dual_nli.py

# 2. Synthetic validation reported in §5 of the paper (50 pairs, instant)
python synthetic_validation.py

# 3. Regenerate figures
python figures/fig1_softmax_trap.py
python figures/fig2_validation_scatter.py
```

The synthetic validation runs in milliseconds and requires only the
Python standard library. It produces:

| Class | n | mean(T) | mean(F) | mean(T+F) | % T+F>1 (dual-NLI) | % T+F>1 (single-NLI sim) |
|--|--|--|--|--|--|--|
| Entailment | 10 | 0.943 | 0.000 | 0.943 | 0% | 0% |
| Contradiction | 10 | 0.120 | 0.650 | 0.770 | 0% | 0% |
| Neutral | 10 | 0.029 | 0.065 | 0.094 | 0% | 0% |
| Paraconsistent | 20 | 0.464 | 0.872 | 1.337 | 100% | 0% |

The single-NLI softmax simulation flags 0/50 pairs as paraconsistent —
exactly as Theorem 1 predicts. The dual-NLI protocol recovers all 20
hand-crafted paraconsistent pairs (100%).

## Running the protocol with real NLI models

```bash
pip install transformers torch

python -c "
from dual_nli import DualNLI
extractor = DualNLI(backend='huggingface')
score = extractor.score(
    response='Paris is the capital of France, but it is also not the capital.',
    ground_truth='Paris is the capital of France.'
)
print(f'T = {score.T:.3f}, F = {score.F:.3f}, T+F = {score.sum_TF:.3f}')
print(f'Paraconsistent: {score.paraconsistent}')
"
```

Default Models:
- **Model A** (T from entailment): `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`
- **Model B** (F from contradiction): `ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli`

Both are public Hugging Face checkpoints. First run downloads ~2.5 GB.

## Reproducing the figures and table in the paper

The figures and Table 1 in the paper are generated directly from
`validation_results.csv`. After running `synthetic_validation.py` and the
two figure scripts, all paper artifacts are reproducible without any
external dependency beyond `matplotlib` (only needed for the figures).

## Forthcoming companion paper

Empirical evaluation of the protocol on three frontier LLMs
(Claude Opus 4.7, GPT-5, Gemini 3) across TruthfulQA, HaluEval, and
FActScore is the subject of a forthcoming companion paper. That paper
will report the rate at which real LLM outputs occupy the paraconsistent
regime and the F1 lift achievable by augmenting standard detectors with
the dual-NLI feature.

## Citing

If you use this code or the protocol, please cite the paper:

```bibtex
@unpublished{leyva2026hallucination,
  author       = {Leyva-V{\'a}zquez, Maikel Yelandi and Smarandache, Florentin},
  title        = {Hallucination Detection Beyond Softmax: A Theoretical Note
                  and an Open Dual-NLI Protocol},
  year         = {2026},
  note         = {Preprint},
  url          = {https://github.com/mleyvaz/hallucination-beyond-softmax}
}
```

## License

MIT — see [LICENSE](LICENSE).

## Contact

- Maikel Yelandi Leyva-Vázquez · `mleyvaz@gmail.com` ·
  ORCID [0000-0002-0569-4932](https://orcid.org/0000-0002-0569-4932)
- Florentin Smarandache · `smarand@unm.edu` ·
  ORCID [0000-0002-5560-5926](https://orcid.org/0000-0002-5560-5926)
