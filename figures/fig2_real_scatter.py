# -*- coding: utf-8 -*-
"""Figure 2 (v1.1): real-model validation scatter in the (T, F) unit square."""
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

a1 = list(csv.DictReader(open(os.path.join(ROOT, 'validation_results_real.csv'), encoding='utf8')))
a2 = list(csv.DictReader(open(os.path.join(ROOT, 'validation_results_a2.csv'), encoding='utf8')))

fig, ax = plt.subplots(figsize=(6.0, 5.2))

# region de la regla de margen: T+F>1 y min(T,F)>=0.15
xs = np.linspace(0, 1, 300)
ax.fill_between(xs, np.maximum(1 - xs, 0.15), 1.0,
                where=(xs >= 0.15), color='#D65F5F', alpha=0.10, lw=0,
                label='margin decision region')
ax.plot([0, 1], [1, 0], 'k--', lw=1.2, label='softmax boundary  T+F = 1')

# controles (experimento A, dual real)
ent = [(float(r['T_dual']), float(r['F_dual'])) for r in a1 if r['cls'] == 'ENT']
con = [(float(r['T_dual']), float(r['F_dual'])) for r in a1 if r['cls'] == 'CON']
ax.scatter(*zip(*ent), marker='^', s=48, c='#4878CF', label='entailment controls (n=10)')
ax.scatter(*zip(*con), marker='s', s=42, c='#6ACC64', label='contradiction controls (n=10)')

# E2: evidencia conflictiva
hol = [(float(r['T_holistic']), float(r['F_holistic'])) for r in a2]
dec = [(float(r['T_decomp']), float(r['F_decomp'])) for r in a2]
ax.scatter(*zip(*hol), marker='x', s=52, c='#8C8C8C',
           label='conflicting evidence — holistic (n=20)')
ax.scatter(*zip(*dec), marker='o', s=52, c='#D65F5F', edgecolors='#7a1f1f',
           linewidths=0.8, label='conflicting evidence — decomposed (n=20)')

ax.set_xlabel('T  (entailment, Model A: DeBERTa-v3-large NLI)')
ax.set_ylabel('F  (contradiction, Model B: RoBERTa-large NLI)')
ax.set_xlim(-0.03, 1.03)
ax.set_ylim(-0.03, 1.03)
ax.set_aspect('equal')
ax.legend(loc='upper right', fontsize=8, framealpha=0.95)
ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(HERE, 'fig2_real_scatter.png'), dpi=300)
print('fig2_real_scatter.png generada')
