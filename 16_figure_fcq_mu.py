# -*- coding: utf-8 -*-

"""
16 - Figure supplementaire : craving-trait (FCQ-T-r) x effet mu du reel
=======================================================================

Panneau A : effet mu du reel (central droit, reel - virtuel, valences
            moyennees) en fonction du score FCQ-T-r.
Panneaux B, C : idem pour les aliments appetissants et degoutants.

Droite : regression lineaire (visualisation seulement ; le test est le
rho de Spearman), bande = IC 95 % bootstrap de la droite.
Ligne verticale pointillee : seuil de 50 (Meule, 2018) ; sujets au-dessus
= cercles vides.

Entrees : outputs_correlations/correlations_data.csv (script 14)
          BDDseparee_FCQRT.xlsx (colonne Pre_FCQTR, sujets 1..30 dans l'ordre)
Sorties : outputs_correlations/figure_fcq_mu.png / .pdf
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

DATA_FILE = os.path.join("outputs_correlations", "correlations_data.csv")
FCQ_FILE = "BDDseparee_FCQRT.xlsx"
OUTDIR = "outputs_correlations"
if len(sys.argv) == 4:                      # chemins optionnels en ligne de commande
    DATA_FILE, FCQ_FILE, OUTDIR = sys.argv[1:]
os.makedirs(OUTDIR, exist_ok=True)

CUTOFF = 50
N_BOOT = 5000
SEED = 42
INK, INK2, MUTED = "#1f2328", "#57606a", "#8c959f"
COLOR = "#2a6db0"

d = pd.read_csv(DATA_FILE)
fcq = pd.read_excel(FCQ_FILE)["Pre_FCQTR"].values
d["FCQ"] = fcq[d["inclusion"].values - 1]
rng = np.random.default_rng(SEED)

panels = [
    ("MU_R", "A   All foods", "μ power, real − picture (dB)"),
    ("MU_R_APP", "B   Appetizing foods", "μ power, real − picture (dB)"),
    ("MU_R_DISG", "C   Disgusting foods", "μ power, real − picture (dB)"),
]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2})

fig = plt.figure(figsize=(9, 4.2))
gs = fig.add_gridspec(2, 2, width_ratios=[1.6, 1], wspace=0.3, hspace=0.55)
axes = [fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])]

xs = np.linspace(d.FCQ.min() - 2, d.FCQ.max() + 2, 100)
for ax, (var, title, ylab) in zip(axes, panels):
    x, y = d.FCQ.values, d[var].values
    rho, p = stats.spearmanr(x, y)
    boot_r, lines = [], []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(x), len(x))
        boot_r.append(stats.spearmanr(x[i], y[i]).statistic)
        k, c = np.polyfit(x[i], y[i], 1)
        lines.append(k * xs + c)
    lo, hi = np.nanpercentile(boot_r, [2.5, 97.5])
    band = np.percentile(lines, [2.5, 97.5], axis=0)
    k, c = np.polyfit(x, y, 1)

    ax.fill_between(xs, band[0], band[1], color=COLOR, alpha=0.12, lw=0)
    ax.plot(xs, k * xs + c, color=COLOR, lw=2)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axvline(CUTOFF, color=MUTED, lw=0.8, ls="--")
    above = x >= CUTOFF
    big = ax is axes[0]
    s = 42 if big else 22
    ax.scatter(x[~above], y[~above], s=s, color=COLOR, edgecolor="white", lw=0.8, zorder=3)
    ax.scatter(x[above], y[above], s=s, facecolor="white", edgecolor=COLOR, lw=1.4, zorder=3)
    ax.set_title(title, loc="left", fontsize=10 if big else 9, color=INK, fontweight="bold")
    ax.text(0.03, 0.97, f"ρ = {rho:.2f} [{lo:.2f}, {hi:.2f}]\n" + ("p < .001" if p < .001 else f"p = {p:.3f}".replace("0.", ".")),
            transform=ax.transAxes, va="top", fontsize=9 if big else 8, color=INK)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.set_ylabel(ylab, fontsize=9 if big else 8)
    if ax is not axes[1]:
        ax.set_xlabel("Trait food craving (FCQ-T-r score)", fontsize=9 if big else 8)
    if big:
        ax.text(CUTOFF + 0.6, ax.get_ylim()[0] + 0.03 * np.ptp(ax.get_ylim()),
                "cut-off (Meule, 2018)", fontsize=7.5, color=INK2, rotation=90,
                va="bottom")

fig.savefig(os.path.join(OUTDIR, "figure_fcq_mu.png"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(OUTDIR, "figure_fcq_mu.pdf"), bbox_inches="tight")
print("Figure sauvegardee dans", OUTDIR)
