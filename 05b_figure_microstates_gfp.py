# -*- coding: utf-8 -*-
"""
05b - Figure de l'article : microetats + GFP - Lab Is Coat
==========================================================

A lancer dans le dossier des microetats, APRES les scripts 02, 03 et 04
(meme config.py, memes epoques). Ne modifie aucune sortie existante.

  A - Les cinq cartes (ordre de la sequence grand-average : MS3, MS1, MS4,
      MS5, puis MS2 non interpretee), GEV moyenne dans les ERP individuels.
  B - GFP moyenne des quatre conditions (moyenne des GFP individuelles,
      80-600 ms), avec sous la courbe la segmentation du grand-average
      toutes conditions confondues, et les deux fenetres tardives grisees.
  C - (optionnel, desactive par defaut : PANEL_C = None) contrastes
      intra-sujet dans les fenetres 238-493 et 494-600 ms.

Code couleur commun a l'article : reel = orange, photo = bleu marine ;
appetissant = trait plein, degoutant = tirets.

Sorties : outputs_microstates/publication_figures/Figure_microstates_GFP.png / .pdf
          + valeurs de controle affichees (dz, a comparer a la section 3.2)
"""

import os
import glob
import time

import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy import stats

import config
import microstate_utils as msu

OUTDIR = os.path.join(config.OUTPUT_DIR, "publication_figures")
os.makedirs(OUTDIR, exist_ok=True)
MODEL_FILE = os.path.join(config.OUTPUT_DIR, "microstate_model.npz")
INFO_FILE = os.path.join(config.OUTPUT_DIR, "reference_info.fif")
PARAMETERS_FILE = os.path.join(config.OUTPUT_DIR, "microstate_parameters.csv")
EXCLUDED = getattr(config, "EXCLUDED_SUBJECTS", [])

MAP_ORDER = ["MS3", "MS1", "MS4", "MS5", "MS2"]       # ordre de la sequence
# Couleurs des microetats : volontairement sans orange ni bleu marine, reserves
# au reel et a la photo dans tout l'article.
MAP_COLORS = {"MS3": "#3a9e8f", "MS1": "#8e6bbf", "MS4": "#7a9a3a",
              "MS5": "#c4568a", "MS2": "#bab0ac"}
PANEL_C = None               # None = figure sans panneau C ; "contrasts" ou "conditions" pour le remettre
C_REAL, C_VIRT = "#eb6834", "#0d366b"      # reel = orange, photo = bleu marine
COND_STYLE = {
    "REAL_APP": dict(color=C_REAL, ls="-", label="Real, appetizing"),
    "VIRT_APP": dict(color=C_VIRT, ls="-", label="Picture, appetizing"),
    "REAL_DISG": dict(color=C_REAL, ls="--", label="Real, disgusting"),
    "VIRT_DISG": dict(color=C_VIRT, ls="--", label="Picture, disgusting"),
}
CONDS = list(COND_STYLE)

# ---------------- modele et GEV ----------------
model = msu.load_model(MODEL_FILE)
templates, names = model["templates"], list(model["names"])
ignore_polarity = model["ignore_polarity"]
info = mne.io.read_info(INFO_FILE, verbose=False)
params = pd.read_csv(PARAMETERS_FILE)
params = params[~params.subject.isin(EXCLUDED)]
gev = params.groupby(["subject", "microstate"]).GEV.mean().unstack().mean()

# ---------------- ERP individuels (comme les scripts 03 et 05) ----------------
files = sorted(glob.glob(os.path.join(config.EPOCHS_DIR, "*-epo.fif")))
files = [f for f in files if os.path.basename(f).replace("-epo.fif", "") not in EXCLUDED]
common_channels, sfreq = config.get_dataset_specs(files)

data, subjects = [], []                 # data : (sujets, conditions, canaux, temps)
times = None
for f in files:
    ep = config.prepare_epochs(f, common_channels, sfreq)
    sub = []
    for cond in CONDS:
        vis = config.CONDITIONS[cond]
        try:
            e = mne.concatenate_epochs([ep[vis[0]], ep[vis[1]]], verbose=False)
        except KeyError:
            sub = None
            break
        evk = e.average().crop(config.TMIN, config.TMAX)
        evk.set_eeg_reference("average", projection=False, verbose=False)
        sub.append(evk.data)
        times = evk.times * 1000
    if sub is not None:
        data.append(sub)
        subjects.append(os.path.basename(f).replace("-epo.fif", ""))
data = np.array(data)
n = len(subjects)
print(f"\n{n} sujets")

gfp = np.sqrt((data ** 2).mean(axis=2)) * 1e6          # (sujets, conditions, temps), uV

# segmentation du grand-average toutes conditions (comme le script 03)
ga = data.mean(axis=(0, 1))
seg = msu.backfit(ga, templates, factor=config.SMOOTHING_FACTOR,
                  half_window_size=config.SMOOTHING_HALF_WINDOW, tol=config.SMOOTHING_TOL,
                  min_segment_length=max(1, int(round(config.MIN_SEGMENT_DURATION_MS
                                                      * sfreq / 1000))),
                  ignore_polarity=ignore_polarity)

# fenetres tardives (segments du grand-average, script 03)
ga_win = config.load_ga_windows()
late = sorted(ga_win.values(), key=lambda w: w[0])[-2:]  # 238-493 et 494-600
print("Fenetres tardives :", late)

# ---------------- figure ----------------
if PANEL_C:
    fig = plt.figure(figsize=(11, 9))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1.0, 1.7, 1.5], hspace=0.45)
else:
    fig = plt.figure(figsize=(11, 6.2))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1.0, 1.7], hspace=0.45)

# A - cartes
gsA = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gs[0], wspace=0.1)
vmax = np.abs(templates).max()
for k, ms in enumerate(MAP_ORDER):
    ax = fig.add_subplot(gsA[k])
    mne.viz.plot_topomap(templates[names.index(ms)], info, axes=ax, show=False,
                         vlim=(-vmax, vmax), cmap="RdBu_r", contours=0)
    note = "\n(not interpreted)" if ms == "MS2" else ""
    ax.set_title(f"{ms}\nGEV = {gev[ms]:.2f}{note}", fontsize=9,
                 color=MAP_COLORS[ms] if ms != "MS2" else "0.4", fontweight="bold")
    if k == 0:
        ax.text(-0.25, 1.25, "A", transform=ax.transAxes, fontsize=13, fontweight="bold")

# B - GFP + segmentation
gsB = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1], height_ratios=[6, 1],
                                       hspace=0.05)
axB = fig.add_subplot(gsB[0])
for a, b in late:
    axB.axvspan(a, b, color="0.92", zorder=0)
axB.axvline(late[1][0], color="white", lw=2, zorder=0)   # limite entre les deux fenetres
for ci, cond in enumerate(CONDS):
    st = COND_STYLE[cond]
    axB.plot(times, gfp[:, ci].mean(0), color=st["color"], ls=st["ls"], lw=1.6,
             label=st["label"])
axB.set_xlim(times[0], times[-1])
axB.set_ylabel("GFP (µV)")
axB.tick_params(labelbottom=False)
axB.legend(fontsize=8, ncol=2, frameon=False, loc="upper left")
for side in ["top", "right"]:
    axB.spines[side].set_visible(False)
axB.text(-0.07, 1.02, "B", transform=axB.transAxes, fontsize=13, fontweight="bold")

axS = fig.add_subplot(gsB[1], sharex=axB)
dt = times[1] - times[0]
for t, lab in zip(times, seg):
    axS.axvspan(t, t + dt, color=MAP_COLORS[names[lab]] if lab >= 0 else "white", lw=0)
start = 0
for i in range(1, len(seg) + 1):                  # nom de la carte au centre de chaque segment
    if i == len(seg) or seg[i] != seg[start]:
        if seg[start] >= 0 and times[i - 1] - times[start] > 15:
            axS.text((times[start] + times[i - 1]) / 2, 0.5, names[seg[start]],
                     ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        start = i
axS.set_yticks([])
axS.set_xlabel("Time after stimulus onset (ms)")
for side in ["top", "right", "left"]:
    axS.spines[side].set_visible(False)

# C - effets intra-sujet dans les fenetres tardives
# Les differences entre participants (GFP de 1 a 6 uV) masquent l'effet quand
# on trace les 4 conditions. On trace donc, pour chaque participant, les trois
# contrastes du plan (difference intra-sujet, en uV) :
#   Valence = degoutant - appetissant ; Reality = reel - photo ;
#   Interaction = (reel - photo)appetissant - (reel - photo)degoutant.
# Un effet se lit alors directement : nuage et IC 95 % au-dessus de zero.
# PANEL_C = "contrasts" (defaut), "conditions" (ancienne version) ou None (sans C).
if PANEL_C:
    gsC = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2], wspace=0.3)
rng = np.random.default_rng(0)
EFFECTS = [("Valence", "0.25"), ("Reality", C_REAL), ("Interaction", "0.55")]
for w, (a, b) in enumerate(late):
    m = (times >= a) & (times <= b)
    v = gfp[:, :, m].mean(axis=2)                        # (sujets, conditions)
    ra, va, rd, vd = (v[:, CONDS.index(c)] for c in ["REAL_APP", "VIRT_APP", "REAL_DISG", "VIRT_DISG"])
    eff = {"Valence": (rd + vd) / 2 - (ra + va) / 2,
           "Reality": (ra + rd) / 2 - (va + vd) / 2,
           "Interaction": (ra - va) - (rd - vd)}
    for k, e in eff.items():
        t, p = stats.ttest_1samp(e, 0)
        print(f"{a:.0f}-{b:.0f} ms : {k:11s} {e.mean():+.2f} uV, dz = {e.mean() / e.std(ddof=1):.2f}, "
              f"t({n-1}) = {t:.2f}, p = {p:.2g}")
    if not PANEL_C:
        continue
    ax = fig.add_subplot(gsC[w])
    if PANEL_C == "contrasts":
        for x, (k, col) in enumerate(EFFECTS):
            e = eff[k]
            ax.scatter(x + rng.uniform(-0.13, 0.13, n), e, s=12, color=col, alpha=0.4, lw=0)
            ci95 = stats.t.ppf(.975, n - 1) * e.std(ddof=1) / np.sqrt(n)
            ax.errorbar(x + 0.3, e.mean(), yerr=ci95, fmt="o", color=col, ms=7, capsize=4,
                        lw=1.6)
            dz = e.mean() / e.std(ddof=1)
            ax.text(x + 0.3, 1.0, f"dz = {dz:.2f}", transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=8.5, color=col)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(range(len(EFFECTS)))
        ax.set_xticklabels(["Disgusting −\nappetizing", "Real −\npicture", "Interaction"],
                           fontsize=8.5)
        ax.set_xlim(-0.5, len(EFFECTS) - 0.2)
        ax.set_ylabel("GFP difference (µV)" if w == 0 else "")
        ax.set_title(f"{a:.0f}–{b:.0f} ms", fontsize=10, pad=16)
    else:                                                # ancienne version : 4 conditions
        for ci, cond in enumerate(CONDS):
            st = COND_STYLE[cond]
            x = ci + (0.4 if ci >= 2 else 0)
            ax.scatter(x + rng.uniform(-0.12, 0.12, n), v[:, ci], s=10, color=st["color"],
                       alpha=0.35, lw=0)
            ci95 = stats.t.ppf(.975, n - 1) * v[:, ci].std(ddof=1) / np.sqrt(n)
            ax.errorbar(x + 0.25, v[:, ci].mean(), yerr=ci95, fmt="o", color=st["color"],
                        mfc="white" if st["ls"] == "--" else st["color"], ms=6, capsize=3, lw=1.4)
        ax.set_xticks([0, 1, 2.4, 3.4])
        ax.set_xticklabels(["Real", "Picture", "Real", "Picture"], fontsize=8)
        ax.set_title(f"{a:.0f}–{b:.0f} ms", fontsize=10)
        ax.set_ylabel("Mean GFP (µV)" if w == 0 else "")
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    if w == 0:
        ax.text(-0.17, 1.08, "C", transform=ax.transAxes, fontsize=13, fontweight="bold")

for ext in ["png", "pdf"]:
    fname = os.path.abspath(os.path.join(OUTDIR, f"Figure_microstates_GFP.{ext}"))
    try:
        fig.savefig(fname, dpi=300, bbox_inches="tight")
    except OSError:                       # fichier ouvert dans une visionneuse
        fname = fname.replace(f".{ext}", f"_{time.strftime('%H%M%S')}.{ext}")
        fig.savefig(fname, dpi=300, bbox_inches="tight")
    print("Figure :", fname)
plt.close(fig)
