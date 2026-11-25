# -*- coding: utf-8 -*-
"""
12c - Figure de l'article : effet du reel sur la puissance mu - Lab Is Coat
=========================================================================

A lancer dans le meme dossier que les scripts 11 et 12, APRES le script 11.
Relit les TFR (outputs_tfr/tfr/*_tfr.npz) ; ne modifie aucune sortie.
Memes ROI, bandes, fenetres et parametres de clusters que le script 12
(meme graine -> memes clusters).

  A - Cartes temps-frequence REAL - PICTURE (valence moyennee), ROI
      centrales gauche et droite, avec le contour des clusters p < .05.
  B - Puissance mu centrale droite dans la fenetre principale :
      chaque participant (REAL vs PICTURE) + moyennes, t et dz.
  C - Decours de la puissance mu (8-13 Hz), REAL vs PICTURE, gauche et
      droite, +/- SEM ; fenetre principale en jaune.
  D - Topographies de la difference mu REAL - PICTURE par tranches de
      250 ms ; electrodes des deux ROI centrales en noir.

Sortie : outputs_tfr/figures/Figure_tfr_mu.png / .pdf
"""

import os
import glob
import time
import inspect

import numpy as np
import mne
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from mne.stats import permutation_cluster_1samp_test

OUTPUT_DIR = "outputs_tfr"
TFR_DIR = os.path.join(OUTPUT_DIR, "tfr")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

MU = (8, 13)
ANALYSIS_WIN = (0.3, 1.75)
CLUSTER_WIN = (0.0, 1.75)
SHOW_WIN = (-0.5, 1.75)                 # periode affichee (cartes et decours)
BIN_EDGES = np.arange(0.25, 1.7501, 0.25)
N_PERMUTATIONS, CLUSTER_P, SEED = 5000, 0.05, 42
ROIS = {
    "Left central": ["E59", "E52", "E51", "E65", "E66", "E58"],
    "Right central": ["E183", "E184", "E196", "E182", "E164", "E195"],
}
REAL, VIRT = ["REAL_APP", "REAL_DISG"], ["VIRT_APP", "VIRT_DISG"]
C_REAL, C_VIRT = "#eb6834", "#0d366b"      # reel = orange, photo = bleu marine (code de l'article)
# Echelle divergente des cartes (A et D) : bleu marine = plus de puissance pour la
# photo, blanc = pas de difference, orange = plus de puissance pour le reel.
# Mettre DIFF_CMAP = "RdBu_r" pour revenir a l'echelle rouge-bleu classique.
from matplotlib.colors import LinearSegmentedColormap
DIFF_CMAP = LinearSegmentedColormap.from_list("picture_real", [C_VIRT, "#f7f7f7", C_REAL])
RNG_KW = "rng" if "rng" in inspect.signature(permutation_cluster_1samp_test).parameters else "seed"

# ---------------- chargement (comme le script 12) ----------------
files = sorted(glob.glob(os.path.join(TFR_DIR, "*_tfr.npz")))
subjects, data, chs = [], [], []
freqs = times = conditions = info0 = None
for f in files:
    z = np.load(f, allow_pickle=False)
    s = os.path.basename(f).replace("_tfr.npz", "")
    if times is None:
        freqs, times, conditions = z["freqs"], z["times"], list(z["conditions"])
        info0 = mne.io.read_info(os.path.join(TFR_DIR, f"{s}-info.fif"), verbose=False)
    subjects.append(s)
    data.append(z["power_db"])
    chs.append([str(c) for c in z["ch_names"]])
common = [c for c in chs[0] if all(c in cs for cs in chs[1:])]
P = np.stack([d[:, [cs.index(c) for c in common]] for d, cs in zip(data, chs)])
info = mne.pick_info(info0, [info0.ch_names.index(c) for c in common])
ci = {c: i for i, c in enumerate(conditions)}
n = len(subjects)
print(f"{n} sujets")


def fmt_p(p):
    return "p < .001" if p < .001 else f"p = {p:.3f}".replace("0.", ".")


def reality_diff(X):
    """X : (sujets, conditions, ...) -> REAL - PICTURE, valence moyennee."""
    return X[:, [ci[c] for c in REAL]].mean(1) - X[:, [ci[c] for c in VIRT]].mean(1)


roi_idx = {r: [common.index(c) for c in chans if c in common] for r, chans in ROIS.items()}
roi_tf = {r: P[:, :, idx].mean(axis=2) for r, idx in roi_idx.items()}   # (suj, cond, f, t)
fmu = (freqs >= MU[0]) & (freqs <= MU[1])
show = (times >= SHOW_WIN[0]) & (times <= SHOW_WIN[1])
cw = (times >= CLUSTER_WIN[0]) & (times <= CLUSTER_WIN[1])

# ---------------- clusters temps x frequence (comme le script 12) ----------------
t_thr = stats.t.ppf(1 - CLUSTER_P / 2, n - 1)
masks = {}
for roi in ROIS:
    D = reality_diff(roi_tf[roi])[..., cw]
    T, clusters, pv, _ = permutation_cluster_1samp_test(
        D, threshold=t_thr, n_permutations=N_PERMUTATIONS, tail=0, out_type="mask",
        verbose=False, **{RNG_KW: SEED})
    m = np.zeros((len(freqs), len(times)), bool)
    for cl, p in zip(clusters, pv):
        if p < .05:
            m[:, cw] |= cl
    masks[roi] = m
    print(f"{roi} : clusters p < .05 -> {[round(float(p), 3) for p in pv if p < .05] or 'aucun'}")

# ---------------- figure ----------------
fig = plt.figure(figsize=(12, 10.5))
gs = gridspec.GridSpec(3, 1, height_ratios=[1.25, 1.0, 0.75], hspace=0.42)

# A + B
gs1 = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gs[0],
                                       width_ratios=[1, 1, 0.05, 0.28, 0.62], wspace=0.3)
lim = 1.5
ext = [times[show][0], times[show][-1], freqs[0] - .5, freqs[-1] + .5]
for k, roi in enumerate(ROIS):
    ax = fig.add_subplot(gs1[k])
    M = reality_diff(roi_tf[roi]).mean(0)[:, show]
    im = ax.imshow(M, aspect="auto", origin="lower", extent=ext, cmap=DIFF_CMAP,
                   vmin=-lim, vmax=lim)
    if masks[roi].any():
        ax.contour(times[show], freqs, masks[roi][:, show], levels=[.5], colors="k",
                   linewidths=1.2)
    for y in MU:
        ax.axhline(y, color="0.3", lw=.6, ls="--")
    ax.axvline(0, color="k", lw=.8, ls=":")
    ax.set_title(f"{roi}: real − picture", fontsize=10)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)" if k == 0 else "")
    if k == 0:
        ax.text(-0.22, 1.06, "A", transform=ax.transAxes, fontsize=13, fontweight="bold")
cb = fig.colorbar(im, cax=fig.add_subplot(gs1[2]))     # barre de couleur dans sa colonne
cb.set_label("dB", fontsize=9)

axB = fig.add_subplot(gs1[4])                            # colonne 3 vide : espace pour l'axe y de B
am = (times >= ANALYSIS_WIN[0]) & (times <= ANALYSIS_WIN[1])
mu_r = roi_tf["Right central"][:, :, fmu][..., am].mean(axis=(2, 3))     # (suj, cond)
r = mu_r[:, [ci[c] for c in REAL]].mean(1)
v = mu_r[:, [ci[c] for c in VIRT]].mean(1)
for a, b in zip(v, r):
    axB.plot([0, 1], [a, b], color="0.75", lw=.8, zorder=1)
axB.scatter(np.zeros(n), v, s=14, color=C_VIRT, zorder=2)
axB.scatter(np.ones(n), r, s=14, color=C_REAL, zorder=2)
axB.plot([0, 1], [v.mean(), r.mean()], color="k", lw=2.2, marker="o", ms=7, zorder=3)
t, p = stats.ttest_rel(r, v)
dz = (r - v).mean() / (r - v).std(ddof=1)
print(f"Mu central droit, fenetre principale : REAL {r.mean():.2f} | PICTURE {v.mean():.2f} dB, "
      f"t({n-1}) = {t:.2f}, p = {p:.4f}, dz = {dz:.2f}")
axB.set_xticks([0, 1])
axB.set_xticklabels(["Picture", "Real"])
axB.set_xlim(-0.4, 1.4)
axB.set_ylabel("μ power, right central (dB)")
axB.set_title(f"{ANALYSIS_WIN[0]}–{ANALYSIS_WIN[1]} s\nt({n-1}) = {t:.2f}, {fmt_p(p)}, "
              f"dz = {dz:.2f}", fontsize=9)
for side in ["top", "right"]:
    axB.spines[side].set_visible(False)
axB.text(-0.45, 1.06, "B", transform=axB.transAxes, fontsize=13, fontweight="bold")

# C - decours mu
gs2 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1], wspace=0.2)
axes_c = []
for k, roi in enumerate(ROIS):
    ax = fig.add_subplot(gs2[k], sharey=axes_c[0] if axes_c else None)
    axes_c.append(ax)
    X = roi_tf[roi][:, :, fmu].mean(axis=2)                           # (suj, cond, t)
    for conds, col, lab in [(REAL, C_REAL, "Real"), (VIRT, C_VIRT, "Picture")]:
        y = X[:, [ci[c] for c in conds]].mean(1)[:, show]
        m, se = y.mean(0), y.std(0, ddof=1) / np.sqrt(n)
        ax.plot(times[show], m, color=col, lw=1.6, label=lab)
        ax.fill_between(times[show], m - se, m + se, color=col, alpha=.15, lw=0)
    ax.axvspan(*ANALYSIS_WIN, color="gold", alpha=.10, lw=0)
    ax.axhline(0, color="k", lw=.6)
    ax.axvline(0, color="k", lw=.8, ls=":")
    ax.set_title(f"{roi}, μ 8–13 Hz", fontsize=10)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Power (dB)" if k == 0 else "")
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    if k == 0:
        ax.legend(frameon=False, fontsize=9, loc="lower left")
        ax.text(-0.12, 1.06, "C", transform=ax.transAxes, fontsize=13, fontweight="bold")

# D - topographies
gs3 = gridspec.GridSpecFromSubplotSpec(1, len(BIN_EDGES), subplot_spec=gs[2],
                                       width_ratios=[1] * (len(BIN_EDGES) - 1) + [0.12],
                                       wspace=0.1)
mu_diff = reality_diff(P[:, :, :, fmu].mean(axis=3)).mean(0)              # (canaux, t)
maps = [mu_diff[:, (times >= a) & (times < b)].mean(1)
        for a, b in zip(BIN_EDGES[:-1], BIN_EDGES[1:])]
vmax = np.abs(maps).max()
roi_mask = np.zeros(len(common), bool)
for idx in roi_idx.values():
    roi_mask[idx] = True
for k, (mp, a, b) in enumerate(zip(maps, BIN_EDGES[:-1], BIN_EDGES[1:])):
    ax = fig.add_subplot(gs3[k])
    im, _ = mne.viz.plot_topomap(mp, info, axes=ax, show=False, cmap=DIFF_CMAP,
                                 vlim=(-vmax, vmax), contours=0, sensors=False, mask=roi_mask,
                                 mask_params=dict(marker="o", markerfacecolor="k",
                                                  markeredgecolor="k", markersize=2.5))
    ax.set_title(f"{a:.2f}–{b:.2f} s", fontsize=9)
    if k == 0:
        ax.text(-0.3, 1.15, "D", transform=ax.transAxes, fontsize=13, fontweight="bold")
cax = fig.add_subplot(gs3[-1])
cb = fig.colorbar(im, cax=cax)
cb.set_label("dB", fontsize=8)
cb.ax.tick_params(labelsize=7)

for ext_ in ["png", "pdf"]:
    fname = os.path.abspath(os.path.join(FIG_DIR, f"Figure_tfr_mu.{ext_}"))
    try:
        fig.savefig(fname, dpi=300, bbox_inches="tight")
    except OSError:                       # fichier ouvert dans une visionneuse
        fname = fname.replace(f".{ext_}", f"_{time.strftime('%H%M%S')}.{ext_}")
        fig.savefig(fname, dpi=300, bbox_inches="tight")
    print("Figure :", fname)
plt.close(fig)
