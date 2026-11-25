# -*- coding: utf-8 -*-
"""
17c - Figure S1 : activite sensorimotrice avant le stimulus - Lab Is Coat
========================================================================

A lancer dans le meme dossier que les scripts 11, 12 et 17, APRES le script 11.
Relit les TFR (outputs_tfr/tfr/*_tfr.npz) ; ne modifie aucune sortie.

  A - Cartes temps-frequence (toutes conditions confondues, dB / ligne de base
      commune -2.0 a -1.5 s), ROI centrales gauche et droite, de -1.5 a 1.75 s.
      Trait gris : appui bouton median (-0.9 s, zone grisee : -1.0 a -0.8 s) ;
      pointilles : apparition du stimulus ; crochet : fenetre pre-stimulus.
  B - Decours mu (8-13 Hz) et beta basse (14-20 Hz), gauche et droite,
      toutes conditions confondues, +/- SEM.
  C - Lateralisation beta basse pre-stimulus (droite - gauche, -0.9 a -0.3 s)
      x effet du reel sur mu central droit (0.3-1.75 s) : Spearman, IC
      bootstrap, BF01 (comme le script 17).

Sortie : outputs_tfr/figures/Figure_S1_prestim.png / .pdf
"""

import os
import glob
import time

import numpy as np
import mne
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats, integrate

OUTPUT_DIR = "outputs_tfr"
TFR_DIR = os.path.join(OUTPUT_DIR, "tfr")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

SHOW_WIN = (-1.5, 1.75)
PRESS, PRESS_RANGE = -0.9, (-1.0, -0.8)
PRESTIM_WIN = (-0.9, -0.3)
MAIN_WIN = (0.3, 1.75)
BANDS = {"mu": (8, 13), "beta_low": (14, 20)}
BAND_LABEL = {"mu": "μ 8–13 Hz", "beta_low": "Low β 14–20 Hz"}
ROIS = {
    "Left central": ["E59", "E52", "E51", "E65", "E66", "E58"],
    "Right central": ["E183", "E184", "E196", "E182", "E164", "E195"],
}
ROI_COL = {"Left central": "#2a6db0", "Right central": "#c0392b"}
REAL, VIRT = ["REAL_APP", "REAL_DISG"], ["VIRT_APP", "VIRT_DISG"]
N_BOOT, SEED = 5000, 42

# ---------------- chargement (comme le script 12) ----------------
files = sorted(glob.glob(os.path.join(TFR_DIR, "*_tfr.npz")))
subjects, data, chs = [], [], []
freqs = times = conditions = None
for f in files:
    z = np.load(f, allow_pickle=False)
    if times is None:
        freqs, times, conditions = z["freqs"], z["times"], list(z["conditions"])
    subjects.append(os.path.basename(f).replace("_tfr.npz", ""))
    data.append(z["power_db"])
    chs.append([str(c) for c in z["ch_names"]])
common = [c for c in chs[0] if all(c in cs for cs in chs[1:])]
P = np.stack([d[:, [cs.index(c) for c in common]] for d, cs in zip(data, chs)])
ci = {c: i for i, c in enumerate(conditions)}
n = len(subjects)
print(f"{n} sujets")

roi_tf = {r: P[:, :, [common.index(c) for c in chans if c in common]].mean(axis=2)
          for r, chans in ROIS.items()}                       # (suj, cond, f, t)
show = (times >= SHOW_WIN[0]) & (times <= SHOW_WIN[1])


def band_power(roi, band, win=None, conds=None):
    X = roi_tf[roi][:, [ci[c] for c in (conds or conditions)]].mean(1)   # (suj, f, t)
    X = X[:, (freqs >= BANDS[band][0]) & (freqs <= BANDS[band][1])].mean(1)
    if win is None:
        return X                                                          # (suj, t)
    return X[:, (times >= win[0]) & (times <= win[1])].mean(1)           # (suj,)


def bf10_correlation(r, n):
    lik = lambda rho: (1 - rho ** 2) ** ((n - 1) / 2) / (1 - rho * r) ** ((2 * n - 3) / 2)
    return 0.5 * integrate.quad(lik, -1, 1, limit=200)[0]


def mark_events(ax, with_window=True):
    ax.axvspan(*PRESS_RANGE, color="0.85", alpha=.6, lw=0)
    ax.axvline(PRESS, color="0.35", lw=1)
    ax.axvline(0, color="k", lw=.8, ls=":")
    if with_window:
        ax.axvspan(*PRESTIM_WIN, color="gold", alpha=.12, lw=0)


# ---------------- figure ----------------
fig = plt.figure(figsize=(12, 8.6))
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1.05], hspace=0.38)

# A - cartes TF
gsA = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0], width_ratios=[1, 1, 0.04],
                                       wspace=0.15)
lim = 2.0
ext = [times[show][0], times[show][-1], freqs[0] - .5, freqs[-1] + .5]
for k, roi in enumerate(ROIS):
    ax = fig.add_subplot(gsA[k])
    M = roi_tf[roi].mean(axis=(0, 1))[:, show]
    im = ax.imshow(M, aspect="auto", origin="lower", extent=ext, cmap="RdBu_r",
                   vmin=-lim, vmax=lim)
    ax.axvline(PRESS, color="0.25", lw=1.2)
    ax.axvline(0, color="k", lw=.8, ls=":")
    for y in [8, 13, 20]:
        ax.axhline(y, color="0.3", lw=.5, ls="--")
    ax.annotate("", xy=(PRESTIM_WIN[0], freqs[-1] - 1), xytext=(PRESTIM_WIN[1], freqs[-1] - 1),
                arrowprops=dict(arrowstyle="|-|", color="k", lw=1, shrinkA=0, shrinkB=0))
    ax.text(PRESS, freqs[-1] + 1.2, "press", ha="center", fontsize=8, color="0.25")
    ax.text(0, freqs[-1] + 1.2, "onset", ha="center", fontsize=8)
    ax.set_title(f"{roi}, all conditions", fontsize=10, pad=16)
    ax.set_xlabel("Time from stimulus onset (s)")
    ax.set_ylabel("Frequency (Hz)" if k == 0 else "")
    if k == 1:
        ax.tick_params(labelleft=False)
    if k == 0:
        ax.text(-0.13, 1.1, "A", transform=ax.transAxes, fontsize=13, fontweight="bold")
cb = fig.colorbar(im, cax=fig.add_subplot(gsA[2]))
cb.set_label("Power (dB vs. baseline)", fontsize=8)

# B - decours ; C - nuage de points
gsB = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1], width_ratios=[1, 1, 0.9],
                                       wspace=0.32)
axes_b = []
for k, band in enumerate(BANDS):
    ax = fig.add_subplot(gsB[k], sharey=axes_b[0] if axes_b else None)
    axes_b.append(ax)
    mark_events(ax)
    for roi in ROIS:
        y = band_power(roi, band)[:, show]
        m, se = y.mean(0), y.std(0, ddof=1) / np.sqrt(n)
        ax.plot(times[show], m, color=ROI_COL[roi], lw=1.6, label=roi)
        ax.fill_between(times[show], m - se, m + se, color=ROI_COL[roi], alpha=.15, lw=0)
    ax.axhline(0, color="k", lw=.6)
    ax.set_title(BAND_LABEL[band], fontsize=10)
    ax.set_xlabel("Time from stimulus onset (s)")
    ax.set_ylabel("Power (dB vs. baseline)" if k == 0 else "")
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    if k == 0:
        ax.legend(frameon=False, fontsize=8, loc="lower left")
        ax.text(-0.2, 1.06, "B", transform=ax.transAxes, fontsize=13, fontweight="bold")

lat = band_power("Right central", "beta_low", PRESTIM_WIN) - \
    band_power("Left central", "beta_low", PRESTIM_WIN)
mu_r = band_power("Right central", "mu", MAIN_WIN, REAL) - \
    band_power("Right central", "mu", MAIN_WIN, VIRT)
rho, p = stats.spearmanr(lat, mu_r)
rng = np.random.default_rng(SEED)
boot = [stats.spearmanr(lat[i], mu_r[i]).statistic
        for i in (rng.integers(0, n, n) for _ in range(N_BOOT))]
lo, hi = np.nanpercentile(boot, [2.5, 97.5])
bf01 = 1 / bf10_correlation(rho, n)
t, pt = stats.ttest_1samp(lat, 0)
print(f"Lateralisation beta basse : {lat.mean():+.2f} ± {lat.std(ddof=1):.2f} dB, "
      f"t({n-1}) = {t:.2f}, p = {pt:.4f}, {(lat > 0).sum()}/{n} > 0")
print(f"rho = {rho:.3f} [{lo:.2f}, {hi:.2f}], p = {p:.3f}, BF01 = {bf01:.2f}")

axC = fig.add_subplot(gsB[2])
axC.scatter(lat, mu_r, s=28, color="#2a6db0", edgecolor="white", lw=.8, zorder=3)
kf, c0 = np.polyfit(lat, mu_r, 1)
xs = np.linspace(lat.min(), lat.max(), 50)
axC.plot(xs, kf * xs + c0, color="#2a6db0", lw=1.3)
axC.axhline(0, color="0.5", lw=.6)
axC.axvline(0, color="0.5", lw=.6)
ptxt = f"p = {p:.2f}".replace("0.", ".")
axC.text(0.03, 0.97, f"ρ = {rho:.2f} [{lo:.2f}, {hi:.2f}]\n{ptxt}, BF01 = {bf01:.1f}",
         transform=axC.transAxes, va="top", fontsize=8.5)
axC.set_xlabel("Pre-stimulus low-β lateralization,\nright − left central (dB)", fontsize=9)
axC.set_ylabel("μ power, real − picture,\nright central (dB)", fontsize=9)
for side in ["top", "right"]:
    axC.spines[side].set_visible(False)
axC.text(-0.3, 1.06, "C", transform=axC.transAxes, fontsize=13, fontweight="bold")

for ext_ in ["png", "pdf"]:
    fname = os.path.abspath(os.path.join(FIG_DIR, f"Figure_S1_prestim.{ext_}"))
    try:
        fig.savefig(fname, dpi=300, bbox_inches="tight")
    except OSError:                       # fichier ouvert dans une visionneuse
        fname = fname.replace(f".{ext_}", f"_{time.strftime('%H%M%S')}.{ext_}")
        fig.savefig(fname, dpi=300, bbox_inches="tight")
    print("Figure :", fname)
plt.close(fig)
